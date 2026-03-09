# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.calendar.app import (
    CalendarResourcesServer,
    CalendarResourcesServerConfig,
    CalendarVerifyRequest,
)


class TestApp:
    def _create_server(self):
        """Helper to create server instance."""
        config = CalendarResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        return CalendarResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_real_request(self, response_content, exp_cal_state, request_id=1, turn_index=None, total_turns=None):
        """Helper to create real request with NeMoGymResponse."""
        # Create real NeMoGymResponse object
        response = NeMoGymResponse(
            id=f"resp_test_{request_id}",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": f"msg_test_{request_id}",
                    "content": [
                        {
                            "annotations": [],
                            "text": response_content,
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        # Create real request object
        return CalendarVerifyRequest(
            id=request_id,
            exp_cal_state=exp_cal_state,
            responses_create_params={"input": []},
            response=response,
            turn_index=turn_index,
            total_turns=total_turns,
        )

    def _run_verify_test(self, real_request, expected_reward):
        """Helper to run verify method and check results."""
        server = self._create_server()

        # Run the actual verify method with real objects
        result = asyncio.run(server.verify(real_request))

        # Check the actual results
        assert result.reward == expected_reward

    def test_sanity(self) -> None:
        """Test that we can create a server instance."""
        config = CalendarResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        CalendarResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_valid_response_single_event(self):
        """Test valid response with correctly scheduled single event."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        response_content = '[{"event_id": 0, "start_time": "2pm", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_valid_response_multiple_events(self):
        """Test valid response with multiple correctly scheduled events."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "1": {
                "event_id": 1,
                "duration": 90,
                "constraint": "between 12pm and 2pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
        }
        response_content = (
            '[{"event_id": 0, "start_time": "2pm", "duration": 60}, '
            '{"event_id": 1, "start_time": "12pm", "duration": 90}]'
        )
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_think_tag_present(self):
        """Test that response with <think> tag gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        response_content = (
            '<think>Planning the schedule...</think>[{"event_id": 0, "start_time": "2pm", "duration": 60}]'
        )
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_no_events_expected(self):
        """Test valid response when no events are expected."""
        exp_cal_state = {}
        response_content = "No events to schedule."
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_invalid_json_extraction(self):
        """Test that invalid JSON gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        response_content = "This is not valid JSON at all"
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_empty_calendar_state(self):
        """Test that empty calendar state when events expected gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        response_content = "[]"
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_wrong_number_of_events_too_few(self):
        """Test that having fewer events than expected gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "1": {
                "event_id": 1,
                "duration": 90,
                "constraint": "between 12pm and 2pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
        }
        response_content = '[{"event_id": 0, "start_time": "2pm", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_wrong_number_of_events_too_many(self):
        """Test that having more events than expected gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        response_content = (
            '[{"event_id": 0, "start_time": "2pm", "duration": 60}, '
            '{"event_id": 1, "start_time": "12pm", "duration": 90}]'
        )
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_time_conflict_between_events(self):
        """Test that overlapping events get reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": None,
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "1": {
                "event_id": 1,
                "duration": 60,
                "constraint": None,
                "min_time": "10:00",
                "max_time": "16:00",
            },
        }
        # Events overlap: 2pm-3pm and 2:30pm-3:30pm
        response_content = (
            '[{"event_id": 0, "start_time": "2pm", "duration": 60}, '
            '{"event_id": 1, "start_time": "2:30pm", "duration": 60}]'
        )
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_constraint_before_violation(self):
        """Test that violating 'before' constraint gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 45,
                "constraint": "before 12:45pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event ends at 1pm, which is after 12:45pm
        response_content = '[{"event_id": 0, "start_time": "12:15pm", "duration": 45}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_constraint_before_valid(self):
        """Test that satisfying 'before' constraint gets reward 1."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 45,
                "constraint": "before 12:45pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event ends at 12:00pm, which is before 12:45pm
        response_content = '[{"event_id": 0, "start_time": "11:15am", "duration": 45}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_constraint_after_violation(self):
        """Test that violating 'after' constraint gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "after 2pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 1:30pm, which is before 2pm
        response_content = '[{"event_id": 0, "start_time": "1:30pm", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_constraint_after_valid(self):
        """Test that satisfying 'after' constraint gets reward 1."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "after 2pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 2:30pm, which is after 2pm
        response_content = '[{"event_id": 0, "start_time": "2:30pm", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_constraint_between_violation_starts_too_early(self):
        """Test that starting before the 'between' window gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 1:30pm, before 2pm
        response_content = '[{"event_id": 0, "start_time": "1:30pm", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_constraint_between_violation_ends_too_late(self):
        """Test that ending after the 'between' window gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 90,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 3pm and ends at 4:30pm, after 4pm
        response_content = '[{"event_id": 0, "start_time": "3pm", "duration": 90}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_constraint_between_valid(self):
        """Test that satisfying 'between' constraint gets reward 1."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 90,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 2pm and ends at 3:30pm, within window
        response_content = '[{"event_id": 0, "start_time": "2pm", "duration": 90}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_constraint_at_violation(self):
        """Test that not starting at exact time for 'at' constraint gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 30,
                "constraint": "at 12:15pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 12:30pm instead of 12:15pm
        response_content = '[{"event_id": 0, "start_time": "12:30pm", "duration": 30}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_constraint_at_valid(self):
        """Test that starting at exact time for 'at' constraint gets reward 1."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 30,
                "constraint": "at 12:15pm",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts exactly at 12:15pm
        response_content = '[{"event_id": 0, "start_time": "12:15pm", "duration": 30}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_event_starts_before_min_time(self):
        """Test that event starting before min_time gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": None,
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 9am, before min_time of 10am
        response_content = '[{"event_id": 0, "start_time": "9am", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_event_ends_after_max_time(self):
        """Test that event ending after max_time gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": None,
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event starts at 3:30pm and ends at 4:30pm, after max_time of 4pm (16:00)
        response_content = '[{"event_id": 0, "start_time": "3:30pm", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_wrong_duration(self):
        """Test that wrong event duration gets reward 0."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": None,
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        # Event has duration 90 instead of expected 60
        response_content = '[{"event_id": 0, "start_time": "2pm", "duration": 90}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 0)

    def test_constraint_none_valid(self):
        """Test that event with no constraint (None) is valid when in time window."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": None,
                "min_time": "10:00",
                "max_time": "16:00",
            }
        }
        response_content = '[{"event_id": 0, "start_time": "2pm", "duration": 60}]'
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)

    def test_complex_multi_event_schedule(self):
        """Test complex schedule with multiple events and different constraints."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "1": {
                "event_id": 1,
                "duration": 90,
                "constraint": "between 12pm and 2pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "2": {
                "event_id": 2,
                "duration": 45,
                "constraint": "before 12pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "3": {
                "event_id": 3,
                "duration": 30,
                "constraint": "at 10:00",
                "min_time": "10:00",
                "max_time": "16:00",
            },
        }
        response_content = (
            '[{"event_id": 0, "start_time": "2:30pm", "duration": 60}, '
            '{"event_id": 1, "start_time": "12:30pm", "duration": 90}, '
            '{"event_id": 2, "start_time": "10:30am", "duration": 45}, '
            '{"event_id": 3, "start_time": "10:00", "duration": 30}]'
        )
        real_request = self._create_real_request(response_content, exp_cal_state)
        self._run_verify_test(real_request, 1)


class TestPerTurnVerify:
    """Tests for per-turn verification in multi-turn mode."""

    def _create_server(self):
        config = CalendarResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        return CalendarResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_request(self, response_content, exp_cal_state, turn_index=None, total_turns=None):
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": response_content, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        return CalendarVerifyRequest(
            id=1,
            exp_cal_state=exp_cal_state,
            responses_create_params={"input": []},
            response=response,
            turn_index=turn_index,
            total_turns=total_turns,
        )

    def test_intermediate_turn_valid_structure(self):
        """Intermediate turn with valid calendar structure should get reward 1."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 10am and 11:45am",
                "min_time": "10:00",
                "max_time": "16:00",
            },
            "1": {
                "event_id": 1,
                "duration": 60,
                "constraint": "before 1:45pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
        }
        # Turn 0: only event 0 exists, no constraint check needed
        response_content = '[{"event_id": 0, "event_name": "Meeting", "start_time": "10:00", "duration": 60}]'
        request = self._create_request(response_content, exp_cal_state, turn_index=0, total_turns=5)
        server = self._create_server()
        result = asyncio.run(server.verify(request))
        assert result.reward == 1

    def test_intermediate_turn_conflict_fails(self):
        """Intermediate turn with conflicting events should get reward 0."""
        exp_cal_state = {
            "0": {"event_id": 0, "duration": 60, "constraint": None, "min_time": "10:00", "max_time": "16:00"},
            "1": {"event_id": 1, "duration": 60, "constraint": None, "min_time": "10:00", "max_time": "16:00"},
        }
        response_content = (
            '[{"event_id": 0, "start_time": "10:00", "duration": 60}, '
            '{"event_id": 1, "start_time": "10:30", "duration": 60}]'
        )
        request = self._create_request(response_content, exp_cal_state, turn_index=1, total_turns=5)
        server = self._create_server()
        result = asyncio.run(server.verify(request))
        assert result.reward == 0

    def test_intermediate_turn_wrong_duration_fails(self):
        """Intermediate turn with wrong duration should get reward 0."""
        exp_cal_state = {
            "0": {"event_id": 0, "duration": 60, "constraint": None, "min_time": "10:00", "max_time": "16:00"},
        }
        response_content = '[{"event_id": 0, "start_time": "10:00", "duration": 90}]'
        request = self._create_request(response_content, exp_cal_state, turn_index=0, total_turns=3)
        server = self._create_server()
        result = asyncio.run(server.verify(request))
        assert result.reward == 0

    def test_intermediate_turn_out_of_bounds_fails(self):
        """Intermediate turn with event out of time bounds should get reward 0."""
        exp_cal_state = {
            "0": {"event_id": 0, "duration": 60, "constraint": None, "min_time": "10:00", "max_time": "16:00"},
        }
        response_content = '[{"event_id": 0, "start_time": "09:00", "duration": 60}]'
        request = self._create_request(response_content, exp_cal_state, turn_index=0, total_turns=3)
        server = self._create_server()
        result = asyncio.run(server.verify(request))
        assert result.reward == 0

    def test_intermediate_turn_text_only_response(self):
        """Intermediate turn with text-only response (no JSON) should pass."""
        exp_cal_state = {
            "0": {"event_id": 0, "duration": 60, "constraint": None, "min_time": "10:00", "max_time": "16:00"},
        }
        response_content = "Sure, I can help with that! Let me add it to your calendar."
        request = self._create_request(response_content, exp_cal_state, turn_index=0, total_turns=5)
        server = self._create_server()
        result = asyncio.run(server.verify(request))
        assert result.reward == 1

    def test_final_turn_uses_full_verification(self):
        """Last turn should use full constraint verification."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
        }
        # Event at 10am violates "between 2pm and 4pm" constraint
        response_content = '[{"event_id": 0, "start_time": "10:00", "duration": 60}]'

        # As final turn (turn_index == total_turns - 1): should fail
        request_final = self._create_request(response_content, exp_cal_state, turn_index=4, total_turns=5)
        server = self._create_server()
        result_final = asyncio.run(server.verify(request_final))
        assert result_final.reward == 0

        # As intermediate turn: should pass (structural check only)
        request_intermediate = self._create_request(response_content, exp_cal_state, turn_index=1, total_turns=5)
        result_intermediate = asyncio.run(server.verify(request_intermediate))
        assert result_intermediate.reward == 1

    def test_no_turn_index_uses_full_verification(self):
        """Request without turn_index should use full verification (backwards compat)."""
        exp_cal_state = {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "between 2pm and 4pm",
                "min_time": "10:00",
                "max_time": "16:00",
            },
        }
        response_content = '[{"event_id": 0, "start_time": "10:00", "duration": 60}]'
        request = self._create_request(response_content, exp_cal_state)
        server = self._create_server()
        result = asyncio.run(server.verify(request))
        assert result.reward == 0
