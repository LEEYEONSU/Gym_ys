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
import json
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    SimpleAgent,
    SimpleAgentConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.return_value = json.dumps(mock_response_data)
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200
        server.server_client.post.assert_called_with(
            server_name="my server name",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
            ),
            cookies=None,
        )

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert expected_responses_dict == actual_responses_dict

    async def test_responses_continues_on_reasoning_only(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_reasoning_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "status": "completed",
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        mock_response_chat_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(mock_response_reasoning_data), json.dumps(mock_response_chat_data)]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        expected_calls = [
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
                ),
                cookies=None,
            ),
            call().ok.__bool__(),
            call().read(),
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        NeMoGymEasyInputMessage(content="hello", role="user", type="message"),
                        NeMoGymResponseReasoningItem(
                            id="msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                            summary=[NeMoGymSummary(text="I'm thinking how to respond", type="summary_text")],
                            type="reasoning",
                            encrypted_content=None,
                            status="completed",
                        ),
                    ]
                ),
                cookies=dotjson_mock.cookies,
            ),
            call().ok.__bool__(),
            call().read(),
            call().cookies.items(),
            call().cookies.items().__iter__(),
            call().cookies.items().__len__(),
        ]
        server.server_client.post.assert_has_calls(expected_calls)

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "encrypted_content": None,
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "type": "reasoning",
                },
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert expected_responses_dict == actual_responses_dict


class TestSplitConversationTurns:
    def test_basic_multi_turn(self):
        input_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Add event A"},
            {"role": "assistant", "content": "Done. Calendar: [A]"},
            {"role": "user", "content": "Add event B"},
            {"role": "assistant", "content": "Done. Calendar: [A, B]"},
            {"role": "user", "content": "Move event A"},
        ]
        system_msgs, user_msgs = SimpleAgent._split_conversation_turns(input_messages)
        assert len(system_msgs) == 1
        assert system_msgs[0]["role"] == "system"
        assert len(user_msgs) == 3
        assert user_msgs[0]["content"] == "Add event A"
        assert user_msgs[1]["content"] == "Add event B"
        assert user_msgs[2]["content"] == "Move event A"

    def test_single_turn(self):
        input_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system_msgs, user_msgs = SimpleAgent._split_conversation_turns(input_messages)
        assert len(system_msgs) == 1
        assert len(user_msgs) == 1

    def test_no_system_message(self):
        input_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        system_msgs, user_msgs = SimpleAgent._split_conversation_turns(input_messages)
        assert len(system_msgs) == 0
        assert len(user_msgs) == 2

    def test_multiple_system_messages(self):
        input_messages = [
            {"role": "system", "content": "System 1"},
            {"role": "developer", "content": "Developer msg"},
            {"role": "user", "content": "Hello"},
        ]
        system_msgs, user_msgs = SimpleAgent._split_conversation_turns(input_messages)
        assert len(system_msgs) == 2
        assert len(user_msgs) == 1


class TestMultiTurnConfig:
    def test_config_defaults(self):
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        assert config.multi_turn is False
        assert config.return_transitions is False

    def test_config_multi_turn_enabled(self):
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            multi_turn=True,
            return_transitions=True,
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        assert config.multi_turn is True
        assert config.return_transitions is True


class TestMultiTurnRun:
    @pytest.fixture
    def mock_model_response(self):
        return {
            "id": "resp_test",
            "created_at": 1000.0,
            "model": "test_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "Model response", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

    @pytest.fixture
    def mock_verify_response(self):
        return {
            "reward": 1.0,
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp_test",
                "created_at": 1000.0,
                "model": "test_model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }

    @pytest.fixture
    def mock_seed_response(self):
        return {}

    async def test_multi_turn_run_splits_turns(self, mock_model_response, mock_verify_response, mock_seed_response):
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="test_agent",
            multi_turn=True,
            return_transitions=True,
            resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))

        seed_mock = AsyncMock()
        seed_mock.read.return_value = json.dumps(mock_seed_response)
        seed_mock.cookies = MagicMock()
        seed_mock.cookies.items.return_value = []

        model_mock = AsyncMock()
        model_mock.read.return_value = json.dumps(mock_model_response)
        model_mock.cookies = MagicMock()
        model_mock.cookies.items.return_value = []

        verify_mock = AsyncMock()
        verify_mock.read.return_value = json.dumps(mock_verify_response)
        verify_mock.cookies = MagicMock()
        verify_mock.cookies.items.return_value = []

        server.server_client.post = AsyncMock(side_effect=[seed_mock, model_mock, model_mock, verify_mock])

        mock_request = MagicMock()
        mock_request.cookies = {}

        from responses_api_agents.simple_agent.app import SimpleAgentRunRequest

        body = SimpleAgentRunRequest(
            responses_create_params={
                "input": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Add event A"},
                    {"role": "assistant", "content": "Done."},
                    {"role": "user", "content": "Add event B"},
                ]
            }
        )

        result = await server.run(mock_request, body)
        assert result.reward == 1.0
        assert result.all_turns is not None
        assert len(result.all_turns) == 2
        assert result.total_turns == 2
        assert result.all_turns[0]["turn_index"] == 0
        assert result.all_turns[1]["turn_index"] == 1

    async def test_single_turn_run_returns_transitions(self, mock_model_response, mock_verify_response):
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="test_agent",
            multi_turn=False,
            return_transitions=True,
            resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))

        seed_mock = AsyncMock()
        seed_mock.read.return_value = json.dumps({})
        seed_mock.cookies = MagicMock()
        seed_mock.cookies.items.return_value = []

        model_mock = AsyncMock()
        model_mock.read.return_value = json.dumps(mock_model_response)
        model_mock.cookies = MagicMock()
        model_mock.cookies.items.return_value = []

        verify_mock = AsyncMock()
        verify_mock.read.return_value = json.dumps(mock_verify_response)
        verify_mock.cookies = MagicMock()
        verify_mock.cookies.items.return_value = []

        server.server_client.post = AsyncMock(side_effect=[seed_mock, model_mock, verify_mock])

        mock_request = MagicMock()
        mock_request.cookies = {}

        from responses_api_agents.simple_agent.app import SimpleAgentRunRequest

        body = SimpleAgentRunRequest(
            responses_create_params={
                "input": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        result = await server.run(mock_request, body)
        assert result.reward == 1.0
        assert result.all_turns is not None
        assert len(result.all_turns) == 1
        assert result.total_turns == 1

    async def test_single_turn_no_transitions_when_disabled(self, mock_model_response, mock_verify_response):
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="test_agent",
            multi_turn=False,
            return_transitions=False,
            resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))

        seed_mock = AsyncMock()
        seed_mock.read.return_value = json.dumps({})
        seed_mock.cookies = MagicMock()
        seed_mock.cookies.items.return_value = []

        model_mock = AsyncMock()
        model_mock.read.return_value = json.dumps(mock_model_response)
        model_mock.cookies = MagicMock()
        model_mock.cookies.items.return_value = []

        verify_mock = AsyncMock()
        verify_mock.read.return_value = json.dumps(mock_verify_response)
        verify_mock.cookies = MagicMock()
        verify_mock.cookies.items.return_value = []

        server.server_client.post = AsyncMock(side_effect=[seed_mock, model_mock, verify_mock])

        mock_request = MagicMock()
        mock_request.cookies = {}

        from responses_api_agents.simple_agent.app import SimpleAgentRunRequest

        body = SimpleAgentRunRequest(
            responses_create_params={
                "input": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        result = await server.run(mock_request, body)
        assert result.reward == 1.0
        assert result.all_turns is None

    async def test_multi_turn_per_turn_verify(self, mock_model_response, mock_seed_response):
        """Test that per_turn_verify calls /verify after each turn and records per-turn rewards."""
        per_turn_verify_response_turn0 = {
            "reward": 0.5,
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp_test",
                "created_at": 1000.0,
                "model": "test_model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        per_turn_verify_response_turn1 = {
            "reward": 1.0,
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp_test",
                "created_at": 1000.0,
                "model": "test_model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        final_verify_response = {
            "reward": 1.0,
            "responses_create_params": {"input": []},
            "response": {
                "id": "resp_test",
                "created_at": 1000.0,
                "model": "test_model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }

        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="test_agent",
            multi_turn=True,
            per_turn_verify=True,
            return_transitions=True,
            resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))

        seed_mock = AsyncMock()
        seed_mock.read.return_value = json.dumps(mock_seed_response)
        seed_mock.cookies = MagicMock()
        seed_mock.cookies.items.return_value = []

        model_mock = AsyncMock()
        model_mock.read.return_value = json.dumps(mock_model_response)
        model_mock.cookies = MagicMock()
        model_mock.cookies.items.return_value = []

        verify_turn0_mock = AsyncMock()
        verify_turn0_mock.read.return_value = json.dumps(per_turn_verify_response_turn0)
        verify_turn0_mock.cookies = MagicMock()
        verify_turn0_mock.cookies.items.return_value = []

        verify_turn1_mock = AsyncMock()
        verify_turn1_mock.read.return_value = json.dumps(per_turn_verify_response_turn1)
        verify_turn1_mock.cookies = MagicMock()
        verify_turn1_mock.cookies.items.return_value = []

        final_verify_mock = AsyncMock()
        final_verify_mock.read.return_value = json.dumps(final_verify_response)
        final_verify_mock.cookies = MagicMock()
        final_verify_mock.cookies.items.return_value = []

        # Call order: seed, model(turn0), verify(turn0), model(turn1), verify(turn1), final_verify
        server.server_client.post = AsyncMock(
            side_effect=[
                seed_mock,
                model_mock,
                verify_turn0_mock,
                model_mock,
                verify_turn1_mock,
                final_verify_mock,
            ]
        )

        mock_request = MagicMock()
        mock_request.cookies = {}

        from responses_api_agents.simple_agent.app import SimpleAgentRunRequest

        body = SimpleAgentRunRequest(
            responses_create_params={
                "input": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Add event A"},
                    {"role": "assistant", "content": "Done."},
                    {"role": "user", "content": "Add event B"},
                ]
            }
        )

        result = await server.run(mock_request, body)

        # Final reward from the last /verify
        assert result.reward == 1.0
        assert result.all_turns is not None
        assert len(result.all_turns) == 2
        assert result.total_turns == 2

        # Per-turn rewards
        assert result.all_turns[0]["reward"] == 0.5
        assert result.all_turns[0]["verify_response"]["reward"] == 0.5
        assert result.all_turns[1]["reward"] == 1.0
        assert result.all_turns[1]["verify_response"]["reward"] == 1.0

        # Verify that /verify was called with turn_index and total_turns
        verify_calls = [c for c in server.server_client.post.call_args_list if c.kwargs.get("url_path") == "/verify"]
        assert len(verify_calls) == 3  # 2 per-turn + 1 final
        assert verify_calls[0].kwargs["json"]["turn_index"] == 0
        assert verify_calls[0].kwargs["json"]["total_turns"] == 2
        assert verify_calls[1].kwargs["json"]["turn_index"] == 1
        assert verify_calls[1].kwargs["json"]["total_turns"] == 2
