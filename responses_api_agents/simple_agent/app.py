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
import logging
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    multi_turn: bool = False
    return_transitions: bool = False


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    all_turns: Optional[List[Dict[str, Any]]] = None
    total_turns: int = 0


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig

    @staticmethod
    def _split_conversation_turns(input_messages: list) -> tuple:
        """Split multi-turn conversation input into system context and per-turn user messages.

        For multi-turn conversations (e.g. calendar), the input contains interleaved
        user and assistant messages: [system, user1, asst1, user2, asst2, ..., userN].
        This method extracts the system/developer prefix and the list of user messages.
        Ground-truth assistant messages are skipped so the model generates each turn on-policy.

        Returns:
            (system_messages, user_messages): Lists of message dicts.
        """
        system_messages = []
        user_messages = []
        found_first_user = False

        for msg in input_messages:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role in ("system", "developer") and not found_first_user:
                system_messages.append(msg)
            elif role == "user":
                found_first_user = True
                user_messages.append(msg)
            # Skip assistant messages - model will generate these on-policy

        return system_messages, user_messages

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            # We raise for status here since we expect model calls to always work.
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)

            if not usage:
                usage = model_response.usage

            if usage:
                usage.input_tokens += model_response.usage.input_tokens
                usage.output_tokens += model_response.usage.output_tokens
                usage.total_tokens += model_response.usage.total_tokens

                # TODO support more advanced token details
                usage.input_tokens_details.cached_tokens = 0
                usage.output_tokens_details.reasoning_tokens = 0

            if model_response.incomplete_details and model_response.incomplete_details.reason == "max_output_tokens":
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                    cookies=resources_server_cookies,
                )
                # We don't raise for status here since it's a valid return for the API to error e.g. if the model outputs an invalid call or something.
                resources_server_cookies = api_response.cookies

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=(await api_response.content.read()).decode(),
                )
                new_outputs.append(tool_response)

            # Check if max steps is not None and if we have exhausted it.
            if self.config.max_steps and step >= self.config.max_steps:
                break

        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)

        model_response.output = new_outputs
        model_response.usage = usage
        return model_response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        all_turns: List[Dict[str, Any]] = []
        final_response_json = None

        if self.config.multi_turn:
            # Multi-turn conversation mode: split the conversation history into individual turns
            # and generate each assistant response on-policy. This enables per-turn RL training
            # for benchmarks like calendar where input has [system, user1, asst1, user2, ..., userN].
            params_dict = body.responses_create_params.model_dump()
            system_msgs, user_msgs = self._split_conversation_turns(params_dict.get("input", []))

            if not user_msgs:
                raise ValueError("No user messages found in multi-turn input")

            accumulated = list(system_msgs)

            for turn_idx, user_msg in enumerate(user_msgs):
                current_input = accumulated + [user_msg]
                current_params = {**params_dict, "input": current_input}

                LOG.info("Multi-turn: generating turn %d/%d", turn_idx + 1, len(user_msgs))

                resp = await self.server_client.post(
                    server_name=self.config.name,
                    url_path="/v1/responses",
                    json=current_params,
                    cookies=cookies,
                )
                await raise_for_status(resp)
                cookies = resp.cookies
                response_json = await get_response_json(resp)

                all_turns.append(
                    {
                        "turn_index": turn_idx,
                        "input": current_input,
                        "response": response_json,
                    }
                )

                # Use model's own output as context for next turn (on-policy generation)
                accumulated = current_input + response_json.get("output", [])
                final_response_json = response_json
        else:
            # Single-turn mode (original behavior): pass full input to /v1/responses
            # which handles the tool-calling loop internally via responses().
            resp = await self.server_client.post(
                server_name=self.config.name,
                url_path="/v1/responses",
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(resp)
            cookies = resp.cookies
            final_response_json = await get_response_json(resp)

            all_turns.append(
                {
                    "turn_index": 0,
                    "response": final_response_json,
                }
            )

        verify_request = SimpleAgentVerifyRequest.model_validate(body.model_dump() | {"response": final_response_json})

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        result = SimpleAgentVerifyResponse.model_validate(await get_response_json(verify_response))

        if self.config.return_transitions:
            result.all_turns = all_turns
            result.total_turns = len(all_turns)

        return result


if __name__ == "__main__":
    SimpleAgent.run_webserver()
