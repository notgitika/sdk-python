"""OpenAI Responses API model provider.

- Docs: https://platform.openai.com/docs/api-reference/responses
"""

import json
import logging
from typing import Any, AsyncGenerator, Optional, Type, TypeVar, Union

import openai  # noqa: F401
from pydantic import BaseModel
from typing_extensions import override

from ..types.content import ContentBlock, Messages
from ..types.streaming import StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse
from .openai import OpenAIModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIResponsesModel(OpenAIModel):
    """OpenAI Responses API model provider implementation."""

    @classmethod
    def format_request_message_content(cls, content: ContentBlock, message_role: str = "user") -> dict[str, Any]:
        """Format a Responses API compatible content block."""
        try:
            base_content = super().format_request_message_content(content)
            if message_role == "assistant":
                type_mapping = {
                    "text": "output_text",
                }
            else:
                type_mapping = {"text": "input_text", "image_url": "input_image", "file": "input_file"}

            if base_content["type"] in type_mapping:
                base_content["type"] = type_mapping[base_content["type"]]

            if base_content["type"] == "input_image" and isinstance(base_content.get("image_url"), dict):
                if "url" in base_content["image_url"]:
                    base_content["image_url"] = base_content["image_url"]["url"]

            return base_content

        except Exception as e:
            raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type") from e

    @classmethod
    def format_request_message_tool_call(cls, tool_use: ToolUse) -> dict[str, Any]:
        """Format a Responses API compatible function call."""
        return {
            "type": "function_call",
            "call_id": tool_use["toolUseId"],
            "name": tool_use["name"],
            "arguments": json.dumps(tool_use["input"]),
        }

    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Responses API compatible function call output."""
        output_parts = []
        for content in tool_result["content"]:
            if "json" in content:
                output_parts.append(json.dumps(content["json"]))
            elif "text" in content:
                output_parts.append(content["text"])

        return {
            "type": "function_call_output",
            "call_id": tool_result["toolUseId"],
            "output": "\n".join(output_parts),
        }

    def _filter_params_for_responses_api(self, include_max_tokens: bool = True) -> dict[str, Any]:
        """Filter and transform parameters for Responses API endpoints."""
        params = self.config.get("params", {})
        if not isinstance(params, dict):
            params = {}
        supported_params: dict[str, Any] = {}

        if include_max_tokens:
            param_mapping = {
                "max_tokens": "max_output_tokens",  # Transform to Responses API format
                "temperature": "temperature",
                "top_p": "top_p",
                "stream": "stream",
            }
        else:
            # For responses.parse (doesn't support max_tokens)
            param_mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                # Note: max_tokens not supported by responses.parse
            }

        for old_param, new_param in param_mapping.items():
            if old_param in params:
                supported_params[new_param] = params[old_param]

        return supported_params

    @classmethod
    def format_request_messages(cls, messages: Messages, system_prompt: Optional[str] = None) -> dict[str, Any]:  # type: ignore[override]
        """Format messages for the Responses API."""
        result: dict[str, Any] = {}

        if system_prompt:
            result["instructions"] = system_prompt

        input_items: list[dict[str, Any]] = []

        for message in messages:
            if message["role"] in ["user", "assistant"]:
                regular_content = []
                tool_calls = []
                tool_results = []

                for content in message["content"]:
                    if "toolUse" in content:
                        tool_calls.append(cls.format_request_message_tool_call(content["toolUse"]))
                    elif "toolResult" in content:
                        tool_results.append(cls.format_request_tool_message(content["toolResult"]))
                    else:
                        # Pass the message role to format_request_message_content
                        regular_content.append(cls.format_request_message_content(content, message["role"]))

                if regular_content:
                    input_items.append({"type": "message", "role": message["role"], "content": regular_content})

                input_items.extend(tool_calls)
                input_items.extend(tool_results)

        if (
            len(input_items) == 1
            and input_items[0].get("type") == "message"
            and isinstance(input_items[0].get("content"), list)
            and len(input_items[0]["content"]) == 1
        ):
            content_list = input_items[0]["content"]
            if isinstance(content_list, list) and len(content_list) > 0:
                content_item = content_list[0]
                if isinstance(content_item, dict) and content_item.get("type") == "input_text":
                    result["input"] = content_item.get("text", "")
                else:
                    result["input"] = input_items
            else:
                result["input"] = input_items
        elif input_items:
            result["input"] = input_items

        return result

    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a Responses API compatible request."""
        message_data = self.format_request_messages(messages, system_prompt)

        request = {
            "model": self.config["model_id"],
            "stream": True,  # Default to streaming
        }

        # Add message data
        if "instructions" in message_data:
            request["instructions"] = message_data["instructions"]

        if "input" in message_data:
            request["input"] = message_data["input"]

        supported_params = self._filter_params_for_responses_api(include_max_tokens=True)
        request.update(supported_params)

        # Always include tools (empty list if none provided)
        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "parameters": tool_spec["inputSchema"]["json"],
                }
                for tool_spec in tool_specs
            ]
        else:
            request["tools"] = []

        return request

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the OpenAI Responses API model."""
        request = self.format_request(messages, tool_specs, system_prompt)
        response = await self.client.responses.create(**request)

        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        tool_calls: dict[str, dict[str, Any]] = {}
        final_usage = None
        has_text_content = False

        async for event in response:
            if hasattr(event, "type"):
                if event.type == "response.output_text.delta":
                    # Text content streaming
                    if hasattr(event, "delta") and isinstance(event.delta, str):
                        has_text_content = True
                        yield self.format_chunk(
                            {"chunk_type": "content_delta", "data_type": "text", "data": event.delta}
                        )

                elif event.type == "response.output_item.added":
                    # Tool call started
                    if hasattr(event, "item") and hasattr(event.item, "type") and event.item.type == "function_call":
                        call_id = getattr(event.item, "call_id", "unknown")
                        tool_calls[call_id] = {
                            "name": getattr(event.item, "name", ""),
                            "arguments": "",
                            "call_id": call_id,
                            "item_id": getattr(event.item, "id", ""),
                        }

                elif event.type == "response.function_call_arguments.delta":
                    # Tool arguments streaming - match by item_id
                    if hasattr(event, "delta") and hasattr(event, "item_id"):
                        for _call_id, call_info in tool_calls.items():
                            if call_info["item_id"] == event.item_id:
                                call_info["arguments"] += event.delta
                                break

                elif event.type == "response.completed":
                    # Response complete
                    if hasattr(event, "response") and hasattr(event.response, "usage"):
                        final_usage = event.response.usage
                    break

        # Close text content if we had any
        if has_text_content:
            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

        # Yield tool calls if any
        for call_info in tool_calls.values():
            mock_tool_call = type(
                "MockToolCall",
                (),
                {
                    "function": type(
                        "MockFunction", (), {"name": call_info["name"], "arguments": call_info["arguments"]}
                    )(),
                    "id": call_info["call_id"],
                },
            )()

            yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call})
            yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call})
            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        finish_reason = "tool_calls" if tool_calls else "stop"
        yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason})

        if final_usage:
            usage_data = type(
                "Usage",
                (),
                {
                    "prompt_tokens": getattr(final_usage, "input_tokens", 0),
                    "completion_tokens": getattr(final_usage, "output_tokens", 0),
                    "total_tokens": getattr(final_usage, "total_tokens", 0),
                },
            )()
            yield self.format_chunk({"chunk_type": "metadata", "data": usage_data})

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model using Responses API."""
        request = self.format_request_messages(prompt, system_prompt)

        supported_params = self._filter_params_for_responses_api(include_max_tokens=False)

        request.update(
            {
                "model": self.config["model_id"],
                **supported_params,
            }
        )

        parsed_response = await self.client.responses.parse(text_format=output_model, **request)

        parsed = parsed_response.output_parsed
        if not parsed:
            raise ValueError("No structured output found in Responses API response.")
        yield {"output": parsed}
