import unittest.mock

import pydantic
import pytest

import strands
from strands.models.openai_responses import OpenAIResponsesModel


@pytest.fixture
def openai_client_cls():
    with unittest.mock.patch.object(strands.models.openai_responses.openai, "AsyncOpenAI") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def openai_client(openai_client_cls):
    return openai_client_cls.return_value


@pytest.fixture
def model_id():
    return "gpt-4o-mini"


@pytest.fixture
def model(openai_client, model_id):
    _ = openai_client
    return OpenAIResponsesModel(model_id=model_id, params={"max_tokens": 100, "temperature": 0.7})


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "Hello world!"}]}]


@pytest.fixture
def tool_specs():
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                },
            },
        },
    ]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant."


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        message: str
        confidence: float

    return TestOutputModel


def test__init__(openai_client_cls, model_id):
    model = OpenAIResponsesModel({"api_key": "test-key"}, model_id=model_id, params={"temperature": 0.5})

    tru_config = model.get_config()
    exp_config = {"model_id": "gpt-4o-mini", "params": {"temperature": 0.5}}

    assert tru_config == exp_config
    openai_client_cls.assert_called_once_with(api_key="test-key")


def test_update_config(model, model_id):
    model.update_config(model_id=model_id, params={"temperature": 0.8})

    tru_config = model.get_config()
    assert tru_config["model_id"] == model_id
    assert tru_config["params"]["temperature"] == 0.8


@pytest.mark.parametrize(
    "content, exp_result",
    [
        # Document - should transform to input_file
        (
            {
                "document": {
                    "format": "pdf",
                    "name": "test doc",
                    "source": {"bytes": b"document"},
                },
            },
            {
                "file": {
                    "file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=",
                    "filename": "test doc",
                },
                "type": "input_file",  # Responses API uses input_file instead of file
            },
        ),
        # Image - should transform to input_image
        (
            {
                "image": {
                    "format": "jpg",
                    "source": {"bytes": b"image"},
                },
            },
            {
                "image_url": "data:image/jpeg;base64,aW1hZ2U=",
                "type": "input_image",
            },
        ),
        # Text - should transform to input_text
        (
            {"text": "hello"},
            {"type": "input_text", "text": "hello"},  # Responses API uses input_text instead of text
        ),
    ],
)
def test_format_request_message_content(content, exp_result):
    tru_result = OpenAIResponsesModel.format_request_message_content(content)
    assert tru_result == exp_result


def test_format_request_message_content_unsupported_type():
    content = {"unsupported": {}}

    with pytest.raises(TypeError, match="content_type=<unsupported> | unsupported type"):
        OpenAIResponsesModel.format_request_message_content(content)


def test_format_request_message_tool_call():
    tool_use = {
        "input": {"expression": "2+2"},
        "name": "calculator",
        "toolUseId": "c1",
    }

    tru_result = OpenAIResponsesModel.format_request_message_tool_call(tool_use)
    exp_result = {
        "type": "function_call",  # Responses API expects "function_call", not "function"
        "call_id": "c1",
        "name": "calculator",
        "arguments": '{"expression": "2+2"}',
    }
    assert tru_result == exp_result


def test_format_request_tool_message():
    tool_result = {
        "content": [{"text": "4"}, {"json": {"result": 4}}],
        "status": "success",
        "toolUseId": "c1",
    }

    tru_result = OpenAIResponsesModel.format_request_tool_message(tool_result)
    exp_result = {
        "type": "function_call_output",
        "call_id": "c1",
        "output": '4\n{"result": 4}',
    }
    assert tru_result == exp_result


def test_format_request_messages_simple(system_prompt):
    messages = [
        {
            "role": "user",
            "content": [{"text": "Hello!"}],
        },
    ]

    tru_result = OpenAIResponsesModel.format_request_messages(messages, system_prompt)
    exp_result = {
        "instructions": system_prompt,
        "input": "Hello!",  # Simple single text input
    }
    assert tru_result == exp_result


def test_format_request_messages_complex(system_prompt):
    messages = [
        {
            "role": "user",
            "content": [{"text": "Calculate this"}],
        },
        {
            "role": "assistant",
            "content": [
                {"text": "I'll help"},
                {
                    "toolUse": {
                        "input": {"expression": "2+2"},
                        "name": "calculator",
                        "toolUseId": "c1",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "c1", "status": "success", "content": [{"text": "4"}]}}],
        },
    ]

    tru_result = OpenAIResponsesModel.format_request_messages(messages, system_prompt)

    # Should be complex format with array input
    assert "instructions" in tru_result
    assert tru_result["instructions"] == system_prompt
    assert "input" in tru_result
    assert isinstance(tru_result["input"], list)

    # Check that function calls and outputs are properly formatted
    input_items = tru_result["input"]
    function_call_found = any(item.get("type") == "function_call" for item in input_items)  # Updated to "function_call"
    function_output_found = any(item.get("type") == "function_call_output" for item in input_items)
    assert function_call_found
    assert function_output_found


def test_format_request(model, messages, tool_specs, system_prompt):
    tru_request = model.format_request(messages, tool_specs, system_prompt)

    exp_request = {
        "model": "gpt-4o-mini",
        "stream": True,
        "temperature": 0.7,
        "max_output_tokens": 100,  # max_tokens should be transformed to max_output_tokens
        "instructions": system_prompt,
        "input": "Hello world!",
        "tools": [
            {
                "type": "function",
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                },
            },
        ],
    }
    assert tru_request == exp_request


def test_format_request_parameter_filtering(model, messages):
    # Test that unsupported parameters are filtered out
    model.config["params"] = {
        "temperature": 0.7,
        "max_tokens": 100,
        "presence_penalty": 0.1,  # Not supported by Responses API
        "frequency_penalty": 0.1,  # Not supported by Responses API
        "logprobs": True,  # Not supported by Responses API
    }

    tru_request = model.format_request(messages)

    # Should only include supported parameters
    assert "temperature" in tru_request
    assert "max_output_tokens" in tru_request  # Transformed from max_tokens
    assert "presence_penalty" not in tru_request
    assert "frequency_penalty" not in tru_request
    assert "logprobs" not in tru_request


@pytest.mark.asyncio
async def test_stream(openai_client, model, agenerator, alist):
    # Mock Responses API events
    mock_event_1 = unittest.mock.Mock(type="response.created")
    mock_event_2 = unittest.mock.Mock(type="response.in_progress")
    mock_event_3 = unittest.mock.Mock(type="response.output_item.added")
    mock_event_4 = unittest.mock.Mock(type="response.content_part.added")

    # Text delta events
    mock_event_5 = unittest.mock.Mock(type="response.output_text.delta", delta="Hello")
    mock_event_6 = unittest.mock.Mock(type="response.output_text.delta", delta=" world")
    mock_event_7 = unittest.mock.Mock(type="response.output_text.delta", delta="!")

    mock_event_8 = unittest.mock.Mock(type="response.output_text.done")
    mock_event_9 = unittest.mock.Mock(type="response.content_part.done")
    mock_event_10 = unittest.mock.Mock(type="response.output_item.done")

    # Final event with usage
    mock_usage = unittest.mock.Mock(input_tokens=26, output_tokens=4, total_tokens=30)
    mock_response = unittest.mock.Mock(usage=mock_usage)
    mock_event_11 = unittest.mock.Mock(type="response.completed", response=mock_response)

    openai_client.responses.create = unittest.mock.AsyncMock(
        return_value=agenerator(
            [
                mock_event_1,
                mock_event_2,
                mock_event_3,
                mock_event_4,
                mock_event_5,
                mock_event_6,
                mock_event_7,
                mock_event_8,
                mock_event_9,
                mock_event_10,
                mock_event_11,
            ]
        )
    )

    messages = [{"role": "user", "content": [{"text": "Say hello"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    # Expected events from our streaming implementation
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"delta": {"text": " world"}}},
        {"contentBlockDelta": {"delta": {"text": "!"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 26,
                    "outputTokens": 4,
                    "totalTokens": 30,
                },
                "metrics": {"latencyMs": 0},
            }
        },
    ]

    assert len(tru_events) == len(exp_events)
    for i, (tru_event, exp_event) in enumerate(zip(tru_events, exp_events, strict=False)):
        assert tru_event == exp_event, f"Event {i} mismatch"

    # Verify that responses.create was called with correct parameters
    expected_request = {
        "model": "gpt-4o-mini",
        "stream": True,
        "temperature": 0.7,
        "max_output_tokens": 100,
        "input": "Say hello",
        "tools": [],
    }
    openai_client.responses.create.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_with_function_calls(openai_client, model, tool_specs, agenerator, alist):
    # Mock the correct Responses API event sequence
    mock_tool_item = unittest.mock.Mock(type="function_call", name="calculator", call_id="call_123", id="fc_123")
    mock_item_added_event = unittest.mock.Mock(type="response.output_item.added", item=mock_tool_item)

    mock_arguments_delta_event = unittest.mock.Mock(
        type="response.function_call_arguments.delta", delta='{"expression": "2+2"}', item_id="fc_123"
    )

    mock_usage = unittest.mock.Mock(input_tokens=20, output_tokens=10, total_tokens=30)
    mock_response = unittest.mock.Mock(usage=mock_usage)
    mock_final_event = unittest.mock.Mock(type="response.completed", response=mock_response)

    openai_client.responses.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_item_added_event, mock_arguments_delta_event, mock_final_event])
    )

    messages = [{"role": "user", "content": [{"text": "Calculate 2+2"}]}]
    response = model.stream(messages, tool_specs)
    tru_events = await alist(response)

    # Should include function call events in the correct format from format_chunk
    tool_content_start_found = any(
        "contentBlockStart" in event and "toolUse" in event["contentBlockStart"].get("start", {})
        for event in tru_events
    )
    tool_content_delta_found = any(
        "contentBlockDelta" in event and "toolUse" in event["contentBlockDelta"].get("delta", {})
        for event in tru_events
    )
    tool_use_stop_reason = any(
        "messageStop" in event and event["messageStop"]["stopReason"] == "tool_use" for event in tru_events
    )

    assert tool_content_start_found
    assert tool_content_delta_found
    assert tool_use_stop_reason


@pytest.mark.asyncio
async def test_structured_output_responses_api(openai_client, model, test_output_model_cls):
    # Mock Responses API structured output response
    mock_parsed_output = test_output_model_cls(message="Hello", confidence=0.95)
    mock_response = unittest.mock.Mock(output_parsed=mock_parsed_output)
    openai_client.responses.parse = unittest.mock.AsyncMock(return_value=mock_response)

    openai_client.responses.create = unittest.mock.AsyncMock(return_value=mock_response)

    messages = [{"role": "user", "content": [{"text": "Generate a response"}]}]

    result_stream = model.structured_output(test_output_model_cls, messages)
    results = []
    async for result in result_stream:
        results.append(result)

    assert len(results) == 1
    assert "output" in results[0]
    assert isinstance(results[0]["output"], test_output_model_cls)
    assert results[0]["output"].message == "Hello"
    assert results[0]["output"].confidence == 0.95
