# ============================================================================
# LLM API Interface for Heval.jl
# ============================================================================

"""
    LLMConfig

Configuration for LLM API calls.
"""
Base.@kwdef struct LLMConfig
    api_key::String
    model::String = "gpt-4o"
    base_url::String = "https://api.openai.com/v1"
    max_tokens::Int = 4096
    temperature::Float64 = 0.1
end

"""
    tools_to_openai_format(tools)

Convert Tool objects to OpenAI function calling format.
"""
function _compact_json_schema(value)
    if value isa Dict
        compact = Dict{String, Any}()
        for (k, v) in pairs(value)
            key = String(k)
            if key == "description"
                continue
            end
            compact[key] = _compact_json_schema(v)
        end
        return compact
    elseif value isa AbstractVector
        return Any[_compact_json_schema(v) for v in value]
    end
    return value
end

function _compact_tool_description(desc::String)
    compact = replace(strip(desc), r"\s+" => " ")
    return length(compact) <= 140 ? compact : string(first(compact, 137), "...")
end

function tools_to_openai_format(tools::Vector{Tool})
    return [
        Dict(
            "type" => "function",
            "function" => Dict(
                "name" => t.name,
                "description" => _compact_tool_description(t.description),
                "parameters" => _compact_json_schema(t.parameters)
            )
        )
        for t in tools
    ]
end

"""
    messages_to_openai_format(messages)

Convert Message objects to OpenAI API format.
"""
function messages_to_openai_format(messages::Vector{Message})
    result = Dict{String, Any}[]

    for msg in messages
        d = Dict{String, Any}("role" => msg.role)

        if !isnothing(msg.content)
            d["content"] = msg.content
        elseif msg.role == "assistant"
            # OpenAI API requires explicit null content for assistant messages with tool_calls
            d["content"] = nothing
        end

        if !isnothing(msg.tool_calls)
            d["tool_calls"] = [
                Dict(
                    "id" => tc.id,
                    "type" => "function",
                    "function" => Dict(
                        "name" => tc.name,
                        "arguments" => JSON3.write(tc.arguments)
                    )
                )
                for tc in msg.tool_calls
            ]
        end

        if !isnothing(msg.tool_call_id)
            d["tool_call_id"] = msg.tool_call_id
        end

        push!(result, d)
    end

    return result
end

"""
    call_llm(config, messages, tools)

Make an API call to the LLM.
"""
function call_llm(config::LLMConfig, messages::Vector{Message}, tools::Vector{Tool})
    tools_spec = tools_to_openai_format(tools)
    messages_spec = messages_to_openai_format(messages)

    body = Dict(
        "model" => config.model,
        "messages" => messages_spec,
        "max_tokens" => config.max_tokens,
        "temperature" => config.temperature
    )

    if !isempty(tools)
        body["tools"] = tools_spec
        body["tool_choice"] = "auto"
    end

    headers = [
        "Authorization" => "Bearer $(config.api_key)",
        "Content-Type" => "application/json"
    ]

    response = HTTP.post(
        "$(config.base_url)/chat/completions",
        headers,
        JSON3.write(body);
        status_exception = true
    )

    return JSON3.read(response.body)
end

"""
    parse_llm_response(response)

Parse LLM response into Message and tool calls.
"""
function parse_llm_response(response)
    choice = response["choices"][1]
    message = choice["message"]

    content = get(message, "content", nothing)
    tool_calls = nothing

    if haskey(message, "tool_calls") && !isnothing(message["tool_calls"])
        tool_calls = ToolCall[]
        for tc in message["tool_calls"]
            push!(tool_calls, ToolCall(
                tc["id"],
                tc["function"]["name"],
                JSON3.read(tc["function"]["arguments"], Dict{String, Any})
            ))
        end
    end

    return Message("assistant", content, tool_calls, nothing)
end

"""
    _parse_sse_stream(io, on_token) -> String

Parse an OpenAI SSE stream, calling `on_token(delta)` for each content chunk.
Returns the accumulated full text.
"""
function _parse_sse_stream(io::IO, on_token::Function)
    accumulated = IOBuffer()
    while !eof(io)
        line = readline(io)
        isempty(line) && continue
        startswith(line, "data: ") || continue
        payload = line[7:end]
        payload == "[DONE]" && break
        try
            chunk = JSON3.read(payload)
            delta = get(chunk["choices"][1]["delta"], "content", nothing)
            if !isnothing(delta) && !isempty(delta)
                write(accumulated, delta)
                on_token(delta)
            end
        catch
            # Skip malformed chunks
        end
    end
    return String(take!(accumulated))
end

"""
    call_llm_streaming(config, messages, on_token) -> String

Make a streaming API call to the LLM (no tools). Calls `on_token(delta)` for
each text chunk. Returns the accumulated full text.

Falls back to non-streaming `call_llm` on any streaming error.
"""
function call_llm_streaming(config::LLMConfig, messages::Vector{Message},
                             on_token::Function)
    messages_spec = messages_to_openai_format(messages)

    body = Dict(
        "model" => config.model,
        "messages" => messages_spec,
        "max_tokens" => config.max_tokens,
        "temperature" => config.temperature,
        "stream" => true
    )

    headers = [
        "Authorization" => "Bearer $(config.api_key)",
        "Content-Type" => "application/json"
    ]

    try
        accumulated = ""
        HTTP.open("POST", "$(config.base_url)/chat/completions", headers;
                  body = JSON3.write(body), status_exception = true) do io
            accumulated = _parse_sse_stream(io, on_token)
        end
        return accumulated
    catch e
        @warn "Streaming failed, falling back to non-streaming call" exception=(e, catch_backtrace())
        response = call_llm(config, messages, Tool[])
        msg = parse_llm_response(response)
        text = something(msg.content, "")
        if !isempty(text)
            on_token(text)
        end
        return text
    end
end

"""
    format_tool_result(tool_call_id, result)

Create a tool result message.
"""
function format_tool_result(tool_call_id::String, result::Dict)
    return Message("tool", JSON3.write(result), nothing, tool_call_id)
end
