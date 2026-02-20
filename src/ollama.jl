# ============================================================================
# Ollama LLM Interface for Heval.jl
# ============================================================================

"""
    OllamaConfig

Configuration for Ollama API calls.

Supports both the native Ollama API (`/api/chat`) and the OpenAI-compatible
endpoint (`/v1/chat/completions`).

# Fields
- `model::String` - Model name (e.g., "llama3.1", "qwen2.5", "mistral")
- `host::String` - Ollama server address (default: "http://localhost:11434")
- `max_tokens::Int` - Maximum tokens in response (default: 4096)
- `temperature::Float64` - Sampling temperature (default: 0.1)
- `use_openai_compat::Bool` - Use OpenAI-compatible endpoint (default: false)
"""
Base.@kwdef struct OllamaConfig
    model::String = "llama3.1"
    host::String = "http://localhost:11434"
    max_tokens::Int = 4096
    temperature::Float64 = 0.1
    use_openai_compat::Bool = false
end

# ============================================================================
# Native Ollama API
# ============================================================================

"""
    tools_to_ollama_format(tools)

Convert Tool objects to Ollama's native tool calling format.
"""
function tools_to_ollama_format(tools::Vector{Tool})
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
    messages_to_ollama_format(messages)

Convert Message objects to Ollama's native message format.
"""
function messages_to_ollama_format(messages::Vector{Message})
    result = Dict{String, Any}[]

    for msg in messages
        d = Dict{String, Any}("role" => msg.role)

        if !isnothing(msg.content)
            d["content"] = msg.content
        elseif msg.role == "assistant"
            d["content"] = ""
        end

        if !isnothing(msg.tool_calls)
            d["tool_calls"] = [
                Dict(
                    "function" => Dict(
                        "name" => tc.name,
                        "arguments" => tc.arguments
                    )
                )
                for tc in msg.tool_calls
            ]
        end

        push!(result, d)
    end

    return result
end

"""
    call_ollama(config, messages, tools)

Make an API call to Ollama using the native `/api/chat` endpoint.
"""
function call_ollama(config::OllamaConfig, messages::Vector{Message}, tools::Vector{Tool})
    if config.use_openai_compat
        return call_ollama_openai_compat(config, messages, tools)
    end

    messages_spec = messages_to_ollama_format(messages)

    body = Dict{String, Any}(
        "model" => config.model,
        "messages" => messages_spec,
        "stream" => false,
        "options" => Dict(
            "temperature" => config.temperature,
            "num_predict" => config.max_tokens
        )
    )

    if !isempty(tools)
        body["tools"] = tools_to_ollama_format(tools)
    end

    headers = ["Content-Type" => "application/json"]

    response = HTTP.post(
        "$(config.host)/api/chat",
        headers,
        JSON3.write(body);
        status_exception = true
    )

    return JSON3.read(response.body)
end

"""
    call_ollama_openai_compat(config, messages, tools)

Make an API call using Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`).
"""
function call_ollama_openai_compat(config::OllamaConfig, messages::Vector{Message}, tools::Vector{Tool})
    tools_spec = tools_to_openai_format(tools)
    messages_spec = messages_to_openai_format(messages)

    body = Dict{String, Any}(
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
        "Authorization" => "Bearer ollama",
        "Content-Type" => "application/json"
    ]

    response = HTTP.post(
        "$(config.host)/v1/chat/completions",
        headers,
        JSON3.write(body);
        status_exception = true
    )

    return JSON3.read(response.body)
end

"""
    parse_ollama_response(config, response)

Parse Ollama response into Message and tool calls.
Handles both native and OpenAI-compatible formats.
"""
function parse_ollama_response(config::OllamaConfig, response)
    if config.use_openai_compat
        # OpenAI-compatible format: same as parse_llm_response
        return parse_llm_response(response)
    end

    # Native Ollama format
    message = response["message"]

    content = get(message, "content", nothing)
    if content isa AbstractString && isempty(strip(content))
        content = nothing
    end

    tool_calls = nothing

    if haskey(message, "tool_calls") && !isnothing(message["tool_calls"])
        tc_list = message["tool_calls"]
        if length(tc_list) > 0
            tool_calls = ToolCall[]
            for tc in tc_list
                fn = tc["function"]
                # Ollama native format: arguments is already a dict, not a JSON string
                args = if fn["arguments"] isa AbstractString
                    JSON3.read(fn["arguments"], Dict{String, Any})
                else
                    Dict{String, Any}(String(k) => v for (k, v) in pairs(fn["arguments"]))
                end

                push!(tool_calls, ToolCall(
                    string(uuid4()),  # Ollama native API doesn't provide tool call IDs
                    String(fn["name"]),
                    args
                ))
            end
        end
    end

    return Message("assistant", content, tool_calls, nothing)
end

"""
    _parse_ollama_stream(io, on_token) -> String

Parse an Ollama native JSON-lines stream, calling `on_token(delta)` for each
content chunk. Returns the accumulated full text.
"""
function _parse_ollama_stream(io::IO, on_token::Function)
    accumulated = IOBuffer()
    while !eof(io)
        line = readline(io)
        isempty(line) && continue
        try
            chunk = JSON3.read(line)
            content = get(get(chunk, "message", Dict()), "content", "")
            if !isempty(content)
                write(accumulated, content)
                on_token(content)
            end
            get(chunk, "done", false) && break
        catch
            # Skip malformed lines
        end
    end
    return String(take!(accumulated))
end

"""
    call_ollama_streaming(config, messages, on_token) -> String

Make a streaming API call to Ollama (no tools). Calls `on_token(delta)` for
each text chunk. Returns the accumulated full text.

Uses the native `/api/chat` endpoint with `stream: true` when not in
OpenAI-compat mode, otherwise delegates to `call_llm_streaming`.

Falls back to non-streaming `call_ollama` on any streaming error.
"""
function call_ollama_streaming(config::OllamaConfig, messages::Vector{Message},
                                on_token::Function)
    if config.use_openai_compat
        llm_config = LLMConfig(
            api_key = "ollama",
            model = config.model,
            base_url = "$(config.host)/v1",
            max_tokens = config.max_tokens,
            temperature = config.temperature
        )
        return call_llm_streaming(llm_config, messages, on_token)
    end

    messages_spec = messages_to_ollama_format(messages)

    body = Dict{String, Any}(
        "model" => config.model,
        "messages" => messages_spec,
        "stream" => true,
        "options" => Dict(
            "temperature" => config.temperature,
            "num_predict" => config.max_tokens
        )
    )

    headers = ["Content-Type" => "application/json"]

    try
        accumulated = ""
        HTTP.open("POST", "$(config.host)/api/chat", headers;
                  body = JSON3.write(body), status_exception = true) do io
            accumulated = _parse_ollama_stream(io, on_token)
        end
        return accumulated
    catch e
        @warn "Streaming failed, falling back to non-streaming call" exception=(e, catch_backtrace())
        response = call_ollama(config, messages, Tool[])
        msg = parse_ollama_response(config, response)
        text = something(msg.content, "")
        if !isempty(text)
            on_token(text)
        end
        return text
    end
end

# ============================================================================
# Ollama HevalAgent Constructor
# ============================================================================

"""
    HevalAgent(::Val{:ollama}; model="llama3.1", host="http://localhost:11434", kwargs...)

Create a HevalAgent configured for Ollama.

# Arguments
- `model::String` - Ollama model name (default: "llama3.1")
- `host::String` - Ollama server address (default: "http://localhost:11434")
- `use_openai_compat::Bool` - Use OpenAI-compatible endpoint (default: false)
- `max_retries::Int` - Maximum retries (default: 3)

# Example
```julia
# Using native Ollama API (recommended)
agent = HevalAgent(Val(:ollama); model="llama3.1")

# Using OpenAI-compatible endpoint
agent = HevalAgent(Val(:ollama); model="qwen2.5", use_openai_compat=true)

# Custom server
agent = HevalAgent(Val(:ollama); model="mistral", host="http://myserver:11434")
```
"""
function HevalAgent(::Val{:ollama};
    model::String = "llama3.1",
    host::String = "http://localhost:11434",
    use_openai_compat::Bool = false,
    max_retries::Int = 3
)
    ollama_config = OllamaConfig(
        model = model,
        host = host,
        use_openai_compat = use_openai_compat
    )

    if use_openai_compat
        # Use existing OpenAI-compatible path
        config = LLMConfig(
            api_key = "ollama",
            model = model,
            base_url = "$(host)/v1"
        )
        state = AgentState()
        system_prompt = build_system_prompt()
        agent = HevalAgent(config, state, Tool[], Dict{String, Tool}(), system_prompt, max_retries)
        register_tools!(agent)
        return agent
    else
        # Use native Ollama path with a wrapper agent
        config = LLMConfig(
            api_key = "ollama",
            model = model,
            base_url = host
        )
        state = AgentState()
        system_prompt = build_system_prompt()
        agent = OllamaAgent(config, ollama_config, state, Tool[], Dict{String, Tool}(), system_prompt, max_retries)
        register_ollama_tools!(agent)
        return agent
    end
end

# ============================================================================
# OllamaAgent - Native Ollama API Agent
# ============================================================================

"""
    OllamaAgent

AI-powered forecasting agent using Ollama's native API.
Shares the same analysis/query interface as HevalAgent.
"""
mutable struct OllamaAgent
    config::LLMConfig           # kept for compatibility
    ollama_config::OllamaConfig
    state::AgentState
    tools::Vector{Tool}
    tool_index::Dict{String, Tool}
    system_prompt::String
    max_retries::Int
end

function register_ollama_tools!(agent::OllamaAgent)
    agent.tools = [
        create_analyze_features_tool(agent.state),
        create_cross_validate_tool(agent.state),
        create_forecast_tool(agent.state),
        create_anomaly_tool(agent.state),
        create_decompose_tool(agent.state),
        create_unit_root_test_tool(agent.state),
        create_compare_models_tool(agent.state)
    ]
    agent.tool_index = Dict(t.name => t for t in agent.tools)
end

function execute_ollama_tool(agent::OllamaAgent, name::String, args::Dict)
    tool = get(agent.tool_index, name, nothing)
    if !isnothing(tool)
        return tool.fn(args)
    end
    return Dict("error" => "Unknown tool: $name")
end

"""
    _call_llm_for_interpretation(agent::OllamaAgent, prompt::String) -> String

Make a single Ollama LLM call for interpretation (no tools).
"""
function _call_llm_for_interpretation(agent::OllamaAgent, prompt::String)
    messages = [
        Message("system", "You are Heval, a concise forecasting assistant."),
        Message("user", prompt)
    ]
    response = call_ollama(agent.ollama_config, messages, Tool[])
    assistant_msg = parse_ollama_response(agent.ollama_config, response)
    return something(assistant_msg.content, "")
end

"""
    _run_fast_pipeline(agent::OllamaAgent, query; max_retries=2, on_progress=nothing) -> AgentResult

Run the deterministic pipeline + one Ollama LLM call for interpretation.
"""
function _run_fast_pipeline(agent::OllamaAgent, query::Union{String, Nothing};
                             max_retries::Int=2,
                             on_progress::ProgressCallback=nothing,
                             stream::Union{IO, Nothing}=nothing)
    state = agent.state
    _run_deterministic_steps!(state; max_retries=max_retries, on_progress=on_progress)

    prompt = _build_interpretation_prompt(state; query=query)
    output = try
        _emit(on_progress, AgentEvent(llm_start, 1; message="LLM interpretation"))
        result = if !isnothing(stream)
            messages = [
                Message("system", "You are Heval, a concise forecasting assistant."),
                Message("user", prompt)
            ]
            text = call_ollama_streaming(agent.ollama_config, messages,
                token -> print(stream, token))
            flush(stream)
            println(stream)
            text
        else
            _call_llm_for_interpretation(agent, prompt)
        end
        _emit(on_progress, AgentEvent(llm_done, 1; message="Interpretation done"))
        result
    catch e
        @warn "LLM interpretation failed, using programmatic summary" exception=(e, catch_backtrace())
        _build_results_summary(state; query=query)
    end

    _emit(on_progress, AgentEvent(agent_done, 0; message="Fast pipeline finished"))

    # Persist conversation for follow-up queries
    state.conversation_history = [
        Message("system", "You are Heval, a concise forecasting assistant."),
        Message("user", prompt),
        Message("assistant", output)
    ]

    return AgentResult(
        output,
        state.features,
        state.accuracy,
        state.forecasts,
        state.anomalies,
        state.best_model,
        _compute_beats_baseline(state)
    )
end

"""
    analyze(agent::OllamaAgent, data; h=nothing, m=nothing, query=nothing, mode=:fast, groupby=nothing, ...)

Run the full forecasting workflow using Ollama.

# Arguments
- `data` - Time series data (NamedTuple, Vector, Dict, or Tables.jl table for panel data)
- `h::Int` - Forecast horizon (default: 2*m)
- `m::Int` - Seasonal period (default: 12)
- `query::String` - Natural language instructions (optional)
- `mode::Symbol` - Pipeline mode (default: `:fast`)
  - `:fast` — deterministic pipeline + 1 LLM call for interpretation
  - `:local` — deterministic pipeline only, no LLM calls
  - `:agentic` — full LLM-driven agent loop
- `groupby` - Column(s) for panel data grouping (optional)
- `date` - Date column name for panel data (default: :date)
- `target` - Target column name for panel data (default: :value)
"""
function analyze(
    agent::OllamaAgent,
    data;
    h::Union{Int, Nothing} = nothing,
    m::Union{Int, Nothing} = nothing,
    query::Union{String, Nothing} = nothing,
    mode::Symbol = :fast,
    groupby::Union{Vector{Symbol}, Symbol, Nothing} = nothing,
    date::Union{Symbol, Nothing} = nothing,
    target::Union{Symbol, Nothing} = nothing,
    on_progress::ProgressCallback = nothing,
    stream::Union{IO, Nothing} = nothing
)
    # Validate mode
    if !(mode in (:fast, :local, :agentic))
        error("Unknown mode: :$mode. Use :fast, :local, or :agentic.")
    end

    # Set defaults
    m = isnothing(m) ? 12 : m
    h = isnothing(h) ? 2 * m : h

    # Detect panel vs single-series
    is_panel = !isnothing(groupby)

    if is_panel
        if mode != :agentic
            @warn "Panel data requires :agentic mode; ignoring mode=:$mode"
        end
        return _analyze_ollama_panel(agent, data; h=h, m=m, query=query,
                                      groupby=groupby isa Symbol ? [groupby] : groupby,
                                      date_col=something(date, :date),
                                      target_col=something(target, :value),
                                      on_progress=on_progress,
                                      stream=stream)
    end

    # ── Single-series path ─────────────────────────────────────────────────
    dates, values = parse_input_data(data)

    # Initialize state
    agent.state = AgentState()
    agent.state.dates = dates
    agent.state.values = values
    agent.state.seasonal_period = m
    agent.state.horizon = h

    # Re-register tools with new state
    agent.system_prompt = build_system_prompt(; is_panel=false)
    register_ollama_tools!(agent)

    # Route based on mode
    if mode == :local
        return _run_local_pipeline(agent.state;
                                    max_retries=agent.max_retries,
                                    on_progress=on_progress)
    elseif mode == :fast
        return _run_fast_pipeline(agent, query;
                                   max_retries=agent.max_retries,
                                   on_progress=on_progress,
                                   stream=stream)
    else  # :agentic
        # Build user prompt
        data_summary = """
Data summary:
- Length: $(length(values)) observations
- Date range: $(isnothing(dates) ? "not provided" : "$(dates[1]) to $(dates[end])")
- Mean: $(round(mean(values), digits=2))
- Std: $(round(std(values), digits=2))
- Min: $(round(minimum(values), digits=2))
- Max: $(round(maximum(values), digits=2))
- Seasonal period (m): $m
- Forecast horizon (h): $h
"""

        user_prompt = """
$data_summary

$(isnothing(query) ? "Please analyze this time series and generate forecasts." : "User request: $query")

Follow the workflow: analyze_features → cross_validate → generate_forecast → detect_anomalies
"""

        return _run_ollama_agent_loop(agent, user_prompt; on_progress=on_progress, stream=stream)
    end
end

function _analyze_ollama_panel(agent::OllamaAgent, data;
                                h::Int, m::Int,
                                query::Union{String, Nothing},
                                groupby::Vector{Symbol},
                                date_col::Symbol, target_col::Symbol,
                                on_progress::ProgressCallback=nothing,
                                stream::Union{IO, Nothing}=nothing)
    # Initialize state with panel
    agent.state = AgentState()
    agent.state.seasonal_period = m
    agent.state.horizon = h
    agent.state.panel = PanelState(data; groups=groupby, date_col=date_col, target_col=target_col)

    # Also extract values for standard tools
    if Tables.istable(data)
        ct = Tables.columntable(data)
        if target_col in propertynames(ct)
            agent.state.values = Float64.(ct[target_col])
        end
        if date_col in propertynames(ct)
            try
                agent.state.dates = collect(ct[date_col])
            catch end
        end
    end

    # Register tools with panel support
    agent.system_prompt = build_system_prompt(; is_panel=true)
    register_ollama_tools!(agent)
    register_ollama_panel_tools!(agent)

    # Count groups for prompt
    n_groups = "unknown"
    try
        ct = Tables.columntable(data)
        group_keys = Set{String}()
        n_rows = length(ct[first(propertynames(ct))])
        for i in 1:n_rows
            key_parts = [string(ct[g][i]) for g in groupby]
            push!(group_keys, join(key_parts, "|"))
        end
        n_groups = string(length(group_keys))
    catch end

    user_prompt = """
Panel data summary:
- Grouping columns: $(join(String.(groupby), ", "))
- Number of groups: $n_groups
- Date column: $date_col
- Target column: $target_col
- Seasonal period (m): $m
- Forecast horizon (h): $h

$(isnothing(query) ? "Please analyze this panel data and generate forecasts." : "User request: $query")

For panel data, use panel_analyze → analyze_features → cross_validate or panel_fit → detect_anomalies
"""

    return _run_ollama_agent_loop(agent, user_prompt; on_progress=on_progress, stream=stream)
end

function _run_ollama_agent_loop(agent::OllamaAgent, user_prompt::String;
                                 on_progress::ProgressCallback=nothing,
                                 stream::Union{IO, Nothing}=nothing)
    call_fn = (msgs, tools) -> call_ollama(agent.ollama_config, msgs, tools)
    parse_fn = resp -> parse_ollama_response(agent.ollama_config, resp)
    execute_fn = (name, args) -> execute_ollama_tool(agent, name, args)
    return _run_generic_agent_loop(agent.state, agent.tools, agent.system_prompt,
                                    agent.max_retries, user_prompt,
                                    call_fn, parse_fn, execute_fn;
                                    on_progress=on_progress,
                                    stream=stream)
end

function register_ollama_panel_tools!(agent::OllamaAgent)
    state = agent.state
    panel_tools = [create_panel_analyze_tool(state), create_panel_fit_tool(state)]
    for t in panel_tools
        push!(agent.tools, t)
        agent.tool_index[t.name] = t
    end
end

"""
    query(agent::OllamaAgent, question)

Ask a follow-up question about the analysis using Ollama.
"""
function query(agent::OllamaAgent, question::String;
               on_progress::ProgressCallback=nothing,
               stream::Union{IO, Nothing}=nothing)
    call_fn = (msgs, tools) -> call_ollama(agent.ollama_config, msgs, tools)
    parse_fn = resp -> parse_ollama_response(agent.ollama_config, resp)
    execute_fn = (name, args) -> execute_ollama_tool(agent, name, args)
    return _generic_query(agent.state, agent.tools, agent.system_prompt, question,
                          call_fn, parse_fn, execute_fn;
                          on_progress=on_progress,
                          stream=stream)
end

"""
    clear_history(agent::OllamaAgent)

Clear the agent's conversation history and state.
"""
function clear_history(agent::OllamaAgent)
    agent.state = AgentState()
    register_ollama_tools!(agent)
    return nothing
end

# ============================================================================
# Ollama Utilities
# ============================================================================

"""
    list_ollama_models(; host="http://localhost:11434")

List available models on the Ollama server.

# Returns
- `Vector{String}` of model names

# Example
```julia
models = list_ollama_models()
for m in models
    println(m)
end
```
"""
function list_ollama_models(; host::String = "http://localhost:11434")
    response = HTTP.get("$(host)/api/tags"; status_exception = true)
    data = JSON3.read(response.body)
    models = String[]
    if haskey(data, "models")
        for model in data["models"]
            push!(models, String(model["name"]))
        end
    end
    return models
end

"""
    check_ollama_connection(; host="http://localhost:11434")

Check if Ollama server is running and accessible.

# Returns
- `Bool` - true if server is reachable
"""
function check_ollama_connection(; host::String = "http://localhost:11434")
    try
        HTTP.get(host; status_exception = true, connect_timeout = 5)
        return true
    catch
        return false
    end
end
