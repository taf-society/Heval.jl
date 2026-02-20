# ============================================================================
# Core Agent Implementation for Heval.jl
# ============================================================================

import Tables

"""
    HevalAgent

AI-powered forecasting agent.

"Heval" means "friend" in Kurdish - your companion for time series forecasting.

# Fields
- `config::LLMConfig` - LLM API configuration
- `state::AgentState` - Agent state (data, results, history)
- `tools::Vector{Tool}` - Available tools
- `system_prompt::String` - System instructions
- `max_retries::Int` - Max retry attempts

# Example
```julia
agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])
result = analyze(agent, data; h=12, query="Forecast sales")
```
"""
mutable struct HevalAgent
    config::LLMConfig
    state::AgentState
    tools::Vector{Tool}
    tool_index::Dict{String, Tool}
    system_prompt::String
    max_retries::Int
end

"""
    HevalAgent(; api_key, model="gpt-4o", max_retries=3)

Create a new HevalAgent.

# Arguments
- `api_key::String` - OpenAI API key (required)
- `model::String` - Model name (default: "gpt-4o")
- `base_url::String` - API base URL (default: OpenAI)
- `max_retries::Int` - Maximum retries (default: 3)
"""
function HevalAgent(;
    api_key::String,
    model::String = "gpt-4o",
    base_url::String = "https://api.openai.com/v1",
    max_retries::Int = 3
)
    config = LLMConfig(
        api_key = api_key,
        model = model,
        base_url = base_url
    )

    state = AgentState()
    system_prompt = build_system_prompt()

    agent = HevalAgent(config, state, Tool[], Dict{String, Tool}(), system_prompt, max_retries)

    # Register tools
    register_tools!(agent)

    return agent
end

function build_system_prompt(; is_panel::Bool=false)
    base = """
You are Heval, a forecasting assistant powered by Durbyn.jl.

Use tools for calculations; do not invent metrics or forecasts.
Prioritize accuracy and concise explanations.

Model guidance:
- Strong seasonality: ETS, HoltWinters, ARIMA, TBATS, BATS
- Weak/no seasonality: SES, Holt, Theta, ARIMA, ARAR
- Intermittent demand: Croston
- Growth/adoption: Diffusion

Selection rules:
- MASE is the primary selection metric.
- Include SNaive as baseline and prefer models that beat SNaive.
- Report uncertainty and data limitations clearly.
"""

    panel_section = if is_panel
        """

Panel workflow:
1. Use `panel_analyze` with grouping columns.
2. Run `analyze_features` for overall characteristics.
3. Use `cross_validate` and/or `panel_fit` for candidate models.
4. Summarize findings across groups.
"""
    else
        ""
    end

    workflow = """

Required workflow:
1. `analyze_features`
2. `cross_validate` with suitable candidates (include SNaive)
3. `generate_forecast` with best model
4. `detect_anomalies` with chosen model

Response requirements:
- Explain key reasoning briefly and directly answer the user's request.
- If best model does not beat SNaive, test additional models.
"""

    return base * panel_section * workflow
end

function register_tools!(agent::HevalAgent)
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

# ============================================================================
# Main API
# ============================================================================

"""
    analyze(agent, data; h=nothing, m=nothing, query=nothing, mode=:fast, groupby=nothing, ...)

Run the full forecasting workflow.

# Arguments
- `agent::HevalAgent` - The agent instance
- `data` - Time series data. Can be:
  - `NamedTuple` with `:date` and `:value` fields
  - `Vector{<:Real}` of values
  - `Dict` with "date" and "value" keys
  - Any Tables.jl-compatible table (for panel data)
- `h::Int` - Forecast horizon (default: 2*m)
- `m::Int` - Seasonal period (default: 12)
- `query::String` - Natural language instructions (optional)
- `mode::Symbol` - Pipeline mode (default: `:fast`)
  - `:fast` — deterministic pipeline + 1 LLM call for interpretation (~5x faster)
  - `:local` — deterministic pipeline only, no LLM calls (batch/CI use)
  - `:agentic` — full LLM-driven agent loop (5+ LLM calls, exploratory)
- `groupby` - Column(s) for panel data grouping (optional)
- `date` - Date column name for panel data (default: :date)
- `target` - Target column name for panel data (default: :value)

# Returns
- `AgentResult` with forecasts, accuracy metrics, and analysis

# Examples
```julia
# Fast mode (default) — 1 LLM call
data = (date = Date.(2020, 1:36), value = rand(36) .* 100)
result = analyze(agent, data; h=12, query="Forecast next year")

# Local mode — no LLM needed
result = analyze(agent, data; h=12, mode=:local)

# Agentic mode — full LLM loop (backward compat)
result = analyze(agent, data; h=12, mode=:agentic)

# Panel data
panel_data = (date=dates, store=stores, value=values)
result = analyze(agent, panel_data; h=12, m=12, groupby=[:store])
```
"""
function analyze(
    agent::HevalAgent,
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
        return _analyze_panel(agent, data; h=h, m=m, query=query,
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
    register_tools!(agent)

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

        return _run_agent_loop(agent, user_prompt; on_progress=on_progress, stream=stream)
    end
end

function _analyze_panel(agent::HevalAgent, data;
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

    # Also extract a representative single series for feature analysis
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
    register_tools!(agent)
    register_panel_tools!(agent)

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

    return _run_agent_loop(agent, user_prompt; on_progress=on_progress, stream=stream)
end

"""
    _emit(cb, event)

Safely invoke a progress callback. Swallows any errors so a broken callback
never crashes the agent loop.
"""
function _emit(cb::ProgressCallback, event::AgentEvent)
    isnothing(cb) && return
    try
        cb(event)
    catch
    end
    return
end

"""
    _run_generic_agent_loop(state, tools, system_prompt, max_retries, user_prompt, call_fn, parse_fn, execute_fn; on_progress=nothing)

Shared agent loop used by both HevalAgent and OllamaAgent.
`call_fn(messages, tools)` → raw response
`parse_fn(response)` → Message
`execute_fn(name, args)` → result Dict
"""
function _run_generic_agent_loop(state::AgentState, tools::Vector{Tool},
                                  system_prompt::String, max_retries::Int,
                                  user_prompt::String,
                                  call_fn::Function, parse_fn::Function, execute_fn::Function;
                                  on_progress::ProgressCallback=nothing,
                                  stream::Union{IO, Nothing}=nothing,
                                  stream_fn::Union{Function, Nothing}=nothing)
    messages = [
        Message("system", system_prompt),
        Message("user", user_prompt)
    ]

    final_output = ""
    max_tool_rounds = 20

    for retry in 1:max_retries
        for _round in 1:max_tool_rounds
            _emit(on_progress, AgentEvent(llm_start, _round; message="Calling LLM (round $_round)"))
            response = call_fn(messages, tools)
            assistant_msg = parse_fn(response)
            _emit(on_progress, AgentEvent(llm_done, _round; message="LLM responded"))

            if !isnothing(assistant_msg.tool_calls) && !isempty(assistant_msg.tool_calls)
                push!(messages, assistant_msg)

                for tc in assistant_msg.tool_calls
                    _emit(on_progress, AgentEvent(tool_start, _round; tool_name=tc.name, message="Running $(tc.name)"))
                    result = execute_fn(tc.name, tc.arguments)
                    push!(messages, format_tool_result(tc.id, result))
                    _emit(on_progress, AgentEvent(tool_done, _round; tool_name=tc.name, message="Completed $(tc.name)"))
                end

                continue
            end

            final_output = something(assistant_msg.content, "")
            push!(messages, assistant_msg)

            # Write final response to stream if set
            if !isnothing(stream) && !isempty(final_output)
                print(stream, final_output)
                println(stream)
                flush(stream)
            end

            break
        end

        if !isempty(state.accuracy) && retry < max_retries
            best_mase = minimum(met.mase for met in Base.values(state.accuracy))
            naive_mase = haskey(state.accuracy, "SNaive") ?
                         state.accuracy["SNaive"].mase : Inf

            if best_mase >= naive_mase
                _emit(on_progress, AgentEvent(retry, retry; message="Retry $retry: best model didn't beat SNaive"))
                push!(messages, Message("user",
                    "The best model doesn't beat the SNaive baseline. " *
                    "Please try additional models (e.g., ARIMA, ETS, Theta, ARAR, BATS) " *
                    "and select one that outperforms SNaive."
                ))
                continue
            end
        end

        break
    end

    _emit(on_progress, AgentEvent(agent_done, 0; message="Agent loop finished"))

    # Persist conversation for follow-up queries
    state.conversation_history = messages

    return AgentResult(
        final_output,
        state.features,
        state.accuracy,
        state.forecasts,
        state.anomalies,
        state.best_model,
        _compute_beats_baseline(state)
    )
end

"""
    _generic_query(state, tools, system_prompt, question, call_fn, parse_fn, execute_fn; on_progress=nothing)

Shared query loop used by both HevalAgent and OllamaAgent.
"""
function _generic_query(state::AgentState, tools::Vector{Tool},
                        system_prompt::String, question::String,
                        call_fn::Function, parse_fn::Function, execute_fn::Function;
                        on_progress::ProgressCallback=nothing,
                        stream::Union{IO, Nothing}=nothing,
                        stream_fn::Union{Function, Nothing}=nothing)
    if isnothing(state.values) && isnothing(state.panel)
        error("No analysis has been run. Call analyze() first.")
    end

    # Use existing conversation history if available, otherwise build fresh context
    if !isempty(state.conversation_history)
        messages = copy(state.conversation_history)
        push!(messages, Message("user", question))
    else
        context_parts = ["Previous analysis context:"]

        if !isnothing(state.features)
            f = state.features
            push!(context_parts, "- Data: $(f.length) obs, trend=$(f.trend_strength), seasonality=$(f.seasonality_strength)")
            if f.stationarity != "unknown"
                push!(context_parts, "- Stationarity: $(f.stationarity)")
            end
        end

        if !isempty(state.accuracy) && !isnothing(state.best_model)
            push!(context_parts, "- Models evaluated: $(join(keys(state.accuracy), ", "))")
            push!(context_parts, "- Best model: $(state.best_model) (MASE=$(state.accuracy[state.best_model].mase))")
        end

        if !isnothing(state.forecasts)
            fc = state.forecasts
            push!(context_parts, "- Forecast: $(fc.horizon) periods using $(fc.model)")
        end

        if !isempty(state.anomalies)
            push!(context_parts, "- Anomalies detected: $(length(state.anomalies))")
        end

        if !isnothing(state.panel)
            push!(context_parts, "- Panel data: groups=$(join(String.(state.panel.groups), ", "))")
        end

        context = join(context_parts, "\n")

        messages = [
            Message("system", system_prompt),
            Message("user", "$context\n\nUser question: $question")
        ]
    end

    _round = 1
    _emit(on_progress, AgentEvent(llm_start, _round; message="Calling LLM"))
    response = call_fn(messages, tools)
    assistant_msg = parse_fn(response)
    _emit(on_progress, AgentEvent(llm_done, _round; message="LLM responded"))

    while !isnothing(assistant_msg.tool_calls) && !isempty(assistant_msg.tool_calls)
        push!(messages, assistant_msg)

        for tc in assistant_msg.tool_calls
            _emit(on_progress, AgentEvent(tool_start, _round; tool_name=tc.name, message="Running $(tc.name)"))
            result = execute_fn(tc.name, tc.arguments)
            push!(messages, format_tool_result(tc.id, result))
            _emit(on_progress, AgentEvent(tool_done, _round; tool_name=tc.name, message="Completed $(tc.name)"))
        end

        _round += 1
        _emit(on_progress, AgentEvent(llm_start, _round; message="Calling LLM (round $_round)"))
        response = call_fn(messages, tools)
        assistant_msg = parse_fn(response)
        _emit(on_progress, AgentEvent(llm_done, _round; message="LLM responded"))
    end

    final_text = something(assistant_msg.content, "")
    push!(messages, assistant_msg)

    # Write final response to stream if set
    if !isnothing(stream) && !isempty(final_text)
        print(stream, final_text)
        println(stream)
        flush(stream)
    end

    # Persist conversation for subsequent queries
    state.conversation_history = messages

    _emit(on_progress, AgentEvent(agent_done, _round; message="Query finished"))
    return QueryResult(final_text)
end

# ============================================================================
# Fast / Local Pipeline Helpers
# ============================================================================

"""
    _compute_beats_baseline(state::AgentState) -> Bool

Check if the best model beats the SNaive baseline on MASE.
"""
function _compute_beats_baseline(state::AgentState)
    if !isempty(state.accuracy) && !isnothing(state.best_model)
        best_mase = state.accuracy[state.best_model].mase
        naive_mase = get(state.accuracy, "SNaive", AccuracyMetrics(model="SNaive")).mase
        return best_mase < naive_mase
    end
    return false
end

"""
    _run_deterministic_steps!(state::AgentState; max_retries::Int=2, on_progress=nothing)

Run the deterministic pipeline: features → model selection → CV (with retry) → forecast → anomalies.
Modifies `state` in place.
"""
function _run_deterministic_steps!(state::AgentState;
                                    max_retries::Int=2,
                                    on_progress::ProgressCallback=nothing)
    # Step 1: Analyze features
    _emit(on_progress, AgentEvent(tool_start, 1; tool_name="analyze_features", message="Analyzing features"))
    tool_analyze_features(state, Dict{String, Any}())
    _emit(on_progress, AgentEvent(tool_done, 1; tool_name="analyze_features", message="Features analyzed"))

    if isnothing(state.features)
        return
    end

    # Step 2: Select candidate models and cross-validate
    candidates = _select_candidate_models(state.features)
    _emit(on_progress, AgentEvent(tool_start, 2; tool_name="cross_validate", message="Cross-validating $(length(candidates)) models"))
    tool_cross_validate(state, Dict{String, Any}("models" => candidates))
    _emit(on_progress, AgentEvent(tool_done, 2; tool_name="cross_validate", message="Cross-validation done"))

    # Step 2b: Retry if best doesn't beat SNaive
    for retry in 1:max_retries
        if _compute_beats_baseline(state)
            break
        end
        already_tried = collect(keys(state.accuracy))
        extra = _expand_candidate_models(already_tried, state.features)
        if isempty(extra)
            break
        end
        _emit(on_progress, AgentEvent(Heval.retry, retry; message="Retry $retry: trying $(length(extra)) more models"))
        tool_cross_validate(state, Dict{String, Any}("models" => extra))
    end

    # Step 3: Generate forecast with best model
    best = state.best_model
    if !isnothing(best)
        _emit(on_progress, AgentEvent(tool_start, 3; tool_name="generate_forecast", message="Forecasting with $best"))
        tool_generate_forecast(state, Dict{String, Any}("model" => best))
        _emit(on_progress, AgentEvent(tool_done, 3; tool_name="generate_forecast", message="Forecast generated"))

        # Step 4: Detect anomalies
        _emit(on_progress, AgentEvent(tool_start, 4; tool_name="detect_anomalies", message="Detecting anomalies"))
        tool_detect_anomalies(state, Dict{String, Any}("model" => best))
        _emit(on_progress, AgentEvent(tool_done, 4; tool_name="detect_anomalies", message="Anomaly detection done"))
    end

    return
end

"""
    _build_results_summary(state::AgentState; query=nothing) -> String

Build a programmatic text summary of the analysis results (used by :local mode).
"""
function _build_results_summary(state::AgentState; query::Union{String, Nothing}=nothing)
    parts = String[]

    # Features
    if !isnothing(state.features)
        f = state.features
        push!(parts, "Series: $(f.length) observations, trend=$(f.trend_strength), seasonality=$(f.seasonality_strength), stationarity=$(f.stationarity)")
    end

    # Accuracy
    if !isempty(state.accuracy)
        sorted = sort(collect(state.accuracy), by=x -> x.second.mase)
        lines = ["  $(k): MASE=$(v.mase), RMSE=$(v.rmse), MAE=$(v.mae), MAPE=$(v.mape)%" for (k, v) in sorted]
        push!(parts, "Model comparison (by MASE):\n" * join(lines, "\n"))

        if !isnothing(state.best_model)
            bb = _compute_beats_baseline(state)
            push!(parts, "Best model: $(state.best_model) (MASE=$(state.accuracy[state.best_model].mase), $(bb ? "beats" : "does not beat") SNaive)")
        end
    end

    # Forecast
    if !isnothing(state.forecasts)
        fc = state.forecasts
        push!(parts, "Forecast: $(fc.horizon) periods ahead using $(fc.model), range $(round(minimum(fc.point_forecasts), digits=2)) to $(round(maximum(fc.point_forecasts), digits=2))")
    end

    # Anomalies
    if !isempty(state.anomalies)
        n_anom = length(state.anomalies)
        push!(parts, "Anomalies: $n_anom detected")
    else
        push!(parts, "Anomalies: none detected")
    end

    return join(parts, "\n\n")
end

"""
    _build_interpretation_prompt(state::AgentState; query=nothing) -> String

Build a compressed prompt for a single LLM interpretation call in :fast mode.
"""
function _build_interpretation_prompt(state::AgentState; query::Union{String, Nothing}=nothing)
    summary = _build_results_summary(state; query=query)

    request = isnothing(query) ? "Provide a concise analysis summary with key insights and recommendations." :
                                  "User request: $query"

    return """You are Heval, a forecasting assistant. Based on the analysis results below, provide a clear and concise interpretation.

Analysis Results:
$summary

$request

Be direct. Highlight the most important findings: model performance, forecast direction, any anomalies, and actionable recommendations."""
end

"""
    _run_local_pipeline(state::AgentState; max_retries=2, on_progress=nothing) -> AgentResult

Run the deterministic pipeline with a programmatic summary (no LLM calls).
"""
function _run_local_pipeline(state::AgentState;
                              max_retries::Int=2,
                              on_progress::ProgressCallback=nothing)
    _run_deterministic_steps!(state; max_retries=max_retries, on_progress=on_progress)
    output = _build_results_summary(state)
    _emit(on_progress, AgentEvent(agent_done, 0; message="Local pipeline finished"))

    # Persist minimal conversation for follow-up queries
    state.conversation_history = [
        Message("system", "You are Heval, a concise forecasting assistant."),
        Message("user", "Analysis results:\n$output"),
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
    _call_llm_for_interpretation(agent::HevalAgent, prompt::String) -> String

Make a single LLM call for interpretation (no tools).
"""
function _call_llm_for_interpretation(agent::HevalAgent, prompt::String)
    messages = [
        Message("system", "You are Heval, a concise forecasting assistant."),
        Message("user", prompt)
    ]
    response = call_llm(agent.config, messages, Tool[])
    assistant_msg = parse_llm_response(response)
    return something(assistant_msg.content, "")
end

"""
    _run_fast_pipeline(agent::HevalAgent, query; max_retries=2, on_progress=nothing) -> AgentResult

Run the deterministic pipeline + one LLM call for interpretation.
"""
function _run_fast_pipeline(agent::HevalAgent, query::Union{String, Nothing};
                             max_retries::Int=2,
                             on_progress::ProgressCallback=nothing,
                             stream::Union{IO, Nothing}=nothing)
    state = agent.state
    _run_deterministic_steps!(state; max_retries=max_retries, on_progress=on_progress)

    # Single LLM call for interpretation
    prompt = _build_interpretation_prompt(state; query=query)
    output = try
        _emit(on_progress, AgentEvent(llm_start, 1; message="LLM interpretation"))
        result = if !isnothing(stream)
            messages = [
                Message("system", "You are Heval, a concise forecasting assistant."),
                Message("user", prompt)
            ]
            text = call_llm_streaming(agent.config, messages,
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

# ── HevalAgent delegates ──────────────────────────────────────────────────

function _run_agent_loop(agent::HevalAgent, user_prompt::String;
                         on_progress::ProgressCallback=nothing,
                         stream::Union{IO, Nothing}=nothing)
    call_fn = (msgs, tools) -> call_llm(agent.config, msgs, tools)
    parse_fn = parse_llm_response
    execute_fn = (name, args) -> execute_tool(agent, name, args)
    return _run_generic_agent_loop(agent.state, agent.tools, agent.system_prompt,
                                    agent.max_retries, user_prompt,
                                    call_fn, parse_fn, execute_fn;
                                    on_progress=on_progress,
                                    stream=stream)
end

function execute_tool(agent::HevalAgent, name::String, args::Dict)
    tool = get(agent.tool_index, name, nothing)
    if !isnothing(tool)
        return tool.fn(args)
    end
    return Dict("error" => "Unknown tool: $name")
end

function parse_input_data(data)
    if data isa NamedTuple
        dates = haskey(data, :date) ? collect(data.date) : nothing
        values = haskey(data, :value) ? Float64.(collect(data.value)) :
                 haskey(data, :y) ? Float64.(collect(data.y)) : nothing

        if isnothing(values)
            for k in keys(data)
                if k != :date && eltype(data[k]) <: Real
                    values = Float64.(collect(data[k]))
                    break
                end
            end
        end

        return dates, values

    elseif data isa Vector
        return nothing, Float64.(data)

    elseif data isa Dict
        dates = haskey(data, "date") ? data["date"] :
                haskey(data, :date) ? data[:date] : nothing
        values = haskey(data, "value") ? Float64.(data["value"]) :
                 haskey(data, :value) ? Float64.(data[:value]) :
                 haskey(data, "y") ? Float64.(data["y"]) : nothing

        return dates, values
    else
        error("Unsupported data format. Use NamedTuple, Vector, or Dict.")
    end
end

"""
    query(agent, question)

Ask a follow-up question about the analysis.
"""
function query(agent::HevalAgent, question::String;
               on_progress::ProgressCallback=nothing,
               stream::Union{IO, Nothing}=nothing)
    call_fn = (msgs, tools) -> call_llm(agent.config, msgs, tools)
    parse_fn = parse_llm_response
    execute_fn = (name, args) -> execute_tool(agent, name, args)
    return _generic_query(agent.state, agent.tools, agent.system_prompt, question,
                          call_fn, parse_fn, execute_fn;
                          on_progress=on_progress,
                          stream=stream)
end

"""
    clear_history(agent)

Clear the agent's conversation history and state.
"""
function clear_history(agent::HevalAgent)
    agent.state = AgentState()
    agent.system_prompt = build_system_prompt(; is_panel=false)
    register_tools!(agent)
    return nothing
end
