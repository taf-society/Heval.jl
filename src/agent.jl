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
    analyze(agent, data; h=nothing, m=nothing, query=nothing, groupby=nothing, ...)

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
- `groupby` - Column(s) for panel data grouping (optional)
- `date` - Date column name for panel data (default: :date)
- `target` - Target column name for panel data (default: :value)

# Returns
- `AgentResult` with forecasts, accuracy metrics, and analysis

# Examples
```julia
# Single series
data = (date = Date.(2020, 1:36), value = rand(36) .* 100)
result = analyze(agent, data; h=12, query="Forecast next year")

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
    groupby::Union{Vector{Symbol}, Symbol, Nothing} = nothing,
    date::Union{Symbol, Nothing} = nothing,
    target::Union{Symbol, Nothing} = nothing
)
    # Set defaults
    m = isnothing(m) ? 12 : m
    h = isnothing(h) ? 2 * m : h

    # Detect panel vs single-series
    is_panel = !isnothing(groupby)

    if is_panel
        return _analyze_panel(agent, data; h=h, m=m, query=query,
                              groupby=groupby isa Symbol ? [groupby] : groupby,
                              date_col=something(date, :date),
                              target_col=something(target, :value))
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

    return _run_agent_loop(agent, user_prompt)
end

function _analyze_panel(agent::HevalAgent, data;
                        h::Int, m::Int,
                        query::Union{String, Nothing},
                        groupby::Vector{Symbol},
                        date_col::Symbol, target_col::Symbol)
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

    return _run_agent_loop(agent, user_prompt)
end

"""
    _run_generic_agent_loop(state, tools, system_prompt, max_retries, user_prompt, call_fn, parse_fn, execute_fn)

Shared agent loop used by both HevalAgent and OllamaAgent.
`call_fn(messages, tools)` → raw response
`parse_fn(response)` → Message
`execute_fn(name, args)` → result Dict
"""
function _run_generic_agent_loop(state::AgentState, tools::Vector{Tool},
                                  system_prompt::String, max_retries::Int,
                                  user_prompt::String,
                                  call_fn::Function, parse_fn::Function, execute_fn::Function)
    messages = [
        Message("system", system_prompt),
        Message("user", user_prompt)
    ]

    final_output = ""
    max_tool_rounds = 20

    for retry in 1:max_retries
        for _round in 1:max_tool_rounds
            response = call_fn(messages, tools)
            assistant_msg = parse_fn(response)

            if !isnothing(assistant_msg.tool_calls) && !isempty(assistant_msg.tool_calls)
                push!(messages, assistant_msg)

                for tc in assistant_msg.tool_calls
                    result = execute_fn(tc.name, tc.arguments)
                    push!(messages, format_tool_result(tc.id, result))
                end

                continue
            end

            final_output = something(assistant_msg.content, "")
            push!(messages, assistant_msg)
            break
        end

        if !isempty(state.accuracy) && retry < max_retries
            best_mase = minimum(met.mase for met in Base.values(state.accuracy))
            naive_mase = haskey(state.accuracy, "SNaive") ?
                         state.accuracy["SNaive"].mase : Inf

            if best_mase >= naive_mase
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

    beats_baseline = if !isempty(state.accuracy) && !isnothing(state.best_model)
        best_mase = state.accuracy[state.best_model].mase
        naive_mase = get(state.accuracy, "SNaive", AccuracyMetrics(model="SNaive")).mase
        best_mase < naive_mase
    else
        false
    end

    return AgentResult(
        final_output,
        state.features,
        state.accuracy,
        state.forecasts,
        state.anomalies,
        state.best_model,
        beats_baseline
    )
end

"""
    _generic_query(state, tools, system_prompt, question, call_fn, parse_fn, execute_fn)

Shared query loop used by both HevalAgent and OllamaAgent.
"""
function _generic_query(state::AgentState, tools::Vector{Tool},
                        system_prompt::String, question::String,
                        call_fn::Function, parse_fn::Function, execute_fn::Function)
    if isnothing(state.values) && isnothing(state.panel)
        error("No analysis has been run. Call analyze() first.")
    end

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

    response = call_fn(messages, tools)
    assistant_msg = parse_fn(response)

    while !isnothing(assistant_msg.tool_calls) && !isempty(assistant_msg.tool_calls)
        push!(messages, assistant_msg)

        for tc in assistant_msg.tool_calls
            result = execute_fn(tc.name, tc.arguments)
            push!(messages, format_tool_result(tc.id, result))
        end

        response = call_fn(messages, tools)
        assistant_msg = parse_fn(response)
    end

    return QueryResult(something(assistant_msg.content, ""))
end

# ── HevalAgent delegates ──────────────────────────────────────────────────

function _run_agent_loop(agent::HevalAgent, user_prompt::String)
    call_fn = (msgs, tools) -> call_llm(agent.config, msgs, tools)
    parse_fn = parse_llm_response
    execute_fn = (name, args) -> execute_tool(agent, name, args)
    return _run_generic_agent_loop(agent.state, agent.tools, agent.system_prompt,
                                    agent.max_retries, user_prompt,
                                    call_fn, parse_fn, execute_fn)
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
function query(agent::HevalAgent, question::String)
    call_fn = (msgs, tools) -> call_llm(agent.config, msgs, tools)
    parse_fn = parse_llm_response
    execute_fn = (name, args) -> execute_tool(agent, name, args)
    return _generic_query(agent.state, agent.tools, agent.system_prompt, question,
                          call_fn, parse_fn, execute_fn)
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
