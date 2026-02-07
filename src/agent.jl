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

    agent = HevalAgent(config, state, Tool[], system_prompt, max_retries)

    # Register tools
    register_tools!(agent)

    return agent
end

function build_system_prompt(; is_panel::Bool=false)
    base = """
You are Heval, an AI forecasting assistant powered by Durbyn.jl — a comprehensive Julia forecasting library.
"Heval" means "friend" in Kurdish - you are a helpful companion for forecasting tasks.

## Available Models (16 total):
- **ARIMA**: Auto ARIMA with seasonal components (SARIMA). Best for stationary series with clear AR/MA structure.
- **ETS**: Exponential Smoothing with automatic component selection (Error, Trend, Seasonal). Versatile general-purpose model.
- **BATS**: Box-Cox, ARMA errors, Trend, Seasonal. Good for integer seasonal periods with complex structure.
- **TBATS**: Trigonometric BATS. Handles non-integer seasonal periods (e.g., 52.18 weeks/year), multiple seasonalities.
- **Theta**: Theta method (STM/OTM/DSTM/DOTM variants). Competitive on M-competition benchmarks.
- **SES**: Simple Exponential Smoothing. Baseline for no-trend, no-seasonality data.
- **Holt**: Holt's linear trend method. Good for trending data without seasonality.
- **HoltWinters**: Holt-Winters with additive/multiplicative seasonality.
- **Croston**: Intermittent demand forecasting (classic, SBA, SBJ variants).
- **ARAR**: AutoRegressive with Adaptive Reduction. Adaptive AR modeling.
- **ARARMA**: ARAR + ARMA. Combines adaptive reduction with short-memory ARMA.
- **Diffusion**: S-curve growth models (Bass, Gompertz, GSGompertz, Weibull). For technology adoption/market penetration.
- **Naive**: Last value repeated. Simplest baseline.
- **SNaive**: Seasonal naive. Repeats last seasonal cycle. Primary baseline for comparison.
- **RW**: Random walk with optional drift.
- **Meanf**: Historical mean. Constant forecast.

## Available Tools:

### Core Analysis Tools:
1. **analyze_features**: Analyze time series using STL decomposition and unit root tests.
   - Detects trend, seasonality, stationarity, intermittency
   - Uses Durbyn Stats: STL, seasonal_strength, ndiffs, nsdiffs
   - CALL THIS FIRST

2. **cross_validate**: Time series cross-validation with expanding window.
   - Tests any combination of the 16 models
   - Returns MASE, RMSE, MAE, MAPE
   - MASE is primary metric (scale-independent, lower is better)

3. **generate_forecast**: Generate forecasts with model-specific prediction intervals.
   - Returns point forecasts + 80% and 95% confidence intervals
   - Intervals are model-specific (not simplified approximations)

4. **detect_anomalies**: Residual-based outlier detection.
   - Uses actual model residuals from Durbyn fitted models
   - Z-score threshold (default: 3.0)

### Advanced Tools:
5. **decompose**: STL or MSTL decomposition into trend + seasonal + remainder.
   - Use to understand seasonal patterns and trend structure
   - MSTL for multiple seasonal periods

6. **unit_root_test**: ADF and KPSS stationarity tests.
   - Recommends differencing orders (d, D) for ARIMA
   - Helps decide if differencing is needed

7. **compare_models**: In-sample model comparison using AIC/BIC.
   - Complements cross_validate (out-of-sample) with information criteria
   - Quick way to rank models without full CV
"""

    panel_section = if is_panel
        """

### Panel Data Tools:
8. **panel_analyze**: Analyze multi-series panel data.
   - Constructs PanelData with grouping, date handling, time gap filling
   - Reports number of groups, series lengths, patterns

9. **panel_fit**: Fit models across all groups.
   - Automatically fits model collection to each group
   - Generates grouped forecasts

## PANEL WORKFLOW:
1. Call `panel_analyze` with grouping columns
2. Call `analyze_features` for overall data characteristics
3. Call `cross_validate` or `panel_fit` with candidate models
4. Interpret results across groups
"""
    else
        ""
    end

    workflow = """

## WORKFLOW (Follow these steps IN ORDER):

### Step 1: Feature Analysis (REQUIRED)
- Call `analyze_features` to understand the data
- Note: trend strength, seasonality, stationarity, intermittency
- Use ndiffs/nsdiffs to understand differencing needs

### Step 2: Model Selection (REQUIRED)
- Call `cross_validate` with appropriate models:
  * Strong seasonality → ETS, HoltWinters, ARIMA, TBATS, BATS
  * Weak seasonality → SES, Holt, Theta, ARIMA, ARAR
  * Intermittent demand → Croston (with SBA variant)
  * Growth/adoption → Diffusion
  * Always include SNaive as baseline
- Select model with lowest MASE
- IMPORTANT: Best model MUST beat SNaive baseline

### Step 3: Forecasting (REQUIRED)
- Call `generate_forecast` with the best model
- Prediction intervals are model-specific (proper, not simplified)
- Interpret the forecast trend and uncertainty

### Step 4: Anomaly Detection (REQUIRED)
- Call `detect_anomalies` with the best model
- Uses actual model residuals from Durbyn
- Explain how anomalies affect forecast reliability

## OUTPUT REQUIREMENTS:
- Explain reasoning at each step
- Provide technical details about the selected model
- Interpret forecasts in practical terms
- Answer user questions directly
- If best model doesn't beat SNaive, try additional models

## IMPORTANT GUIDELINES:
- Be concise but thorough
- Use MASE for model comparison (scale-independent)
- All 16 models are production-quality Durbyn implementations
- Acknowledge uncertainty in forecasts
- If data is insufficient, explain limitations
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

function _run_agent_loop(agent::HevalAgent, user_prompt::String)
    messages = [
        Message("system", agent.system_prompt),
        Message("user", user_prompt)
    ]

    # Agent loop: separate tool-call rounds from baseline-beating retries
    final_output = ""
    max_tool_rounds = 20

    for retry in 1:agent.max_retries
        for _round in 1:max_tool_rounds
            response = call_llm(agent.config, messages, agent.tools)
            assistant_msg = parse_llm_response(response)

            if !isnothing(assistant_msg.tool_calls) && !isempty(assistant_msg.tool_calls)
                push!(messages, assistant_msg)

                for tc in assistant_msg.tool_calls
                    result = execute_tool(agent, tc.name, tc.arguments)
                    push!(messages, format_tool_result(tc.id, result))
                end

                continue
            end

            final_output = something(assistant_msg.content, "")
            push!(messages, assistant_msg)
            break
        end

        # Validate: check if best model beats baseline
        if !isempty(agent.state.accuracy) && retry < agent.max_retries
            best_mase = minimum(met.mase for met in Base.values(agent.state.accuracy))
            naive_mase = haskey(agent.state.accuracy, "SNaive") ?
                         agent.state.accuracy["SNaive"].mase : Inf

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

    # Build result
    beats_baseline = if !isempty(agent.state.accuracy) && !isnothing(agent.state.best_model)
        best_mase = agent.state.accuracy[agent.state.best_model].mase
        naive_mase = get(agent.state.accuracy, "SNaive", AccuracyMetrics(model="SNaive")).mase
        best_mase < naive_mase
    else
        false
    end

    return AgentResult(
        final_output,
        agent.state.features,
        agent.state.accuracy,
        agent.state.forecasts,
        agent.state.anomalies,
        agent.state.best_model,
        beats_baseline
    )
end

function execute_tool(agent::HevalAgent, name::String, args::Dict)
    for tool in agent.tools
        if tool.name == name
            return tool.fn(args)
        end
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
    if isnothing(agent.state.values) && isnothing(agent.state.panel)
        error("No analysis has been run. Call analyze() first.")
    end

    context_parts = ["Previous analysis context:"]

    if !isnothing(agent.state.features)
        f = agent.state.features
        push!(context_parts, "- Data: $(f.length) obs, trend=$(f.trend_strength), seasonality=$(f.seasonality_strength)")
        if f.stationarity != "unknown"
            push!(context_parts, "- Stationarity: $(f.stationarity)")
        end
    end

    if !isempty(agent.state.accuracy) && !isnothing(agent.state.best_model)
        push!(context_parts, "- Models evaluated: $(join(keys(agent.state.accuracy), ", "))")
        push!(context_parts, "- Best model: $(agent.state.best_model) (MASE=$(agent.state.accuracy[agent.state.best_model].mase))")
    end

    if !isnothing(agent.state.forecasts)
        fc = agent.state.forecasts
        push!(context_parts, "- Forecast: $(fc.horizon) periods using $(fc.model)")
    end

    if !isempty(agent.state.anomalies)
        push!(context_parts, "- Anomalies detected: $(length(agent.state.anomalies))")
    end

    if !isnothing(agent.state.panel)
        push!(context_parts, "- Panel data: groups=$(join(String.(agent.state.panel.groups), ", "))")
    end

    context = join(context_parts, "\n")

    messages = [
        Message("system", agent.system_prompt),
        Message("user", "$context\n\nUser question: $question")
    ]

    response = call_llm(agent.config, messages, agent.tools)
    assistant_msg = parse_llm_response(response)

    while !isnothing(assistant_msg.tool_calls) && !isempty(assistant_msg.tool_calls)
        push!(messages, assistant_msg)

        for tc in assistant_msg.tool_calls
            result = execute_tool(agent, tc.name, tc.arguments)
            push!(messages, format_tool_result(tc.id, result))
        end

        response = call_llm(agent.config, messages, agent.tools)
        assistant_msg = parse_llm_response(response)
    end

    return QueryResult(something(assistant_msg.content, ""))
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
