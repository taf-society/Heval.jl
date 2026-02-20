# ============================================================================
# Type Definitions for Heval.jl
# ============================================================================

"""
    Tool

Represents a callable tool that the LLM can invoke.
"""
struct Tool
    name::String
    description::String
    parameters::Dict{String, Any}
    fn::Function
end

"""
    ToolCall

Represents a tool call request from the LLM.
"""
struct ToolCall
    id::String
    name::String
    arguments::Dict{String, Any}
end

"""
    Message

A message in the conversation.
"""
struct Message
    role::String  # "system", "user", "assistant", "tool"
    content::Union{String, Nothing}
    tool_calls::Union{Vector{ToolCall}, Nothing}
    tool_call_id::Union{String, Nothing}
end

Message(role::String, content::String) = Message(role, content, nothing, nothing)

"""
    ForecastOutput

Structured output containing forecast results.
"""
Base.@kwdef mutable struct ForecastOutput
    model::String = ""
    horizon::Int = 0
    point_forecasts::Vector{Float64} = Float64[]
    lower_80::Vector{Float64} = Float64[]
    upper_80::Vector{Float64} = Float64[]
    lower_95::Vector{Float64} = Float64[]
    upper_95::Vector{Float64} = Float64[]
    dates::Vector{Date} = Date[]
end

"""
    AccuracyMetrics

Model evaluation metrics.
"""
Base.@kwdef struct AccuracyMetrics
    model::String
    mase::Float64 = Inf
    rmse::Float64 = Inf
    mae::Float64 = Inf
    mape::Float64 = Inf
end

"""
    SeriesFeatures

Time series characteristics extracted from data.
"""
Base.@kwdef mutable struct SeriesFeatures
    length::Int = 0
    mean::Float64 = 0.0
    std::Float64 = 0.0
    trend_strength::String = "unknown"
    trend_slope::Float64 = 0.0
    seasonality_strength::String = "unknown"
    seasonal_period::Int = 1
    seasonal_acf::Float64 = 0.0
    is_intermittent::Bool = false
    zero_fraction::Float64 = 0.0
    recommendations::Vector{String} = String[]
    # Durbyn diagnostics
    ndiffs::Int = 0
    nsdiffs::Int = 0
    stationarity::String = "unknown"
end

"""
    AnomalyResult

Detected anomaly information.
"""
struct AnomalyResult
    index::Int
    date::Union{Date, Nothing}
    value::Float64
    z_score::Float64
end

"""
    PanelState

State for panel data (multi-series) analysis.
"""
mutable struct PanelState
    raw_data::Any
    groups::Vector{Symbol}
    date_col::Symbol
    target_col::Symbol
    panel::Any                                              # Durbyn.PanelData
    group_features::Dict{String, SeriesFeatures}
    group_accuracy::Dict{String, Dict{String, AccuracyMetrics}}
    group_forecasts::Any                                    # Durbyn.GroupedForecasts
end

function PanelState(data; groups::Vector{Symbol}, date_col::Symbol=:date, target_col::Symbol=:value)
    PanelState(data, groups, date_col, target_col, nothing,
               Dict{String, SeriesFeatures}(),
               Dict{String, Dict{String, AccuracyMetrics}}(),
               nothing)
end

"""
    AgentState

Mutable state maintained across agent interactions.
"""
mutable struct AgentState
    # Data
    dates::Union{Vector{Date}, Nothing}
    values::Union{Vector{Float64}, Nothing}
    seasonal_period::Int
    horizon::Int

    # Results
    features::Union{SeriesFeatures, Nothing}
    accuracy::Dict{String, AccuracyMetrics}
    forecasts::Union{ForecastOutput, Nothing}
    anomalies::Vector{AnomalyResult}
    best_model::Union{String, Nothing}

    # Panel data
    panel::Union{PanelState, Nothing}

    # Durbyn fitted models (keep references for residuals, re-forecasting, etc.)
    fitted_models::Dict{String, Any}

    # Conversation
    conversation_history::Vector{Message}
end

function AgentState()
    AgentState(
        nothing, nothing, 12, 24,                               # Data defaults
        nothing, Dict{String, AccuracyMetrics}(), nothing,      # Features, accuracy, forecasts
        AnomalyResult[], nothing,                               # Anomalies, best_model
        nothing,                                                # Panel
        Dict{String, Any}(),                                    # Fitted models
        Message[]                                               # Conversation
    )
end

"""
    AgentResult

Result returned from analyze().
"""
struct AgentResult
    output::String
    features::Union{SeriesFeatures, Nothing}
    accuracy::Dict{String, AccuracyMetrics}
    forecasts::Union{ForecastOutput, Nothing}
    anomalies::Vector{AnomalyResult}
    best_model::Union{String, Nothing}
    beats_baseline::Bool
end

"""
    QueryResult

Result returned from query(). Wraps a plain string with pretty display.
"""
struct QueryResult
    content::String
end

# ============================================================================
# Progress / Streaming Events
# ============================================================================

"""
    AgentEventKind

Kind of event emitted during an agent loop.
"""
@enum AgentEventKind begin
    llm_start      # LLM call starting
    llm_done       # LLM call completed
    tool_start     # Tool execution starting
    tool_done      # Tool execution completed
    retry          # Retry loop triggered
    agent_done     # Agent loop finished
end

"""
    AgentEvent

A progress event emitted during the agent loop.

# Fields
- `kind::AgentEventKind` — event type
- `round::Int` — current tool round
- `tool_name::String` — tool name (empty for non-tool events)
- `message::String` — human-readable description
"""
struct AgentEvent
    kind::AgentEventKind
    round::Int
    tool_name::String
    message::String
end

AgentEvent(kind::AgentEventKind, round::Int; tool_name::String="", message::String="") =
    AgentEvent(kind, round, tool_name, message)

"""
    ProgressCallback

Type alias for the optional progress callback: either a `Function` or `nothing`.
"""
const ProgressCallback = Union{Function, Nothing}
