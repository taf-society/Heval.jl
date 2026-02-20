# Agent Architecture

!!! tip "Start with Quick Start"
    If you're new to Heval, start with the [Quick Start](quickstart.md) guide for basic usage. This page covers the internal architecture.

## Overview

**HevalAgent** is the core of Heval.jl. It orchestrates an LLM-powered forecasting workflow by:

1. Receiving time series data and a natural language query
2. Sending the data summary + tool definitions to an LLM
3. The LLM calls tools (analyze_features, cross_validate, etc.) as needed
4. Tool results are returned to the LLM for interpretation
5. The LLM generates a final analysis with model selection and forecasts
6. Validation ensures the best model beats the SNaive baseline

---

## Agent Types

Heval provides two agent types sharing the same API:

### HevalAgent (OpenAI)

```julia
agent = HevalAgent(;
    api_key::String,              # Required: OpenAI API key
    model::String = "gpt-4o",    # LLM model name
    base_url::String = "https://api.openai.com/v1",  # API endpoint
    max_retries::Int = 3          # Retry attempts for baseline validation
)
```

### OllamaAgent (Local Models)

Created via the `Val(:ollama)` dispatch:

```julia
agent = HevalAgent(Val(:ollama);
    model::String = "llama3.1",                    # Ollama model name
    host::String = "http://localhost:11434",        # Ollama server
    use_openai_compat::Bool = false,               # Use /v1 endpoint
    max_retries::Int = 3
)
```

Both agents share the same `analyze`, `query`, and `clear_history` interface.

---

## The Agent Loop

The core loop separates **tool-calling rounds** from **baseline-beating retries**:

```
For each retry (1 to max_retries):
    For each tool round (1 to 20):
        1. Send messages + tools to LLM
        2. If LLM returns tool calls → execute tools, append results, continue
        3. If LLM returns text → capture as final output, break

    Validate: does best model beat SNaive?
        Yes → return result
        No  → inject retry message, try again
```

This design ensures:
- The LLM can call as many tools as needed (up to 20 rounds)
- If the best model doesn't beat the baseline, the agent retries with more models
- Tool-call rounds and validation retries are independent

### Tool-Call Flow Example

```
User: "Forecast next year's sales"
  │
  ├─ LLM calls: analyze_features()
  │   └─ Returns: trend=strong, seasonality=moderate, m=12
  │
  ├─ LLM calls: cross_validate(models=["ARIMA","ETS","Theta","SNaive"])
  │   └─ Returns: MASE scores, best=ETS (0.72), SNaive (1.0)
  │
  ├─ LLM calls: generate_forecast(model="ETS", h=12)
  │   └─ Returns: point forecasts + 80%/95% prediction intervals
  │
  ├─ LLM calls: detect_anomalies(model="ETS")
  │   └─ Returns: 2 anomalies detected
  │
  └─ LLM returns: Final analysis text explaining model choice and forecasts
```

---

## Main API

### `analyze(agent, data; ...)`

The primary entry point for forecasting analysis.

```julia
result = analyze(agent, data;
    h::Int = nothing,       # Forecast horizon (default: 2*m)
    m::Int = nothing,       # Seasonal period (default: 12)
    query::String = nothing, # Natural language instructions
    groupby = nothing,      # Column(s) for panel data grouping
    date = nothing,         # Date column name (default: :date)
    target = nothing        # Target column name (default: :value)
)
```

**Data Formats:**

| Format | Example | Notes |
|--------|---------|-------|
| NamedTuple | `(date=dates, value=values)` | Recommended |
| Vector | `[1.0, 2.0, 3.0]` | Dates auto-generated |
| Dict | `Dict("date"=>dates, "value"=>values)` | String or Symbol keys |
| Tables.jl | Any Tables.jl table | Required for panel data |

**Returns:** `AgentResult`

### `query(agent, question)`

Ask follow-up questions using the full conversation history from `analyze()` or previous `query()` calls. The agent sees the entire prior conversation, so it understands references like "explain it for a manager" without re-running tools.

```julia
answer = query(agent, "Why did you choose ETS over ARIMA?")
answer = query(agent, "Explain that for a non-technical audience")  # knows what "that" refers to
```

If no conversation history is available (e.g., after a fresh `AgentState`), the agent falls back to building a context summary from the analysis state.

The agent has access to all tools during follow-up queries, so it can perform additional analyses if needed (e.g., "try ARIMA instead").

**Returns:** `QueryResult`

### `clear_history(agent)`

Reset the agent's state and conversation history for a new analysis.

```julia
clear_history(agent)
```

---

## AgentResult

The result returned from `analyze()` contains:

| Field | Type | Description |
|-------|------|-------------|
| `output` | `String` | LLM's analysis narrative |
| `features` | `SeriesFeatures` | Extracted time series features |
| `accuracy` | `Dict{String, AccuracyMetrics}` | Model evaluation results |
| `forecasts` | `ForecastOutput` | Generated forecasts with CIs |
| `anomalies` | `Vector{AnomalyResult}` | Detected anomalies |
| `best_model` | `String` | Name of best performing model |
| `beats_baseline` | `Bool` | Whether best model beats SNaive |

### Pretty Display

`AgentResult` displays a rich, formatted summary:

- **REPL**: Color-coded tables with PASS/FAIL badge, model accuracy, forecasts, anomalies
- **Jupyter**: Styled HTML with collapsible sections and highlighted best model
- **Programmatic**: Access any field directly (e.g., `result.forecasts.point_forecasts`)

---

## QueryResult

The result from `query()` wraps a string with pretty display:

```julia
answer = query(agent, "Explain the seasonal pattern")

# REPL: displays bordered box with word wrapping
# Jupyter: styled HTML container

# String interoperable
println(answer)                      # prints raw text
msg = "Answer: " * string(answer)    # concatenation works
```

---

## State Management

The agent maintains an `AgentState` across the analysis workflow:

```julia
agent.state.dates                 # Input dates
agent.state.values                # Input values
agent.state.seasonal_period       # m
agent.state.horizon               # h
agent.state.features              # SeriesFeatures (after analyze_features)
agent.state.accuracy              # Dict of AccuracyMetrics (after cross_validate)
agent.state.forecasts             # ForecastOutput (after generate_forecast)
agent.state.anomalies             # Vector{AnomalyResult} (after detect_anomalies)
agent.state.best_model            # Best model name
agent.state.fitted_models         # Dict of Durbyn fitted model objects
agent.state.panel                 # PanelState (for multi-series)
agent.state.conversation_history  # Vector{Message} — persisted for follow-up queries
```

State is reset on each `analyze()` call or via `clear_history()`. Conversation history is automatically saved after `analyze()` and each `query()` call, enabling context-aware follow-up questions.

---

## System Prompt

The agent's system prompt includes:
- All 16 available models with descriptions
- All 7 core tools with parameter schemas
- Panel data tools (when groupby is specified)
- A structured workflow (features → CV → forecast → anomalies)
- Output requirements and guidelines

The prompt is automatically adjusted based on whether the analysis is single-series or panel data.

---

## Validation Logic

After the agent completes its analysis, Heval validates the result:

1. Check if `cross_validate` was called and produced accuracy metrics
2. Find the model with the lowest MASE
3. Compare against SNaive's MASE
4. If the best model doesn't beat SNaive, inject a retry message asking the LLM to try additional models
5. Retry up to `max_retries` times (default: 3)

This ensures the forecasting results are meaningful — a model that can't beat the seasonal naive baseline isn't worth using.
