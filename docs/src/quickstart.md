# Quick Start

## Installation

Install the development version:

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society/Heval.jl")
```

## Setup

Heval requires an LLM backend. Choose one:

### Option 1: OpenAI (Recommended)

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

```julia
using Heval

agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])
```

### Option 2: Ollama (Local, No API Key)

Install [Ollama](https://ollama.ai), pull a model, and start the server:

```bash
ollama pull llama3.1
ollama serve
```

```julia
using Heval

# Native Ollama API
agent = HevalAgent(Val(:ollama); model="llama3.1")

# Or OpenAI-compatible endpoint
agent = HevalAgent(Val(:ollama); model="qwen2.5", use_openai_compat=true)
```

See the [Ollama Integration](ollama.md) guide for detailed setup.

---

## Example 1: Basic Forecasting

```julia
using Heval
using Dates

# Create agent
agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

# Monthly sales data
data = (
    date = Date(2020,1):Month(1):Date(2023,12),
    value = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195, 210, 225,
             130, 145, 160, 165, 155, 172, 185, 180, 198, 215, 235, 250,
             145, 160, 178, 185, 172, 195, 210, 205, 228, 248, 270, 290,
             162, 180, 200, 210, 195, 225, 245, 240, 268, 295, 325, 360]
)

# Run analysis — forecasts next 12 months
result = analyze(agent, data; h=12, query="Forecast next year's monthly sales")
```

The agent will automatically:
1. Analyze features (trend, seasonality, stationarity)
2. Cross-validate multiple models (ARIMA, ETS, Theta, etc.)
3. Generate forecasts with prediction intervals
4. Detect anomalies in the historical data

### Inspecting Results

```julia
# Best model and whether it beats SNaive baseline
result.best_model       # e.g., "ETS"
result.beats_baseline   # true

# Series features
result.features.trend_strength        # "strong"
result.features.seasonality_strength  # "moderate"
result.features.stationarity          # "non-stationary (d=1 suggested)"

# Model accuracy (sorted by MASE)
for (name, metrics) in result.accuracy
    println("$(name): MASE=$(metrics.mase), RMSE=$(metrics.rmse)")
end

# Forecast values
fc = result.forecasts
fc.point_forecasts   # Vector{Float64} of point forecasts
fc.lower_95          # Lower 95% prediction interval
fc.upper_95          # Upper 95% prediction interval
fc.dates             # Forecast dates

# Anomalies
for a in result.anomalies
    println("$(a.date): value=$(a.value), z-score=$(a.z_score)")
end

# Full LLM analysis text
println(result.output)
```

---

## Example 2: Follow-Up Questions

After running an analysis, ask follow-up questions:

```julia
answer = query(agent, "Why did you choose this model over ARIMA?")
# Displays a formatted box in the REPL, styled HTML in Jupyter

answer = query(agent, "What would happen if the trend continues?")

answer = query(agent, "Are there any seasonal patterns I should worry about?")
```

`QueryResult` is string-interoperable:

```julia
println(answer)                  # Print raw text
msg = "Analysis: " * string(answer)  # String concatenation
```

---

## Example 3: Intermittent Demand

For sparse data with many zeros, Heval automatically recommends Croston methods:

```julia
demand = (
    date = Date(2023,1):Day(1):Date(2023,90),
    value = [0,0,5,0,0,0,2,0,0,0,0,3,0,0,1,0,0,0,0,4,
             0,0,0,6,0,0,0,0,2,0,0,0,0,0,3,0,0,0,1,0,
             0,0,0,0,5,0,0,0,0,0,0,2,0,0,0,0,4,0,0,0,
             0,3,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,0,0,
             0,0,0,3,0,0,0,0,0,1]
)

result = analyze(agent, demand; m=7, h=30,
    query="This is intermittent demand data. Forecast the next month.")
```

---

## Example 4: Custom Model Request

Direct the agent to use specific models:

```julia
result = analyze(agent, data; h=12,
    query="Use Theta method for forecasting and compare with ETS and ARIMA")
```

---

## Example 5: Panel Data (Multiple Series)

Analyze multiple time series simultaneously:

```julia
using Heval
using Dates

# Multi-store sales data
dates = repeat(Date(2020,1):Month(1):Date(2022,12), 3)
stores = vcat(fill("Store_A", 36), fill("Store_B", 36), fill("Store_C", 36))
values = vcat(
    100 .+ 10 .* sin.(1:36) .+ 2 .* randn(36),   # Store A
    200 .+ 15 .* sin.(1:36) .+ 3 .* randn(36),   # Store B
    50 .+ 5 .* sin.(1:36) .+ randn(36)            # Store C
)

panel_data = (date=dates, store=stores, value=values)

agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

result = analyze(agent, panel_data;
    h=12, m=12,
    groupby=[:store],
    query="Forecast next year for all stores"
)
```

See the [Panel Data](panel.md) guide for more details.

---

## Example 6: Using a Different LLM Model

```julia
# Use GPT-4o-mini for faster, cheaper analysis
agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"], model="gpt-4o-mini")

# Use a custom OpenAI-compatible API
agent = HevalAgent(
    api_key="your-key",
    model="your-model",
    base_url="https://your-api.example.com/v1"
)
```

---

## Resetting the Agent

Clear state between analyses:

```julia
clear_history(agent)

# Now run a new analysis
result = analyze(agent, new_data; h=6)
```

---

## Next Steps

- **[Agent Architecture](agent.md)** — Understand how the agent loop works
- **[Available Models](models.md)** — Learn about all 16 forecasting models
- **[Analysis Tools](tools.md)** — Deep dive into feature analysis, CV, and anomaly detection
- **[Panel Data](panel.md)** — Multi-series forecasting with grouping
- **[Ollama Integration](ollama.md)** — Run Heval with local models
- **[Display & Formatting](display.md)** — Customize output display
