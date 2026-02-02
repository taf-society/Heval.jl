# Heval.jl

> "Heval" means "friend" in Kurdish - your companion for time series forecasting.

**Heval.jl** is an AI-powered forecasting agent for Julia. It combines large language models (LLMs) with time series analysis to automate and explain forecasting workflows.

## Features

- 🤖 **AI-Powered Analysis**: Natural language interface for forecasting tasks
- 📊 **Automatic Model Selection**: Evaluates multiple models and selects the best
- 📈 **Built-in Forecasting Models**: Arima, ETS, ARARMA, TBATS, BATS, SES, Holt, Theta, Naive, Seasonal Naive, and more
- 🔍 **Anomaly Detection**: Identifies outliers using residual analysis
- 💬 **Interactive Queries**: Ask follow-up questions about your analysis
- 🔌 **Extensible**: Designed to integrate with [Durbyn.jl](https://github.com/taf-society//Durbyn.jl) for advanced models

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society//Heval.jl")
```

## Quick Start

```julia
using Heval
using Dates

# Create agent with your OpenAI API key
agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

# Prepare time series data
data = (
    date = Date(2020, 1, 1):Month(1):Date(2022, 12, 1),
    value = 100 .+ 10 .* sin.(1:36) .+ 2 .* randn(36)
)

# Run analysis
result = analyze(agent, data; h=12, query="Forecast next year's values")

# View results
println(result.output)           # LLM's analysis
println(result.best_model)       # Best performing model
println(result.forecasts)        # Forecast values
println(result.beats_baseline)   # Whether it beats SNaive

# Ask follow-up questions
answer = query(agent, "Why did you choose this model?")
println(answer)
```

## Workflow

Heval follows a structured forecasting workflow:

```
┌─────────────────────────────────────────────────────────────┐
│  1. FEATURE ANALYSIS                                        │
│     - Trend detection                                       │
│     - Seasonality assessment                                │
│     - Intermittency check                                   │
│     → Generates model recommendations                       │
├─────────────────────────────────────────────────────────────┤
│  2. MODEL SELECTION                                         │
│     - Cross-validation with multiple models                 │
│     - MASE-based comparison (scale-independent)             │
│     - Must beat SNaive baseline                             │
├─────────────────────────────────────────────────────────────┤
│  3. FORECASTING                                             │
│     - Generate point forecasts                              │
│     - Compute prediction intervals (80%, 95%)               │
├─────────────────────────────────────────────────────────────┤
│  4. ANOMALY DETECTION                                       │
│     - Residual-based outlier detection                      │
│     - Z-score thresholding                                  │
└─────────────────────────────────────────────────────────────┘
```

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `SES` | Simple Exponential Smoothing | No trend, no seasonality |
| `Holt` | Holt's Linear Trend | Trend, no seasonality |
| `Theta` | Theta Method | General purpose |
| `Naive` | Last observation | Baseline |
| `SNaive` | Seasonal naive | Seasonal baseline |
| `Meanf` | Historical mean | Simple baseline |
| `ARIMA`* | Auto-regressive integrated moving average | Complex patterns |
| `ETS`* | Error-Trend-Seasonality | Exponential smoothing |
| `BATS`* | Multi-seasonal (integer periods) | Multiple seasonalities |
| `TBATS`* | Multi-seasonal (Fourier) | Non-integer seasonalities |
| `HoltWinters`* | Seasonal Holt-Winters | Trend + seasonality |
| `Croston`* | Intermittent demand | Sparse data |

*Models marked with * use simplified implementations. For full implementations, integrate with Durbyn.jl.

## API Reference

### `HevalAgent`

```julia
agent = HevalAgent(;
    api_key::String,           # Required: OpenAI API key
    model::String = "gpt-4o",  # LLM model to use
    base_url::String = "https://api.openai.com/v1",
    max_retries::Int = 3       # Retry attempts for validation
)
```

### `analyze`

```julia
result = analyze(agent, data;
    h::Int = nothing,          # Forecast horizon (default: 2*m)
    m::Int = nothing,          # Seasonal period (default: 12)
    query::String = nothing    # Natural language instructions
)
```

**Returns**: `AgentResult` with fields:
- `output::String` - LLM's analysis text
- `features::SeriesFeatures` - Extracted time series features
- `accuracy::Dict{String, AccuracyMetrics}` - Model evaluation results
- `forecasts::ForecastOutput` - Generated forecasts
- `anomalies::Vector{AnomalyResult}` - Detected anomalies
- `best_model::String` - Name of best performing model
- `beats_baseline::Bool` - Whether best model beats SNaive

### `query`

```julia
answer = query(agent, "Your question here")
```

Ask follow-up questions about the analysis results.

### `clear_history`

```julia
clear_history(agent)
```

Reset agent state for a new analysis.

## Data Formats

Heval accepts multiple input formats:

```julia
# NamedTuple (recommended)
data = (date = Date.(2020, 1:12), value = rand(12))

# Vector (dates will be auto-generated)
data = rand(12)

# Dict
data = Dict("date" => Date.(2020, 1:12), "value" => rand(12))
```

## Integration with Durbyn.jl

For advanced models (ARIMA, ETS, BATS, TBATS, etc.), Heval can integrate with [Durbyn.jl](https://github.com/taf-society//Durbyn.jl):

```julia
using Heval
using Durbyn

# When Durbyn is loaded, Heval automatically uses its implementations
agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

# Now ARIMA, ETS, etc. use full Durbyn implementations
result = analyze(agent, data; h=12)
```

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Examples

### Basic Forecasting

```julia
using Heval, Dates

agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

# Monthly sales data
sales = (
    date = Date(2020,1):Month(1):Date(2023,12),
    value = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195, 210, 225,
             130, 145, 160, 165, 155, 172, 185, 180, 198, 215, 235, 250,
             145, 160, 178, 185, 172, 195, 210, 205, 228, 248, 270, 290,
             162, 180, 200, 210, 195, 225, 245, 240, 268, 295, 325, 360]
)

result = analyze(agent, sales; h=12, query="Forecast next year's monthly sales")
```

### Intermittent Demand

```julia
# Sparse demand data (many zeros)
demand = (
    date = Date(2023,1):Day(1):Date(2023,90),
    value = [0,0,5,0,0,0,2,0,0,0,0,3,0,0,1,0,0,0,0,4,  # ... more zeros
             0,0,0,6,0,0,0,0,2,0,0,0,0,0,3,0,0,0,1,0,
             0,0,0,0,5,0,0,0,0,0,0,2,0,0,0,0,4,0,0,0,
             0,3,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,0,0,
             0,0,0,3,0,0,0,0,0,1]
)

result = analyze(agent, demand; m=7, h=30,
    query="This is intermittent demand data. Forecast the next month.")
```

### Custom Model Request

```julia
result = analyze(agent, data; h=12,
    query="Use Theta method for forecasting, I want to compare with ETS")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      HevalAgent                             │
├─────────────────────────────────────────────────────────────┤
│  LLMConfig           │ API configuration                    │
│  AgentState          │ Data, results, history               │
│  Tools               │ analyze_features, cross_validate,    │
│                      │ generate_forecast, detect_anomalies  │
│  System Prompt       │ Workflow instructions                │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Tool Calling Loop                                          │
│  1. LLM receives prompt + tool definitions                  │
│  2. LLM calls tools (analyze_features, etc.)                │
│  3. Tool results returned to LLM                            │
│  4. LLM generates final analysis                            │
│  5. Validation: best model must beat SNaive                 │
│  6. Retry if validation fails                               │
└─────────────────────────────────────────────────────────────┘
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

*"Heval" (هەڤال) - Kurdish for "friend, companion"*
