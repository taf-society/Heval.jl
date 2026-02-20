# Heval.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://taf-society.github.io/Heval.jl/dev/) [![Build Status](https://github.com/taf-society/Heval.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taf-society/Heval.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/taf-society/Heval.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/taf-society/Heval.jl)

## About

**Heval** is an AI-powered forecasting agent for Julia. It combines large language models (LLMs) with [Durbyn.jl](https://github.com/taf-society/Durbyn.jl) — a comprehensive time series forecasting library — to automate and explain forecasting workflows through natural language.

Heval — Kurdish for "friend", embodies the idea of a helpful companion that guides you through the complexity of time series analysis with the clarity and rigor of production-grade statistical models.

This package is currently under development and is part of the **TAFS Forecasting Ecosystem**, an open-source initiative.

## About TAFS

**TAFS (Time Series Analysis and Forecasting Society)** is a non-profit association registered as a **"Verein"** in Vienna, Austria. The organization connects a global audience of academics, experts, practitioners, and students to engage, share, learn, and innovate in the fields of data science and artificial intelligence, with a particular focus on time-series analysis, forecasting, and decision science. [TAFS](https://taf-society.org/)

TAFS's mission includes:

-   **Connecting**: Hosting events and discussion groups to establish connections and build a community of like-minded individuals.
-   **Learning**: Providing a platform to learn about the latest research, real-world problems, and applications.
-   **Sharing**: Inviting experts, academics, practitioners, and others to present and discuss problems, research, and solutions.
-   **Innovating**: Supporting the transfer of research into solutions and helping to drive innovations.

As a registered non-profit association under Austrian law, TAFS ensures that all contributions remain fully open source and cannot be privatized or commercialized. [TAFS](https://taf-society.org/)

## License

The Heval package is licensed under the **MIT License**, allowing for open-source distribution and collaboration.

## Installation

Heval is still in development. For the latest development version, install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society/Heval.jl")
```

## How It Works

Heval provides an AI agent that orchestrates a structured forecasting workflow. You give it time series data and a question — the agent uses LLM-guided tool calls to analyze features, select models, generate forecasts, and detect anomalies.

1. **Feature Analysis** — STL decomposition, trend/seasonality detection, stationarity tests
2. **Model Selection** — Cross-validation with MASE-based comparison across 16 models
3. **Forecasting** — Point forecasts with model-specific 80% and 95% prediction intervals
4. **Anomaly Detection** — Residual-based outlier detection using Z-score thresholding
5. **Explanation** — Natural language interpretation of results

All 16 forecasting models are powered by **Durbyn.jl** — production-quality implementations of ARIMA, ETS, BATS, TBATS, Theta, Croston, ARAR, and more.

## Available Models (16 total)

| Model | Description | Best For |
|-------|-------------|----------|
| **ARIMA** | Auto ARIMA with seasonal components | Complex AR/MA patterns |
| **ETS** | Error-Trend-Seasonality (automatic) | General purpose |
| **BATS** | Box-Cox, ARMA, Trend, Seasonal | Integer seasonal periods |
| **TBATS** | Trigonometric BATS | Non-integer seasonalities |
| **Theta** | Theta method (STM/OTM/DSTM/DOTM) | Competition benchmarks |
| **SES** | Simple Exponential Smoothing | No trend, no seasonality |
| **Holt** | Holt's linear trend | Trending, no seasonality |
| **HoltWinters** | Seasonal Holt-Winters | Trend + seasonality |
| **Croston** | Intermittent demand (SBA/SBJ) | Sparse demand data |
| **ARAR** | Adaptive AR with memory reduction | Adaptive autoregression |
| **ARARMA** | ARAR + ARMA | Adaptive + short-memory |
| **Diffusion** | S-curve growth (Bass, Gompertz) | Technology adoption |
| **Naive** | Last value repeated | Simplest baseline |
| **SNaive** | Seasonal naive | Seasonal baseline |
| **RW** | Random walk with optional drift | Stochastic baseline |
| **Meanf** | Historical mean | Constant forecast |

---

## Quick Start

### OpenAI

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

# Run full analysis
result = analyze(agent, data; h=12, query="Forecast next year's values")

# Inspect results
result.best_model       # Best performing model name
result.beats_baseline   # Whether it beats SNaive
result.forecasts        # ForecastOutput with point forecasts + CIs
result.features         # SeriesFeatures (trend, seasonality, etc.)
result.anomalies        # Detected outliers

# Ask follow-up questions
answer = query(agent, "Why did you choose this model?")
```

### Ollama (Local, No API Key)

```julia
using Heval

# Using native Ollama API
agent = HevalAgent(Val(:ollama); model="llama3.1")

# Using OpenAI-compatible endpoint
agent = HevalAgent(Val(:ollama); model="qwen2.5", use_openai_compat=true)

# Same API as OpenAI agent
result = analyze(agent, data; h=12, query="Forecast next year")
```

---

## Complete Workflow Example

```julia
using Heval
using Dates

agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

# Monthly sales data (4 years)
sales = (
    date = Date(2020,1):Month(1):Date(2023,12),
    value = [120, 135, 148, 152, 141, 158, 170, 165, 180, 195, 210, 225,
             130, 145, 160, 165, 155, 172, 185, 180, 198, 215, 235, 250,
             145, 160, 178, 185, 172, 195, 210, 205, 228, 248, 270, 290,
             162, 180, 200, 210, 195, 225, 245, 240, 268, 295, 325, 360]
)

# 1. Run analysis
result = analyze(agent, sales; h=12, query="Forecast next year's monthly sales")

# 2. Check model performance
println("Best model: $(result.best_model)")
println("Beats baseline: $(result.beats_baseline)")

# Model accuracy table (sorted by MASE)
for (name, metrics) in sort(collect(result.accuracy), by=x -> x[2].mase)
    marker = name == result.best_model ? " *" : ""
    println("  $(name)$(marker): MASE=$(metrics.mase), RMSE=$(metrics.rmse)")
end

# 3. Access forecasts
fc = result.forecasts
for i in 1:fc.horizon
    println("$(fc.dates[i]): $(round(fc.point_forecasts[i], digits=1)) " *
            "[$(round(fc.lower_95[i], digits=1)), $(round(fc.upper_95[i], digits=1))]")
end

# 4. Check anomalies
for a in result.anomalies
    println("Anomaly at $(a.date): value=$(a.value), z-score=$(round(a.z_score, digits=2))")
end

# 5. Ask follow-up questions (conversation history is preserved automatically)
answer = query(agent, "What drives the seasonal pattern?")
println(answer)

answer = query(agent, "Explain that for a non-technical audience")  # knows what "that" refers to
println(answer)
```

---

## Panel Data (Multiple Series)

Analyze multiple time series simultaneously:

```julia
using Heval
using Dates

# Multi-store sales data
dates = repeat(Date(2020,1):Month(1):Date(2022,12), 3)
stores = vcat(fill("Store_A", 36), fill("Store_B", 36), fill("Store_C", 36))
values = vcat(
    100 .+ 10 .* sin.(1:36) .+ 2 .* randn(36),
    200 .+ 15 .* sin.(1:36) .+ 3 .* randn(36),
    50 .+ 5 .* sin.(1:36) .+ randn(36)
)

panel_data = (date=dates, store=stores, value=values)

agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

result = analyze(agent, panel_data;
    h=12, m=12,
    groupby=[:store],
    query="Forecast next year for all stores"
)
```

---

## Data Formats

Heval accepts multiple input formats:

```julia
# NamedTuple (recommended)
data = (date = Date.(2020, 1:12), value = rand(12))

# Vector (dates auto-generated)
data = rand(12)

# Dict
data = Dict("date" => Date.(2020, 1:12), "value" => rand(12))

# Tables.jl-compatible (for panel data)
data = (date=dates, store=stores, value=values)
```

---

## Intermittent Demand

For sparse data with many zeros:

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

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        HevalAgent                            │
├──────────────────────────────────────────────────────────────┤
│  LLMConfig / OllamaConfig   │ API configuration             │
│  AgentState                  │ Data, results, history        │
│  Tools (7 core + 2 panel)   │ analyze_features, cross_      │
│                              │ validate, generate_forecast,  │
│                              │ detect_anomalies, decompose,  │
│                              │ unit_root_test, compare_models│
│  System Prompt               │ Workflow instructions         │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Tool Calling Loop                                           │
│  1. LLM receives prompt + tool definitions                   │
│  2. LLM calls tools (analyze_features, etc.)                 │
│  3. Tool results returned to LLM                             │
│  4. LLM generates final analysis                             │
│  5. Validation: best model must beat SNaive                  │
│  6. Retry if validation fails (up to max_retries)            │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Durbyn.jl (16 Models)                                       │
│  ARIMA, ETS, BATS, TBATS, Theta, SES, Holt, HoltWinters,   │
│  Croston, ARAR, ARARMA, Diffusion, Naive, SNaive, RW, Meanf│
└──────────────────────────────────────────────────────────────┘
```

---

## What's Next

- **[Quick Start](https://taf-society.github.io/Heval.jl/dev/quickstart/)** — Get started with your first analysis
- **User Guide**:
  - [Agent Architecture](https://taf-society.github.io/Heval.jl/dev/agent/) — How the agent works under the hood
  - [Available Models](https://taf-society.github.io/Heval.jl/dev/models/) — Detailed guide to all 16 forecasting models
  - [Analysis Tools](https://taf-society.github.io/Heval.jl/dev/tools/) — Feature analysis, cross-validation, anomaly detection
  - [Panel Data](https://taf-society.github.io/Heval.jl/dev/panel/) — Multi-series forecasting with grouping
  - [Ollama Integration](https://taf-society.github.io/Heval.jl/dev/ollama/) — Local LLM setup and usage
  - [Display & Formatting](https://taf-society.github.io/Heval.jl/dev/display/) — Rich output in REPL and Jupyter
- **[API Reference](https://taf-society.github.io/Heval.jl/dev/api/)** — Complete API documentation
