# Heval.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://taf-society.github.io/Heval.jl/dev/) [![Build Status](https://github.com/taf-society/Heval.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/taf-society/Heval.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/taf-society/Heval.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/taf-society/Heval.jl)

**Heval** is an AI-powered forecasting agent for Julia. It combines large language models (LLMs) with [Durbyn.jl](https://github.com/taf-society/Durbyn.jl) — a comprehensive time series forecasting library — to automate and explain forecasting workflows through natural language.

Heval — Kurdish for "friend", embodies the idea of a helpful companion that guides you through the complexity of time series analysis with the clarity and rigor of production-grade statistical models.

> This site documents the development version. After your first tagged release, see **stable** docs for the latest release.

---

## About TAFS

**TAFS (Time Series Analysis and Forecasting Society)** is a non-profit association ("Verein") in Vienna, Austria. It connects academics, experts, practitioners, and students focused on time-series, forecasting, and decision science. Contributions remain fully open source.
Learn more at [taf-society.org](https://taf-society.org/).

---

## Installation

Heval is under active development. For the latest dev version:

```julia
using Pkg
Pkg.add(url="https://github.com/taf-society/Heval.jl")
```

!!! tip "Requirement: OpenAI API Key or Ollama"
    Heval requires an LLM backend. You can use either:
    - **OpenAI API** — Set `ENV["OPENAI_API_KEY"]` with your API key
    - **Ollama** — Run a local model server (no API key required)

    See the [Ollama Integration](ollama.md) guide for local model setup.

---

## How It Works

Heval provides an AI agent that orchestrates a structured forecasting workflow. You give it data and a question — the agent uses LLM-guided tool calls to analyze features, select models, generate forecasts, and detect anomalies.

```
┌────────────────────────────────────────────────────────────────┐
│                        HevalAgent                              │
│                                                                │
│  1. Feature Analysis    → STL decomposition, unit root tests   │
│  2. Model Selection     → Cross-validation with 16 models      │
│  3. Forecasting         → Point forecasts + prediction intervals│
│  4. Anomaly Detection   → Residual-based outlier detection     │
│  5. Natural Language     → Explain results in plain English     │
└────────────────────────────────────────────────────────────────┘
```

All 16 forecasting models are powered by **Durbyn.jl** — production-quality implementations of ARIMA, ETS, BATS, TBATS, Theta, Croston, ARAR, and more.

---

## Quick Example

```julia
using Heval
using Dates

# Create agent with OpenAI
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

The `AgentResult` displays a rich, formatted summary in both the REPL and Jupyter notebooks — including model accuracy tables, forecast tables with confidence intervals, and anomaly lists.

---

## Available Models (16 total)

Heval has access to 16 forecasting models, all backed by Durbyn.jl:

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

See the [Available Models](models.md) guide for detailed descriptions of each model.

---

## Key Features

<div class="feature-grid">
<div class="card">
<h3>AI-Powered Analysis</h3>
Natural language interface for forecasting tasks. Ask questions, get explanations, and iterate on your analysis interactively.
</div>
<div class="card">
<h3>Automatic Model Selection</h3>
Cross-validation with MASE-based comparison across 16 models. The best model must beat the SNaive baseline.
</div>
<div class="card">
<h3>Production Models</h3>
All models are full Durbyn.jl implementations — not toy versions. ARIMA, ETS, BATS, TBATS, Theta, Croston, and more.
</div>
<div class="card">
<h3>Panel Data Support</h3>
Analyze multiple time series at once with automatic grouping, parallel fitting, and grouped forecasts.
</div>
<div class="card">
<h3>Anomaly Detection</h3>
Residual-based outlier detection using actual model residuals and Z-score thresholding.
</div>
<div class="card">
<h3>Local LLM Support</h3>
Use Ollama for fully local, private forecasting with no API keys required. Supports Llama, Qwen, Mistral, and more.
</div>
</div>

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

## Built with Zana.jl

This package was developed with the help of [Zana.jl](https://taf-society.github.io/Zana.jl/dev/) — a terminal-based AI coding assistant for Julia that integrates LLMs directly into your development workflow.

---

## What's Next

- **[Quick Start](quickstart.md)** — Get started with your first analysis
- **User Guide** pages:
  - [Agent Architecture](agent.md) — How the agent works under the hood
  - [Available Models](models.md) — Detailed guide to all 16 forecasting models
  - [Analysis Tools](tools.md) — Tools the agent uses (features, CV, forecasting, anomalies)
  - [Panel Data](panel.md) — Multi-series analysis with grouping
  - [Ollama Integration](ollama.md) — Local LLM setup and usage
  - [Display & Formatting](display.md) — Rich output in REPL and Jupyter
- **[API Reference](api.md)** — Complete API documentation
