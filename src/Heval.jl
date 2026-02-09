"""
    Heval.jl

AI-powered forecasting agent for time series analysis, powered by Durbyn.jl.

"Heval" means "friend" in Kurdish - your companion for forecasting tasks.

# Quick Start
```julia
using Heval

# Create agent
agent = HevalAgent(api_key=ENV["OPENAI_API_KEY"])

# Analyze single time series
data = (date = Date.(2020, 1:24), value = rand(24) .* 100)
result = analyze(agent, data; h=12, query="Forecast next year")

# Panel data
panel = (date=dates, store=stores, value=values)
result = analyze(agent, panel; h=12, m=12, groupby=[:store])

# Ask follow-up questions
answer = query(agent, "Which model performed best?")
```
"""
module Heval

using Dates
using HTTP
using JSON3
using Statistics
using Random
using UUIDs
using Durbyn
import Tables

# Type definitions
include("types.jl")

# LLM interface
include("llm.jl")

# Durbyn bridge layer (model name → spec → fit → forecast)
include("durbyn_bridge.jl")

# Tool implementations (single-series + advanced)
include("tools.jl")

# Panel data tools
include("tools_panel.jl")

# Core agent
include("agent.jl")

# Ollama interface
include("ollama.jl")

# Pretty display for user-facing types (after all agent types are defined)
include("display.jl")

# Exports
export HevalAgent, OllamaAgent
export analyze, query, clear_history
export AgentResult, QueryResult, ForecastOutput, PanelState
export OllamaConfig
export list_ollama_models, check_ollama_connection
export AVAILABLE_MODELS

end # module
