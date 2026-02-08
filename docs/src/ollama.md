# Ollama Integration

Heval supports **Ollama** as a fully local LLM backend — no API keys, no cloud, no data leaving your machine. This is ideal for sensitive data, offline environments, or when you want to avoid API costs.

---

## Setup

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai).

### 2. Pull a Model

```bash
# Recommended models for tool-calling
ollama pull llama3.1        # Meta's Llama 3.1 (8B, good balance)
ollama pull qwen2.5         # Alibaba's Qwen 2.5 (strong tool-calling)
ollama pull mistral          # Mistral 7B (fast, capable)
```

### 3. Start the Server

```bash
ollama serve
```

The server runs at `http://localhost:11434` by default.

### 4. Verify Connection

```julia
using Heval

# Check if Ollama is running
check_ollama_connection()  # true

# List available models
models = list_ollama_models()
for m in models
    println(m)
end
```

---

## Creating an Ollama Agent

### Native Ollama API (Recommended)

```julia
using Heval

# Default: Llama 3.1 with native API
agent = HevalAgent(Val(:ollama))

# Specify model
agent = HevalAgent(Val(:ollama); model="qwen2.5")

# Custom server address
agent = HevalAgent(Val(:ollama);
    model="llama3.1",
    host="http://myserver:11434"
)
```

### OpenAI-Compatible Endpoint

Some models work better with the OpenAI-compatible endpoint:

```julia
agent = HevalAgent(Val(:ollama);
    model="qwen2.5",
    use_openai_compat=true
)
```

This uses Ollama's `/v1/chat/completions` endpoint instead of the native `/api/chat`.

---

## Usage

The Ollama agent shares the **exact same API** as the OpenAI agent:

```julia
using Heval
using Dates

agent = HevalAgent(Val(:ollama); model="llama3.1")

data = (
    date = Date(2020,1):Month(1):Date(2022,12),
    value = 100 .+ 10 .* sin.(1:36) .+ 2 .* randn(36)
)

# Analyze — same interface as OpenAI agent
result = analyze(agent, data; h=12, query="Forecast next year")

# Follow-up questions
answer = query(agent, "Why did you choose this model?")

# Reset
clear_history(agent)
```

### Panel Data

Panel data works the same way:

```julia
result = analyze(agent, panel_data;
    h=12, m=12,
    groupby=[:store],
    query="Forecast all stores"
)
```

---

## OllamaConfig

The underlying configuration for Ollama connections:

```julia
OllamaConfig(;
    model::String = "llama3.1",              # Model name
    host::String = "http://localhost:11434",  # Server address
    max_tokens::Int = 4096,                  # Max response tokens
    temperature::Float64 = 0.1,              # Sampling temperature
    use_openai_compat::Bool = false          # Use /v1 endpoint
)
```

---

## API Endpoints

Heval supports two Ollama endpoints:

| Endpoint | Path | When to Use |
|----------|------|-------------|
| **Native** | `/api/chat` | Default, recommended for most models |
| **OpenAI-compat** | `/v1/chat/completions` | If native tool-calling has issues |

The native API sends tool definitions in Ollama's format. The OpenAI-compatible endpoint uses the standard OpenAI tool-calling format.

---

## Model Recommendations

For tool-calling with Heval, these models work well:

| Model | Size | Tool-Calling | Speed | Notes |
|-------|------|--------------|-------|-------|
| **llama3.1** | 8B | Good | Fast | Best balance of quality and speed |
| **qwen2.5** | 7B | Excellent | Fast | Strong structured output |
| **mistral** | 7B | Good | Fast | Reliable baseline |
| **llama3.1:70b** | 70B | Excellent | Slow | Best quality, needs GPU |
| **qwen2.5:72b** | 72B | Excellent | Slow | Top-tier tool-calling |

!!! info "Model Size vs Quality"
    Larger models (70B+) produce better analysis narratives and more reliable tool-calling, but require significant GPU memory. The 7-8B models are practical for most use cases.

---

## Utility Functions

### `list_ollama_models`

List all models available on the Ollama server:

```julia
models = list_ollama_models()
# ["llama3.1:latest", "qwen2.5:latest", "mistral:latest"]

# Custom host
models = list_ollama_models(host="http://myserver:11434")
```

### `check_ollama_connection`

Check if the Ollama server is reachable:

```julia
is_running = check_ollama_connection()  # true/false

# Custom host
is_running = check_ollama_connection(host="http://myserver:11434")
```

---

## Troubleshooting

**Problem:** `check_ollama_connection()` returns `false`
- **Solution:** Start the Ollama server with `ollama serve`. Check that the port (11434) isn't blocked.

**Problem:** Tool calls fail or produce errors
- **Solution:** Try `use_openai_compat=true` or switch to a model with better tool-calling support (e.g., `qwen2.5`).

**Problem:** Analysis is slow
- **Solution:** Use a smaller model (7-8B). Ensure Ollama is using GPU acceleration (`ollama ps` to check).

**Problem:** Model produces poor analysis
- **Solution:** Try a larger model or switch to OpenAI for complex analyses. Local models may need more retries.
