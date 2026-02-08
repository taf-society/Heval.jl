# Panel Data

Heval supports **panel data** (multiple time series) analysis through Durbyn.jl's `PanelData` infrastructure. This enables forecasting across multiple series — such as sales by store, demand by product, or metrics by region — in a single analysis call.

---

## Quick Example

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

## How Panel Analysis Works

When you pass `groupby` to `analyze()`, Heval switches to the panel data path:

1. **Panel Construction** — Wraps data in a `PanelData` object with group keys, date column, and seasonal period
2. **Feature Analysis** — Extracts features from the combined data and per-group summaries
3. **Model Fitting** — Fits model collection across all groups (leveraging Durbyn's parallel fitting)
4. **Forecasting** — Generates grouped forecasts for each series
5. **Interpretation** — The LLM summarizes results across groups

### Panel Tools

Two additional tools are registered for panel analysis:

**panel_analyze** — Constructs the PanelData object and reports:
- Number of groups
- Series lengths
- Group columns and date configuration

**panel_fit** — Fits one or more models across all groups:
- Builds a Durbyn model collection from model names
- Calls `Durbyn.fit(collection, panel)` for parallel fitting
- Generates grouped forecasts with `Durbyn.forecast(fitted, h=h)`

---

## Parameters

```julia
result = analyze(agent, data;
    h = 12,                    # Forecast horizon
    m = 12,                    # Seasonal period
    groupby = [:store],        # Grouping column(s) — triggers panel mode
    date = :date,              # Date column name (default: :date)
    target = :value,           # Target column name (default: :value)
    query = "Analyze all stores"
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `groupby` | `Symbol` or `Vector{Symbol}` | Column(s) to group by |
| `date` | `Symbol` | Date column name |
| `target` | `Symbol` | Target variable column name |

---

## Multiple Grouping Columns

You can group by multiple columns for hierarchical data:

```julia
# Region × Product grouping
data = (
    date = dates,
    region = regions,
    product = products,
    value = values
)

result = analyze(agent, data;
    h=6, m=12,
    groupby=[:region, :product],
    query="Forecast demand by region and product"
)
```

---

## Panel Workflow

The agent follows this workflow for panel data:

```
1. panel_analyze    ← Construct PanelData, count groups
2. analyze_features ← Overall data characteristics
3. panel_fit        ← Fit models across all groups
4. Interpret        ← Summarize results across groups
```

The standard single-series tools (cross_validate, detect_anomalies) are also available and operate on the aggregated data.

---

## PanelState

Panel analysis state is stored in `agent.state.panel`:

```julia
agent.state.panel.raw_data         # Original input data
agent.state.panel.groups           # Grouping column symbols
agent.state.panel.date_col         # Date column symbol
agent.state.panel.target_col       # Target column symbol
agent.state.panel.panel            # Durbyn.PanelData object
agent.state.panel.group_features   # Per-group SeriesFeatures
agent.state.panel.group_accuracy   # Per-group accuracy metrics
agent.state.panel.group_forecasts  # Durbyn grouped forecasts
```

---

## Data Formats

Panel data should be in **long format** with at least three columns:
- A **date** column
- One or more **grouping** columns
- A **target** (value) column

```julia
# Long format (required)
data = (
    date = [Date(2020,1), Date(2020,2), Date(2020,1), Date(2020,2)],
    store = ["A", "A", "B", "B"],
    value = [100, 110, 200, 220]
)
```

Any Tables.jl-compatible format works (NamedTuple of vectors, DataFrame, etc.).

!!! tip "Wide to Long"
    If your data is in wide format (one column per series), reshape it to long format before passing to Heval. Durbyn's `TableOps.pivot_longer` can help with this.
