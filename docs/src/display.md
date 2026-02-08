# Display & Formatting

Heval provides **rich display** for all result types in both the REPL and Jupyter notebooks. This page describes how results are rendered and how to access the underlying data.

---

## AgentResult Display

### REPL (text/plain)

In the Julia REPL, `AgentResult` displays a structured summary:

```
Heval Analysis Result
═══════════════════════════════════════════════════════
  Best model: ETS (MASE: 0.7234)  PASS

── Series Features ────────────────────────────────────
  Length: 48 obs  |  Period: 12
  Trend: strong (slope: 0.0234)
  Seasonality: moderate (ACF: 0.45)
  Stationarity: non-stationary (d=1, D=0)

── Model Accuracy ─────────────────────────────────────
  Model      │ MASE   │ RMSE   │ MAE    │ MAPE
  ───────────┼────────┼────────┼────────┼────────
  ETS *      │ 0.7234 │ 12.45  │ 8.91   │ 4.52%
  ARIMA      │ 0.7891 │ 13.21  │ 9.56   │ 4.89%
  Theta      │ 0.8123 │ 14.02  │ 10.12  │ 5.12%
  SNaive     │ 1.0    │ 18.34  │ 14.21  │ 7.23%
  (* = best)

── Forecast (ETS, h=12) ──────────────────────────────
  Date        │ Point  │ 80% CI            │ 95% CI
  ────────────┼────────┼───────────────────┼──────────
  2024-01-01  │ 175.2  │ [162.1, 188.3]    │ [155.4, 195.0]
  2024-02-01  │ 180.5  │ [165.8, 195.2]    │ [158.1, 202.9]
  ...         │ ...    │ ...               │ ...
  2024-12-01  │ 215.8  │ [190.2, 241.4]    │ [176.8, 254.8]

── Anomalies (2 detected) ────────────────────────────
  2022-03-01: value=148.0, z=3.21
  2023-08-01: value=240.0, z=-3.05

── Analysis ───────────────────────────────────────────
  The ETS model with automatic component selection...
  (1523 chars total — access result.output for full text)
```

Features:
- Color-coded **PASS/FAIL** badge (green/red in color terminals)
- Best model highlighted with cyan and star marker
- Tables with Unicode box-drawing characters
- Forecast table with dates, point forecasts, and confidence intervals
- Truncated LLM analysis (full text in `result.output`)

### Jupyter (text/html)

In Jupyter notebooks, `AgentResult` renders as styled HTML with:
- Blue highlighted best model row
- Green/red PASS/FAIL badge
- Scrollable analysis narrative
- Responsive table layout

---

## QueryResult Display

### REPL

`QueryResult` displays as a bordered box with word wrapping:

```
╭─ Heval ──────────────────────────────────────────────╮
│ The ETS model was selected because it achieved the   │
│ lowest MASE score (0.7234) in cross-validation. It   │
│ outperformed ARIMA and Theta, likely because the     │
│ data exhibits multiplicative seasonality which ETS   │
│ handles natively through its M component.            │
╰──────────────────────────────────────────────────────╯
```

### Jupyter

Renders as a styled HTML container with the "Heval" header.

### String Interoperability

`QueryResult` is fully string-interoperable:

```julia
answer = query(agent, "Why this model?")

# Print raw text
println(answer)

# Convert to string
s = string(answer)

# String concatenation
msg = "The agent said: " * string(answer)

# Works with print/string/show
Base.print(io, answer)  # prints content
Base.string(answer)     # returns content string
```

---

## ForecastOutput Display

### REPL

```
Forecast (ETS, h=12)
  Date        │ Point  │ 80% CI            │ 95% CI
  ────────────┼────────┼───────────────────┼──────────
  2024-01-01  │ 175.2  │ [162.1, 188.3]    │ [155.4, 195.0]
  2024-02-01  │ 180.5  │ [165.8, 195.2]    │ [158.1, 202.9]
  ...
  2024-12-01  │ 215.8  │ [190.2, 241.4]    │ [176.8, 254.8]
  (6 of 12 rows shown)
```

For forecasts with more than 6 periods, shows the first 5 and last row with an ellipsis.

### Jupyter

Styled HTML table with date, point forecast, and confidence interval columns.

---

## SeriesFeatures Display

### REPL

```
Series Features
  Length: 48 obs  |  Period: 12
  Mean: 185.42  |  Std: 45.21
  Trend: strong (slope: 0.0234)
  Seasonality: moderate (ACF: 0.45)
  Stationarity: non-stationary (d=1, D=0)
  Intermittent: yes (zero fraction: 0.35)
  Recommendations: Croston/SBA, ETS, HoltWinters
```

### Jupyter

Key-value HTML layout with labels and values.

---

## AccuracyMetrics Display

### REPL

```
AccuracyMetrics — ETS
  MASE: 0.7234
  RMSE: 12.45
  MAE:  8.91
  MAPE: 4.52%
```

### Compact (single-line)

```
AccuracyMetrics(ETS: MASE=0.7234)
```

---

## Compact Show

All types have a single-line compact representation:

```julia
AccuracyMetrics(ETS: MASE=0.7234)
SeriesFeatures(n=48, trend=strong)
Anomaly(2022-03-01, z=3.21)
ForecastOutput(ETS, h=12)
AgentResult(best=ETS, MASE=0.7234)
QueryResult("The ETS model was selected because...")
```

---

## Accessing Raw Data

All display types are regular Julia structs — access fields directly:

```julia
result = analyze(agent, data; h=12)

# Forecast data
fc = result.forecasts
fc.point_forecasts    # Vector{Float64}
fc.lower_95           # Vector{Float64}
fc.upper_95           # Vector{Float64}
fc.dates              # Vector{Date}
fc.model              # String
fc.horizon            # Int

# Features
f = result.features
f.trend_strength      # String
f.seasonal_period     # Int
f.ndiffs              # Int

# Accuracy
for (name, m) in result.accuracy
    println("$(name): MASE=$(m.mase)")
end

# Anomalies
for a in result.anomalies
    println("$(a.date): z=$(a.z_score)")
end
```
