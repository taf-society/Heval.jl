# Analysis Tools

Heval's agent operates through a set of **tools** — functions that the LLM can call to perform analysis steps. Each tool receives parameters from the LLM and returns structured results back to it.

!!! tip "You Don't Call Tools Directly"
    Tools are called by the LLM agent as part of the automated workflow. You interact with Heval through `analyze()` and `query()` — the agent decides which tools to invoke and in what order.

---

## Core Tools

### analyze_features

Analyzes time series characteristics using STL decomposition and unit root tests.

**What it does:**
- Computes trend strength and slope via MSTL decomposition
- Measures seasonal strength (from Durbyn's `seasonal_strength`)
- Tests stationarity via `ndiffs` and `nsdiffs`
- Detects intermittency (zero fraction > 30%)
- Generates model recommendations based on features

**Returns:**

| Field | Description |
|-------|-------------|
| `trend_strength` | "strong", "moderate", or "weak" |
| `trend_slope` | Linear slope of trend component |
| `seasonality_strength` | "strong", "moderate", or "weak" |
| `seasonal_acf` | Seasonal autocorrelation (or seasonal strength value) |
| `is_intermittent` | Whether >30% of values are zero |
| `stationarity` | "stationary" or "non-stationary (d=N suggested)" |
| `ndiffs` | Recommended regular differencing order |
| `nsdiffs` | Recommended seasonal differencing order |
| `recommendations` | List of suggested models |

**Stored in:** `agent.state.features` as `SeriesFeatures`

---

### cross_validate

Evaluates models using **time series cross-validation** with an expanding window.

**How it works:**
1. Splits data into expanding training windows and fixed-size test sets
2. For each window, fits the model on training data and forecasts the test period
3. Computes error metrics per window, then averages across windows
4. MASE is computed per-window using the training set's seasonal naive errors

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | `Array{String}` | Models to evaluate (required) |
| `n_windows` | `Int` | Number of CV windows (default: 3) |

**Metrics returned:**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MASE** | Mean Absolute Scaled Error | < 1 = better than SNaive |
| **RMSE** | Root Mean Square Error | Scale-dependent |
| **MAE** | Mean Absolute Error | Scale-dependent |
| **MAPE** | Mean Absolute Percentage Error | Percentage-based |

**Stored in:** `agent.state.accuracy` as `Dict{String, AccuracyMetrics}`

!!! info "Why Per-Window MASE?"
    MASE is computed per cross-validation window and then averaged — not computed on concatenated predictions. This preserves the seasonal lag relationship used for scaling.

---

### generate_forecast

Generates forecasts using a specified model via Durbyn.

**What it does:**
1. Fits the model to the full dataset using Durbyn's `fit()` + `forecast()`
2. Extracts **model-specific prediction intervals** (80% and 95%)
3. Infers forecast dates from the data's actual date spacing
4. Caches the fitted model for subsequent anomaly detection

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `String` | Model name (required) |
| `h` | `Int` | Forecast horizon (default: stored horizon) |

**Returns:**
- Point forecasts with dates
- 80% and 95% confidence intervals (model-specific, not simplified)
- Forecast trend direction and summary statistics

**Stored in:** `agent.state.forecasts` as `ForecastOutput`

---

### detect_anomalies

Detects outliers using **residual-based analysis**.

**How it works:**
1. Retrieves residuals from the Durbyn fitted model (cached from `generate_forecast`)
2. If no fitted model exists, fits the specified model first
3. Computes Z-scores of residuals: ``z_i = (r_i - \bar{r}) / \sigma_r``
4. Flags observations where ``|z_i| > \text{threshold}``

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `String` | Model for residuals (required) |
| `threshold` | `Float64` | Z-score threshold (default: 3.0) |

**Fallback chain:**
1. Use cached fitted model residuals
2. Fit the model now and extract residuals
3. Use STL decomposition remainder
4. Use deviations from mean (last resort)

**Stored in:** `agent.state.anomalies` as `Vector{AnomalyResult}`

---

## Advanced Tools

### decompose

Decomposes the time series into **trend, seasonal, and remainder** components.

**Methods:**
- **STL** (default): Seasonal and Trend decomposition using Loess. Single seasonal period.
- **MSTL**: Multiple Seasonal-Trend decomposition. Handles multiple seasonal periods.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `method` | `String` | "stl" or "mstl" (default: "stl") |

**Returns:**
- Trend summary (mean, range, direction)
- Seasonal summary (strength, amplitude, number of components for MSTL)
- Remainder summary (mean, std)

---

### unit_root_test

Runs **stationarity tests** and recommends differencing orders.

**Tests available:**

| Test | Null Hypothesis | Reject means |
|------|-----------------|--------------|
| **ADF** | Unit root present | Series is stationary |
| **KPSS** | Series is stationary | Series is non-stationary |

!!! info "ADF + KPSS Together"
    Using both tests together helps resolve ambiguous cases. When ADF rejects (stationary) and KPSS doesn't reject (stationary), you have strong evidence of stationarity. When both are ambiguous, differencing may be appropriate.

**Also returns:**
- `recommended_d`: Regular differencing order (from `ndiffs`)
- `recommended_D`: Seasonal differencing order (from `nsdiffs`)

---

### compare_models

Compares models using **in-sample information criteria** (AIC, BIC).

This complements `cross_validate` (out-of-sample) with in-sample metrics:

| Criterion | Description |
|-----------|-------------|
| **AIC** | Akaike Information Criterion |
| **AICc** | Corrected AIC (for small samples) |
| **BIC** | Bayesian Information Criterion |
| **sigma2** | Residual variance |

Also reports in-sample RMSE and MAE from model residuals.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | `Array{String}` | Models to compare (required) |

---

## Recommended Workflow

The agent follows this workflow by default:

```
1. analyze_features          ← Understand data characteristics
2. cross_validate            ← Select best model (out-of-sample)
3. generate_forecast         ← Produce forecasts with CIs
4. detect_anomalies          ← Flag outliers
```

Optional advanced steps:
- `decompose` — Deeper understanding of seasonal structure
- `unit_root_test` — Detailed stationarity analysis for ARIMA modeling
- `compare_models` — In-sample comparison as a complement to CV

---

## Model Recommendations

After analyzing features, the agent generates model recommendations based on:

| Data Characteristic | Recommended Models |
|--------------------|--------------------|
| Intermittent (>30% zeros) | Croston/SBA |
| Strong seasonality | ETS, HoltWinters, ARIMA, TBATS, BATS |
| Weak/no seasonality | SES, Holt, Theta, ARIMA, ARAR |
| Strong trend | Holt (damped), Theta (optimized) |
| Always included | SNaive (baseline) |
