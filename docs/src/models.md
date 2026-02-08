# Available Models

Heval has access to 16 forecasting models, all powered by [Durbyn.jl](https://github.com/taf-society/Durbyn.jl). The agent selects the best model automatically through cross-validation, but you can also guide model selection through natural language queries.

!!! tip "Model Selection"
    You don't need to choose a model yourself. The agent's `cross_validate` tool evaluates multiple candidates and selects the one with the lowest MASE. The best model must beat the **SNaive** baseline.

---

## Exponential Smoothing Family

### SES — Simple Exponential Smoothing

Best for data with **no trend and no seasonality**. Produces flat forecasts at a weighted average of recent observations.

```math
\begin{aligned}
\hat{y}_{t+h|t} &= \ell_t \\
\ell_t &= \alpha y_t + (1-\alpha)\ell_{t-1}
\end{aligned}
```

**When to use:** Short series, no clear patterns, baseline comparison.

### Holt — Holt's Linear Trend

Extends SES with a **linear trend** component. Optionally damped to prevent forecast "runaway".

```math
\begin{aligned}
\hat{y}_{t+h|t} &= \ell_t + h b_t \\
\ell_t &= \alpha y_t + (1-\alpha)(\ell_{t-1} + b_{t-1}) \\
b_t &= \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1}
\end{aligned}
```

**When to use:** Trending data without seasonality.

### HoltWinters — Seasonal Holt-Winters

Adds a **seasonal component** to Holt's method. Supports both additive and multiplicative seasonality.

**When to use:** Data with both trend and seasonality. Choose multiplicative when seasonal amplitude grows with the level.

### ETS — Error-Trend-Seasonality

The most general exponential smoothing framework. Automatically selects the best combination of:
- **Error**: Additive (A) or Multiplicative (M)
- **Trend**: None (N), Additive (A), Additive Damped (Ad)
- **Seasonality**: None (N), Additive (A), Multiplicative (M)

The "Z" (automatic) option lets Durbyn search all admissible combinations.

**When to use:** General-purpose model — a strong first choice for most time series.

---

## ARIMA Family

### ARIMA — Auto-Regressive Integrated Moving Average

Combines **autoregression** (AR), **differencing** (I), and **moving average** (MA) components. Durbyn's `auto_arima` searches over order combinations to minimize AIC.

Seasonal ARIMA (SARIMA) includes seasonal AR, differencing, and MA terms: ``ARIMA(p,d,q)(P,D,Q)[m]``

**When to use:** Complex stationary/differenced patterns, data with clear autocorrelation structure.

### ARAR — Adaptive AR with Memory Reduction

The ARAR algorithm (Brockwell & Davis) applies **memory-shortening transformations** before fitting an AR model. This makes it effective for long-memory and non-stationary series without explicit differencing.

**When to use:** Long-memory time series, when ARIMA's fixed differencing doesn't capture the dynamics well.

### ARARMA — Adaptive AR + ARMA

Combines ARAR's memory-shortening with an **ARMA model** for the whitened series (Parzen). This gives the flexibility of ARMA after adaptive preprocessing.

**When to use:** When ARAR alone doesn't capture short-range dependencies.

---

## Multi-Seasonal Models

### BATS — Box-Cox, ARMA errors, Trend, Seasonal

Handles complex seasonal patterns with **integer seasonal periods**. Includes optional Box-Cox transformation and ARMA error correction.

Following De Livera, Hyndman & Snyder (2011):

```math
y_t^{(\omega)} = \ell_{t-1} + \phi b_{t-1} + \sum_{i=1}^{T} s_{t-m_i}^{(i)} + d_t
```

**When to use:** Multiple integer seasonal periods (e.g., hourly data with daily and weekly patterns).

### TBATS — Trigonometric BATS

Extension of BATS using **Fourier representation** for seasonal components. This allows:
- **Non-integer seasonal periods** (e.g., 52.18 weeks/year)
- **Very long cycles** efficiently (``O(k)`` instead of ``O(m)``)
- **Dual calendar effects** (e.g., Gregorian + Hijri)

**When to use:** Non-integer seasonality, very long seasonal periods, multiple complex cycles.

---

## Theta Methods

### Theta — Theta Method (STM/OTM/DSTM/DOTM)

The Theta method decomposes the series into **theta lines** — modifications of the original series that emphasize different characteristics. Durbyn implements the Dynamic Optimised Theta Method (Fiorucci et al., 2016).

Variants:
- **STM**: Standard Theta Method
- **OTM**: Optimised Theta Method
- **DSTM**: Dynamic Standard Theta Method
- **DOTM**: Dynamic Optimised Theta Method

**When to use:** Competition-proven method, good general-purpose performance, especially for M-competition-style data.

---

## Intermittent Demand

### Croston — Intermittent Demand Forecasting

Specialized for time series with **many zero values and occasional non-zero spikes**. Separately models demand size and inter-demand interval.

Variants:
- **Classic**: Original Croston (1972) method
- **SBA**: Syntetos-Boylan Approximation — bias-corrected, generally best
- **SBJ**: Shale-Boylan-Johnston correction — alternative bias correction

**When to use:** Spare parts inventory, slow-moving retail items, any series with >30% zeros.

---

## Growth Models

### Diffusion — S-Curve Growth Models

Models technology adoption and market penetration using S-curve functions:
- **Bass**: Innovation + imitation diffusion
- **Gompertz**: Asymmetric S-curve
- **GSGompertz**: Generalized shifted Gompertz
- **Weibull**: Flexible hazard-based growth

**When to use:** Product lifecycle forecasting, technology adoption curves, market saturation analysis.

---

## Baseline Methods

### Naive

Repeats the **last observed value** for all forecast horizons. The simplest possible forecast.

```math
\hat{y}_{t+h|t} = y_t
```

### SNaive — Seasonal Naive

Repeats the **last seasonal cycle**. This is the primary baseline — all models are compared against it.

```math
\hat{y}_{t+h|t} = y_{t+h-m}
```

!!! info "Why SNaive Matters"
    Heval requires the best model to beat SNaive. A model that can't outperform "repeat last year's pattern" isn't providing value. This is a standard benchmark in forecasting competitions (M3, M4, M5).

### RW — Random Walk

Random walk model, optionally with **drift** (constant trend). Equivalent to ARIMA(0,1,0).

### Meanf — Historical Mean

Forecasts the **historical mean** for all horizons. Useful as a baseline for stationary series.

---

## Model Selection Criteria

The agent uses **MASE (Mean Absolute Scaled Error)** as the primary selection metric:

```math
\text{MASE} = \frac{\text{MAE}_{\text{forecast}}}{\text{MAE}_{\text{seasonal naive}}}
```

- MASE < 1: Model is better than seasonal naive
- MASE = 1: Model is equivalent to seasonal naive
- MASE > 1: Model is worse than seasonal naive

MASE is preferred because it is:
- **Scale-independent** — works across different data magnitudes
- **Symmetric** — treats over- and under-prediction equally
- **Interpretable** — directly compares to the seasonal naive baseline

Additional metrics reported: **RMSE**, **MAE**, **MAPE**.

---

## Guiding Model Selection

You can influence which models the agent considers:

```julia
# Let the agent decide (default)
result = analyze(agent, data; h=12)

# Suggest specific models
result = analyze(agent, data; h=12,
    query="Compare ARIMA, ETS, and Theta for this data")

# Request intermittent demand methods
result = analyze(agent, data; h=12,
    query="This is sparse demand data, use Croston methods")

# Force a specific model
result = analyze(agent, data; h=12,
    query="Use BATS for forecasting — this data has daily and weekly seasonality")
```

---

## References

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- De Livera, A.M., Hyndman, R.J., & Snyder, R.D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. *Journal of the American Statistical Association*, 106(496), 1513-1527.
- Fiorucci, J.A., Pellegrini, T.R., Louzada, F., Petropoulos, F., & Koehler, A.B. (2016). Models for optimising the theta method and their relationship to state space models. *International Journal of Forecasting*, 32(4), 1151-1161.
- Syntetos, A.A., & Boylan, J.E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303-314.
- Brockwell, P.J., & Davis, R.A. (1991). *Time Series: Theory and Methods* (2nd ed.). Springer.
