# ============================================================================
# Tool Implementations for Heval.jl — Durbyn Backend
# ============================================================================

using Durbyn
import Durbyn

# LLMs sometimes send numbers as strings; coerce safely.
_to_float(x::Real) = Float64(x)
_to_float(x::AbstractString) = parse(Float64, x)
_to_float(x, default::Float64) = try _to_float(x) catch; default end

_to_int(x::Real) = Int(round(x))
_to_int(x::AbstractString) = parse(Int, x)
_to_int(x, default::Int) = try _to_int(x) catch; default end

# Approximate p-value for ADF test by interpolation between critical values.
# ADF is a left-tail test: more negative tau → stronger evidence against unit root.
function _adf_approx_pvalue(tau_stat::Float64, cvals::Vector{Float64}, clevels::Vector{Float64})
    # Sort by critical value (most negative first) for interpolation
    perm = sortperm(cvals)
    sorted_cvals = cvals[perm]
    sorted_clevels = clevels[perm]

    if tau_stat <= sorted_cvals[1]
        return sorted_clevels[1]  # more extreme than smallest critical value
    elseif tau_stat >= sorted_cvals[end]
        return 1.0 - sorted_clevels[end]  # well above largest critical value
    end

    # Linear interpolation between critical values
    for i in 1:length(sorted_cvals)-1
        if sorted_cvals[i] <= tau_stat <= sorted_cvals[i+1]
            t = (tau_stat - sorted_cvals[i]) / (sorted_cvals[i+1] - sorted_cvals[i])
            return sorted_clevels[i] + t * (sorted_clevels[i+1] - sorted_clevels[i])
        end
    end
    return sorted_clevels[end]
end

# ============================================================================
# Tool: analyze_features
# ============================================================================

function create_analyze_features_tool(state::AgentState)
    Tool(
        "analyze_features",
        "Analyze time series characteristics including trend, seasonality, stationarity, and intermittency using STL decomposition and unit root tests. Call this FIRST to understand the data.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "compute_acf" => Dict(
                    "type" => "boolean",
                    "description" => "Whether to compute autocorrelation features (default: true)"
                )
            ),
            "required" => String[]
        ),
        function(args)
            tool_analyze_features(state, args)
        end
    )
end

function tool_analyze_features(state::AgentState, args::Dict)
    if isnothing(state.values)
        return Dict("error" => "No data loaded. Provide data to analyze().")
    end

    y = Float64.(state.values)
    m = state.seasonal_period
    n = length(y)

    features = SeriesFeatures()
    features.length = n
    features.mean = mean(y)
    features.std = std(y)
    features.seasonal_period = m

    # Trend analysis via MSTL decomposition
    if n > 2m && m > 1
        try
            mstl_result = Durbyn.Stats.mstl(y, m)
            trend_component = mstl_result.trend
            seasonal_component = mstl_result.seasonals[1]
            remainder = mstl_result.remainder

            # Compute trend slope from trend component
            n_trend = length(trend_component)
            x = collect(1.0:n_trend)
            x_mean = mean(x)
            t_mean = mean(trend_component)
            num = sum((x .- x_mean) .* (trend_component .- t_mean))
            den = sum((x .- x_mean) .^ 2)
            slope = den > 0 ? num / den : 0.0
            normalized_slope = t_mean != 0 ? slope / abs(t_mean) * n_trend : slope
            features.trend_slope = slope

            features.trend_strength = if abs(normalized_slope) > 0.5
                "strong"
            elseif abs(normalized_slope) > 0.1
                "moderate"
            else
                "weak"
            end

            # Seasonal strength from MSTL
            try
                ss_vec = Durbyn.Stats.seasonal_strength(mstl_result)
                ss = isempty(ss_vec) ? 0.0 : ss_vec[1]
                features.seasonality_strength = if ss > 0.6
                    "strong"
                elseif ss > 0.3
                    "moderate"
                else
                    "weak"
                end
                features.seasonal_acf = ss
            catch
                # Fallback to ACF-based seasonality
                acf_result = Durbyn.acf(y, m)
                seasonal_acf_val = acf_result.acf[m + 1]  # acf array is 0-indexed (lag 0, 1, ..., m)
                features.seasonal_acf = seasonal_acf_val
                features.seasonality_strength = if seasonal_acf_val > 0.6
                    "strong"
                elseif seasonal_acf_val > 0.3
                    "moderate"
                else
                    "weak"
                end
            end
        catch e
            # Fallback: simple trend detection
            features.trend_slope = 0.0
            features.trend_strength = "unknown"
            features.seasonality_strength = "unknown"
        end
    elseif n > 4
        # Not enough data for STL, use simple regression
        x = collect(1.0:n)
        x_mean = mean(x)
        y_mean = mean(y)
        num = sum((x .- x_mean) .* (y .- y_mean))
        den = sum((x .- x_mean) .^ 2)
        slope = den > 0 ? num / den : 0.0
        normalized_slope = y_mean != 0 ? slope / abs(y_mean) * n : slope
        features.trend_slope = slope
        features.trend_strength = if abs(normalized_slope) > 0.5
            "strong"
        elseif abs(normalized_slope) > 0.1
            "moderate"
        else
            "weak"
        end
        features.seasonality_strength = "insufficient_data"
    else
        features.trend_strength = "insufficient_data"
        features.seasonality_strength = "insufficient_data"
    end

    # Unit root tests for stationarity
    if n >= 10
        try
            nd = Durbyn.Stats.ndiffs(y)
            features.ndiffs = nd
            features.stationarity = nd == 0 ? "stationary" : "non-stationary (d=$nd suggested)"
        catch
            features.stationarity = "unknown"
        end
    end

    # Seasonal differencing
    if n > 2m && m > 1
        try
            nsd = Durbyn.Stats.nsdiffs(y, m)
            features.nsdiffs = nsd
        catch
            features.nsdiffs = 0
        end
    end

    # Intermittency check
    features.zero_fraction = sum(y .== 0) / n
    features.is_intermittent = features.zero_fraction > 0.3

    # Generate recommendations
    features.recommendations = generate_recommendations(features)

    # Store in state
    state.features = features

    return Dict(
        "status" => "success",
        "features" => Dict(
            "length" => features.length,
            "mean" => round(features.mean, digits=2),
            "std" => round(features.std, digits=2),
            "trend_strength" => features.trend_strength,
            "trend_slope" => round(features.trend_slope, digits=4),
            "seasonality_strength" => features.seasonality_strength,
            "seasonal_acf" => round(features.seasonal_acf, digits=3),
            "seasonal_period" => features.seasonal_period,
            "is_intermittent" => features.is_intermittent,
            "zero_fraction" => round(features.zero_fraction, digits=3),
            "stationarity" => features.stationarity,
            "ndiffs" => features.ndiffs,
            "nsdiffs" => features.nsdiffs
        ),
        "recommendations" => features.recommendations
    )
end

function generate_recommendations(features::SeriesFeatures)
    recs = String[]

    if features.is_intermittent
        push!(recs, "Croston/SBA (intermittent demand - $(round(features.zero_fraction*100, digits=1))% zeros)")
    end

    if features.seasonality_strength in ["strong", "moderate"]
        push!(recs, "ETS, HoltWinters, ARIMA with seasonal terms ($(features.seasonality_strength) seasonality)")
        push!(recs, "TBATS for complex/multiple seasonalities")
        push!(recs, "BATS for integer seasonal periods")
    else
        push!(recs, "SES, Holt, Theta, ARIMA (weak/no seasonality)")
        push!(recs, "ARAR for adaptive autoregressive modeling")
    end

    if features.trend_strength == "strong"
        push!(recs, "Holt with damping for strong trend")
        push!(recs, "Theta (optimized) for trend+level decomposition")
    end

    push!(recs, "SNaive as baseline (always include for comparison)")

    return recs
end

# ============================================================================
# Tool: cross_validate
# ============================================================================

function create_cross_validate_tool(state::AgentState)
    Tool(
        "cross_validate",
        "Evaluate models using time series cross-validation with Durbyn. Returns MASE, RMSE, MAE, MAPE for each model. Available models: $(join(AVAILABLE_MODELS, ", "))",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "models" => Dict(
                    "type" => "array",
                    "items" => Dict("type" => "string"),
                    "description" => "List of model names to evaluate"
                ),
                "n_windows" => Dict(
                    "type" => "integer",
                    "description" => "Number of CV windows (default: 3)"
                )
            ),
            "required" => ["models"]
        ),
        function(args)
            tool_cross_validate(state, args)
        end
    )
end

function tool_cross_validate(state::AgentState, args::Dict)
    if isnothing(state.values)
        return Dict("error" => "No data loaded")
    end

    model_names = get(args, "models", String[])
    n_windows = _to_int(get(args, "n_windows", 3), 3)

    if isempty(model_names)
        return Dict("error" => "No models specified")
    end

    # Validate models
    for name in model_names
        if !(name in AVAILABLE_MODELS)
            return Dict("error" => "Unknown model: $name. Available: $(join(AVAILABLE_MODELS, ", "))")
        end
    end

    y = Float64.(state.values)
    m = state.seasonal_period
    h = state.horizon
    n = length(y)

    # Time series CV: expanding window
    window_size = max(2m, div(n, 2))

    results = Dict{String, AccuracyMetrics}()

    for model_name in model_names
        window_maes = Float64[]
        window_rmses = Float64[]
        window_mapes = Float64[]
        window_mases = Float64[]

        for w in 1:n_windows
            train_end = window_size + (w - 1) * div(n - window_size - h, max(n_windows - 1, 1))
            test_end = min(train_end + h, n)

            if train_end >= n || test_end > n
                continue
            end

            train = y[1:train_end]
            test = y[train_end+1:test_end]
            test_h = length(test)

            if test_h == 0
                continue
            end

            # Generate forecast using Durbyn
            fc = _cv_forecast(train, test_h, m, model_name, state.dates)

            # Compute per-window metrics
            push!(window_maes, mean(abs.(test .- fc)))
            push!(window_rmses, sqrt(mean((test .- fc) .^ 2)))
            push!(window_mapes, mean(abs.((test .- fc) ./ max.(abs.(test), 1e-10))) * 100)

            # MASE using training set for seasonal naive baseline
            n_train = length(train)
            if n_train > m
                naive_errors = abs.(train[m+1:end] .- train[1:end-m])
                mae_naive = mean(naive_errors)
                mae_fc = mean(abs.(test .- fc))
                mase_val = mae_naive > 0 ? mae_fc / mae_naive : Inf
                push!(window_mases, mase_val)
            else
                push!(window_mases, Inf)
            end
        end

        if !isempty(window_maes)
            results[model_name] = AccuracyMetrics(
                model = model_name,
                mase = round(mean(window_mases), digits=4),
                rmse = round(mean(window_rmses), digits=2),
                mae = round(mean(window_maes), digits=2),
                mape = round(mean(window_mapes), digits=2)
            )
        end
    end

    # Store results
    state.accuracy = results

    # Find best model
    if !isempty(results)
        best = argmin(k -> results[k].mase, collect(keys(results)))
        state.best_model = best

        naive_mase = haskey(results, "SNaive") ? results["SNaive"].mase : Inf
        beats_naive = results[best].mase < naive_mase

        # Format results table
        table_lines = ["Model | MASE | RMSE | MAE | MAPE"]
        push!(table_lines, repeat("-", 50))

        for (name, metrics) in sort(collect(results), by=x -> x[2].mase)
            marker = name == best ? " *" : ""
            push!(table_lines, "$(name)$(marker) | $(metrics.mase) | $(metrics.rmse) | $(metrics.mae) | $(metrics.mape)%")
        end

        return Dict(
            "status" => "success",
            "results" => Dict(k => Dict(
                "mase" => v.mase, "rmse" => v.rmse, "mae" => v.mae, "mape" => v.mape
            ) for (k, v) in results),
            "best_model" => best,
            "best_mase" => results[best].mase,
            "beats_snaive" => beats_naive,
            "table" => join(table_lines, "\n")
        )
    else
        return Dict("error" => "No valid results computed")
    end
end

"""
    _cv_forecast(train, h, m, model_name, dates)

Generate a forecast for cross-validation using Durbyn.
Returns a Vector{Float64} of point forecasts.
"""
function _cv_forecast(train::Vector{Float64}, h::Int, m::Int, model_name::String,
                      dates::Union{Vector{Date}, Nothing})
    n = length(train)

    # Build dates for training window
    train_dates = if !isnothing(dates) && length(dates) >= n
        dates[1:n]
    else
        # Synthetic monthly dates
        [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]
    end

    try
        result = durbyn_fit_forecast(model_name, train_dates, train, h, m)
        return Float64.(result.forecast.mean)
    catch e
        # If a model fails, fall back to seasonal naive
        return _fallback_snaive(train, h, m)
    end
end

function _fallback_snaive(y::Vector{Float64}, h::Int, m::Int)
    n = length(y)
    forecasts = zeros(h)
    for i in 1:h
        idx = n - m + ((i - 1) % m) + 1
        if idx > 0 && idx <= n
            forecasts[i] = y[idx]
        else
            forecasts[i] = y[end]
        end
    end
    return forecasts
end

# ============================================================================
# Tool: generate_forecast
# ============================================================================

function create_forecast_tool(state::AgentState)
    Tool(
        "generate_forecast",
        "Generate forecasts using the specified model via Durbyn. Returns point forecasts with model-specific prediction intervals.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "model" => Dict(
                    "type" => "string",
                    "description" => "Model name to use"
                ),
                "h" => Dict(
                    "type" => "integer",
                    "description" => "Forecast horizon (default: use stored horizon)"
                )
            ),
            "required" => ["model"]
        ),
        function(args)
            tool_generate_forecast(state, args)
        end
    )
end

function tool_generate_forecast(state::AgentState, args::Dict)
    if isnothing(state.values)
        return Dict("error" => "No data loaded")
    end

    model_name = get(args, "model", nothing)
    h = _to_int(get(args, "h", state.horizon), state.horizon)

    if isnothing(model_name)
        return Dict("error" => "Model name required")
    end

    if !(model_name in AVAILABLE_MODELS)
        return Dict("error" => "Unknown model: $model_name")
    end

    y = Float64.(state.values)
    m = state.seasonal_period
    dates = state.dates

    # Use Durbyn fit + forecast for proper model-specific PIs
    try
        result = durbyn_fit_forecast(model_name, dates, y, h, m)
        fc = result.forecast

        # Store fitted model reference for anomaly detection, re-forecasting
        state.fitted_models[model_name] = result.fitted

        # Extract results
        fc_data = extract_forecast_result(fc, dates, h)

        # Generate forecast dates
        fc_dates = fc_data["dates"]

        # Store forecasts in state
        state.forecasts = ForecastOutput(
            model = model_name,
            horizon = h,
            point_forecasts = fc_data["point_forecasts"],
            lower_80 = fc_data["lower_80"],
            upper_80 = fc_data["upper_80"],
            lower_95 = fc_data["lower_95"],
            upper_95 = fc_data["upper_95"],
            dates = fc_dates
        )

        # Format output
        point_fc = fc_data["point_forecasts"]
        lower_95 = fc_data["lower_95"]
        upper_95 = fc_data["upper_95"]

        fc_summary = if !isempty(fc_dates)
            [string(fc_dates[i], ": ", round(point_fc[i], digits=2),
                    !isempty(lower_95) && !isempty(upper_95) ?
                    " [$(round(lower_95[i], digits=2)), $(round(upper_95[i], digits=2))]" : "")
             for i in 1:min(5, h)]
        else
            ["t+$i: $(round(point_fc[i], digits=2))" for i in 1:min(5, h)]
        end

        if h > 5
            push!(fc_summary, "... ($(h - 5) more periods)")
        end

        return Dict(
            "status" => "success",
            "model" => model_name,
            "method" => fc_data["method"],
            "horizon" => h,
            "forecasts" => join(fc_summary, "\n"),
            "summary" => Dict(
                "mean_forecast" => round(mean(point_fc), digits=2),
                "min_forecast" => round(minimum(point_fc), digits=2),
                "max_forecast" => round(maximum(point_fc), digits=2),
                "trend" => point_fc[end] > point_fc[1] ? "increasing" : "decreasing"
            )
        )
    catch e
        return Dict("error" => "Failed to forecast with $model_name: $(sprint(showerror, e))")
    end
end

# ============================================================================
# Tool: detect_anomalies
# ============================================================================

function create_anomaly_tool(state::AgentState)
    Tool(
        "detect_anomalies",
        "Detect anomalies/outliers in the time series using model residual analysis.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "model" => Dict(
                    "type" => "string",
                    "description" => "Model to use for residual calculation (default: STL decomposition)"
                ),
                "threshold" => Dict(
                    "type" => "number",
                    "description" => "Z-score threshold (default: 3.0)"
                )
            ),
            "required" => ["model"]
        ),
        function(args)
            tool_detect_anomalies(state, args)
        end
    )
end

function tool_detect_anomalies(state::AgentState, args::Dict)
    if isnothing(state.values)
        return Dict("error" => "No data loaded")
    end

    model_name = get(args, "model", "SES")
    threshold = _to_float(get(args, "threshold", 3.0), 3.0)

    y = Float64.(state.values)
    m = state.seasonal_period
    n = length(y)

    # Get residuals from Durbyn fitted model if available
    resid = nothing

    if haskey(state.fitted_models, model_name)
        try
            fitted_model = state.fitted_models[model_name]
            resid = Float64.(Durbyn.residuals(fitted_model))
        catch
        end
    end

    # Fallback: fit the model now and get residuals
    if isnothing(resid)
        try
            dates = !isnothing(state.dates) ? state.dates :
                    [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]
            fitted_model = durbyn_fit(model_name, dates, y, m)
            resid = Float64.(Durbyn.residuals(fitted_model))
            state.fitted_models[model_name] = fitted_model
        catch
            # Last resort: use STL remainder as residuals
            if n > 2m && m > 1
                try
                    stl_result = Durbyn.Stats.stl(y, m; s_window="periodic")
                    resid = Float64.(stl_result.time_series.remainder)
                catch
                end
            end
        end
    end

    # Ultimate fallback: diff-based residuals
    if isnothing(resid) || isempty(resid)
        resid = y .- mean(y)
    end

    # Z-score based anomaly detection
    mu = mean(resid)
    sigma = std(resid)

    if sigma < 1e-10
        return Dict(
            "status" => "success",
            "n_anomalies" => 0,
            "message" => "No variation in residuals - no anomalies detected"
        )
    end

    z_scores = (resid .- mu) ./ sigma
    anomaly_idx = findall(abs.(z_scores) .> threshold)

    # Build anomaly results
    anomalies = AnomalyResult[]
    for idx in anomaly_idx
        date = !isnothing(state.dates) && idx <= length(state.dates) ? state.dates[idx] : nothing
        push!(anomalies, AnomalyResult(idx, date, y[idx], z_scores[idx]))
    end

    state.anomalies = anomalies

    # Format output
    anomaly_details = [
        isnothing(a.date) ?
            "Index $(a.index): value=$(round(a.value, digits=2)), z=$(round(a.z_score, digits=2))" :
            "$(a.date): value=$(round(a.value, digits=2)), z=$(round(a.z_score, digits=2))"
        for a in anomalies[1:min(10, length(anomalies))]
    ]

    return Dict(
        "status" => "success",
        "model" => model_name,
        "threshold" => threshold,
        "n_anomalies" => length(anomalies),
        "anomaly_rate" => round(length(anomalies) / n * 100, digits=2),
        "anomalies" => anomaly_details,
        "summary" => if isempty(anomalies)
            "No anomalies detected at threshold=$threshold"
        else
            "Found $(length(anomalies)) anomalies ($(round(length(anomalies)/n*100, digits=1))% of data)"
        end
    )
end

# ============================================================================
# Tool: decompose
# ============================================================================

function create_decompose_tool(state::AgentState)
    Tool(
        "decompose",
        "Decompose the time series into trend, seasonal, and remainder components using STL or MSTL decomposition.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "method" => Dict(
                    "type" => "string",
                    "description" => "Decomposition method: 'stl' or 'mstl' (default: 'stl')"
                )
            ),
            "required" => String[]
        ),
        function(args)
            tool_decompose(state, args)
        end
    )
end

function tool_decompose(state::AgentState, args::Dict)
    if isnothing(state.values)
        return Dict("error" => "No data loaded")
    end

    method = lowercase(get(args, "method", "stl"))
    y = Float64.(state.values)
    m = state.seasonal_period
    n = length(y)

    if n <= 2m
        return Dict("error" => "Need at least 2*m=$(2m) observations for decomposition, have $n")
    end

    try
        if method == "mstl"
            result = Durbyn.Stats.mstl(y, m)
            return Dict(
                "status" => "success",
                "method" => "MSTL",
                "trend_summary" => Dict(
                    "mean" => round(mean(result.trend), digits=2),
                    "range" => round(maximum(result.trend) - minimum(result.trend), digits=2)
                ),
                "seasonal_summary" => Dict(
                    "n_components" => length(result.seasonals),
                    "strengths" => [round(std(s) / std(y), digits=3) for s in result.seasonals]
                ),
                "remainder_summary" => Dict(
                    "mean" => round(mean(result.remainder), digits=4),
                    "std" => round(std(result.remainder), digits=2)
                )
            )
        else
            result = Durbyn.Stats.stl(y, m; s_window="periodic")
            trend = result.time_series.trend
            seasonal = result.time_series.seasonal
            remainder = result.time_series.remainder

            # seasonal_strength needs MSTLResult, compute via mstl
            ss = try
                ss_vec = Durbyn.Stats.seasonal_strength(Durbyn.Stats.mstl(y, m))
                isempty(ss_vec) ? NaN : ss_vec[1]
            catch
                NaN
            end

            return Dict(
                "status" => "success",
                "method" => "STL",
                "trend_summary" => Dict(
                    "mean" => round(mean(trend), digits=2),
                    "range" => round(maximum(trend) - minimum(trend), digits=2),
                    "direction" => trend[end] > trend[1] ? "increasing" : "decreasing"
                ),
                "seasonal_summary" => Dict(
                    "strength" => round(ss, digits=3),
                    "amplitude" => round(maximum(seasonal) - minimum(seasonal), digits=2)
                ),
                "remainder_summary" => Dict(
                    "mean" => round(mean(remainder), digits=4),
                    "std" => round(std(remainder), digits=2)
                )
            )
        end
    catch e
        return Dict("error" => "Decomposition failed: $(sprint(showerror, e))")
    end
end

# ============================================================================
# Tool: unit_root_test
# ============================================================================

function create_unit_root_test_tool(state::AgentState)
    Tool(
        "unit_root_test",
        "Run stationarity tests (ADF, KPSS) and recommend differencing orders for ARIMA modeling.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "tests" => Dict(
                    "type" => "array",
                    "items" => Dict("type" => "string"),
                    "description" => "Tests to run: 'adf', 'kpss' (default: both)"
                )
            ),
            "required" => String[]
        ),
        function(args)
            tool_unit_root_test(state, args)
        end
    )
end

function tool_unit_root_test(state::AgentState, args::Dict)
    if isnothing(state.values)
        return Dict("error" => "No data loaded")
    end

    tests = get(args, "tests", ["adf", "kpss"])
    y = Float64.(state.values)
    m = state.seasonal_period

    results = Dict{String, Any}()

    for test_name in tests
        tname = lowercase(test_name)
        try
            if tname == "adf"
                result = Durbyn.Stats.adf(y)
                # ADF teststat is a NamedMatrix; tau statistic is in [1,1]
                tau_stat = result.teststat.data[1, 1]
                # Approximate p-value from critical values (left-tail test)
                tau_cvals = Float64[result.cval[1, j] for j in 1:length(result.clevels)]
                p_val = _adf_approx_pvalue(tau_stat, tau_cvals, result.clevels)
                results["adf"] = Dict(
                    "statistic" => round(tau_stat, digits=4),
                    "p_value" => round(p_val, digits=4),
                    "conclusion" => p_val < 0.05 ? "stationary (reject unit root)" : "non-stationary (cannot reject unit root)"
                )
            elseif tname == "kpss"
                result = Durbyn.Stats.kpss(y)
                # KPSS teststat is a plain Float64; pvalue() exists
                p_val = Durbyn.Stats.pvalue(result)
                results["kpss"] = Dict(
                    "statistic" => round(result.teststat, digits=4),
                    "p_value" => round(p_val, digits=4),
                    "conclusion" => p_val < 0.05 ? "non-stationary (reject stationarity)" : "stationary (cannot reject stationarity)"
                )
            else
                results[tname] = Dict("error" => "Unknown test: $tname. Use 'adf' or 'kpss'.")
            end
        catch e
            results[tname] = Dict("error" => "Test failed: $(sprint(showerror, e))")
        end
    end

    # Recommended differencing
    try
        nd = Durbyn.Stats.ndiffs(y)
        results["recommended_d"] = nd
    catch end

    if m > 1 && length(y) > 2m
        try
            nsd = Durbyn.Stats.nsdiffs(y, m)
            results["recommended_D"] = nsd
        catch end
    end

    return Dict("status" => "success", "results" => results)
end

# ============================================================================
# Tool: compare_models
# ============================================================================

function create_compare_models_tool(state::AgentState)
    Tool(
        "compare_models",
        "Fit multiple models to the full dataset and compare them using information criteria (AIC, BIC) and in-sample metrics. This complements cross_validate which uses out-of-sample metrics.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "models" => Dict(
                    "type" => "array",
                    "items" => Dict("type" => "string"),
                    "description" => "List of model names to compare"
                )
            ),
            "required" => ["models"]
        ),
        function(args)
            tool_compare_models(state, args)
        end
    )
end

function tool_compare_models(state::AgentState, args::Dict)
    if isnothing(state.values)
        return Dict("error" => "No data loaded")
    end

    model_names = get(args, "models", String[])
    if isempty(model_names)
        return Dict("error" => "No models specified")
    end

    y = Float64.(state.values)
    m = state.seasonal_period
    n = length(y)

    dates = !isnothing(state.dates) ? state.dates :
            [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]

    comparison = Dict{String, Any}()

    for model_name in model_names
        try
            fitted_model = durbyn_fit(model_name, dates, y, m)
            state.fitted_models[model_name] = fitted_model

            info = Dict{String, Any}("model" => model_name)

            # Try to extract information criteria
            for field in [:aic, :aicc, :bic, :sigma2]
                try
                    val = getproperty(fitted_model.fit, field)
                    if !isnothing(val) && val isa Real
                        info[string(field)] = round(Float64(val), digits=2)
                    end
                catch end
            end

            # In-sample residual metrics
            try
                resid = Float64.(Durbyn.residuals(fitted_model))
                info["rmse_insample"] = round(sqrt(mean(resid .^ 2)), digits=2)
                info["mae_insample"] = round(mean(abs.(resid)), digits=2)
            catch end

            comparison[model_name] = info
        catch e
            comparison[model_name] = Dict("model" => model_name, "error" => sprint(showerror, e))
        end
    end

    return Dict(
        "status" => "success",
        "comparison" => comparison,
        "n_models" => length(comparison)
    )
end
