# ============================================================================
# Durbyn Bridge Layer
#
# Maps between Heval's string-based LLM interface and Durbyn's typed
# spec/fit/forecast API. This is the only file that knows about both worlds.
# ============================================================================

using Durbyn
import Tables

# ── Available Models ───────────────────────────────────────────────────────

const AVAILABLE_MODELS = [
    "ARIMA", "ETS", "BATS", "TBATS", "Theta",
    "SES", "Holt", "HoltWinters", "Croston",
    "ARAR", "ARARMA", "Diffusion",
    "Naive", "SNaive", "RW", "Meanf"
]

# ── Model Spec Builders ───────────────────────────────────────────────────
# Each function returns a Durbyn AbstractModelSpec from a target symbol.

function build_spec(model_name::String, target::Symbol; m::Union{Int,Nothing}=nothing, kwargs...)
    name = uppercase(model_name)
    if name == "ARIMA"
        return _build_arima_spec(target; m=m, kwargs...)
    elseif name == "ETS"
        return _build_ets_spec(target; m=m, kwargs...)
    elseif name == "BATS"
        return _build_bats_spec(target; m=m, kwargs...)
    elseif name == "TBATS"
        return _build_tbats_spec(target; m=m, kwargs...)
    elseif name == "THETA"
        return _build_theta_spec(target; m=m, kwargs...)
    elseif name == "SES"
        return _build_ses_spec(target; m=m, kwargs...)
    elseif name == "HOLT"
        return _build_holt_spec(target; m=m, kwargs...)
    elseif name == "HOLTWINTERS"
        return _build_holtwinters_spec(target; m=m, kwargs...)
    elseif name == "CROSTON"
        return _build_croston_spec(target; m=m, kwargs...)
    elseif name == "ARAR"
        return _build_arar_spec(target; kwargs...)
    elseif name == "ARARMA"
        return _build_ararma_spec(target; kwargs...)
    elseif name == "DIFFUSION"
        return _build_diffusion_spec(target; kwargs...)
    elseif name == "NAIVE"
        return _build_naive_spec(target; m=m, kwargs...)
    elseif name == "SNAIVE"
        return _build_snaive_spec(target; m=m, kwargs...)
    elseif name == "RW"
        return _build_rw_spec(target; m=m, kwargs...)
    elseif name == "MEANF"
        return _build_meanf_spec(target; m=m, kwargs...)
    else
        error("Unknown model: $model_name. Available: $(join(AVAILABLE_MODELS, ", "))")
    end
end

function _build_arima_spec(target::Symbol; m=nothing, kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.p(), Durbyn.q(), Durbyn.P(), Durbyn.Q()
    ])
    return Durbyn.ArimaSpec(formula; m=m)
end

function _build_ets_spec(target::Symbol; m=nothing, kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.e("Z"), Durbyn.t("Z"), Durbyn.s("Z")
    ])
    return Durbyn.EtsSpec(formula; m=m)
end

function _build_bats_spec(target::Symbol; m=nothing, kwargs...)
    sp = isnothing(m) ? nothing : m
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.bats(; seasonal_periods=sp)
    ])
    return Durbyn.BatsSpec(formula)
end

function _build_tbats_spec(target::Symbol; m=nothing, kwargs...)
    sp = isnothing(m) ? nothing : Float64(m)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.tbats(; seasonal_periods=sp)
    ])
    return Durbyn.TbatsSpec(formula)
end

function _build_theta_spec(target::Symbol; m=nothing, kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.theta()
    ])
    return Durbyn.ThetaSpec(formula; m=m)
end

function _build_ses_spec(target::Symbol; m=nothing, kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.ses()
    ])
    return Durbyn.SesSpec(formula; m=m)
end

function _build_holt_spec(target::Symbol; m=nothing, kwargs...)
    damped = get(kwargs, :damped, nothing)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.holt(; damped=damped)
    ])
    return Durbyn.HoltSpec(formula; m=m)
end

function _build_holtwinters_spec(target::Symbol; m=nothing, kwargs...)
    seasonal = get(kwargs, :seasonal, "additive")
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.hw(; seasonal=seasonal)
    ])
    return Durbyn.HoltWintersSpec(formula; m=m)
end

function _build_croston_spec(target::Symbol; m=nothing, kwargs...)
    method = get(kwargs, :method, "sba")
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.croston(; method=method)
    ])
    return Durbyn.CrostonSpec(formula; m=m)
end

function _build_arar_spec(target::Symbol; kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.arar()
    ])
    return Durbyn.ArarSpec(formula)
end

function _build_ararma_spec(target::Symbol; kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.p(), Durbyn.q()
    ])
    return Durbyn.ArarmaSpec(formula)
end

function _build_diffusion_spec(target::Symbol; kwargs...)
    model_type = get(kwargs, :model, nothing)
    term = Durbyn.Grammar.diffusion(; model=model_type)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[term])
    return Durbyn.DiffusionSpec(formula)
end

function _build_naive_spec(target::Symbol; m=nothing, kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.naive_term()
    ])
    return Durbyn.NaiveSpec(formula; m=m)
end

function _build_snaive_spec(target::Symbol; m=nothing, kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.snaive_term()
    ])
    return Durbyn.SnaiveSpec(formula; m=m)
end

function _build_rw_spec(target::Symbol; m=nothing, kwargs...)
    use_drift = get(kwargs, :drift, false)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.rw_term(; drift=use_drift)
    ])
    return Durbyn.RwSpec(formula; m=m)
end

function _build_meanf_spec(target::Symbol; m=nothing, kwargs...)
    formula = Durbyn.ModelFormula(target, Durbyn.AbstractTerm[
        Durbyn.meanf_term()
    ])
    return Durbyn.MeanfSpec(formula; m=m)
end

# ── Single-Series Fit + Forecast ──────────────────────────────────────────

function durbyn_fit(model_name::String, dates::Vector{Date}, values::Vector{Float64}, m::Int;
                    target::Symbol=:value)
    data = (; date=dates, value=values)
    spec = build_spec(model_name, target; m=m)
    return Durbyn.fit(spec, data; m=m)
end

function durbyn_forecast(fitted_model, h::Int; level::Vector{<:Real}=[80, 95])
    return Durbyn.forecast(fitted_model; h=h, level=level)
end

function durbyn_fit_forecast(model_name::String, dates::Vector{Date}, values::Vector{Float64},
                             h::Int, m::Int; target::Symbol=:value, level::Vector{<:Real}=[80, 95])
    fitted = durbyn_fit(model_name, dates, values, m; target=target)
    fc = durbyn_forecast(fitted, h; level=level)
    return (fitted=fitted, forecast=fc)
end

# ── Forecast Result Extraction ────────────────────────────────────────────
# Convert a Durbyn Forecast object into Heval-friendly Dicts for the LLM.

function extract_forecast_result(fc, dates::Vector{Date}, h::Int)
    point_forecasts = Float64.(fc.mean)

    # Extract prediction intervals
    lower_80 = Float64[]
    upper_80 = Float64[]
    lower_95 = Float64[]
    upper_95 = Float64[]

    levels = fc.level
    for (i, lvl) in enumerate(levels)
        lvl_val = lvl >= 1 ? lvl : lvl * 100
        lo = _extract_pi_vector(fc.lower, i, h)
        hi = _extract_pi_vector(fc.upper, i, h)
        if round(Int, lvl_val) == 80
            lower_80 = lo
            upper_80 = hi
        elseif round(Int, lvl_val) == 95
            lower_95 = lo
            upper_95 = hi
        end
    end

    # Infer forecast dates from data dates
    fc_dates = if length(dates) >= 2
        date_diff = dates[end] - dates[end-1]
        [dates[end] + i * date_diff for i in 1:h]
    else
        [dates[end] + Dates.Day(i) for i in 1:h]
    end

    # Extract fitted values and residuals if available
    fitted_vals = try Float64.(fc.fitted) catch; Float64[] end
    resid_vals = try Float64.(fc.residuals) catch; Float64[] end

    return Dict{String, Any}(
        "point_forecasts" => point_forecasts,
        "lower_80" => lower_80,
        "upper_80" => upper_80,
        "lower_95" => lower_95,
        "upper_95" => upper_95,
        "dates" => fc_dates,
        "method" => try fc.method catch; "unknown" end,
        "fitted_values" => fitted_vals,
        "residuals" => resid_vals,
    )
end

function _extract_pi_vector(bounds, level_idx::Int, h::Int)
    if bounds isa AbstractMatrix
        return Float64[bounds[i, level_idx] for i in 1:h]
    elseif bounds isa AbstractVector
        if level_idx <= length(bounds)
            element = bounds[level_idx]
            if element isa AbstractVector
                return Float64.(element)
            elseif element isa Number
                return fill(Float64(element), h)
            end
        end
        # Fallback: single-level vector
        if length(bounds) == h && level_idx == 1
            return Float64.(bounds)
        end
    end
    return Float64[]
end

# ── Model Collection for Cross-Validation ─────────────────────────────────

function build_model_collection(model_names::Vector{String}, target::Symbol; m::Union{Int,Nothing}=nothing)
    specs = [build_spec(name, target; m=m) for name in model_names]
    return Durbyn.model(specs...; names=model_names)
end

# ── Panel Data Construction ───────────────────────────────────────────────

function build_panel(data; groupby::Vector{Symbol}, date::Symbol=:date,
                     m::Union{Int,Nothing}=nothing, frequency::Union{Symbol,Nothing}=nothing,
                     fill_time::Bool=false, balanced::Bool=false, target::Union{Symbol,Nothing}=nothing)
    return Durbyn.PanelData(data; groupby=groupby, date=date, m=m,
                            frequency=frequency, fill_time=fill_time, balanced=balanced,
                            target=target)
end

# ── Stats Bridge ──────────────────────────────────────────────────────────
# Expose Durbyn.Stats diagnostics through simple function calls.

function compute_acf(y::AbstractVector, m::Int, max_lag::Union{Int,Nothing}=nothing)
    result = Durbyn.acf(y, m, max_lag)
    return result.values
end

function compute_stl(y::AbstractVector, m::Int)
    return Durbyn.Stats.stl(y, m; s_window="periodic")
end

function compute_seasonal_strength(y::AbstractVector, m::Int)
    mstl_result = Durbyn.Stats.mstl(y, m)
    return Durbyn.Stats.seasonal_strength(mstl_result)
end

function run_adf_test(y::AbstractVector)
    return Durbyn.Stats.adf(y)
end

function run_kpss_test(y::AbstractVector)
    return Durbyn.Stats.kpss(y)
end

function compute_ndiffs(y::AbstractVector)
    return Durbyn.Stats.ndiffs(y)
end

function compute_nsdiffs(y::AbstractVector, m::Int)
    return Durbyn.Stats.nsdiffs(y, m)
end

function compute_decompose(y::AbstractVector, m::Int)
    return Durbyn.Stats.decompose(x=y, m=m)
end

function compute_mstl(y::AbstractVector, m::Union{Int, Vector{Int}})
    return Durbyn.Stats.mstl(y, m)
end
