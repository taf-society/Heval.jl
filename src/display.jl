# ============================================================================
# Pretty Display for Heval.jl Types
# ============================================================================

using Dates: Date, Dates

# ── Helper Utilities ─────────────────────────────────────────────────────────

_round4(x::Float64) = isfinite(x) ? round(x, digits=4) : x
_round2(x::Float64) = isfinite(x) ? round(x, digits=2) : x

function _truncate_text(s::AbstractString, maxlen::Int=200)
    length(s) <= maxlen && return s
    return s[1:prevind(s, maxlen)] * "…"
end

function _fmt_ci(lo::Float64, hi::Float64)
    isfinite(lo) && isfinite(hi) || return "—"
    return "[$(round(lo, digits=1)), $(round(hi, digits=1))]"
end

function _section_header(io::IO, title::String, width::Int=51)
    use_color = get(io, :color, false)::Bool
    bar = "─" ^ max(width - length(title) - 3, 4)
    println(io, "")
    println(io, "── ", title, " ", bar)
end

function _pass_fail_str(io::IO, beats::Bool)
    use_color = get(io, :color, false)::Bool
    if use_color
        if beats
            printstyled(io, "PASS"; color=:green, bold=true)
        else
            printstyled(io, "FAIL"; color=:red, bold=true)
        end
    else
        print(io, beats ? "PASS" : "FAIL")
    end
end

function _fmt_metric(x::Float64; pct::Bool=false)
    !isfinite(x) && return "—"
    s = string(round(x, digits=pct ? 2 : 4))
    pct && (s *= "%")
    return s
end

# ── Compact show (single-line) ──────────────────────────────────────────────

function Base.show(io::IO, m::AccuracyMetrics)
    print(io, "AccuracyMetrics($(m.model): MASE=$(_round4(m.mase)))")
end

function Base.show(io::IO, f::SeriesFeatures)
    print(io, "SeriesFeatures(n=$(f.length), trend=$(f.trend_strength))")
end

function Base.show(io::IO, a::AnomalyResult)
    dstr = isnothing(a.date) ? "idx=$(a.index)" : string(a.date)
    print(io, "Anomaly($(dstr), z=$(round(a.z_score, digits=2)))")
end

function Base.show(io::IO, fc::ForecastOutput)
    print(io, "ForecastOutput($(fc.model), h=$(fc.horizon))")
end

function Base.show(io::IO, r::AgentResult)
    bm = isnothing(r.best_model) ? "none" : r.best_model
    mase_str = if !isnothing(r.best_model) && haskey(r.accuracy, r.best_model)
        string(_round4(r.accuracy[r.best_model].mase))
    else
        "—"
    end
    print(io, "AgentResult(best=$(bm), MASE=$(mase_str))")
end

# ── MIME"text/plain" verbose display ─────────────────────────────────────────

# AccuracyMetrics
function Base.show(io::IO, ::MIME"text/plain", m::AccuracyMetrics)
    println(io, "AccuracyMetrics — $(m.model)")
    println(io, "  MASE: $(_fmt_metric(m.mase))")
    println(io, "  RMSE: $(_fmt_metric(m.rmse))")
    println(io, "  MAE:  $(_fmt_metric(m.mae))")
    print(io,   "  MAPE: $(_fmt_metric(m.mape; pct=true))")
end

# SeriesFeatures
function Base.show(io::IO, ::MIME"text/plain", f::SeriesFeatures)
    println(io, "Series Features")
    println(io, "  Length: $(f.length) obs  |  Period: $(f.seasonal_period)")
    println(io, "  Mean: $(_round2(f.mean))  |  Std: $(_round2(f.std))")
    slope_str = f.trend_slope != 0.0 ? " (slope: $(_round4(f.trend_slope)))" : ""
    println(io, "  Trend: $(f.trend_strength)$(slope_str)")
    acf_str = f.seasonal_acf != 0.0 ? " (ACF: $(_round2(f.seasonal_acf)))" : ""
    println(io, "  Seasonality: $(f.seasonality_strength)$(acf_str)")
    stat_str = "$(f.stationarity)"
    if f.ndiffs > 0 || f.nsdiffs > 0
        stat_str *= " (d=$(f.ndiffs), D=$(f.nsdiffs))"
    end
    println(io, "  Stationarity: $(stat_str)")
    if f.is_intermittent
        println(io, "  Intermittent: yes (zero fraction: $(_round2(f.zero_fraction)))")
    end
    if !isempty(f.recommendations)
        print(io, "  Recommendations: ", join(f.recommendations, ", "))
    end
end

# ForecastOutput
function Base.show(io::IO, ::MIME"text/plain", fc::ForecastOutput)
    println(io, "Forecast ($(fc.model), h=$(fc.horizon))")
    isempty(fc.point_forecasts) && return

    has_80 = !isempty(fc.lower_80) && !isempty(fc.upper_80)
    has_95 = !isempty(fc.lower_95) && !isempty(fc.upper_95)
    has_dates = !isempty(fc.dates)

    # Decide which rows to show
    n = length(fc.point_forecasts)
    if n > 6
        show_idxs = vcat(1:5, n)
        ellipsis_after = 5
    else
        show_idxs = collect(1:n)
        ellipsis_after = -1
    end

    # Header
    hdr = "  "
    hdr *= rpad(has_dates ? "Date" : "#", 12) * "│ " * rpad("Point", 7)
    has_80 && (hdr *= "│ " * rpad("80% CI", 18))
    has_95 && (hdr *= "│ " * rpad("95% CI", 18))
    println(io, hdr)

    sep = "  " * "─"^12 * "┼" * "─"^8
    has_80 && (sep *= "┼" * "─"^19)
    has_95 && (sep *= "┼" * "─"^19)
    println(io, sep)

    for (j, i) in enumerate(show_idxs)
        if j > 1 && show_idxs[j-1] != i - 1
            row = "  " * rpad("...", 12) * "│ " * rpad("...", 7)
            has_80 && (row *= "│ " * rpad("...", 18))
            has_95 && (row *= "│ " * rpad("...", 18))
            println(io, row)
        end
        dstr = has_dates ? string(fc.dates[i]) : string(i)
        row = "  " * rpad(dstr, 12) * "│ " * rpad(string(_round2(fc.point_forecasts[i])), 7)
        has_80 && (row *= "│ " * rpad(_fmt_ci(fc.lower_80[i], fc.upper_80[i]), 18))
        has_95 && (row *= "│ " * rpad(_fmt_ci(fc.lower_95[i], fc.upper_95[i]), 18))
        println(io, row)
    end

    if n > 6
        print(io, "  ($(length(show_idxs)) of $(n) rows shown)")
    end
end

# AgentResult (composite)
function Base.show(io::IO, ::MIME"text/plain", r::AgentResult)
    use_color = get(io, :color, false)::Bool
    W = 51

    # ── Header ──
    if use_color
        printstyled(io, "Heval Analysis Result\n"; bold=true)
    else
        println(io, "Heval Analysis Result")
    end
    println(io, "═" ^ W)

    # Best model summary
    if !isnothing(r.best_model)
        mase_str = if haskey(r.accuracy, r.best_model)
            "MASE: $(_round4(r.accuracy[r.best_model].mase))"
        else
            ""
        end
        print(io, "  Best model: $(r.best_model)")
        !isempty(mase_str) && print(io, " (", mase_str, ")")
        print(io, "  ")
        _pass_fail_str(io, r.beats_baseline)
        println(io)
    end

    # ── Series Features ──
    if !isnothing(r.features)
        f = r.features
        _section_header(io, "Series Features", W)
        println(io, "  Length: $(f.length) obs  |  Period: $(f.seasonal_period)")
        slope_str = f.trend_slope != 0.0 ? " (slope: $(_round4(f.trend_slope)))" : ""
        println(io, "  Trend: $(f.trend_strength)$(slope_str)")
        acf_str = f.seasonal_acf != 0.0 ? " (ACF: $(_round2(f.seasonal_acf)))" : ""
        println(io, "  Seasonality: $(f.seasonality_strength)$(acf_str)")
        stat_str = "$(f.stationarity)"
        if f.ndiffs > 0 || f.nsdiffs > 0
            stat_str *= " (d=$(f.ndiffs), D=$(f.nsdiffs))"
        end
        println(io, "  Stationarity: $(stat_str)")
    end

    # ── Model Accuracy ──
    if !isempty(r.accuracy)
        _section_header(io, "Model Accuracy", W)
        sorted = sort(collect(values(r.accuracy)); by=m -> m.mase)

        # Compute column widths
        name_w = max(7, maximum(length(m.model) for m in sorted) + 2)
        hdr = "  " * rpad("Model", name_w) * "│ " * rpad("MASE", 7) *
              "│ " * rpad("RMSE", 7) * "│ " * rpad("MAE", 7) * "│ " * "MAPE"
        println(io, hdr)
        sep = "  " * "─"^name_w * "┼" * "─"^8 * "┼" * "─"^8 * "┼" * "─"^8 * "┼" * "─"^8
        println(io, sep)

        for m in sorted
            star = (!isnothing(r.best_model) && m.model == r.best_model) ? " *" : ""
            name = m.model * star
            row = "  "
            if use_color && !isempty(star)
                printstyled(io, row * rpad(name, name_w); color=:cyan)
            else
                print(io, row * rpad(name, name_w))
            end
            println(io, "│ " * rpad(_fmt_metric(m.mase), 7) *
                        "│ " * rpad(_fmt_metric(m.rmse), 7) *
                        "│ " * rpad(_fmt_metric(m.mae), 7) *
                        "│ " * _fmt_metric(m.mape; pct=true))
        end
        println(io, "  (* = best)")
    end

    # ── Forecast ──
    if !isnothing(r.forecasts) && !isempty(r.forecasts.point_forecasts)
        fc = r.forecasts
        _section_header(io, "Forecast ($(fc.model), h=$(fc.horizon))", W)

        has_80 = !isempty(fc.lower_80) && !isempty(fc.upper_80)
        has_95 = !isempty(fc.lower_95) && !isempty(fc.upper_95)
        has_dates = !isempty(fc.dates)

        n = length(fc.point_forecasts)
        if n > 6
            show_idxs = vcat(1:5, n)
        else
            show_idxs = collect(1:n)
        end

        hdr = "  " * rpad(has_dates ? "Date" : "#", 12) * "│ " * rpad("Point", 7)
        has_80 && (hdr *= "│ " * rpad("80% CI", 18))
        has_95 && (hdr *= "│ " * rpad("95% CI", 18))
        println(io, hdr)

        sep = "  " * "─"^12 * "┼" * "─"^8
        has_80 && (sep *= "┼" * "─"^19)
        has_95 && (sep *= "┼" * "─"^19)
        println(io, sep)

        for (j, i) in enumerate(show_idxs)
            if j > 1 && show_idxs[j-1] != i - 1
                row = "  " * rpad("...", 12) * "│ " * rpad("...", 7)
                has_80 && (row *= "│ " * rpad("...", 18))
                has_95 && (row *= "│ " * rpad("...", 18))
                println(io, row)
            end
            dstr = has_dates ? string(fc.dates[i]) : string(i)
            row = "  " * rpad(dstr, 12) * "│ " * rpad(string(_round2(fc.point_forecasts[i])), 7)
            has_80 && (row *= "│ " * rpad(_fmt_ci(fc.lower_80[i], fc.upper_80[i]), 18))
            has_95 && (row *= "│ " * rpad(_fmt_ci(fc.lower_95[i], fc.upper_95[i]), 18))
            println(io, row)
        end

        if n > 6
            println(io, "  ($(length(show_idxs)) of $(n) rows shown)")
        end
    end

    # ── Anomalies ──
    if !isempty(r.anomalies)
        _section_header(io, "Anomalies ($(length(r.anomalies)) detected)", W)
        for a in r.anomalies
            dstr = isnothing(a.date) ? "idx=$(a.index)" : string(a.date)
            println(io, "  $(dstr): value=$(_round2(a.value)), z=$(round(a.z_score, digits=2))")
        end
    end

    # ── LLM Analysis ──
    if !isempty(r.output)
        _section_header(io, "Analysis", W)
        truncated = _truncate_text(r.output, 200)
        println(io, "  ", replace(truncated, "\n" => "\n  "))
        if length(r.output) > 200
            print(io, "  ($(length(r.output)) chars total — access result.output for full text)")
        end
    end
end

# ── MIME"text/html" ──────────────────────────────────────────────────────────

const _HTML_STYLE = """
<style>
.heval-result { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 720px; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 8px 0; }
.heval-result h3 { margin: 0 0 8px 0; color: #2c3e50; }
.heval-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }
.heval-pass { background: #d4edda; color: #155724; }
.heval-fail { background: #f8d7da; color: #721c24; }
.heval-section { margin-top: 12px; }
.heval-section h4 { margin: 0 0 6px 0; color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 4px; }
.heval-table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
.heval-table th { background: #f8f9fa; text-align: left; padding: 6px 10px; border: 1px solid #dee2e6; }
.heval-table td { padding: 6px 10px; border: 1px solid #dee2e6; }
.heval-table tr:nth-child(even) { background: #f8f9fa; }
.heval-best { background: #EBF5FB !important; }
.heval-narrative { max-height: 200px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 0.9em; white-space: pre-wrap; }
.heval-kv { font-size: 0.9em; line-height: 1.6; }
</style>"""

function _html_escape(s::AbstractString)
    s = replace(s, "&" => "&amp;")
    s = replace(s, "<" => "&lt;")
    s = replace(s, ">" => "&gt;")
    s = replace(s, "\"" => "&quot;")
    return s
end

# ForecastOutput HTML
function Base.show(io::IO, ::MIME"text/html", fc::ForecastOutput)
    println(io, _HTML_STYLE)
    println(io, "<div class=\"heval-result\">")
    println(io, "<h3>Forecast — $(fc.model), h=$(fc.horizon)</h3>")

    if !isempty(fc.point_forecasts)
        has_80 = !isempty(fc.lower_80) && !isempty(fc.upper_80)
        has_95 = !isempty(fc.lower_95) && !isempty(fc.upper_95)
        has_dates = !isempty(fc.dates)

        println(io, "<table class=\"heval-table\">")
        print(io, "<tr>")
        print(io, "<th>$(has_dates ? "Date" : "#")</th><th>Point</th>")
        has_80 && print(io, "<th>80% CI</th>")
        has_95 && print(io, "<th>95% CI</th>")
        println(io, "</tr>")

        n = length(fc.point_forecasts)
        if n > 6
            show_idxs = vcat(1:5, n)
        else
            show_idxs = collect(1:n)
        end

        for (j, i) in enumerate(show_idxs)
            if j > 1 && show_idxs[j-1] != i - 1
                ncols = 2 + has_80 + has_95
                println(io, "<tr><td colspan=\"$(ncols)\" style=\"text-align:center\">⋮</td></tr>")
            end
            dstr = has_dates ? string(fc.dates[i]) : string(i)
            print(io, "<tr><td>$(dstr)</td><td>$(_round2(fc.point_forecasts[i]))</td>")
            has_80 && print(io, "<td>$(_fmt_ci(fc.lower_80[i], fc.upper_80[i]))</td>")
            has_95 && print(io, "<td>$(_fmt_ci(fc.lower_95[i], fc.upper_95[i]))</td>")
            println(io, "</tr>")
        end
        println(io, "</table>")
        n > 6 && println(io, "<p style=\"font-size:0.85em;color:#666\">$(length(show_idxs)) of $(n) rows shown</p>")
    end
    println(io, "</div>")
end

# SeriesFeatures HTML
function Base.show(io::IO, ::MIME"text/html", f::SeriesFeatures)
    println(io, _HTML_STYLE)
    println(io, "<div class=\"heval-result\">")
    println(io, "<h3>Series Features</h3>")
    println(io, "<div class=\"heval-kv\">")
    println(io, "<b>Length:</b> $(f.length) obs &nbsp;|&nbsp; <b>Period:</b> $(f.seasonal_period)<br>")
    println(io, "<b>Mean:</b> $(_round2(f.mean)) &nbsp;|&nbsp; <b>Std:</b> $(_round2(f.std))<br>")
    slope_str = f.trend_slope != 0.0 ? " (slope: $(_round4(f.trend_slope)))" : ""
    println(io, "<b>Trend:</b> $(f.trend_strength)$(slope_str)<br>")
    acf_str = f.seasonal_acf != 0.0 ? " (ACF: $(_round2(f.seasonal_acf)))" : ""
    println(io, "<b>Seasonality:</b> $(f.seasonality_strength)$(acf_str)<br>")
    stat_str = "$(f.stationarity)"
    if f.ndiffs > 0 || f.nsdiffs > 0
        stat_str *= " (d=$(f.ndiffs), D=$(f.nsdiffs))"
    end
    println(io, "<b>Stationarity:</b> $(stat_str)")
    println(io, "</div></div>")
end

# AgentResult HTML
function Base.show(io::IO, ::MIME"text/html", r::AgentResult)
    println(io, _HTML_STYLE)
    println(io, "<div class=\"heval-result\">")

    # Header with badge
    badge_class = r.beats_baseline ? "heval-pass" : "heval-fail"
    badge_text = r.beats_baseline ? "PASS" : "FAIL"
    print(io, "<h3>Heval Analysis Result")
    if !isnothing(r.best_model)
        mase_str = if haskey(r.accuracy, r.best_model)
            " (MASE: $(_round4(r.accuracy[r.best_model].mase)))"
        else
            ""
        end
        print(io, " — Best: $(r.best_model)$(mase_str)")
    end
    println(io, " <span class=\"heval-badge $(badge_class)\">$(badge_text)</span></h3>")

    # ── Features ──
    if !isnothing(r.features)
        f = r.features
        println(io, "<div class=\"heval-section\"><h4>Series Features</h4>")
        println(io, "<div class=\"heval-kv\">")
        println(io, "<b>Length:</b> $(f.length) obs &nbsp;|&nbsp; <b>Period:</b> $(f.seasonal_period)<br>")
        slope_str = f.trend_slope != 0.0 ? " (slope: $(_round4(f.trend_slope)))" : ""
        println(io, "<b>Trend:</b> $(f.trend_strength)$(slope_str)<br>")
        acf_str = f.seasonal_acf != 0.0 ? " (ACF: $(_round2(f.seasonal_acf)))" : ""
        println(io, "<b>Seasonality:</b> $(f.seasonality_strength)$(acf_str)<br>")
        stat_str = "$(f.stationarity)"
        if f.ndiffs > 0 || f.nsdiffs > 0
            stat_str *= " (d=$(f.ndiffs), D=$(f.nsdiffs))"
        end
        println(io, "<b>Stationarity:</b> $(stat_str)")
        println(io, "</div></div>")
    end

    # ── Accuracy ──
    if !isempty(r.accuracy)
        sorted = sort(collect(values(r.accuracy)); by=m -> m.mase)
        println(io, "<div class=\"heval-section\"><h4>Model Accuracy</h4>")
        println(io, "<table class=\"heval-table\">")
        println(io, "<tr><th>Model</th><th>MASE</th><th>RMSE</th><th>MAE</th><th>MAPE</th></tr>")
        for m in sorted
            is_best = !isnothing(r.best_model) && m.model == r.best_model
            cls = is_best ? " class=\"heval-best\"" : ""
            star = is_best ? " ★" : ""
            println(io, "<tr$(cls)><td><b>$(m.model)$(star)</b></td>" *
                        "<td>$(_fmt_metric(m.mase))</td>" *
                        "<td>$(_fmt_metric(m.rmse))</td>" *
                        "<td>$(_fmt_metric(m.mae))</td>" *
                        "<td>$(_fmt_metric(m.mape; pct=true))</td></tr>")
        end
        println(io, "</table></div>")
    end

    # ── Forecast ──
    if !isnothing(r.forecasts) && !isempty(r.forecasts.point_forecasts)
        fc = r.forecasts
        has_80 = !isempty(fc.lower_80) && !isempty(fc.upper_80)
        has_95 = !isempty(fc.lower_95) && !isempty(fc.upper_95)
        has_dates = !isempty(fc.dates)

        println(io, "<div class=\"heval-section\"><h4>Forecast ($(fc.model), h=$(fc.horizon))</h4>")
        println(io, "<table class=\"heval-table\">")
        print(io, "<tr><th>$(has_dates ? "Date" : "#")</th><th>Point</th>")
        has_80 && print(io, "<th>80% CI</th>")
        has_95 && print(io, "<th>95% CI</th>")
        println(io, "</tr>")

        n = length(fc.point_forecasts)
        if n > 6
            show_idxs = vcat(1:5, n)
        else
            show_idxs = collect(1:n)
        end

        for (j, i) in enumerate(show_idxs)
            if j > 1 && show_idxs[j-1] != i - 1
                ncols = 2 + has_80 + has_95
                println(io, "<tr><td colspan=\"$(ncols)\" style=\"text-align:center\">⋮</td></tr>")
            end
            dstr = has_dates ? string(fc.dates[i]) : string(i)
            print(io, "<tr><td>$(dstr)</td><td>$(_round2(fc.point_forecasts[i]))</td>")
            has_80 && print(io, "<td>$(_fmt_ci(fc.lower_80[i], fc.upper_80[i]))</td>")
            has_95 && print(io, "<td>$(_fmt_ci(fc.lower_95[i], fc.upper_95[i]))</td>")
            println(io, "</tr>")
        end
        println(io, "</table>")
        n > 6 && println(io, "<p style=\"font-size:0.85em;color:#666\">$(length(show_idxs)) of $(n) rows shown</p>")
        println(io, "</div>")
    end

    # ── Anomalies ──
    if !isempty(r.anomalies)
        println(io, "<div class=\"heval-section\"><h4>Anomalies ($(length(r.anomalies)) detected)</h4>")
        println(io, "<table class=\"heval-table\">")
        println(io, "<tr><th>Date/Index</th><th>Value</th><th>Z-score</th></tr>")
        for a in r.anomalies
            dstr = isnothing(a.date) ? "idx=$(a.index)" : string(a.date)
            println(io, "<tr><td>$(dstr)</td><td>$(_round2(a.value))</td><td>$(round(a.z_score, digits=2))</td></tr>")
        end
        println(io, "</table></div>")
    end

    # ── LLM narrative ──
    if !isempty(r.output)
        println(io, "<div class=\"heval-section\"><h4>Analysis</h4>")
        println(io, "<div class=\"heval-narrative\">$(_html_escape(r.output))</div></div>")
    end

    println(io, "</div>")
end

# ── QueryResult ──────────────────────────────────────────────────────────────

# String interop
Base.string(qr::QueryResult) = qr.content
Base.print(io::IO, qr::QueryResult) = print(io, qr.content)

# Compact show
function Base.show(io::IO, qr::QueryResult)
    print(io, "QueryResult(\"", _truncate_text(qr.content, 50), "\")")
end

# Word-wrap helper for box display
function _word_wrap(text::AbstractString, width::Int)
    lines = String[]
    for paragraph in split(text, '\n')
        if isempty(paragraph)
            push!(lines, "")
            continue
        end
        words = split(paragraph)
        isempty(words) && (push!(lines, ""); continue)
        current = string(words[1])
        for w in words[2:end]
            if length(current) + 1 + length(w) > width
                push!(lines, current)
                current = string(w)
            else
                current *= " " * w
            end
        end
        push!(lines, current)
    end
    return lines
end

# text/plain — box layout
function Base.show(io::IO, ::MIME"text/plain", qr::QueryResult)
    use_color = get(io, :color, false)::Bool
    max_inner = 72
    min_inner = 54

    wrapped = _word_wrap(qr.content, max_inner - 4)  # 4 = "│ " + " │" padding chars
    content_width = isempty(wrapped) ? 0 : maximum(length(l) for l in wrapped)
    inner = clamp(content_width + 4, min_inner, max_inner)

    title = "Heval"
    top_bar_len = inner - length(title) - 4  # 4 = "╭─ " + " " before bar + "╮" overhead
    top_bar_len = max(top_bar_len, 1)

    # Top border
    print(io, "╭─ ")
    if use_color
        printstyled(io, title; bold=true)
    else
        print(io, title)
    end
    print(io, " ")
    print(io, "─" ^ top_bar_len)
    println(io, "╮")

    # Content lines
    for line in wrapped
        padded = rpad(line, inner - 4)  # pad to fill box width minus "│ " and " │"
        println(io, "│ ", padded, " │")
    end
    if isempty(wrapped)
        println(io, "│ ", " " ^ (inner - 4), " │")
    end

    # Bottom border
    print(io, "╰", "─" ^ (inner - 2), "╯")
end

# text/html
function Base.show(io::IO, ::MIME"text/html", qr::QueryResult)
    println(io, _HTML_STYLE)
    println(io, "<div class=\"heval-result\">")
    println(io, "<h4 style=\"margin:0 0 6px 0; color:#2c3e50;\">Heval</h4>")
    println(io, "<div class=\"heval-narrative\">", _html_escape(qr.content), "</div>")
    println(io, "</div>")
end
