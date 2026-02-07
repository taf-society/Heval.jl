using Test
using Heval
using Dates
using Statistics
using Durbyn

@testset "Heval.jl" begin

    @testset "Type Definitions" begin
        # Test SeriesFeatures
        features = Heval.SeriesFeatures()
        @test features.length == 0
        @test features.trend_strength == "unknown"
        @test features.is_intermittent == false
        @test features.stationarity == "unknown"
        @test features.ndiffs == 0
        @test features.nsdiffs == 0

        # Test AgentState
        state = Heval.AgentState()
        @test isnothing(state.values)
        @test isempty(state.accuracy)
        @test isempty(state.conversation_history)
        @test isnothing(state.panel)
        @test isempty(state.fitted_models)

        # Test ForecastOutput
        fc = Heval.ForecastOutput()
        @test fc.model == ""
        @test fc.horizon == 0
        @test isempty(fc.point_forecasts)

        # Test PanelState
        data = (date=[Date(2020,1,1)], store=[:A], value=[1.0])
        ps = Heval.PanelState(data; groups=[:store])
        @test ps.groups == [:store]
        @test ps.date_col == :date
        @test ps.target_col == :value
        @test isnothing(ps.panel)
    end

    @testset "Durbyn Bridge - build_spec" begin
        # Test that specs are constructed for all available models
        for model_name in Heval.AVAILABLE_MODELS
            spec = Heval.build_spec(model_name, :value; m=12)
            @test spec isa Durbyn.AbstractModelSpec
        end

        # Unknown model
        @test_throws ErrorException Heval.build_spec("NoSuchModel", :value)
    end

    @testset "Durbyn Bridge - fit/forecast" begin
        # Create a simple seasonal series
        n = 48
        dates = [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]
        seasonal = [sin(2π * i / 12) * 10 for i in 1:n]
        values = Float64.(100 .+ collect(1:n) .+ seasonal .+ randn(n) .* 2)

        h = 6
        m = 12

        # Test fit + forecast for a few key models
        for model_name in ["SES", "Naive", "SNaive", "Meanf"]
            result = Heval.durbyn_fit_forecast(model_name, dates, values, h, m)
            @test !isnothing(result.fitted)
            @test !isnothing(result.forecast)
            @test length(result.forecast.mean) == h
        end

        # Test extract_forecast_result
        result = Heval.durbyn_fit_forecast("SES", dates, values, h, m)
        fc_data = Heval.extract_forecast_result(result.forecast, dates, h)
        @test length(fc_data["point_forecasts"]) == h
        @test length(fc_data["dates"]) == h
        @test fc_data["dates"][1] > dates[end]
    end

    @testset "Tool: analyze_features" begin
        state = Heval.AgentState()
        # Need enough data for STL (> 2*m)
        n = 48
        seasonal = [sin(2π * i / 12) * 10 for i in 1:n]
        state.values = Float64.(100 .+ collect(1:n) .+ seasonal)
        state.seasonal_period = 12
        state.horizon = 12

        tool = Heval.create_analyze_features_tool(state)
        result = tool.fn(Dict{String, Any}())

        @test result["status"] == "success"
        @test haskey(result, "features")
        @test haskey(result, "recommendations")
        @test !isnothing(state.features)
        @test state.features.length == n

        # Check stationarity was assessed
        @test haskey(result["features"], "stationarity")
        @test haskey(result["features"], "ndiffs")

        # Error case: no data
        state2 = Heval.AgentState()
        tool2 = Heval.create_analyze_features_tool(state2)
        result2 = tool2.fn(Dict{String, Any}())
        @test haskey(result2, "error")
    end

    @testset "Tool: cross_validate" begin
        state = Heval.AgentState()
        n = 48
        seasonal = [sin(2π * i / 12) * 10 for i in 1:n]
        state.values = Float64.(100 .+ collect(1:n) .+ seasonal)
        state.dates = [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]
        state.seasonal_period = 12
        state.horizon = 6

        tool_cv = Heval.create_cross_validate_tool(state)
        result_cv = tool_cv.fn(Dict{String, Any}("models" => ["SES", "SNaive", "Naive"]))

        @test result_cv["status"] == "success"
        @test haskey(result_cv, "best_model")
        @test !isempty(state.accuracy)
        @test !isnothing(state.best_model)

        # Unknown model
        result_bad = tool_cv.fn(Dict{String, Any}("models" => ["NoSuchModel"]))
        @test haskey(result_bad, "error")

        # Empty models
        result_empty = tool_cv.fn(Dict{String, Any}("models" => String[]))
        @test haskey(result_empty, "error")
    end

    @testset "Tool: generate_forecast" begin
        state = Heval.AgentState()
        n = 48
        seasonal = [sin(2π * i / 12) * 10 for i in 1:n]
        state.values = Float64.(100 .+ collect(1:n) .+ seasonal)
        state.dates = [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]
        state.seasonal_period = 12
        state.horizon = 6

        tool_fc = Heval.create_forecast_tool(state)
        result_fc = tool_fc.fn(Dict{String, Any}("model" => "SES", "h" => 6))

        @test result_fc["status"] == "success"
        @test !isnothing(state.forecasts)
        @test state.forecasts.horizon == 6
        @test length(state.forecasts.point_forecasts) == 6
        @test !isempty(state.forecasts.dates)

        # Fitted model should be cached
        @test haskey(state.fitted_models, "SES")

        # Error: no model specified
        result_err = tool_fc.fn(Dict{String, Any}())
        @test haskey(result_err, "error")

        # Error: unknown model
        result_unk = tool_fc.fn(Dict{String, Any}("model" => "UnknownModel"))
        @test haskey(result_unk, "error")
    end

    @testset "Tool: detect_anomalies" begin
        state = Heval.AgentState()
        n = 48
        values = Float64.(100 .+ randn(n) .* 5)
        # Insert a clear outlier
        values[20] = 500.0
        state.values = values
        state.dates = [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]
        state.seasonal_period = 12
        state.horizon = 6

        tool_anom = Heval.create_anomaly_tool(state)
        result_anom = tool_anom.fn(Dict{String, Any}("model" => "SES", "threshold" => 3.0))

        @test result_anom["status"] == "success"
        @test haskey(result_anom, "n_anomalies")
        @test result_anom["n_anomalies"] >= 1  # should detect our outlier
    end

    @testset "Tool: decompose" begin
        state = Heval.AgentState()
        n = 48
        seasonal = [sin(2π * i / 12) * 10 for i in 1:n]
        state.values = Float64.(100 .+ collect(1:n) .+ seasonal)
        state.seasonal_period = 12
        state.horizon = 12

        tool = Heval.create_decompose_tool(state)

        # STL decomposition
        result = tool.fn(Dict{String, Any}("method" => "stl"))
        @test result["status"] == "success"
        @test result["method"] == "STL"
        @test haskey(result, "trend_summary")
        @test haskey(result, "seasonal_summary")
        @test haskey(result, "remainder_summary")

        # Error: insufficient data
        state2 = Heval.AgentState()
        state2.values = Float64[1.0, 2.0, 3.0]
        state2.seasonal_period = 12
        tool2 = Heval.create_decompose_tool(state2)
        result2 = tool2.fn(Dict{String, Any}())
        @test haskey(result2, "error")
    end

    @testset "Tool: unit_root_test" begin
        state = Heval.AgentState()
        state.values = Float64.(cumsum(randn(100)))  # random walk (non-stationary)
        state.seasonal_period = 12
        state.horizon = 12

        tool = Heval.create_unit_root_test_tool(state)
        result = tool.fn(Dict{String, Any}("tests" => ["adf", "kpss"]))

        @test result["status"] == "success"
        @test haskey(result["results"], "adf")
        @test haskey(result["results"], "kpss")

        # ADF result structure
        adf = result["results"]["adf"]
        @test haskey(adf, "statistic")
        @test haskey(adf, "p_value")
        @test haskey(adf, "conclusion")
    end

    @testset "Tool: compare_models" begin
        state = Heval.AgentState()
        n = 48
        seasonal = [sin(2π * i / 12) * 10 for i in 1:n]
        state.values = Float64.(100 .+ collect(1:n) .+ seasonal)
        state.dates = [Date(2020, 1, 1) + Dates.Month(i - 1) for i in 1:n]
        state.seasonal_period = 12
        state.horizon = 6

        tool = Heval.create_compare_models_tool(state)
        result = tool.fn(Dict{String, Any}("models" => ["SES", "Naive"]))

        @test result["status"] == "success"
        @test result["n_models"] == 2
        @test haskey(result["comparison"], "SES")
        @test haskey(result["comparison"], "Naive")
    end

    @testset "Tool Error Handling" begin
        state = Heval.AgentState()

        # Feature analysis with no data loaded
        tool = Heval.create_analyze_features_tool(state)
        result = tool.fn(Dict{String, Any}())
        @test haskey(result, "error")

        # Cross validation with unknown model
        state.values = Float64.(1:20)
        state.seasonal_period = 4
        state.horizon = 4
        tool_cv = Heval.create_cross_validate_tool(state)
        result_cv = tool_cv.fn(Dict{String, Any}("models" => ["NoSuchModel"]))
        @test haskey(result_cv, "error")

        # Forecast with no model specified
        tool_fc = Heval.create_forecast_tool(state)
        result_fc = tool_fc.fn(Dict{String, Any}())
        @test haskey(result_fc, "error")
    end

    @testset "Data Parsing" begin
        # NamedTuple input
        data1 = (date = Date.(2020, 1:12), value = rand(12))
        dates1, values1 = Heval.parse_input_data(data1)
        @test length(dates1) == 12
        @test length(values1) == 12

        # Vector input
        data2 = rand(10)
        dates2, values2 = Heval.parse_input_data(data2)
        @test isnothing(dates2)
        @test length(values2) == 10

        # Dict input
        data3 = Dict("date" => Date.(2020, 1:6), "value" => rand(6))
        dates3, values3 = Heval.parse_input_data(data3)
        @test length(dates3) == 6
        @test length(values3) == 6

        # NamedTuple with :y key
        data4 = (date = Date.(2020, 1:5), y = rand(5))
        dates4, values4 = Heval.parse_input_data(data4)
        @test length(values4) == 5

        # Unsupported format
        @test_throws ErrorException Heval.parse_input_data("not valid")
    end

    @testset "Recommendations" begin
        # Intermittent data
        features_int = Heval.SeriesFeatures(
            is_intermittent = true,
            zero_fraction = 0.4,
            seasonality_strength = "weak"
        )
        recs = Heval.generate_recommendations(features_int)
        @test any(r -> occursin("Croston", r), recs)

        # Strong seasonality
        features_seas = Heval.SeriesFeatures(
            is_intermittent = false,
            seasonality_strength = "strong"
        )
        recs2 = Heval.generate_recommendations(features_seas)
        @test any(r -> occursin("ETS", r), recs2)
        @test any(r -> occursin("TBATS", r), recs2)
        @test any(r -> occursin("BATS", r), recs2)

        # Strong trend
        features_trend = Heval.SeriesFeatures(
            is_intermittent = false,
            seasonality_strength = "weak",
            trend_strength = "strong"
        )
        recs3 = Heval.generate_recommendations(features_trend)
        @test any(r -> occursin("Holt", r), recs3)
        @test any(r -> occursin("Theta", r), recs3)
        @test any(r -> occursin("SNaive", r), recs3)
        @test any(r -> occursin("ARAR", r), recs3)
    end

    @testset "Available Models" begin
        @test "ARIMA" in Heval.AVAILABLE_MODELS
        @test "ETS" in Heval.AVAILABLE_MODELS
        @test "BATS" in Heval.AVAILABLE_MODELS
        @test "TBATS" in Heval.AVAILABLE_MODELS
        @test "Theta" in Heval.AVAILABLE_MODELS
        @test "SES" in Heval.AVAILABLE_MODELS
        @test "Holt" in Heval.AVAILABLE_MODELS
        @test "HoltWinters" in Heval.AVAILABLE_MODELS
        @test "Croston" in Heval.AVAILABLE_MODELS
        @test "ARAR" in Heval.AVAILABLE_MODELS
        @test "ARARMA" in Heval.AVAILABLE_MODELS
        @test "Diffusion" in Heval.AVAILABLE_MODELS
        @test "Naive" in Heval.AVAILABLE_MODELS
        @test "SNaive" in Heval.AVAILABLE_MODELS
        @test "RW" in Heval.AVAILABLE_MODELS
        @test "Meanf" in Heval.AVAILABLE_MODELS
        @test length(Heval.AVAILABLE_MODELS) == 16
    end

    @testset "Forecast Date Inference" begin
        state = Heval.AgentState()
        state.dates = [Date(2020, i, 1) for i in 1:12]
        state.values = Float64.(10 .+ randn(12))
        state.seasonal_period = 12
        state.horizon = 3

        tool_fc = Heval.create_forecast_tool(state)
        result = tool_fc.fn(Dict{String, Any}("model" => "SES", "h" => 3))

        @test result["status"] == "success"
        @test !isempty(state.forecasts.dates)
        @test length(state.forecasts.dates) == 3
        # Check dates are monthly increments from last date
        @test state.forecasts.dates[1] == Date(2020, 12, 1) + (Date(2020, 12, 1) - Date(2020, 11, 1))
    end

    @testset "Ollama Types" begin
        # Test OllamaConfig creation
        config = Heval.OllamaConfig()
        @test config.model == "llama3.1"
        @test config.host == "http://localhost:11434"
        @test config.use_openai_compat == false
        @test config.temperature == 0.1

        # Custom config
        config2 = Heval.OllamaConfig(
            model = "qwen2.5",
            host = "http://myserver:11434",
            use_openai_compat = true
        )
        @test config2.model == "qwen2.5"
        @test config2.use_openai_compat == true
    end

    @testset "Ollama Message Formatting" begin
        messages = [
            Heval.Message("system", "You are a helper."),
            Heval.Message("user", "Hello"),
        ]
        formatted = Heval.messages_to_ollama_format(messages)
        @test length(formatted) == 2
        @test formatted[1]["role"] == "system"
        @test formatted[1]["content"] == "You are a helper."
        @test formatted[2]["role"] == "user"

        # Test assistant message with empty content gets ""
        msg_empty = Heval.Message("assistant", nothing, nothing, nothing)
        formatted2 = Heval.messages_to_ollama_format([msg_empty])
        @test formatted2[1]["content"] == ""
    end

    @testset "Ollama Tool Formatting" begin
        tool = Heval.Tool(
            "test_tool",
            "A test tool",
            Dict("type" => "object", "properties" => Dict()),
            identity
        )
        formatted = Heval.tools_to_ollama_format([tool])
        @test length(formatted) == 1
        @test formatted[1]["type"] == "function"
        @test formatted[1]["function"]["name"] == "test_tool"
    end

    @testset "OpenAI Message Formatting" begin
        tc = Heval.ToolCall("tc1", "some_tool", Dict{String, Any}("arg" => "val"))
        msg = Heval.Message("assistant", nothing, [tc], nothing)
        formatted = Heval.messages_to_openai_format([msg])
        @test formatted[1]["content"] === nothing
        @test haskey(formatted[1], "tool_calls")
    end

    @testset "Panel State Construction" begin
        data = (
            date = repeat([Date(2020, 1, 1), Date(2020, 2, 1), Date(2020, 3, 1)], 2),
            store = vcat(fill(:A, 3), fill(:B, 3)),
            value = Float64[10, 12, 15, 20, 22, 25]
        )

        ps = Heval.PanelState(data; groups=[:store], date_col=:date, target_col=:value)
        @test ps.groups == [:store]
        @test ps.date_col == :date
        @test ps.target_col == :value
        @test isnothing(ps.panel)
        @test isempty(ps.group_features)
    end

    @testset "Display - Compact show" begin
        # AccuracyMetrics
        am = Heval.AccuracyMetrics(model="ETS", mase=0.74, rmse=12.3, mae=8.9, mape=5.2)
        s = sprint(show, am)
        @test occursin("AccuracyMetrics", s)
        @test occursin("ETS", s)
        @test occursin("0.74", s)

        # SeriesFeatures
        sf = Heval.SeriesFeatures(length=120, trend_strength="moderate")
        s = sprint(show, sf)
        @test occursin("SeriesFeatures", s)
        @test occursin("n=120", s)
        @test occursin("moderate", s)

        # AnomalyResult
        ar = Heval.AnomalyResult(5, Date(2023, 3, 15), 245.67, 3.45)
        s = sprint(show, ar)
        @test occursin("Anomaly", s)
        @test occursin("2023-03-15", s)
        @test occursin("3.45", s)

        # AnomalyResult without date
        ar2 = Heval.AnomalyResult(5, nothing, 245.67, 3.45)
        s2 = sprint(show, ar2)
        @test occursin("idx=5", s2)

        # ForecastOutput
        fc = Heval.ForecastOutput(model="ETS", horizon=6)
        s = sprint(show, fc)
        @test occursin("ForecastOutput", s)
        @test occursin("ETS", s)
        @test occursin("h=6", s)

        # AgentResult
        acc = Dict("ETS" => am)
        result = Heval.AgentResult("test output", nothing, acc, nothing,
            Heval.AnomalyResult[], "ETS", true)
        s = sprint(show, result)
        @test occursin("AgentResult", s)
        @test occursin("best=ETS", s)
        @test occursin("0.74", s)

        # AgentResult with no best model
        result2 = Heval.AgentResult("test", nothing, Dict{String, Heval.AccuracyMetrics}(),
            nothing, Heval.AnomalyResult[], nothing, false)
        s2 = sprint(show, result2)
        @test occursin("best=none", s2)
    end

    @testset "Display - text/plain verbose" begin
        # AccuracyMetrics verbose
        am = Heval.AccuracyMetrics(model="ETS", mase=0.74, rmse=12.3, mae=8.9, mape=5.2)
        s = sprint(show, MIME("text/plain"), am)
        @test occursin("MASE:", s)
        @test occursin("RMSE:", s)
        @test occursin("MAE:", s)
        @test occursin("MAPE:", s)

        # SeriesFeatures verbose
        sf = Heval.SeriesFeatures(
            length=120, mean=145.0, std=23.0,
            trend_strength="moderate", trend_slope=0.0234,
            seasonality_strength="strong", seasonal_period=12, seasonal_acf=0.82,
            ndiffs=1, nsdiffs=0, stationarity="non-stationary"
        )
        s = sprint(show, MIME("text/plain"), sf)
        @test occursin("120 obs", s)
        @test occursin("Period: 12", s)
        @test occursin("moderate", s)
        @test occursin("slope:", s)
        @test occursin("strong", s)
        @test occursin("ACF:", s)
        @test occursin("non-stationary", s)
        @test occursin("d=1", s)

        # SeriesFeatures intermittent
        sf_int = Heval.SeriesFeatures(is_intermittent=true, zero_fraction=0.4)
        s_int = sprint(show, MIME("text/plain"), sf_int)
        @test occursin("Intermittent: yes", s_int)

        # ForecastOutput verbose (short, <=6 rows)
        fc = Heval.ForecastOutput(
            model="ETS", horizon=4,
            point_forecasts=[112.0, 115.0, 118.0, 121.0],
            dates=Date.(2024, 1:4),
            lower_95=[101.0, 102.0, 103.0, 103.0],
            upper_95=[123.0, 129.0, 135.0, 139.0]
        )
        s = sprint(show, MIME("text/plain"), fc)
        @test occursin("Forecast (ETS, h=4)", s)
        @test occursin("Date", s)
        @test occursin("Point", s)
        @test occursin("95% CI", s)
        @test occursin("2024-01-01", s)
        @test occursin("112.0", s)

        # ForecastOutput verbose (long, >6 rows, triggers truncation)
        fc_long = Heval.ForecastOutput(
            model="ETS", horizon=12,
            point_forecasts=collect(100.0:10.0:210.0),
            dates=[Date(2024, i, 1) for i in 1:12]
        )
        s_long = sprint(show, MIME("text/plain"), fc_long)
        @test occursin("...", s_long)
        @test occursin("6 of 12 rows shown", s_long)

        # ForecastOutput with no CI columns
        fc_noci = Heval.ForecastOutput(
            model="SES", horizon=3,
            point_forecasts=[10.0, 20.0, 30.0],
            dates=Date.(2024, 1:3)
        )
        s_noci = sprint(show, MIME("text/plain"), fc_noci)
        @test !occursin("80% CI", s_noci)
        @test !occursin("95% CI", s_noci)
    end

    @testset "Display - AgentResult text/plain composite" begin
        features = Heval.SeriesFeatures(
            length=120, mean=145.0, std=23.0,
            trend_strength="moderate", trend_slope=0.0234,
            seasonality_strength="strong", seasonal_period=12, seasonal_acf=0.82,
            ndiffs=1, nsdiffs=0, stationarity="non-stationary"
        )
        accuracy = Dict(
            "ETS" => Heval.AccuracyMetrics(model="ETS", mase=0.74, rmse=12.3, mae=8.9, mape=5.2),
            "SNaive" => Heval.AccuracyMetrics(model="SNaive", mase=1.0, rmse=15.6, mae=11.2, mape=6.9)
        )
        fc = Heval.ForecastOutput(
            model="ETS", horizon=6,
            point_forecasts=Float64[112, 115, 118, 121, 124, 127],
            dates=Date.(2024, 1:6),
            lower_95=Float64[101, 102, 103, 103, 104, 104],
            upper_95=Float64[123, 129, 135, 139, 145, 151]
        )
        anomalies = [
            Heval.AnomalyResult(15, Date(2023, 3, 15), 245.67, 3.45),
            Heval.AnomalyResult(21, Date(2023, 9, 1), 12.34, -3.12)
        ]
        long_text = "The series shows " * "x"^250
        result = Heval.AgentResult(long_text, features, accuracy, fc, anomalies, "ETS", true)

        s = sprint(show, MIME("text/plain"), result)

        # Header
        @test occursin("Heval Analysis Result", s)
        @test occursin("═", s)
        @test occursin("Best model: ETS", s)
        @test occursin("PASS", s)

        # Features section
        @test occursin("Series Features", s)
        @test occursin("120 obs", s)
        @test occursin("moderate", s)

        # Accuracy table
        @test occursin("Model Accuracy", s)
        @test occursin("ETS *", s)
        @test occursin("SNaive", s)
        @test occursin("* = best", s)
        # ETS should come before SNaive (sorted by MASE)
        pos_ets = findfirst("ETS *", s)
        pos_snaive = findfirst("SNaive", s)
        @test !isnothing(pos_ets) && !isnothing(pos_snaive)

        # Forecast section
        @test occursin("Forecast (ETS, h=6)", s)
        @test occursin("2024-01-01", s)

        # Anomalies
        @test occursin("Anomalies (2 detected)", s)
        @test occursin("2023-03-15", s)
        @test occursin("z=3.45", s)

        # Analysis (truncated)
        @test occursin("Analysis", s)
        @test occursin("…", s)
        @test occursin("chars total", s)

        # Nothing/empty handling: no features, no accuracy, no forecasts, no anomalies
        minimal = Heval.AgentResult("", nothing, Dict{String, Heval.AccuracyMetrics}(),
            nothing, Heval.AnomalyResult[], nothing, false)
        s_min = sprint(show, MIME("text/plain"), minimal)
        @test occursin("Heval Analysis Result", s_min)
        @test !occursin("Series Features", s_min)
        @test !occursin("Model Accuracy", s_min)
        @test !occursin("Forecast", s_min)
        @test !occursin("Anomalies", s_min)
        @test !occursin("── Analysis", s_min)
    end

    @testset "Display - text/html" begin
        # ForecastOutput HTML
        fc = Heval.ForecastOutput(
            model="ETS", horizon=4,
            point_forecasts=[112.0, 115.0, 118.0, 121.0],
            dates=Date.(2024, 1:4),
            lower_95=[101.0, 102.0, 103.0, 103.0],
            upper_95=[123.0, 129.0, 135.0, 139.0]
        )
        html = sprint(show, MIME("text/html"), fc)
        @test occursin("<table", html)
        @test occursin("heval-table", html)
        @test occursin("2024-01-01", html)
        @test occursin("112.0", html)

        # SeriesFeatures HTML
        sf = Heval.SeriesFeatures(length=120, seasonal_period=12,
            trend_strength="moderate", seasonality_strength="strong")
        html_sf = sprint(show, MIME("text/html"), sf)
        @test occursin("<div", html_sf)
        @test occursin("120 obs", html_sf)
        @test occursin("moderate", html_sf)

        # AgentResult HTML
        accuracy = Dict(
            "ETS" => Heval.AccuracyMetrics(model="ETS", mase=0.74, rmse=12.3, mae=8.9, mape=5.2),
            "SNaive" => Heval.AccuracyMetrics(model="SNaive", mase=1.0, rmse=15.6, mae=11.2, mape=6.9)
        )
        features = Heval.SeriesFeatures(length=120, seasonal_period=12,
            trend_strength="moderate", seasonality_strength="strong")
        anomalies = [Heval.AnomalyResult(15, Date(2023, 3, 15), 245.67, 3.45)]
        result = Heval.AgentResult("The series shows moderate trend.", features,
            accuracy, fc, anomalies, "ETS", true)

        html_r = sprint(show, MIME("text/html"), result)
        @test occursin("heval-result", html_r)
        @test occursin("heval-pass", html_r)
        @test occursin("Best: ETS", html_r)
        @test occursin("heval-best", html_r)  # best model row highlight
        @test occursin("heval-narrative", html_r)  # LLM text section
        @test occursin("Series Features", html_r)
        @test occursin("Model Accuracy", html_r)
        @test occursin("Anomalies", html_r)
        @test occursin("★", html_r)  # star on best model

        # FAIL badge
        result_fail = Heval.AgentResult("text", nothing, Dict{String, Heval.AccuracyMetrics}(),
            nothing, Heval.AnomalyResult[], nothing, false)
        html_fail = sprint(show, MIME("text/html"), result_fail)
        @test occursin("heval-fail", html_fail)
        @test occursin("FAIL", html_fail)

        # HTML escaping
        result_esc = Heval.AgentResult("<script>alert('xss')</script>", nothing,
            Dict{String, Heval.AccuracyMetrics}(), nothing, Heval.AnomalyResult[], nothing, false)
        html_esc = sprint(show, MIME("text/html"), result_esc)
        @test !occursin("<script>", html_esc)
        @test occursin("&lt;script&gt;", html_esc)

        # Minimal AgentResult HTML (nothing/empty fields)
        minimal = Heval.AgentResult("", nothing, Dict{String, Heval.AccuracyMetrics}(),
            nothing, Heval.AnomalyResult[], nothing, true)
        html_min = sprint(show, MIME("text/html"), minimal)
        @test occursin("heval-result", html_min)
        @test !occursin("Series Features", html_min)
        @test !occursin("Model Accuracy", html_min)
        @test !occursin("<div class=\"heval-narrative\">", html_min)
    end

    @testset "QueryResult - Compact show" begin
        qr = Heval.QueryResult("This data appears to be monthly retail sales figures.")
        s = sprint(show, qr)
        @test occursin("QueryResult", s)
        @test occursin("This data appears to be monthly retail sales fi", s)

        # Long content gets truncated
        qr_long = Heval.QueryResult("x" ^ 100)
        s_long = sprint(show, qr_long)
        @test occursin("…", s_long)
        @test occursin("QueryResult(\"", s_long)
    end

    @testset "QueryResult - text/plain box" begin
        qr = Heval.QueryResult("This data appears to be monthly retail sales.")
        s = sprint(show, MIME("text/plain"), qr)
        @test occursin("╭─", s)
        @test occursin("Heval", s)
        @test occursin("╰", s)
        @test occursin("╮", s)
        @test occursin("╯", s)
        @test occursin("│", s)
        @test occursin("monthly retail sales", s)

        # Long text wraps
        qr_long = Heval.QueryResult("word " ^ 50)
        s_long = sprint(show, MIME("text/plain"), qr_long)
        lines = split(s_long, '\n')
        # Should have multiple content lines
        content_lines = filter(l -> startswith(l, "│"), lines)
        @test length(content_lines) > 1

        # Empty content
        qr_empty = Heval.QueryResult("")
        s_empty = sprint(show, MIME("text/plain"), qr_empty)
        @test occursin("╭─", s_empty)
        @test occursin("Heval", s_empty)
        @test occursin("╯", s_empty)
    end

    @testset "QueryResult - text/html" begin
        qr = Heval.QueryResult("The series shows a strong upward trend.")
        html = sprint(show, MIME("text/html"), qr)
        @test occursin("heval-result", html)
        @test occursin("heval-narrative", html)
        @test occursin("Heval</h4>", html)
        @test occursin("strong upward trend", html)

        # HTML escaping
        qr_esc = Heval.QueryResult("<script>alert('xss')</script>")
        html_esc = sprint(show, MIME("text/html"), qr_esc)
        @test !occursin("<script>", html_esc)
        @test occursin("&lt;script&gt;", html_esc)
    end

    @testset "QueryResult - string interop" begin
        qr = Heval.QueryResult("hello world")
        @test string(qr) == "hello world"
        @test sprint(print, qr) == "hello world"

        # Empty content
        qr_empty = Heval.QueryResult("")
        @test string(qr_empty) == ""
        @test sprint(print, qr_empty) == ""
    end

    @testset "System Prompt Generation" begin
        # Single series prompt
        prompt_single = Heval.build_system_prompt(; is_panel=false)
        @test occursin("Durbyn.jl", prompt_single)
        @test occursin("ARIMA", prompt_single)
        @test occursin("ARAR", prompt_single)
        @test occursin("Diffusion", prompt_single)
        @test occursin("16 total", prompt_single)
        @test !occursin("panel_analyze", prompt_single)

        # Panel prompt
        prompt_panel = Heval.build_system_prompt(; is_panel=true)
        @test occursin("panel_analyze", prompt_panel)
        @test occursin("panel_fit", prompt_panel)
        @test occursin("PANEL WORKFLOW", prompt_panel)
    end

end

println("\nAll tests passed!")
