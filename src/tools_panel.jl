# ============================================================================
# Panel Data Tools for Heval.jl — Durbyn Backend
# ============================================================================

using Durbyn
import Tables

# ============================================================================
# Tool: panel_analyze
# ============================================================================

function create_panel_analyze_tool(state::AgentState)
    Tool(
        "panel_analyze",
        "Analyze panel (multi-series) data. Constructs a PanelData object and extracts per-group features including trend, seasonality, and data summary.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "groups" => Dict(
                    "type" => "array",
                    "items" => Dict("type" => "string"),
                    "description" => "Column names to group by (e.g., ['store', 'region'])"
                ),
                "date" => Dict(
                    "type" => "string",
                    "description" => "Date column name (default: 'date')"
                ),
                "target" => Dict(
                    "type" => "string",
                    "description" => "Target variable column name (default: 'value')"
                ),
                "frequency" => Dict(
                    "type" => "string",
                    "description" => "Data frequency: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly' (optional)"
                ),
                "fill_time" => Dict(
                    "type" => "boolean",
                    "description" => "Fill time gaps (default: false)"
                ),
                "balanced" => Dict(
                    "type" => "boolean",
                    "description" => "Balance panel across groups (default: false)"
                )
            ),
            "required" => ["groups"]
        ),
        function(args)
            tool_panel_analyze(state, args)
        end
    )
end

function tool_panel_analyze(state::AgentState, args::Dict)
    if isnothing(state.panel)
        return Dict("error" => "No panel data loaded. Use analyze() with groupby parameter.")
    end

    groups = Symbol[Symbol(g) for g in get(args, "groups", String.(state.panel.groups))]
    date_col = Symbol(get(args, "date", String(state.panel.date_col)))
    target_col = Symbol(get(args, "target", String(state.panel.target_col)))
    frequency = let f = get(args, "frequency", nothing)
        isnothing(f) ? nothing : Symbol(f)
    end
    fill_time = get(args, "fill_time", false)
    balanced = get(args, "balanced", false)
    m = state.seasonal_period

    try
        # Build PanelData
        panel = build_panel(state.panel.raw_data;
                            groupby=groups, date=date_col, m=m,
                            frequency=frequency, fill_time=fill_time, balanced=balanced,
                            target=target_col)
        state.panel.panel = panel
        state.panel.groups = groups
        state.panel.date_col = date_col
        state.panel.target_col = target_col

        # Count groups and series lengths
        ct = Tables.columntable(state.panel.raw_data)
        n_rows = length(ct[first(propertynames(ct))])

        # Get unique group keys
        group_keys = Set{String}()
        for i in 1:n_rows
            key_parts = [string(ct[g][i]) for g in groups]
            push!(group_keys, join(key_parts, "|"))
        end

        n_groups = length(group_keys)

        return Dict(
            "status" => "success",
            "n_groups" => n_groups,
            "n_rows" => n_rows,
            "group_columns" => String.(groups),
            "date_column" => String(date_col),
            "target_column" => String(target_col),
            "seasonal_period" => m,
            "fill_time" => fill_time,
            "balanced" => balanced,
            "message" => "Panel data constructed with $n_groups groups. Use panel_fit to fit models across all groups."
        )
    catch e
        return Dict("error" => "Failed to construct panel: $(sprint(showerror, e))")
    end
end

# ============================================================================
# Tool: panel_fit
# ============================================================================

function create_panel_fit_tool(state::AgentState)
    Tool(
        "panel_fit",
        "Fit one or more models across all groups in the panel data and generate forecasts.",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "models" => Dict(
                    "type" => "array",
                    "items" => Dict("type" => "string"),
                    "description" => "Model names to fit (e.g., ['ARIMA', 'ETS', 'Theta'])"
                ),
                "h" => Dict(
                    "type" => "integer",
                    "description" => "Forecast horizon (default: use stored horizon)"
                )
            ),
            "required" => ["models"]
        ),
        function(args)
            tool_panel_fit(state, args)
        end
    )
end

function tool_panel_fit(state::AgentState, args::Dict)
    if isnothing(state.panel) || isnothing(state.panel.panel)
        return Dict("error" => "No panel data constructed. Call panel_analyze first.")
    end

    model_names = get(args, "models", String[])
    h = _to_int(get(args, "h", state.horizon), state.horizon)

    if isempty(model_names)
        return Dict("error" => "No models specified")
    end

    # Validate models
    for name in model_names
        if !(name in AVAILABLE_MODELS)
            return Dict("error" => "Unknown model: $name. Available: $(join(AVAILABLE_MODELS, ", "))")
        end
    end

    m = state.seasonal_period
    target = state.panel.target_col

    try
        # Build model collection
        specs = [build_spec(name, target; m=m) for name in model_names]
        collection = Durbyn.model(specs...; names=model_names)

        # Fit across all groups
        fitted_collection = Durbyn.fit(collection, state.panel.panel)

        # Forecast
        fc = Durbyn.forecast(fitted_collection; h=h)
        state.panel.group_forecasts = fc

        # Convert to table for summary
        fc_table = Durbyn.as_table(fc)

        # Extract summary statistics
        n_successful = try length(Durbyn.successful_models(fitted_collection)) catch; 0 end
        n_failed = try length(Durbyn.failed_groups(fitted_collection)) catch; 0 end

        return Dict(
            "status" => "success",
            "models" => model_names,
            "horizon" => h,
            "n_successful_fits" => n_successful,
            "n_failed_fits" => n_failed,
            "message" => "Fitted $(length(model_names)) model(s) across panel groups. Forecasts generated for h=$h periods."
        )
    catch e
        return Dict("error" => "Panel fitting failed: $(sprint(showerror, e))")
    end
end

# ============================================================================
# Register all panel tools
# ============================================================================

function register_panel_tools!(agent)
    state = agent.state
    panel_tools = [create_panel_analyze_tool(state), create_panel_fit_tool(state)]
    for t in panel_tools
        push!(agent.tools, t)
        agent.tool_index[t.name] = t
    end
end
