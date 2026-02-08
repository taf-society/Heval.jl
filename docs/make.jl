using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using Heval

const ON_CI = get(ENV, "CI", "false") == "true"

makedocs(
    sitename = "Heval.jl",
    modules  = [Heval],
    format   = Documenter.HTML(
        prettyurls = ON_CI,
        assets     = ["assets/theme.css"],
        mathengine = Documenter.MathJax3(),
    ),
    pages    = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "User Guide" => Any[
            "Agent Architecture"    => "agent.md",
            "Available Models"      => "models.md",
            "Analysis Tools"        => "tools.md",
            "Panel Data"            => "panel.md",
            "Ollama Integration"    => "ollama.md",
            "Display & Formatting"  => "display.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :none,
)

deploydocs(
    repo      = "github.com/taf-society/Heval.jl",
    devbranch = "main",
)
