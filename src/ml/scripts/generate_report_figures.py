"""Generate paper- and PPT-ready figures from saved experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from patent_model.plotting_style import setup_chinese_fonts
from scripts._cli_utils import positive_int
from scripts.report.decision import _build_decision_summary
from scripts.report.loaders import (
    _load_best_component_frame,
    _load_component_candidates,
    _load_environment_comparison,
    _load_main_runs,
    _load_robustness_model_summary,
    _load_robustness_tables,
)
from scripts.report.paper_plots import (
    _paper_style,
    _plot_best_profile_bars,
    _plot_candidate_component_compare,
    _plot_component_breakdown,
    _plot_decision_dashboard,
    _plot_env_compensation,
    _plot_main_heatmap,
    _plot_metric_small_multiples,
    _plot_pareto_frontier,
    _plot_robustness_curves,
    _plot_robustness_scoreboard,
    _plot_robustness_summary,
    _save_dual_format,
    _save_figure,
)
from scripts.report.ppt_plots import _plot_ppt_overview, _ppt_style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate report figures for the four-component patent-sim experiments.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default="outputs/report_figures_v3sync")
    parser.add_argument("--robustness-dir", default="outputs/environment_compensation_robustness_four_main_svr_ridge")
    parser.add_argument("--robustness-meta-key", default="dynamic_ridge")
    parser.add_argument("--figure-dpi", type=positive_int, default=300)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    setup_chinese_fonts()
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    outputs_root = repo_root / "outputs"
    output_dir = (repo_root / args.output_dir).resolve()
    paper_dir = output_dir / "paper"
    ppt_dir = output_dir / "ppt"
    data_dir = output_dir / "data"

    main_runs = _load_main_runs(outputs_root)
    best_runs = main_runs.sort_values("macro_RMSE_pp").groupby("profile", as_index=False).first()
    best_main_row = main_runs.sort_values("macro_RMSE_pp").iloc[0]
    best_component_frame = _load_best_component_frame(Path(str(best_main_row["run_dir"])))
    env_best = _load_environment_comparison(outputs_root)
    robustness_summary, noise_metrics, pressure_metrics = _load_robustness_tables(repo_root / args.robustness_dir)
    robustness_model_summary = _load_robustness_model_summary(outputs_root)
    decision_summary, top_candidates = _build_decision_summary(main_runs, robustness_model_summary)
    component_candidate_rows = top_candidates[
        top_candidates["recommendation"].isin(["recommended", "baseline", "candidate"])
    ].sort_values(["recommendation", "macro_RMSE_pp"], ascending=[True, True]).head(4)
    component_candidates = _load_component_candidates(component_candidate_rows)

    data_dir.mkdir(parents=True, exist_ok=True)
    main_runs.to_csv(data_dir / "main_grid_all_runs.csv", index=False)
    best_runs.to_csv(data_dir / "main_grid_best_by_profile.csv", index=False)
    env_best.to_csv(data_dir / "environment_best_profile_by_model.csv", index=False)
    robustness_summary.to_csv(data_dir / "robustness_summary.csv", index=False)
    robustness_model_summary.to_csv(data_dir / "robustness_model_summary.csv", index=False)
    decision_summary.to_csv(data_dir / "decision_summary.csv", index=False)
    top_candidates.to_csv(data_dir / "top_candidates.csv", index=False)
    component_candidates.to_csv(data_dir / "candidate_component_metrics.csv", index=False)

    _paper_style()
    fig_paths: list[str] = []
    paper_figures = [
        (_plot_best_profile_bars(best_runs), "fig01_main_best_profile"),
        (_plot_main_heatmap(main_runs), "fig02_main_grid_heatmap"),
        (_plot_component_breakdown(best_component_frame), "fig03_best_component_breakdown"),
        (_plot_env_compensation(env_best), "fig04_environment_compensation_best"),
        (_plot_robustness_summary(robustness_summary), "fig05_robustness_summary"),
        (_plot_robustness_curves(noise_metrics, pressure_metrics, args.robustness_meta_key), "fig06_robustness_curves"),
        (_plot_decision_dashboard(top_candidates), "fig07_decision_dashboard"),
        (_plot_metric_small_multiples(top_candidates), "fig08_metric_small_multiples"),
        (_plot_pareto_frontier(decision_summary), "fig09_pareto_frontier"),
        (_plot_candidate_component_compare(component_candidates), "fig10_candidate_component_compare"),
        (
            _plot_robustness_scoreboard(robustness_model_summary if not robustness_model_summary.empty else decision_summary),
            "fig11_robustness_scoreboard",
        ),
    ]
    for fig, name in paper_figures:
        fig_paths += _save_dual_format(fig, paper_dir / name, args.figure_dpi)

    _ppt_style()
    fig = _plot_ppt_overview(best_runs, env_best, best_component_frame, robustness_summary)
    _save_figure(fig, ppt_dir / "overview.png", args.figure_dpi, close=False)
    _save_figure(fig, ppt_dir / "overview.svg", args.figure_dpi, close=True)

    return {
        "output_dir": str(output_dir),
        "paper_dir": str(paper_dir),
        "ppt_dir": str(ppt_dir),
        "data_dir": str(data_dir),
        "paper_figures": fig_paths,
    }


if __name__ == "__main__":
    main()
