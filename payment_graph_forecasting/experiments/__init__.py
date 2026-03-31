"""Experiment launchers and runners."""


def launch_experiment(spec):
    """Lazy import to keep package imports lightweight."""

    from payment_graph_forecasting.experiments.launcher import launch_experiment as _launch_experiment

    return _launch_experiment(spec)


def run_hpo(model_name, argv=None):
    """Lazy import to keep package imports lightweight."""

    from payment_graph_forecasting.experiments.hpo import run_hpo as _run_hpo

    return _run_hpo(model_name, argv)


__all__ = ["launch_experiment", "run_hpo"]
