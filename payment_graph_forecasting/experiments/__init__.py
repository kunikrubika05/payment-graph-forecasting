"""Experiment launchers and runners."""


def launch_experiment(spec):
    """Lazy import to keep package imports lightweight."""

    from payment_graph_forecasting.experiments.launcher import launch_experiment as _launch_experiment

    return _launch_experiment(spec)


__all__ = ["launch_experiment"]
