from payment_graph_forecasting.evaluation import (
    EvaluationRunResult,
    evaluate_dygformer_model,
    evaluate_eagle_model,
    evaluate_glformer_model,
    evaluate_graphmixer_model,
    evaluate_hyperevent_model,
    evaluate_sg_graphmixer_model,
)


def test_evaluation_api_exports_are_importable():
    assert EvaluationRunResult is not None
    assert callable(evaluate_dygformer_model)
    assert callable(evaluate_graphmixer_model)
    assert callable(evaluate_eagle_model)
    assert callable(evaluate_glformer_model)
    assert callable(evaluate_hyperevent_model)
    assert callable(evaluate_sg_graphmixer_model)


def test_evaluate_graphmixer_model_wraps_legacy_function(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.1, "seed": kwargs["seed"]}

    import src.models.evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_graphmixer_model(seed=11)

    assert result.metrics == {"mrr": 0.1, "seed": 11}


def test_evaluate_eagle_model_wraps_legacy_function(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.2, "num_neighbors": kwargs["num_neighbors"]}

    import src.models.EAGLE.eagle_evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_eagle_model(num_neighbors=9)

    assert result.metrics == {"mrr": 0.2, "num_neighbors": 9}


def test_evaluate_dygformer_model_wraps_legacy_function(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.25, "max_edges": kwargs["max_edges"]}

    import src.models.DyGFormer.dygformer_evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_dygformer_model(max_edges=17)

    assert result.metrics == {"mrr": 0.25, "max_edges": 17}


def test_evaluate_glformer_model_wraps_legacy_function(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.3, "max_edges": kwargs["max_edges"]}

    import src.models.GLFormer.glformer_evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_glformer_model(max_edges=33)

    assert result.metrics == {"mrr": 0.3, "max_edges": 33}


def test_evaluate_glformer_model_dispatches_sampler_backend(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.31, "sampler": kwargs["sampler"]}

    import src.models.GLFormer_cuda.glformer_evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_glformer_model(sampler="cpp-sampler")

    assert result.metrics == {"mrr": 0.31, "sampler": "cpp-sampler"}


def test_evaluate_hyperevent_model_wraps_legacy_function(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.4, "n_latest": kwargs["n_latest"]}

    import src.models.HyperEvent.hyperevent_evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_hyperevent_model(n_latest=8)

    assert result.metrics == {"mrr": 0.4, "n_latest": 8}


def test_evaluate_sg_graphmixer_model_wraps_legacy_function(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.5, "n_negatives": kwargs["n_negatives"]}

    import src.models.sg_graphmixer.evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_sg_graphmixer_model(n_negatives=100)

    assert result.metrics == {"mrr": 0.5, "n_negatives": 100}
