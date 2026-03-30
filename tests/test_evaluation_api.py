from payment_graph_forecasting.evaluation import (
    EvaluationRunResult,
    evaluate_eagle_model,
    evaluate_glformer_model,
    evaluate_graphmixer_model,
)


def test_evaluation_api_exports_are_importable():
    assert EvaluationRunResult is not None
    assert callable(evaluate_graphmixer_model)
    assert callable(evaluate_eagle_model)
    assert callable(evaluate_glformer_model)


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


def test_evaluate_glformer_model_wraps_legacy_function(monkeypatch):
    def _fake_evaluate(**kwargs):
        return {"mrr": 0.3, "max_edges": kwargs["max_edges"]}

    import src.models.GLFormer.glformer_evaluate as legacy_eval

    monkeypatch.setattr(legacy_eval, "evaluate_tgb_style", _fake_evaluate)
    result = evaluate_glformer_model(max_edges=33)

    assert result.metrics == {"mrr": 0.3, "max_edges": 33}
