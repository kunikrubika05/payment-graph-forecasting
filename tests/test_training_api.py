from payment_graph_forecasting.training import (
    TrainingRunResult,
    train_eagle_model,
    train_glformer_model,
    train_graphmixer_model,
    train_hyperevent_model,
    train_sg_graphmixer_model,
)


def test_training_api_exports_are_importable():
    assert TrainingRunResult is not None
    assert callable(train_graphmixer_model)
    assert callable(train_eagle_model)
    assert callable(train_glformer_model)
    assert callable(train_hyperevent_model)
    assert callable(train_sg_graphmixer_model)


def test_train_graphmixer_model_wraps_legacy_function(monkeypatch):
    def _fake_train_graphmixer(**kwargs):
        return "graphmixer-model", {"train_loss": [1.0], "received": kwargs["seed"]}

    import src.models.train as legacy_train

    monkeypatch.setattr(legacy_train, "train_graphmixer", _fake_train_graphmixer)
    result = train_graphmixer_model(seed=7)

    assert result.model == "graphmixer-model"
    assert result.history["received"] == 7


def test_train_eagle_model_wraps_legacy_function(monkeypatch):
    def _fake_train_eagle(**kwargs):
        return "eagle-model", {"train_loss": [0.5], "received": kwargs["batch_size"]}

    import src.models.EAGLE.eagle_train as legacy_train

    monkeypatch.setattr(legacy_train, "train_eagle", _fake_train_eagle)
    result = train_eagle_model(batch_size=16)

    assert result.model == "eagle-model"
    assert result.history["received"] == 16


def test_train_glformer_model_wraps_legacy_function(monkeypatch):
    def _fake_train_glformer(**kwargs):
        return "glformer-model", {"train_loss": [0.25], "received": kwargs["patience"]}

    import src.models.GLFormer.glformer_train as legacy_train

    monkeypatch.setattr(legacy_train, "train_glformer", _fake_train_glformer)
    result = train_glformer_model(patience=3)

    assert result.model == "glformer-model"
    assert result.history["received"] == 3


def test_train_glformer_model_dispatches_sampler_backend(monkeypatch):
    def _fake_train_glformer_cuda(**kwargs):
        return "glformer-cuda-model", {"train_loss": [0.2], "received": kwargs["sampling_backend"]}

    import src.models.GLFormer_cuda.glformer_train as legacy_train

    monkeypatch.setattr(legacy_train, "train_glformer_cuda", _fake_train_glformer_cuda)
    result = train_glformer_model(sampling_backend="cpp")

    assert result.model == "glformer-cuda-model"
    assert result.history["received"] == "cpp"


def test_train_hyperevent_model_wraps_legacy_function(monkeypatch):
    def _fake_train_hyperevent(**kwargs):
        return "hyperevent-model", {"train_loss": [0.2], "received": kwargs["n_neighbor"]}

    import src.models.HyperEvent.hyperevent_train as legacy_train

    monkeypatch.setattr(legacy_train, "train_hyperevent", _fake_train_hyperevent)
    result = train_hyperevent_model(n_neighbor=12)

    assert result.model == "hyperevent-model"
    assert result.history["received"] == 12


def test_train_sg_graphmixer_model_wraps_legacy_function(monkeypatch):
    def _fake_train_graphmixer(**kwargs):
        return "sg-graphmixer-model", {"train_loss": [0.1], "received": kwargs["n_negatives"]}

    import src.models.sg_graphmixer.train as legacy_train

    monkeypatch.setattr(legacy_train, "train_graphmixer", _fake_train_graphmixer)
    result = train_sg_graphmixer_model(n_negatives=100)

    assert result.model == "sg-graphmixer-model"
    assert result.history["received"] == 100
