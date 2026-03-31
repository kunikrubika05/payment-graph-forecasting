import numpy as np

from src.models.HyperEvent.hyperevent_train import ensure_non_empty_relational_sequences


def test_ensure_non_empty_relational_sequences_unmasks_empty_rows():
    vecs = np.zeros((2, 4, 12), dtype=np.float32)
    masks = np.array(
        [
            [True, True, True, True],
            [False, True, True, True],
        ],
        dtype=bool,
    )

    safe_vecs, safe_masks = ensure_non_empty_relational_sequences(vecs, masks)

    assert safe_vecs is vecs
    assert safe_masks[0].tolist() == [False, True, True, True]
    assert safe_masks[1].tolist() == [False, True, True, True]


def test_ensure_non_empty_relational_sequences_preserves_non_empty_masks():
    vecs = np.zeros((1, 3, 12), dtype=np.float32)
    masks = np.array([[False, False, True]], dtype=bool)

    _, safe_masks = ensure_non_empty_relational_sequences(vecs, masks)

    assert np.array_equal(safe_masks, masks)
