import pytest

from impl_recon.utils import data_generation


def test_split_train_val_caseids():
    with pytest.raises(ValueError):  # empty val
        data_generation.split_train_val_caseids(10, -1)
    with pytest.raises(ValueError):  # empty val
        data_generation.split_train_val_caseids(2, 0.01)
    with pytest.raises(ValueError):  # empty train
        data_generation.split_train_val_caseids(10, 10)
    with pytest.raises(ValueError):  # empty train
        data_generation.split_train_val_caseids(2, 0.99)
    with pytest.raises(ValueError):  # negative num cases
        data_generation.split_train_val_caseids(-1, 0.5)
    # Correct usage
    ids_train, ids_valid = data_generation.split_train_val_caseids(100, 0.3)
    assert len(ids_train) + len(ids_valid) == 100
    assert all(x not in ids_valid for x in ids_train)
    assert all(x not in ids_train for x in ids_valid)
    assert len(ids_valid) < len(ids_train)

    # Reproducibility
    splits = [data_generation.split_train_val_caseids(100, 0.3) for _ in range(10)]
    assert all(x[0] == splits[0][0] for x in splits)
    assert all(x[1] == splits[0][1] for x in splits)
