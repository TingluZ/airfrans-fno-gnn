from afrans.splits import make_split


def test_make_split_no_overlap_and_full_coverage():
    names = [f"sim_{i}" for i in range(100)]
    split = make_split(names, val_frac=0.1, seed=0)

    assert set(split["train"]).isdisjoint(split["val"])
    assert len(split["train"]) + len(split["val"]) == len(names)
    assert len(split["val"]) == 10


def test_make_split_deterministic_with_same_seed():
    names = [f"sim_{i}" for i in range(50)]
    a = make_split(names, seed=42)
    b = make_split(names, seed=42)
    assert a["train"] == b["train"]
    assert a["val"] == b["val"]


def test_make_split_different_seeds_differ():
    names = [f"sim_{i}" for i in range(50)]
    a = make_split(names, seed=1)
    b = make_split(names, seed=2)
    assert a["val"] != b["val"]


def test_make_split_names_match_indices():
    names = [f"sim_{i}" for i in range(20)]
    split = make_split(names, val_frac=0.2, seed=0)

    assert split["train_names"] == [names[i] for i in split["train"]]
    assert split["val_names"] == [names[i] for i in split["val"]]
