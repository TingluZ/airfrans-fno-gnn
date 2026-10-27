import numpy as np

from afrans.graph_dataset import edges_from_quads


def _undirected(edge_index):
    return {tuple(sorted(e)) for e in edge_index.T.tolist()}


def test_single_quad_gives_its_four_sides_not_diagonals():
    """VTK stores quad vertices in cyclic order, so the sides are consecutive
    pairs. The diagonals (0-2, 1-3) are not mesh edges."""
    ei = edges_from_quads([[0, 1, 2, 3]], n_nodes=4)

    assert ei.shape == (2, 8)
    assert _undirected(ei) == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_shared_side_is_not_duplicated():
    """Two quads sharing the side 1-4: 4 + 4 sides - 1 shared = 7 undirected."""
    ei = edges_from_quads([[0, 1, 4, 3], [1, 2, 5, 4]], n_nodes=6)

    assert len(_undirected(ei)) == 7
    assert ei.shape[1] == 14


def test_edge_index_is_symmetric_long_and_has_no_self_loops():
    ei = edges_from_quads([[0, 1, 4, 3], [1, 2, 5, 4]], n_nodes=6)

    directed = set(map(tuple, ei.T.tolist()))
    assert all((j, i) in directed for i, j in directed)
    assert all(i != j for i, j in directed)
    assert ei.dtype.is_floating_point is False and str(ei.dtype) == "torch.int64"


def test_works_when_ids_exceed_int32_key_range():
    """The dedupe packs each edge into one integer (lo * n_nodes + hi);
    it must not overflow for realistic mesh sizes."""
    n = 200_000
    ei = edges_from_quads(np.array([[n - 4, n - 3, n - 2, n - 1]], dtype=np.uint32), n_nodes=n)
    assert ei.max().item() == n - 1
    assert ei.shape[1] == 8
