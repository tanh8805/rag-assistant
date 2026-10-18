import pytest

from eval.metrics import recall_at_k, reciprocal_rank

RANKED = ["văn bản A", "có đoạn X ở đây", "văn bản B", "có đoạn Y\nxuống dòng"]


def test_recall_counts_spans_found_in_top_k():
    assert recall_at_k(RANKED, ["đoạn X"], k=1) == 0
    assert recall_at_k(RANKED, ["đoạn X"], k=2) == 1
    assert recall_at_k(RANKED, ["đoạn X", "đoạn Y xuống dòng"], k=2) == 0.5
    assert recall_at_k(RANKED, ["đoạn X", "đoạn Y xuống dòng"], k=4) == 1  # khác xuống dòng vẫn khớp


def test_reciprocal_rank_is_first_hit():
    assert reciprocal_rank(RANKED, ["đoạn X"]) == pytest.approx(1 / 2)
    assert reciprocal_rank(RANKED, ["đoạn Y", "đoạn X"]) == pytest.approx(1 / 2)  # span bất kỳ
    assert reciprocal_rank(RANKED, ["không có"]) == 0
    assert reciprocal_rank([], ["đoạn X"]) == 0
