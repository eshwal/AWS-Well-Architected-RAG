import pytest
from src.service.hybrid import rank_fusion



class TestRankFusion:
    def test_basic_fusion_combines_two_lists(self,make_doc):
        dense = [make_doc("a"), make_doc("b")]
        sparse = [make_doc("b"), make_doc("c")]

        result = rank_fusion([dense, sparse], rrf_const=60, top_k=4)
        result_ids = [r.document.id for r in result]

        # all unique chunks across both lists should appear
        assert set(result_ids) == {"a", "b", "c"}

    def test_score_decays_with_rank_regression(self,make_doc):
        """Regression test for the fixed RRF bug: a chunk ranked higher
        (lower rank index) must contribute a LARGER score than one ranked
        lower. Before the fix, this relationship was inverted."""
        dense = [make_doc("first"), make_doc("second"), make_doc("third")]

        result = rank_fusion([dense], rrf_const=60, top_k=3)
        scores = {r.document.id: r.score for r in result}

        assert scores["first"] > scores["second"] > scores["third"]

    def test_duplicate_chunk_across_lists_scores_are_summed(self,make_doc):
        dense = [make_doc("shared")]
        sparse = [make_doc("shared")]

        result = rank_fusion([dense, sparse], rrf_const=60, top_k=4)

        assert len(result) == 1
        expected_score = (1 / (60 + 1 + 0)) + (1 / (60 + 1 + 0))
        assert result[0].score == pytest.approx(expected_score)

    def test_duplicate_chunk_keeps_first_seen_metadata(self,make_doc):
        dense = [make_doc("x", content="dense version")]
        sparse = [make_doc("x", content="sparse version")]

        result = rank_fusion([dense, sparse], rrf_const=60, top_k=4)

        assert result[0].document.page_content == "dense version"

    def test_top_k_truncates_results(self,make_doc):
        dense = [make_doc(str(i)) for i in range(10)]

        result = rank_fusion([dense], rrf_const=60, top_k=3)

        assert len(result) == 3

    def test_results_sorted_descending_by_score(self,make_doc):
        dense = [make_doc("a"), make_doc("b"), make_doc("c")]
        sparse = [make_doc("c"), make_doc("a"), make_doc("b")]

        result = rank_fusion([dense, sparse], rrf_const=60, top_k=3)
        scores = [r.score for r in result]

        assert scores == sorted(scores, reverse=True)