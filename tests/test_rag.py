import pytest
from unittest.mock import AsyncMock, patch
from src.service.rag import (
    should_use_sparse_fallback,
    format_docs,
    query_compliance_platform,
)
from src.models.document import IngestionChunk, RetrievalChunk




class TestShouldUseSparseFallback:
    def test_matches_doc_code(self):
        assert should_use_sparse_fallback("What does AGENTSEC02 require?") is True

    def test_matches_snake_case_identifier(self):
        assert should_use_sparse_fallback("Explain x_amz_bedrock_agentcore_search") is True

    def test_false_for_normal_query(self):
        assert should_use_sparse_fallback("How do I prevent tool overload?") is False


class TestFormatDocs:
    def test_joins_docs_with_source_and_content(self,make_doc):
        docs = [make_doc(doc_id="1", content="Alpha", source="doc1"),
                make_doc(doc_id="2", content="Beta", source="doc2")]

        result = format_docs(docs)

        assert "doc1" in result and "Alpha" in result
        assert "doc2" in result and "Beta" in result

    def test_handles_missing_source_metadata(self):
        doc = RetrievalChunk(
            document=IngestionChunk(id="1", page_content="text", metadata={}),
            score=1.0,
        )
        result = format_docs([doc])
        assert "Unknown" in result


class TestQueryCompliancePlatform:
    @pytest.mark.anyio
    async def test_dense_success_returns_answer(self,make_doc):
        docs = [make_doc()]
        with patch("src.service.rag.dense_context", new=AsyncMock(return_value=(docs, "context text"))), \
             patch("src.service.rag.generate_answer", new=AsyncMock(return_value="the answer")):

            result = await query_compliance_platform("a question", {"search_mode": "dense"})

        assert result["answer"] == "the answer"
        assert result["error"] is None if "error" in result else "error" not in result
        assert len(result["referenced_metadata"]) == 1

    @pytest.mark.anyio
    async def test_retrieval_failure_returns_graceful_error(self):
        with patch("src.service.rag.dense_context", new=AsyncMock(side_effect=Exception("pinecone down"))):
            result = await query_compliance_platform("a question", {"search_mode": "dense"})

        assert result["answer"] == ""
        assert result["chunks"] == []
        assert "retrieval_failed" in result["error"]

    @pytest.mark.anyio
    async def test_generation_failure_returns_graceful_error_with_docs_preserved(self,make_doc):
        docs = [make_doc()]
        with patch("src.service.rag.dense_context", new=AsyncMock(return_value=(docs, "context text"))), \
             patch("src.service.rag.generate_answer", new=AsyncMock(side_effect=Exception("mistral timeout"))):

            result = await query_compliance_platform("a question", {"search_mode": "dense"})

        assert result["answer"] == ""
        assert "generation_failed" in result["error"]
        # docs were already retrieved, so chunks/sources should still be populated
        assert len(result["chunks"]) == 1

    @pytest.mark.anyio
    async def test_hybrid_mode_uses_hybrid_context(self,make_doc):
        docs = [make_doc()]
        with patch("src.service.rag.hybrid_context", new=AsyncMock(return_value=(docs, "context"))) as mock_hybrid, \
             patch("src.service.rag.generate_answer", new=AsyncMock(return_value="answer")):

            await query_compliance_platform("a question", {"search_mode": "hybrid", "rrf_const": 50})

        mock_hybrid.assert_called_once()