import logging
import time
from typing import Any

from app.models.conversation import RetrievedDocument


logger = logging.getLogger(__name__)


class OpenSearchHybridClient:
    def __init__(
        self,
        host: str | None,
        index_name: str,
        region: str,
        port: int,
        username: str | None,
        password: str | None,
        verify_certs: bool,
        vector_field: str,
        embedding_dimensions: int,
        llm_client: Any,
    ):
        self.host = host
        self.index_name = index_name
        self.region = region
        self.port = port
        self.username = username
        self.password = password
        self.verify_certs = verify_certs
        self.vector_field = vector_field or "embeddings"
        self.embedding_dimensions = embedding_dimensions
        self.llm_client = llm_client
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.host)

    def search(self, query: str, size: int = 5, filters: dict[str, Any] | None = None) -> list[RetrievedDocument]:
        if not self.host:
            logger.info("OpenSearch host is not configured; returning no retrieval results")
            return []

        client = self._get_client()
        if client is None:
            return []

        full_text_results = self._full_text_search(client=client, query=query, size=max(size * 3, 15), filters=filters)
        try:
            query_embedding = self.llm_client.create_embedding(query, dimensions=self.embedding_dimensions)
        except Exception as exc:  # pragma: no cover - network/API failure path
            logger.warning("Failed to generate query embedding; continuing with full-text only: %s", exc)
            query_embedding = None
        semantic_results = []
        if query_embedding:
            semantic_results = self._semantic_search(
                client=client,
                query_embedding=query_embedding,
                size=max(size * 3, 15),
                k=max(size * 4, 20),
                filters=filters,
            )

        return self._reciprocal_rank_fusion(full_text_results, semantic_results, limit=size)

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection
        except ImportError:  # pragma: no cover - depends on local environment
            logger.warning("opensearch-py is not installed; retrieval will remain disabled")
            return None

        if self.username and self.password:
            self._client = OpenSearch(
                hosts=[{"host": self.host, "port": self.port}],
                http_auth=(self.username, self.password),
                use_ssl=True,
                verify_certs=self.verify_certs,
                timeout=30,
            )
            return self._client

        try:
            from opensearchpy import AWSV4SignerAuth
            import boto3
        except ImportError:  # pragma: no cover - depends on local environment
            logger.warning("AWS OpenSearch auth dependencies are missing; retrieval will remain disabled")
            return None

        credentials = boto3.Session(region_name=self.region).get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, "es")
        self._client = OpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=self.verify_certs,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )
        return self._client

    def _full_text_search(self, client, query: str, size: int, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = client.search(index=self.index_name, body=self._full_text_query(query=query, size=size, filters=filters))
                return [
                    {
                        "id": hit.get("_source", {}).get("doc_id", hit["_id"]),
                        "score": hit.get("_score"),
                        "source": hit.get("_source", {}),
                    }
                    for hit in response.get("hits", {}).get("hits", [])
                ]
            except Exception as exc:
                if attempt == max_retries:
                    logger.warning("Full-text search failed after %s attempts: %s", max_retries + 1, exc)
                    return []
                delay_seconds = 2**attempt
                logger.warning(
                    "Full-text search attempt %s/%s failed: %s. Retrying in %ss",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay_seconds,
                )
                time.sleep(delay_seconds)

    def _semantic_search(
        self,
        client,
        query_embedding: list[float],
        size: int,
        k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        preferred_vector_field = self.vector_field
        candidate_fields = [preferred_vector_field]
        if preferred_vector_field != "embeddings":
            candidate_fields.append("embeddings")
        if preferred_vector_field != "embedding":
            candidate_fields.append("embedding")

        for field in candidate_fields:
            results = self._semantic_search_for_field(
                client=client,
                vector_field=field,
                query_embedding=query_embedding,
                size=size,
                k=k,
                filters=filters,
            )
            if results:
                if field != self.vector_field:
                    logger.info("Semantic search succeeded using fallback vector field '%s'", field)
                    self.vector_field = field
                return results
        return []

    def _semantic_search_for_field(
        self,
        client,
        vector_field: str,
        query_embedding: list[float],
        size: int,
        k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = client.search(
                    index=self.index_name,
                    body=self._semantic_query(
                        vector_field=vector_field,
                        query_embedding=query_embedding,
                        size=size,
                        k=k,
                        filters=filters,
                    ),
                )
                return [
                    {
                        "id": hit.get("_source", {}).get("doc_id", hit["_id"]),
                        "score": hit.get("_score"),
                        "source": hit.get("_source", {}),
                    }
                    for hit in response.get("hits", {}).get("hits", [])
                ]
            except Exception as exc:
                error_text = str(exc)
                non_retryable = "not knn_vector type" in error_text or "failed to create query" in error_text
                if non_retryable:
                    logger.warning(
                        "Semantic search failed for vector field '%s': %s",
                        vector_field,
                        exc,
                    )
                    return []

                if attempt == max_retries:
                    logger.warning(
                        "Semantic search failed for vector field '%s' after %s attempts: %s",
                        vector_field,
                        max_retries + 1,
                        exc,
                    )
                    return []

                delay_seconds = 2**attempt
                logger.warning(
                    "Semantic search attempt %s/%s failed on field '%s': %s. Retrying in %ss",
                    attempt + 1,
                    max_retries + 1,
                    vector_field,
                    exc,
                    delay_seconds,
                )
                time.sleep(delay_seconds)

    def _reciprocal_rank_fusion(
        self,
        full_text_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
        limit: int,
        k: int = 60,
    ) -> list[RetrievedDocument]:
        fused_scores: dict[str, float] = {}
        documents: dict[str, dict[str, Any]] = {}
        for rank, result in enumerate(full_text_results, start=1):
            doc_id = result["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            documents.setdefault(doc_id, result["source"])
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            documents.setdefault(doc_id, result["source"])

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [self._to_document(doc_id=doc_id, score=score, source=documents[doc_id]) for doc_id, score in ranked]

    def _to_document(self, doc_id: str, score: float, source: dict[str, Any]) -> RetrievedDocument:
        metadata = source.get("metadata", {})
        content = (
            source.get("content")
            or source.get("normalized_text")
            or source.get("text")
            or source.get("chunk_text")
            or ""
        )
        resolved_doc_id = source.get("doc_id") or source.get("id") or doc_id
        return RetrievedDocument(
            doc_id=resolved_doc_id,
            title=source.get("title") or metadata.get("title") or metadata.get("source_filename"),
            product=source.get("product") or metadata.get("source_folder"),
            model=source.get("model"),
            firmware=source.get("firmware"),
            error_code=source.get("error_code"),
            doc_type=source.get("doc_type"),
            section_title=source.get("section_title"),
            page_number=source.get("page_number"),
            content=content,
            score=score,
            metadata=metadata,
        )

    def _full_text_query(self, query: str, size: int, filters: dict[str, Any] | None) -> dict[str, Any]:
        must: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "normalized_text^3",
                        "text^2",
                        "content^2",
                        "title^2",
                        "section_title^2",
                        "error_code^2",
                        "model^2",
                    ],
                    "fuzziness": "AUTO",
                    "lenient": True,
                }
            }
        ]
        return {
            "size": size,
            "query": {
                "bool": {
                    "must": must,
                    "filter": self._build_filters(filters),
                }
            },
            "_source": [
                "doc_id",
                "title",
                "product",
                "model",
                "firmware",
                "error_code",
                "doc_type",
                "section_title",
                "page_number",
                "content",
                "id",
                "normalized_text",
                "text",
                "tokens",
                "adjacency",
                "metadata",
            ],
        }

    def _semantic_query(
        self,
        vector_field: str,
        query_embedding: list[float],
        size: int,
        k: int,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {
            "size": size,
            "_source": [
                "doc_id",
                "title",
                "product",
                "model",
                "firmware",
                "error_code",
                "doc_type",
                "section_title",
                "page_number",
                "content",
                "id",
                "normalized_text",
                "text",
                "tokens",
                "adjacency",
                "metadata",
            ],
        }
        filter_clauses = self._build_filters(filters)
        if filter_clauses:
            query["query"] = {
                "bool": {
                    "filter": filter_clauses,
                    "must": [
                        {
                            "knn": {
                                vector_field: {
                                    "vector": query_embedding,
                                    "k": k,
                                }
                            }
                        }
                    ],
                }
            }
        else:
            query["query"] = {"knn": {vector_field: {"vector": query_embedding, "k": k}}}
        return query

    def _build_filters(self, filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        clauses = []
        for key, value in (filters or {}).items():
            if value in (None, "", [], {}):
                continue
            clauses.append({"term": {key: value}})
        return clauses
