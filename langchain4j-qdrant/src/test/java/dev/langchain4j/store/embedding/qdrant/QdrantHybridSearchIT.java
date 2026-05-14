package dev.langchain4j.store.embedding.qdrant;

import static dev.langchain4j.internal.Utils.randomUUID;
import static io.qdrant.client.grpc.Collections.Distance.Cosine;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.filter.MetadataFilterBuilder;
import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.CreateCollection;
import io.qdrant.client.grpc.Collections.SparseVectorConfig;
import io.qdrant.client.grpc.Collections.SparseVectorParams;
import io.qdrant.client.grpc.Collections.VectorParams;
import io.qdrant.client.grpc.Collections.VectorsConfig;
import io.qdrant.client.grpc.Collections.VectorParamsMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.qdrant.QdrantContainer;

@Testcontainers
class QdrantHybridSearchIT {

    private static final String COLLECTION_NAME = "langchain4j-hybrid-" + randomUUID();
    private static final String DENSE_VECTOR_NAME = "dense";
    private static final String SPARSE_VECTOR_NAME = "sparse";

    @Container
    private static final QdrantContainer QDRANT_CONTAINER = new QdrantContainer("qdrant/qdrant:latest");

    private static QdrantEmbeddingStore EMBEDDING_STORE;
    private static final EmbeddingModel EMBEDDING_MODEL = new AllMiniLmL6V2QuantizedEmbeddingModel();

    /**
     * Naive term-frequency sparse embedding function for testing.
     * Produces sparse vectors where indices are character hash codes and values are term frequencies.
     */
    private static final QdrantSparseEmbeddingFunction SPARSE_FUNCTION = text -> {
        String[] words = text.toLowerCase().split("\\s+");
        Map<Integer, Float> termFreqs = new HashMap<>();
        for (String word : words) {
            int index = Math.abs(word.hashCode()) % 10000;
            termFreqs.merge(index, 1.0f, Float::sum);
        }
        List<Integer> indices = new ArrayList<>(termFreqs.keySet());
        List<Float> values = indices.stream().map(termFreqs::get).toList();
        return QdrantSparseEmbedding.of(values, indices);
    };

    @BeforeAll
    static void setup() throws InterruptedException, ExecutionException {
        QdrantClient client = new QdrantClient(
                QdrantGrpcClient.newBuilder(QDRANT_CONTAINER.getHost(), QDRANT_CONTAINER.getGrpcPort(), false)
                        .build());

        // Create collection with named dense and sparse vectors
        client.createCollectionAsync(CreateCollection.newBuilder()
                        .setCollectionName(COLLECTION_NAME)
                        .setVectorsConfig(VectorsConfig.newBuilder()
                                .setParamsMap(VectorParamsMap.newBuilder()
                                        .putMap(DENSE_VECTOR_NAME, VectorParams.newBuilder()
                                                .setDistance(Cosine)
                                                .setSize(EMBEDDING_MODEL.dimension())
                                                .build())))
                        .setSparseVectorsConfig(SparseVectorConfig.newBuilder()
                                .putMap(SPARSE_VECTOR_NAME, SparseVectorParams.newBuilder().build()))
                        .build())
                .get();

        client.close();

        EMBEDDING_STORE = QdrantEmbeddingStore.builder()
                .host(QDRANT_CONTAINER.getHost())
                .port(QDRANT_CONTAINER.getGrpcPort())
                .collectionName(COLLECTION_NAME)
                .searchMode(QdrantEmbeddingStore.SearchMode.HYBRID)
                .denseVectorName(DENSE_VECTOR_NAME)
                .sparseVectorName(SPARSE_VECTOR_NAME)
                .sparseEmbeddingFunction(SPARSE_FUNCTION)
                .rrfK(60)
                .prefetchLimit(20)
                .build();
    }

    @AfterAll
    static void teardown() {
        EMBEDDING_STORE.close();
    }

    @BeforeEach
    void clearStore() {
        EMBEDDING_STORE.clearStore();
    }

    @Test
    void should_perform_hybrid_search() {
        // Given
        TextSegment segment1 = TextSegment.from("The quick brown fox jumps over the lazy dog");
        TextSegment segment2 = TextSegment.from("A fast auburn fox leaps across the sleepy hound");
        TextSegment segment3 = TextSegment.from("Java programming language is widely used");

        Embedding embedding1 = EMBEDDING_MODEL.embed(segment1).content();
        Embedding embedding2 = EMBEDDING_MODEL.embed(segment2).content();
        Embedding embedding3 = EMBEDDING_MODEL.embed(segment3).content();

        EMBEDDING_STORE.add(embedding1, segment1);
        EMBEDDING_STORE.add(embedding2, segment2);
        EMBEDDING_STORE.add(embedding3, segment3);

        // When
        String queryText = "quick fox";
        Embedding queryEmbedding = EMBEDDING_MODEL.embed(queryText).content();

        EmbeddingSearchResult<TextSegment> result = EMBEDDING_STORE.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .query(queryText)
                        .maxResults(3)
                        .build());

        // Then
        assertThat(result.matches()).isNotEmpty();
        assertThat(result.matches().get(0).embedded().text())
                .contains("fox");
    }

    @Test
    void should_return_results_ordered_by_hybrid_score() {
        // Given
        TextSegment segment1 = TextSegment.from("Artificial intelligence and machine learning");
        TextSegment segment2 = TextSegment.from("Deep learning neural networks for AI");
        TextSegment segment3 = TextSegment.from("Cooking recipes for Italian pasta");

        Embedding embedding1 = EMBEDDING_MODEL.embed(segment1).content();
        Embedding embedding2 = EMBEDDING_MODEL.embed(segment2).content();
        Embedding embedding3 = EMBEDDING_MODEL.embed(segment3).content();

        EMBEDDING_STORE.add(embedding1, segment1);
        EMBEDDING_STORE.add(embedding2, segment2);
        EMBEDDING_STORE.add(embedding3, segment3);

        // When
        String queryText = "artificial intelligence machine learning";
        Embedding queryEmbedding = EMBEDDING_MODEL.embed(queryText).content();

        EmbeddingSearchResult<TextSegment> result = EMBEDDING_STORE.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .query(queryText)
                        .maxResults(3)
                        .build());

        // Then
        assertThat(result.matches()).hasSizeGreaterThanOrEqualTo(2);
        // Scores should be in descending order
        List<EmbeddingMatch<TextSegment>> matches = result.matches();
        for (int i = 0; i < matches.size() - 1; i++) {
            assertThat(matches.get(i).score()).isGreaterThanOrEqualTo(matches.get(i + 1).score());
        }
    }

    @Test
    void should_apply_metadata_filter_in_hybrid_search() {
        // Given
        TextSegment segment1 = TextSegment.from("Python programming language",
                Metadata.from("category", "programming"));
        TextSegment segment2 = TextSegment.from("Python snake species",
                Metadata.from("category", "animals"));

        Embedding embedding1 = EMBEDDING_MODEL.embed(segment1).content();
        Embedding embedding2 = EMBEDDING_MODEL.embed(segment2).content();

        EMBEDDING_STORE.add(embedding1, segment1);
        EMBEDDING_STORE.add(embedding2, segment2);

        // When - search for "Python" but filter to programming category
        String queryText = "Python";
        Embedding queryEmbedding = EMBEDDING_MODEL.embed(queryText).content();

        EmbeddingSearchResult<TextSegment> result = EMBEDDING_STORE.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .query(queryText)
                        .maxResults(3)
                        .filter(MetadataFilterBuilder.metadataKey("category").isEqualTo("programming"))
                        .build());

        // Then
        assertThat(result.matches()).hasSize(1);
        assertThat(result.matches().get(0).embedded().text()).contains("programming");
    }

    @Test
    void should_throw_when_query_is_null_in_hybrid_mode() {
        Embedding queryEmbedding = EMBEDDING_MODEL.embed("test").content();

        assertThatThrownBy(() -> EMBEDDING_STORE.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .maxResults(3)
                        .build()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("HYBRID");
    }

    @Test
    void should_include_embedding_in_hybrid_results() {
        // Given
        TextSegment segment = TextSegment.from("Test embedding inclusion in hybrid results");
        Embedding embedding = EMBEDDING_MODEL.embed(segment).content();
        EMBEDDING_STORE.add(embedding, segment);

        // When
        String queryText = "embedding hybrid results";
        Embedding queryEmbedding = EMBEDDING_MODEL.embed(queryText).content();

        EmbeddingSearchResult<TextSegment> result = EMBEDDING_STORE.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .query(queryText)
                        .maxResults(1)
                        .build());

        // Then
        assertThat(result.matches()).hasSize(1);
        assertThat(result.matches().get(0).embedding()).isNotNull();
        assertThat(result.matches().get(0).embedding().vector()).hasSize(EMBEDDING_MODEL.dimension());
    }

    @Test
    void should_add_and_search_multiple_documents() {
        // Given
        List<TextSegment> segments = List.of(
                TextSegment.from("Database indexing strategies for performance"),
                TextSegment.from("SQL query optimization techniques"),
                TextSegment.from("NoSQL database design patterns"),
                TextSegment.from("Weather forecast for tomorrow"),
                TextSegment.from("Climate change and global warming"));

        List<Embedding> embeddings = segments.stream()
                .map(s -> EMBEDDING_MODEL.embed(s).content())
                .toList();

        List<String> ids = EMBEDDING_STORE.addAll(embeddings, segments);
        assertThat(ids).hasSize(5);

        // When
        String queryText = "database performance optimization";
        Embedding queryEmbedding = EMBEDDING_MODEL.embed(queryText).content();

        EmbeddingSearchResult<TextSegment> result = EMBEDDING_STORE.search(
                EmbeddingSearchRequest.builder()
                        .queryEmbedding(queryEmbedding)
                        .query(queryText)
                        .maxResults(3)
                        .build());

        // Then
        assertThat(result.matches()).hasSize(3);
        // Top results should be database-related
        assertThat(result.matches().get(0).embedded().text().toLowerCase())
                .containsAnyOf("database", "sql", "indexing");
    }

    @Test
    void should_search_with_vector_mode_on_hybrid_collection() throws InterruptedException, ExecutionException {
        // Given: a separate store configured with VECTOR mode but with sparseEmbeddingFunction
        // This proves you can ingest with sparse vectors and search dense-only without re-ingesting.
        String vectorModeCollection = "langchain4j-vector-on-hybrid-" + randomUUID();

        QdrantClient client = new QdrantClient(
                QdrantGrpcClient.newBuilder(QDRANT_CONTAINER.getHost(), QDRANT_CONTAINER.getGrpcPort(), false)
                        .build());

        client.createCollectionAsync(CreateCollection.newBuilder()
                        .setCollectionName(vectorModeCollection)
                        .setVectorsConfig(VectorsConfig.newBuilder()
                                .setParamsMap(VectorParamsMap.newBuilder()
                                        .putMap(DENSE_VECTOR_NAME, VectorParams.newBuilder()
                                                .setDistance(Cosine)
                                                .setSize(EMBEDDING_MODEL.dimension())
                                                .build())))
                        .setSparseVectorsConfig(SparseVectorConfig.newBuilder()
                                .putMap(SPARSE_VECTOR_NAME, SparseVectorParams.newBuilder().build()))
                        .build())
                .get();
        client.close();

        QdrantEmbeddingStore vectorModeStore = QdrantEmbeddingStore.builder()
                .host(QDRANT_CONTAINER.getHost())
                .port(QDRANT_CONTAINER.getGrpcPort())
                .collectionName(vectorModeCollection)
                .searchMode(QdrantEmbeddingStore.SearchMode.VECTOR)
                .denseVectorName(DENSE_VECTOR_NAME)
                .sparseVectorName(SPARSE_VECTOR_NAME)
                .sparseEmbeddingFunction(SPARSE_FUNCTION)
                .build();

        try {
            TextSegment segment1 = TextSegment.from("The quick brown fox jumps over the lazy dog");
            TextSegment segment2 = TextSegment.from("Java programming language is widely used");

            Embedding embedding1 = EMBEDDING_MODEL.embed(segment1).content();
            Embedding embedding2 = EMBEDDING_MODEL.embed(segment2).content();

            // Ingests with named vectors (dense + sparse)
            vectorModeStore.add(embedding1, segment1);
            vectorModeStore.add(embedding2, segment2);

            // When: searching with VECTOR mode (dense-only, no query text needed)
            Embedding queryEmbedding = EMBEDDING_MODEL.embed("fox dog").content();

            EmbeddingSearchResult<TextSegment> result = vectorModeStore.search(
                    EmbeddingSearchRequest.builder()
                            .queryEmbedding(queryEmbedding)
                            .maxResults(2)
                            .build());

            // Then: should find results using only the dense vector
            assertThat(result.matches()).isNotEmpty();
            assertThat(result.matches().get(0).embedded().text()).contains("fox");
            assertThat(result.matches().get(0).embedding()).isNotNull();
        } finally {
            vectorModeStore.close();
        }
    }
}
