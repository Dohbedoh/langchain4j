package dev.langchain4j.store.embedding.qdrant;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.Utils.isNullOrEmpty;
import static dev.langchain4j.internal.Utils.randomUUID;
import static dev.langchain4j.internal.ValidationUtils.ensureNotEmpty;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;
import static io.qdrant.client.PointIdFactory.id;
import static io.qdrant.client.ValueFactory.value;
import static io.qdrant.client.VectorsFactory.namedVectors;
import static io.qdrant.client.VectorsFactory.vectors;
import static io.qdrant.client.WithPayloadSelectorFactory.enable;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static java.util.Comparator.comparingDouble;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.*;
import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.VectorInputFactory;
import io.qdrant.client.WithVectorsSelectorFactory;
import io.qdrant.client.grpc.Common.Filter;
import io.qdrant.client.grpc.JsonWithInt.Value;
import io.qdrant.client.grpc.Points;
import io.qdrant.client.grpc.Points.DeletePoints;
import io.qdrant.client.grpc.Points.PointStruct;
import io.qdrant.client.grpc.Points.PointsSelector;
import io.qdrant.client.grpc.Points.PrefetchQuery;
import io.qdrant.client.grpc.Points.Query;
import io.qdrant.client.grpc.Points.QueryPoints;
import io.qdrant.client.grpc.Points.Rrf;
import io.qdrant.client.grpc.Points.ScoredPoint;
import io.qdrant.client.grpc.Points.SearchPoints;
import io.qdrant.client.grpc.Points.SparseVector;
import io.qdrant.client.grpc.Points.Vector;
import io.qdrant.client.grpc.Points.VectorOutput;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents a <a href="https://qdrant.tech/">Qdrant</a> collection as an embedding store.
 * Supports storing {@link dev.langchain4j.data.document.Metadata}.
 *
 * <p>Supports two search modes:
 * <ul>
 *   <li>{@link SearchMode#VECTOR} — standard dense vector search (default)</li>
 *   <li>{@link SearchMode#HYBRID} — combines dense and sparse vector search using
 *       Qdrant's server-side <a href="https://qdrant.tech/documentation/concepts/hybrid-queries/">
 *       Reciprocal Rank Fusion (RRF)</a></li>
 * </ul>
 *
 * <h2>Decoupled ingestion and search</h2>
 * <p>Ingestion is driven by the presence of a {@link QdrantSparseEmbeddingFunction}, not
 * by the {@code searchMode}. When a sparse embedding function is configured, both dense and
 * sparse named vectors are stored regardless of the active search mode. This allows switching
 * between {@code VECTOR} and {@code HYBRID} search on the same collection without re-ingesting data.
 *
 * <p>Conversely, data ingested without a sparse embedding function (legacy/unnamed vectors) cannot
 * be searched in {@code HYBRID} mode without re-ingestion.
 *
 * <h2>Hybrid mode requirements</h2>
 * <ul>
 *   <li>The Qdrant collection must be configured with named dense and sparse vectors
 *       matching {@code denseVectorName} and {@code sparseVectorName}</li>
 *   <li>A {@link QdrantSparseEmbeddingFunction} must be provided</li>
 *   <li>{@link dev.langchain4j.store.embedding.EmbeddingSearchRequest#query()} must be
 *       non-blank at search time (used to generate the sparse query vector)</li>
 * </ul>
 */
public class QdrantEmbeddingStore implements EmbeddingStore<TextSegment> {
    private static final Logger log = LoggerFactory.getLogger(QdrantEmbeddingStore.class);

    /**
     * Search modes for the embedding store.
     */
    public enum SearchMode {
        VECTOR,
        HYBRID
    }

    private final QdrantClient client;
    private final String payloadTextKey;
    private final String collectionName;
    private final SearchMode searchMode;
    private final String denseVectorName;
    private final String sparseVectorName;
    private final QdrantSparseEmbeddingFunction sparseEmbeddingFunction;
    private final int rrfK;
    private final int prefetchLimit;

    /**
     * @param collectionName The name of the Qdrant collection.
     * @param host           The host of the Qdrant instance.
     * @param port           The GRPC port of the Qdrant instance.
     * @param useTls         Whether to use TLS(HTTPS).
     * @param payloadTextKey The field name of the text segment in the Qdrant
     *                       payload.
     * @param apiKey         The Qdrant API key to authenticate with.
     */
    public QdrantEmbeddingStore(
            String collectionName,
            String host,
            int port,
            boolean useTls,
            String payloadTextKey,
            @Nullable String apiKey) {
        this(collectionName, host, port, useTls, payloadTextKey, apiKey,
                SearchMode.VECTOR, null, null, null, 60, 20);
    }

    /**
     * @param client         A Qdrant client instance.
     * @param collectionName The name of the Qdrant collection.
     * @param payloadTextKey The field name of the text segment in the Qdrant
     *                       payload.
     */
    public QdrantEmbeddingStore(QdrantClient client, String collectionName, String payloadTextKey) {
        this(client, collectionName, payloadTextKey,
                SearchMode.VECTOR, null, null, null, 60, 20);
    }

    QdrantEmbeddingStore(
            String collectionName,
            String host,
            int port,
            boolean useTls,
            String payloadTextKey,
            @Nullable String apiKey,
            SearchMode searchMode,
            @Nullable String denseVectorName,
            @Nullable String sparseVectorName,
            @Nullable QdrantSparseEmbeddingFunction sparseEmbeddingFunction,
            int rrfK,
            int prefetchLimit) {

        QdrantGrpcClient.Builder grpcClientBuilder = QdrantGrpcClient.newBuilder(host, port, useTls);

        if (apiKey != null) {
            grpcClientBuilder.withApiKey(apiKey);
        }

        this.client = new QdrantClient(grpcClientBuilder.build());
        this.collectionName = collectionName;
        this.payloadTextKey = payloadTextKey;
        this.searchMode = getOrDefault(searchMode, SearchMode.VECTOR);
        this.denseVectorName = getOrDefault(denseVectorName, "dense");
        this.sparseVectorName = getOrDefault(sparseVectorName, "sparse");
        this.sparseEmbeddingFunction = sparseEmbeddingFunction;
        this.rrfK = rrfK;
        this.prefetchLimit = prefetchLimit;
    }

    QdrantEmbeddingStore(
            QdrantClient client,
            String collectionName,
            String payloadTextKey,
            SearchMode searchMode,
            @Nullable String denseVectorName,
            @Nullable String sparseVectorName,
            @Nullable QdrantSparseEmbeddingFunction sparseEmbeddingFunction,
            int rrfK,
            int prefetchLimit) {
        this.client = client;
        this.collectionName = collectionName;
        this.payloadTextKey = payloadTextKey;
        this.searchMode = getOrDefault(searchMode, SearchMode.VECTOR);
        this.denseVectorName = getOrDefault(denseVectorName, "dense");
        this.sparseVectorName = getOrDefault(sparseVectorName, "sparse");
        this.sparseEmbeddingFunction = sparseEmbeddingFunction;
        this.rrfK = rrfK;
        this.prefetchLimit = prefetchLimit;
    }

    @Override
    public String add(Embedding embedding) {
        String id = randomUUID();
        add(id, embedding);
        return id;
    }

    @Override
    public void add(String id, Embedding embedding) {
        addInternal(id, embedding, null);
    }

    @Override
    public String add(Embedding embedding, TextSegment textSegment) {
        String id = randomUUID();
        addInternal(id, embedding, textSegment);
        return id;
    }

    @Override
    public List<String> addAll(List<Embedding> embeddings) {

        List<String> ids = embeddings.stream().map(ignored -> randomUUID()).toList();

        addAll(ids, embeddings, null);

        return ids;
    }

    private void addInternal(String id, Embedding embedding, TextSegment textSegment) {
        addAll(singletonList(id), singletonList(embedding), textSegment == null ? null : singletonList(textSegment));
    }

    @Override
    public void addAll(List<String> ids, List<Embedding> embeddings, List<TextSegment> textSegments)
            throws RuntimeException {
        if (isNullOrEmpty(ids) || isNullOrEmpty(embeddings)) {
            log.info("Empty embeddings - no ops");
            return;
        }
        try {
            List<PointStruct> points = new ArrayList<>(embeddings.size());

            for (int i = 0; i < embeddings.size(); i++) {

                String id = ids.get(i);
                UUID uuid = UUID.fromString(id);
                Embedding embedding = embeddings.get(i);

                PointStruct.Builder pointBuilder = PointStruct.newBuilder().setId(id(uuid));

                if (sparseEmbeddingFunction != null) {
                    pointBuilder.setVectors(buildNamedVectors(embedding, textSegments, i));
                } else {
                    pointBuilder.setVectors(vectors(embedding.vector()));
                }

                if (textSegments != null) {
                    Map<String, Object> metadata =
                            textSegments.get(i).metadata().toMap();

                    Map<String, Value> payload = ValueMapFactory.valueMap(metadata);
                    payload.put(payloadTextKey, value(textSegments.get(i).text()));
                    pointBuilder.putAllPayload(payload);
                }

                points.add(pointBuilder.build());
            }

            client.upsertAsync(collectionName, points).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private Points.Vectors buildNamedVectors(Embedding embedding, List<TextSegment> textSegments, int index) {
        Vector denseVector = Vector.newBuilder()
                .setDense(Points.DenseVector.newBuilder().addAllData(embedding.vectorAsList()))
                .build();

        String text = (textSegments != null) ? textSegments.get(index).text() : null;
        if (text == null || text.isBlank()) {
            log.warn("No text available for sparse embedding at index {}, using dense vector only", index);
            return namedVectors(Map.of(denseVectorName, denseVector));
        }

        QdrantSparseEmbedding sparseEmb = sparseEmbeddingFunction.embed(text);
        Vector sparseVector = Vector.newBuilder()
                .setSparse(SparseVector.newBuilder()
                        .addAllValues(sparseEmb.values())
                        .addAllIndices(sparseEmb.indices()))
                .build();

        return namedVectors(Map.of(denseVectorName, denseVector, sparseVectorName, sparseVector));
    }

    @Override
    public void remove(String id) {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("id cannot be null or blank");
        }
        removeAll(Collections.singleton(id));
    }

    @Override
    public void removeAll(Collection<String> ids) {
        ensureNotEmpty(ids, "ids");
        try {

            Points.PointsIdsList pointsIdsList = Points.PointsIdsList.newBuilder()
                    .addAllIds(ids.stream().map(id -> id(UUID.fromString(id))).toList())
                    .build();
            PointsSelector pointsSelector =
                    PointsSelector.newBuilder().setPoints(pointsIdsList).build();

            client.deleteAsync(DeletePoints.newBuilder()
                            .setCollectionName(collectionName)
                            .setPoints(pointsSelector)
                            .build())
                    .get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void removeAll(dev.langchain4j.store.embedding.filter.Filter filter) {
        ensureNotNull(filter, "filter");
        try {
            Filter qdrantFilter = QdrantFilterConverter.convertExpression(filter);
            PointsSelector pointsSelector =
                    PointsSelector.newBuilder().setFilter(qdrantFilter).build();

            client.deleteAsync(DeletePoints.newBuilder()
                            .setCollectionName(collectionName)
                            .setPoints(pointsSelector)
                            .build())
                    .get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void removeAll() {
        clearStore();
    }

    @Override
    public EmbeddingSearchResult<TextSegment> search(EmbeddingSearchRequest request) {
        return switch (searchMode) {
            case VECTOR -> vectorSearch(request);
            case HYBRID -> hybridSearch(request);
        };
    }

    private EmbeddingSearchResult<TextSegment> vectorSearch(EmbeddingSearchRequest request) {

        if (sparseEmbeddingFunction != null) {
            // Collection uses named vectors — use QueryPoints targeting the dense vector
            return namedVectorSearch(request);
        }

        SearchPoints.Builder searchBuilder = SearchPoints.newBuilder()
                .setCollectionName(collectionName)
                .addAllVector(request.queryEmbedding().vectorAsList())
                .setWithVectors(WithVectorsSelectorFactory.enable(true))
                .setWithPayload(enable(true))
                .setLimit(request.maxResults());

        if (request.filter() != null) {
            Filter filter = QdrantFilterConverter.convertExpression(request.filter());
            searchBuilder.setFilter(filter);
        }

        List<ScoredPoint> results;

        try {
            results = client.searchAsync(searchBuilder.build()).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }

        if (results.isEmpty()) {
            return new EmbeddingSearchResult<>(emptyList());
        }

        List<EmbeddingMatch<TextSegment>> matches = results.stream()
                .map(vector -> toEmbeddingMatch(vector, request.queryEmbedding()))
                .filter(match -> match.score() >= request.minScore())
                .sorted(comparingDouble(EmbeddingMatch::score))
                .collect(toList());

        Collections.reverse(matches);

        return new EmbeddingSearchResult<>(matches);
    }

    private EmbeddingSearchResult<TextSegment> namedVectorSearch(EmbeddingSearchRequest request) {
        QueryPoints.Builder queryBuilder = QueryPoints.newBuilder()
                .setCollectionName(collectionName)
                .setQuery(Query.newBuilder()
                        .setNearest(VectorInputFactory.vectorInput(request.queryEmbedding().vectorAsList())))
                .setUsing(denseVectorName)
                .setWithVectors(WithVectorsSelectorFactory.enable(true))
                .setWithPayload(enable(true))
                .setLimit(request.maxResults());

        if (request.filter() != null) {
            Filter filter = QdrantFilterConverter.convertExpression(request.filter());
            queryBuilder.setFilter(filter);
        }

        List<ScoredPoint> results;

        try {
            results = client.queryAsync(queryBuilder.build()).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }

        if (results.isEmpty()) {
            return new EmbeddingSearchResult<>(emptyList());
        }

        List<EmbeddingMatch<TextSegment>> matches = results.stream()
                .map(this::toNamedVectorEmbeddingMatch)
                .filter(match -> match.score() >= request.minScore())
                .sorted(comparingDouble(EmbeddingMatch::score))
                .collect(toList());

        Collections.reverse(matches);

        return new EmbeddingSearchResult<>(matches);
    }

    private EmbeddingSearchResult<TextSegment> hybridSearch(EmbeddingSearchRequest request) {
        String queryText = request.query();
        if (queryText == null || queryText.isBlank()) {
            throw new IllegalArgumentException(
                    "For HYBRID search mode, query() must be provided in EmbeddingSearchRequest");
        }

        Filter qdrantFilter = request.filter() != null
                ? QdrantFilterConverter.convertExpression(request.filter()) : null;

        // Dense prefetch — filter applied here so irrelevant docs are excluded before fusion
        PrefetchQuery.Builder densePrefetchBuilder = PrefetchQuery.newBuilder()
                .setQuery(Query.newBuilder()
                        .setNearest(VectorInputFactory.vectorInput(request.queryEmbedding().vectorAsList())))
                .setUsing(denseVectorName)
                .setLimit(prefetchLimit);
        if (qdrantFilter != null) {
            densePrefetchBuilder.setFilter(qdrantFilter);
        }

        // Sparse prefetch — same filter
        QdrantSparseEmbedding sparseEmb = sparseEmbeddingFunction.embed(queryText);
        PrefetchQuery.Builder sparsePrefetchBuilder = PrefetchQuery.newBuilder()
                .setQuery(Query.newBuilder()
                        .setNearest(VectorInputFactory.vectorInput(sparseEmb.values(), sparseEmb.indices())))
                .setUsing(sparseVectorName)
                .setLimit(prefetchLimit);
        if (qdrantFilter != null) {
            sparsePrefetchBuilder.setFilter(qdrantFilter);
        }

        // Fusion via server-side RRF
        QueryPoints.Builder queryBuilder = QueryPoints.newBuilder()
                .setCollectionName(collectionName)
                .addPrefetch(densePrefetchBuilder.build())
                .addPrefetch(sparsePrefetchBuilder.build())
                .setQuery(Query.newBuilder().setRrf(Rrf.newBuilder().setK(rrfK)))
                .setWithVectors(WithVectorsSelectorFactory.enable(true))
                .setWithPayload(enable(true))
                .setLimit(request.maxResults());

        if (qdrantFilter != null) {
            queryBuilder.setFilter(qdrantFilter);
        }

        List<ScoredPoint> results;

        try {
            results = client.queryAsync(queryBuilder.build()).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }

        if (results.isEmpty()) {
            return new EmbeddingSearchResult<>(emptyList());
        }

        List<EmbeddingMatch<TextSegment>> matches = results.stream()
                .map(this::toNamedVectorEmbeddingMatch)
                .filter(match -> match.score() >= request.minScore())
                .sorted(comparingDouble(EmbeddingMatch::score))
                .collect(toList());

        Collections.reverse(matches);

        return new EmbeddingSearchResult<>(matches);
    }

    /** Deletes all points from the Qdrant collection. */
    public void clearStore() {
        try {

            Filter emptyFilter = Filter.newBuilder().build();
            PointsSelector allPointsSelector =
                    PointsSelector.newBuilder().setFilter(emptyFilter).build();

            client.deleteAsync(DeletePoints.newBuilder()
                            .setCollectionName(collectionName)
                            .setPoints(allPointsSelector)
                            .build())
                    .get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    /** Closes the underlying GRPC client. */
    public void close() {
        client.close();
    }

    private EmbeddingMatch<TextSegment> toEmbeddingMatch(ScoredPoint scoredPoint, Embedding referenceEmbedding) {
        Map<String, Value> payload = scoredPoint.getPayloadMap();

        Value textSegmentValue = payload.getOrDefault(payloadTextKey, null);

        Map<String, Object> metadata = payload.entrySet().stream()
                .filter(entry -> !entry.getKey().equals(payloadTextKey))
                .collect(toMap(Map.Entry::getKey, entry -> ObjectFactory.object(entry.getValue())));

        Embedding embedding = toEmbedding(scoredPoint.getVectors().getVector());
        double cosineSimilarity = CosineSimilarity.between(embedding, referenceEmbedding);

        return new EmbeddingMatch<>(
                RelevanceScore.fromCosineSimilarity(cosineSimilarity),
                scoredPoint.getId().getUuid(),
                embedding,
                textSegmentValue == null
                        ? null
                        : TextSegment.from(textSegmentValue.getStringValue(), new Metadata(metadata)));
    }

    private EmbeddingMatch<TextSegment> toNamedVectorEmbeddingMatch(ScoredPoint scoredPoint) {
        Map<String, Value> payload = scoredPoint.getPayloadMap();

        Value textSegmentValue = payload.getOrDefault(payloadTextKey, null);

        Map<String, Object> metadata = payload.entrySet().stream()
                .filter(entry -> !entry.getKey().equals(payloadTextKey))
                .collect(toMap(Map.Entry::getKey, entry -> ObjectFactory.object(entry.getValue())));

        // Extract dense embedding from named vectors
        Embedding embedding = null;
        Points.VectorsOutput vectorsOutput = scoredPoint.getVectors();
        if (vectorsOutput.hasVectors()) {
            VectorOutput denseOutput = vectorsOutput.getVectors().getVectorsMap().get(denseVectorName);
            if (denseOutput != null) {
                embedding = toEmbedding(denseOutput);
            }
        } else if (vectorsOutput.hasVector()) {
            embedding = toEmbedding(vectorsOutput.getVector());
        }

        // Use Qdrant's server-computed score directly
        double score = scoredPoint.getScore();

        return new EmbeddingMatch<>(
                score,
                scoredPoint.getId().getUuid(),
                embedding,
                textSegmentValue == null
                        ? null
                        : TextSegment.from(textSegmentValue.getStringValue(), new Metadata(metadata)));
    }

    private static Embedding toEmbedding(Points.VectorOutput vectorOutput) {
        return Embedding.from(getOrDefault(vectorOutput.getDense().getDataList(), vectorOutput.getDataList()));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private String collectionName;
        private String host = "localhost";
        private int port = 6334;
        private boolean useTls = false;
        private String payloadTextKey = "text_segment";
        private String apiKey = null;
        private QdrantClient client = null;
        private SearchMode searchMode = SearchMode.VECTOR;
        private String denseVectorName = "dense";
        private String sparseVectorName = "sparse";
        private QdrantSparseEmbeddingFunction sparseEmbeddingFunction = null;
        private int rrfK = 60;
        private int prefetchLimit = 20;

        /**
         * @param host The host of the Qdrant instance. Defaults to "localhost".
         */
        public Builder host(String host) {
            this.host = host;
            return this;
        }

        /**
         * @param collectionName REQUIRED. The name of the collection.
         */
        public Builder collectionName(String collectionName) {
            this.collectionName = collectionName;
            return this;
        }

        /**
         * @param port The GRPC port of the Qdrant instance. Defaults to 6334.
         */
        public Builder port(int port) {
            this.port = port;
            return this;
        }

        /**
         * @param useTls Whether to use TLS(HTTPS). Defaults to false.
         */
        public Builder useTls(boolean useTls) {
            this.useTls = useTls;
            return this;
        }

        /**
         * @param payloadTextKey The field name of the text segment in the payload.
         *                       Defaults to "text_segment".
         */
        public Builder payloadTextKey(String payloadTextKey) {
            this.payloadTextKey = payloadTextKey;
            return this;
        }

        /**
         * @param apiKey The Qdrant API key to authenticate with. Defaults to null.
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * @param client A Qdrant client instance. Defaults to null.
         */
        public Builder client(QdrantClient client) {
            this.client = client;
            return this;
        }

        /**
         * @param searchMode The search mode. Defaults to {@link SearchMode#VECTOR}.
         *                   When set to {@link SearchMode#HYBRID}, the store combines dense and sparse
         *                   vector search using Qdrant's server-side Reciprocal Rank Fusion.
         *                   Requires {@link #sparseEmbeddingFunction} to be set.
         */
        public Builder searchMode(SearchMode searchMode) {
            this.searchMode = searchMode;
            return this;
        }

        /**
         * @param denseVectorName The name of the dense vector in the Qdrant collection.
         *                        Defaults to "dense". Used when {@link #sparseEmbeddingFunction} is set.
         */
        public Builder denseVectorName(String denseVectorName) {
            this.denseVectorName = denseVectorName;
            return this;
        }

        /**
         * @param sparseVectorName The name of the sparse vector in the Qdrant collection.
         *                         Defaults to "sparse". Used when {@link #sparseEmbeddingFunction} is set.
         */
        public Builder sparseVectorName(String sparseVectorName) {
            this.sparseVectorName = sparseVectorName;
            return this;
        }

        /**
         * @param sparseEmbeddingFunction A function that converts text to sparse embeddings.
         *                                Required when {@link SearchMode#HYBRID} is used.
         *                                When provided, the store always indexes both dense and sparse
         *                                vectors regardless of the search mode, allowing you to switch
         *                                between VECTOR and HYBRID search without re-ingesting data.
         */
        public Builder sparseEmbeddingFunction(QdrantSparseEmbeddingFunction sparseEmbeddingFunction) {
            this.sparseEmbeddingFunction = sparseEmbeddingFunction;
            return this;
        }

        /**
         * @param rrfK The k parameter for Reciprocal Rank Fusion. Defaults to 60.
         *             Higher values give more weight to highly-ranked results.
         */
        public Builder rrfK(int rrfK) {
            this.rrfK = rrfK;
            return this;
        }

        /**
         * @param prefetchLimit The maximum number of results returned by each prefetch query
         *                      before fusion. Defaults to 20.
         */
        public Builder prefetchLimit(int prefetchLimit) {
            this.prefetchLimit = prefetchLimit;
            return this;
        }

        public QdrantEmbeddingStore build() {
            Objects.requireNonNull(collectionName, "collectionName cannot be null");

            if (searchMode == SearchMode.HYBRID) {
                Objects.requireNonNull(sparseEmbeddingFunction,
                        "sparseEmbeddingFunction is required for HYBRID search mode");
            }

            if (client != null) {
                return new QdrantEmbeddingStore(client, collectionName, payloadTextKey,
                        searchMode, denseVectorName, sparseVectorName, sparseEmbeddingFunction,
                        rrfK, prefetchLimit);
            }
            return new QdrantEmbeddingStore(collectionName, host, port, useTls, payloadTextKey, apiKey,
                    searchMode, denseVectorName, sparseVectorName, sparseEmbeddingFunction,
                    rrfK, prefetchLimit);
        }
    }
}
