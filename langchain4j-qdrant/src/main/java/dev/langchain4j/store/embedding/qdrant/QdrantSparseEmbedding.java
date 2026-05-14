package dev.langchain4j.store.embedding.qdrant;

import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

import java.util.List;

/**
 * Represents a sparse vector embedding for use with Qdrant hybrid search.
 * Sparse vectors consist of indices and their corresponding non-zero values,
 * typically produced by models like SPLADE or BM25.
 */
public class QdrantSparseEmbedding {

    private final List<Float> values;
    private final List<Integer> indices;

    public QdrantSparseEmbedding(List<Float> values, List<Integer> indices) {
        this.values = ensureNotNull(values, "values");
        this.indices = ensureNotNull(indices, "indices");
        if (values.size() != indices.size()) {
            throw new IllegalArgumentException(
                    "values and indices must have the same size, got " + values.size() + " and " + indices.size());
        }
        for (int i = 0; i < indices.size(); i++) {
            if (indices.get(i) < 0) {
                throw new IllegalArgumentException(
                        "indices must not contain negative values, found " + indices.get(i) + " at position " + i);
            }
        }
    }

    public static QdrantSparseEmbedding of(List<Float> values, List<Integer> indices) {
        return new QdrantSparseEmbedding(values, indices);
    }

    public List<Float> values() {
        return values;
    }

    public List<Integer> indices() {
        return indices;
    }
}
