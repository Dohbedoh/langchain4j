package dev.langchain4j.store.embedding.qdrant;

/**
 * A function that converts text into a sparse embedding for use with Qdrant hybrid search.
 * Implementations may use SPLADE, BM25, or other sparse encoding strategies.
 */
@FunctionalInterface
public interface QdrantSparseEmbeddingFunction {

    /**
     * Generates a sparse embedding from the given text.
     *
     * @param text the text to embed
     * @return the sparse embedding with indices and values
     */
    QdrantSparseEmbedding embed(String text);
}
