package dev.langchain4j.model.bedrock;

import java.util.HashMap;
import java.util.Map;

/**
 * Configuration for Claude's extended thinking on Bedrock.
 * <p>
 * Use {@link #adaptive()} for Claude 4.6+ models (model dynamically decides how much to think),
 * or {@link #enabled(int)} for Claude 3.7/4.5 models (fixed token budget).
 * <p>
 * This is placed under the {@code "thinking"} key in
 * {@link BedrockChatRequestParameters.Builder#additionalModelRequestFields}.
 *
 * @see BedrockChatRequestParameters.Builder#thinking(BedrockThinking)
 */
public class BedrockThinking {

    private final String type;
    private final Integer budgetTokens;

    private BedrockThinking(String type, Integer budgetTokens) {
        this.type = type;
        this.budgetTokens = budgetTokens;
    }

    /**
     * Adaptive thinking — the model dynamically decides how much to think.
     * Supported on Claude 4.6+ models.
     */
    public static BedrockThinking adaptive() {
        return new BedrockThinking("adaptive", null);
    }

    /**
     * Enabled thinking with a fixed token budget.
     * Supported on Claude 3.7, 4.5 models.
     *
     * @param budgetTokens the maximum number of tokens for thinking (must be positive)
     */
    public static BedrockThinking enabled(int budgetTokens) {
        if (budgetTokens <= 0) {
            throw new IllegalArgumentException("budgetTokens must be positive, got: " + budgetTokens);
        }
        return new BedrockThinking("enabled", budgetTokens);
    }

    Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("type", type);
        if (budgetTokens != null) {
            map.put("budget_tokens", budgetTokens);
        }
        return map;
    }
}
