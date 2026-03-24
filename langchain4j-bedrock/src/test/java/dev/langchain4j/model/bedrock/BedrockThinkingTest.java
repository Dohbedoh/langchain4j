package dev.langchain4j.model.bedrock;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Map;
import org.junit.jupiter.api.Test;

class BedrockThinkingTest {

    @Test
    void adaptive_should_produce_correct_map() {
        // when
        BedrockThinking thinking = BedrockThinking.adaptive();

        // then
        Map<String, Object> map = thinking.toMap();
        assertThat(map).containsEntry("type", "adaptive").hasSize(1);
    }

    @Test
    void enabled_should_produce_correct_map() {
        // when
        BedrockThinking thinking = BedrockThinking.enabled(1024);

        // then
        Map<String, Object> map = thinking.toMap();
        assertThat(map)
                .containsEntry("type", "enabled")
                .containsEntry("budget_tokens", 1024)
                .hasSize(2);
    }

    @Test
    void enabled_should_reject_zero_budget() {
        assertThatThrownBy(() -> BedrockThinking.enabled(0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("budgetTokens must be positive");
    }

    @Test
    void enabled_should_reject_negative_budget() {
        assertThatThrownBy(() -> BedrockThinking.enabled(-1))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("budgetTokens must be positive");
    }
}
