// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "llm_pipeline_base.hpp"
#include "sampler.hpp"
#include "utils.hpp"

namespace ov::genai {

class StatefulLLMPipeline final : public LLMPipelineImplBase {
    ov::InferRequest m_model_runner;
    Sampler m_sampler;

    // Chat scenario specific parameters
    bool is_chat_conversation = false;
    bool m_trust_encoded_history = true;
    ChatHistory m_history;
    std::string m_templated_chat_history = {};
    std::vector<int64_t> m_tokenized_chat_history;
    ov::genai::utils::GenerationChatInputsType m_chat_input_type = ov::genai::utils::GenerationChatInputsType::UNDEF;
    // Tail of previous output in chat mode is missing in KV cache, let's keep it
    std::optional<int64_t> m_last_disappeared_token = std::nullopt;
    // If sequence contains some symbols, which could be ambiguously encoded by tokenizer, we need to trim kv cache
    // If we use beam search sampling with chat mode we need to remove last answer of the model from kv cache and add best answer to history 
    // so, let's keep info about amount of tokens to trim from kv cache and amount of tokens to keep in history
    ov::genai::utils::HistoryRemoveManager m_kv_history_manager = {0, 0};
    size_t m_kv_cache_seq_length_axis = 2;

    void reset_kv_state();
public:

    StatefulLLMPipeline(
        const ov::InferRequest& request,
        const ov::genai::Tokenizer& tokenizer,
        OptionalGenerationConfig generation_config = std::nullopt
    );

    StatefulLLMPipeline(
        const std::filesystem::path& models_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& plugin_config
    );

    // StatefulLLMPipeline(
    //     const std::shared_ptr<ov::Model>& model,
    //     const ov::genai::Tokenizer& tokenizer,
    //     const std::string& device,
    //     const ov::AnyMap& config,
    //     const ov::genai::GenerationConfig& generation_config
    // );

    StatefulLLMPipeline(
        const std::filesystem::path& models_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& config,
        const ov::genai::GenerationConfig& generation_config
    );

    StatefulLLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& plugin_config
    );

    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;
};

} // namespace ov::genai
