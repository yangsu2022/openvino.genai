// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// visual_language_chat with NPU pipeline properties (see inference-with-genai-on-npu docs).
// The language model runs on NPU; vision/text embedding models run on the AUTO device, set to GPU below.
// NPU needs an int4 *symmetric* LLM, e.g. optimum-cli ... --weight-format int4 --sym --ratio 1.0 --group-size -1

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include <filesystem>

ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

int main(int argc, char* argv[]) try {
    if (argc < 3 || argc > 4) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> [DEVICE]");
    }

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);
    std::string device = (argc >= 4) ? argv[3] : "NPU";

    ov::AnyMap properties = {
        {"DEVICE_PROPERTIES", ov::AnyMap{
            {"NPU", ov::AnyMap{
                // Must cover the longest prompt: PaddleOCR-VL-1.6 has <= 1280 image tokens per image (max_pixels / 28^2).
                {"MAX_PROMPT_LEN", 1536},
                // KV cache = MAX_PROMPT_LEN + MIN_RESPONSE_LEN; generation stops silently when it is full.
                {"MIN_RESPONSE_LEN", 512},
                {"GENERATE_HINT", "BEST_PERF"},
                // Prefill an A4 page (~1273 tokens) in one chunk instead of 2 x 1024.
                {"NPUW_LLM_PREFILL_CHUNK_SIZE", "2048"}
            }},
            // Vision/embedding models use AUTO; GenAI defaults it to CPU, which is ~10x slower here.
            {"AUTO", ov::AnyMap{{"MULTI_DEVICE_PRIORITIES", "GPU"}}}
        }},
        // NPU compilation takes minutes; cache it for the next run.
        ov::cache_dir("vlm_cache")
    };
    ov::genai::VLMPipeline pipe(argv[1], device, properties);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::string prompt;

    ov::genai::ChatHistory history;

    std::cout << "question:\n";
    std::getline(std::cin, prompt);

    history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
    ov::genai::VLMDecodedResults decoded_results = pipe.generate(
        history,
        ov::genai::images(rgbs),
        ov::genai::generation_config(generation_config),
        ov::genai::streamer(print_subword)
    );
    history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
    std::cout << "\n----------\n"
                 "question:\n";
    while (std::getline(std::cin, prompt)) {
        history.push_back({{"role", "user"}, {"content", std::move(prompt)}});
        // New images and videos can be passed at each turn
        ov::genai::VLMDecodedResults decoded_results = pipe.generate(
            history,
            ov::genai::generation_config(generation_config),
            ov::genai::streamer(print_subword)
        );
        history.push_back({{"role", "assistant"}, {"content", std::move(decoded_results.texts[0])}});
        std::cout << "\n----------\n"
                     "question:\n";
    }
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
