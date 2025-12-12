// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cxxopts.hpp>
#include <filesystem>

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>
#include "../text_generation/read_prompt_from_file.h"

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vlm", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value(""))
    ("pf,prompt_file", "Read prompt from file", cxxopts::value<std::string>())
    ("i,image", "Image", cxxopts::value<std::string>()->default_value("image.jpg"))
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("mt,max_new_tokens", "Maximal number of new tokens", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
    ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::string prompt;
    if (result.count("prompt") && result.count("prompt_file")) {
        std::cout << "Prompt and prompt file should not exist together!" << std::endl;
        return EXIT_FAILURE;
    } else {
        if (result.count("prompt_file")) {
            prompt = utils::read_prompt(result["prompt_file"].as<std::string>());
        } else {
            prompt = result["prompt"].as<std::string>().empty() ? "What is on the image?" : result["prompt"].as<std::string>();
        }
    }
    if (prompt.empty()) {
        std::cout << "Prompt is empty!" << std::endl;
        return EXIT_FAILURE;
    } 

    const std::string models_path = result["model"].as<std::string>();
    const std::string image_path = result["image"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    std::vector<ov::Tensor> images = utils::load_images(image_path);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();
    config.ignore_eos = true;
    // 默认启用 prompt lookup decoding
    config.num_assistant_tokens = 5;
    config.max_ngram_size = 3;

    std::cout << ov::get_openvino_version() << std::endl;

    std::unique_ptr<ov::genai::VLMPipeline> pipe;
    if (device == "NPU")
        pipe = std::make_unique<ov::genai::VLMPipeline>(models_path, device);
    else {
        // Setting of Scheduler config will trigger usage of ContinuousBatching pipeline, which is not default for Qwen2VL, Qwen2.5VL, Gemma3 due to accuracy issues.
        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.enable_prefix_caching = false;
        scheduler_config.max_num_batched_tokens = std::numeric_limits<std::size_t>::max();
        pipe = std::make_unique<ov::genai::VLMPipeline>(models_path, device, 
            ov::genai::scheduler_config(scheduler_config), 
            ov::genai::prompt_lookup(true));
    }

    auto input_data = pipe->get_tokenizer().encode(prompt);
    size_t prompt_token_size = input_data.input_ids.get_shape()[1];
    std::cout << "Number of images:" << images.size() << ", prompt token size:" << prompt_token_size << std::endl;

    // Warmup (LLM 模式)
    for (size_t i = 0; i < num_warmup; i++) {
        std::cout << "== warmup: " << i << std::endl;
        pipe->generate(prompt, ov::genai::generation_config(config), ov::genai::streamer(print_subword));
    }

    // 主循环 (LLM 模式)
    ov::genai::DecodedResults res = pipe->generate(prompt, ov::genai::generation_config(config));
    auto metrics = res.perf_metrics;
    for (size_t i = 0; i < num_iter - 1; i++) {
        std::cout << "\n== iter:" << i << std::endl;
        res = pipe->generate(prompt, ov::genai::generation_config(config), ov::genai::streamer(print_subword));
        metrics = metrics + res.perf_metrics;
    }


    return 0;
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
