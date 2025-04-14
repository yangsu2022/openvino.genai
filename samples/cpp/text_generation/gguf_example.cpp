// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

#include "gguf_modeling.hpp"

#include "openvino/openvino.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <TOKENIZER_DIR> <OUTPUT_DIR>");
    }

    std::string models_path = argv[1];  
    std::string tokenizer_path = argv[2];
    std::string output_path = argv[3];

    std::cout << "Loading model from: " << models_path << std::endl;
    std::cout << "Loading tokenizer model from: " << tokenizer_path << std::endl;
    std::cout << "Saving converted model to: " << output_path << std::endl;

    auto model = create_from_gguf(models_path);

    std::cout << "Finished create_from_gguf" << std::endl;
    
    ov::Core core;

    ov::genai::Tokenizer tokenizer(tokenizer_path);

    std::cout << "Starting model compiling " << std::endl;
    ov::CompiledModel compiled_model = core.compile_model(model, "GPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    ov::genai::GenerationConfig config;
    std::set<int64_t> single_zero_set = {0};
    config.eos_token_id = 0;   
    config.stop_token_ids = single_zero_set;
    config.max_new_tokens = 100;

    std::cout << "Starting model inferening " << std::endl;
    ov::genai::LLMPipeline pipe(infer_request, tokenizer);

    // ov::genai::GenerationConfig config;
    std::string result = pipe.generate("What is OpenVINO?", ov::genai::generation_config(config));
    std::cout << result << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    ov::save_model(model, output_path + "/openvino_model.xml", false);

    auto duration = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    std::cout << "save_model done. Time: " << duration << "s" << std::endl;

    std::cout << "Model successfully saved to: " << output_path << "/openvino_model.xml" << std::endl;

} catch (const std::exception& error) {
    try {
        std::cerr << "Error: " << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}