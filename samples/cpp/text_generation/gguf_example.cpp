// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

#include "gguf_modeling.hpp"

#include "openvino/openvino.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <OUTPUT_DIR>");
    }

    std::string models_path = argv[1];  
    std::string output_path = argv[2];

    std::cout << "Loading model from: " << models_path << std::endl;
    std::cout << "Saving converted model to: " << output_path << std::endl;

    auto model = create_from_gguf(models_path);

    auto start_time = std::chrono::high_resolution_clock::now();
    
    ov::save_model(model, output_path + "/openvino_model.xml", false);

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
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