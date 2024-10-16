# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)  

project(GenAI)

add_subdirectory(thirdparty/openvino_tokenizers/)  # Build openvino-tokenizers
if(NOT MSVC)  # # Disable the warning from user_ov_extensions
    target_compile_options(sentencepiece-static PRIVATE -Wno-stringop-overflow)
endif()

add_subdirectory(llm/chatglm_cpp)
target_compile_definitions(chatglm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(chatglm PRIVATE openvino_tokenizers)

add_subdirectory(llm/glm4_cpp)
target_compile_definitions(glm4 PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(glm4 PRIVATE openvino_tokenizers)

add_subdirectory(llm/llama_cpp)
target_compile_definitions(llama PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(llama PRIVATE openvino_tokenizers)

add_subdirectory(llm/minicpm_cpp)
target_compile_definitions(minicpm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(minicpm PRIVATE openvino_tokenizers)

add_subdirectory(llm/mistral_cpp)
target_compile_definitions(mistral PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(mistral PRIVATE openvino_tokenizers)

add_subdirectory(llm/qwen_cpp)
target_compile_definitions(qwen PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(qwen PRIVATE openvino_tokenizers)

add_subdirectory(llm/phi_cpp)
target_compile_definitions(phi PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(phi PRIVATE openvino_tokenizers)
