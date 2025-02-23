// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include <nlohmann/json.hpp>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/perf_metrics.hpp"

#include "llm_pipeline_static.hpp"
#include "llm_pipeline_stateful.hpp"
#include "continuous_batching_adapter.hpp"
#include "speculative_decoding/speculative_decoding_impl.hpp"

namespace ov {
namespace genai {

namespace {

/*
* NPU reads some properties from the config file, but when LLMPipeline is initialized
* from the model_str and weights_tensor, there are no files.
* In the later case ModelDesc is stored in properties.
* This function pops ModelDescr from the the properties and returns a pair of updated properties and ModelDescr.
*/
std::pair<ov::AnyMap, ov::genai::static_llm::ModelConfigDesc> split_model_descr(const ov::AnyMap& properties) {
    ov::AnyMap main_properties = properties;
    ov::genai::static_llm::ModelConfigDesc model_descr;

    auto pop_property = [](ov::AnyMap& orig_propertis, const std::string& key, auto& value) {
        if (orig_propertis.find(key) != orig_propertis.end()) {
            value = orig_propertis.at(key).as<std::decay_t<decltype(value)>>();
            orig_propertis.erase(key);
        }
    };
    pop_property(main_properties, "name_or_path", model_descr.name_or_path);
    pop_property(main_properties, "type", model_descr.type);
    pop_property(main_properties, "num_key_value_heads", model_descr.num_key_value_heads);

    return {main_properties, model_descr};
}

} // namespace


std::pair<std::string, Any> streamer(StreamerVariant func) {
    if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&func)) {
        return {utils::STREAMER_ARG_NAME, Any::make<std::shared_ptr<StreamerBase>>(*streamer_obj)};
    } else  {
        auto callback = std::get<std::function<bool(std::string)>>(func);
        return {utils::STREAMER_ARG_NAME, Any::make<std::function<bool(std::string)>>(callback)};
    }
}

std::pair<std::string, Any> generation_config(const GenerationConfig& config) {
    return {utils::CONFIG_ARG_NAME, Any::make<GenerationConfig>(config)};
}

std::pair<std::string, Any> draft_model(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties) {
    auto [plugin_config, scheduler_config] = utils::split_scheduler_config(properties);

    std::filesystem::path openvino_model_name = "openvino_model.xml";
    auto model = utils::singleton_core().read_model(models_path / openvino_model_name, {}, plugin_config);
    auto generation_config = utils::from_config_json_if_exists(models_path);
    auto tokenizer = ov::genai::Tokenizer(models_path);
    return { utils::DRAFT_MODEL_ARG_NAME, Any::make<ModelDesc>(model, tokenizer, device, plugin_config, scheduler_config, generation_config) };
}

std::pair<std::string, Any> draft_model(
    std::string& model_str,
    ov::Tensor& weights_tensor,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config) {
    auto [plugin_config, scheduler_config] = utils::split_scheduler_config(properties);

    auto model = utils::singleton_core().read_model(model_str, weights_tensor);
    return { utils::DRAFT_MODEL_ARG_NAME, Any::make<ModelDesc>(model, tokenizer, device, plugin_config, scheduler_config, generation_config) };
}

// Public LLMPipeline

ov::genai::LLMPipeline::LLMPipeline(
    const ov::InferRequest& request,
    const ov::genai::Tokenizer& tokenizer,
    OptionalGenerationConfig generation_config) {
    auto start_time = std::chrono::steady_clock::now();
    m_pimpl = std::make_unique<StatefulLLMPipeline>(request, tokenizer, generation_config);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();
    if (properties.find(ov::genai::scheduler_config.name()) != properties.end() ||
        properties.find(utils::DRAFT_MODEL_ARG_NAME) != properties.end() ||
        properties.find(ov::genai::prompt_lookup.name()) != properties.end()) {
        auto [plugin_config, scheduler_config] = utils::split_scheduler_config(properties);
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, tokenizer, scheduler_config, device, plugin_config);
    } else if (device == "NPU") {
        m_pimpl = static_llm::LLMPipelineFactory::create(models_path, tokenizer, device, properties);
    } else {
        m_pimpl = std::make_unique<StatefulLLMPipeline>(models_path, tokenizer, device, properties);
    }
    m_pimpl->save_load_time(start_time);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();

    if (properties.find(ov::genai::scheduler_config.name()) != properties.end() ||
        properties.find(utils::DRAFT_MODEL_ARG_NAME) != properties.end() ||
        properties.find(ov::genai::prompt_lookup.name()) != properties.end()) {
        auto [device_properties, scheduler_config] = utils::split_scheduler_config(properties);
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(models_path, scheduler_config, device, device_properties);
    } else if (device == "NPU") {
        m_pimpl = static_llm::LLMPipelineFactory::create(models_path, device, properties);
    } else {
        m_pimpl = std::make_unique<StatefulLLMPipeline>(models_path, device, properties);
    }

    m_pimpl->save_load_time(start_time);
}

ov::genai::LLMPipeline::LLMPipeline(
    const std::string& model_str,
    const ov::Tensor& weights_tensor,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config) {
    auto start_time = std::chrono::steady_clock::now();

    if (properties.find(ov::genai::scheduler_config.name()) != properties.end() ||
        properties.find(utils::DRAFT_MODEL_ARG_NAME) != properties.end() ||
        properties.find(ov::genai::prompt_lookup.name()) != properties.end()){

        auto [device_properties, scheduler_config] = utils::split_scheduler_config(properties);
        m_pimpl = std::make_unique<ContinuousBatchingAdapter>(model_str, weights_tensor,
                                                              tokenizer, scheduler_config, device, device_properties, generation_config);
    } else if (device == "NPU") {
        // TODO: CVS-158771 Currently, it's a workaround. Probably there is a better solution.
        // NPU reads some properties from the config file, but when LLMPipeline is initialized
        // from the model_str and weights_tensor, there is no files.
        // Therefore, we need to pass these properties manually.
        // This is necessary only for NPU, for other plugins can be ommited.
        // Example of usage:
        // ov::AnyMap model_descr_properties = {{"name_or_path", "meta-llama/Llama-2-7b-chat-hf"},
        //                                      {"type", "llama"},
        //                                      {"num_key_value_heads", 32}};
        // ov::genai::LLMPipeline pipe(model_str,..., model_descr_properties);
        // This will convert from AnyMap to ModelDesc.
        auto [filtered_properties, model_descr] = split_model_descr(properties);

        m_pimpl = static_llm::LLMPipelineFactory::create(
            utils::singleton_core().read_model(model_str, weights_tensor),
            model_descr,
            tokenizer,
            device,
            filtered_properties,
            generation_config
        );
    } else {
        m_pimpl = std::make_unique<StatefulLLMPipeline>(
            model_str, // utils::singleton_core().read_model(model_str, weights_tensor),
            tokenizer,
            device,
            properties,
            generation_config);
    }

    m_pimpl->save_load_time(start_time);
}

DecodedResults LLMPipeline::generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer) {
    return m_pimpl->generate(inputs, generation_config, streamer);
}

DecodedResults LLMPipeline::generate(StringInputs text, const ov::AnyMap& config_map) {
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_pimpl->generate(text, config, utils::get_streamer_from_map(config_map));
}

EncodedResults LLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
    return m_pimpl->generate(inputs, generation_config, streamer);
}

EncodedResults LLMPipeline::generate(const EncodedInputs& inputs, const ov::AnyMap& config_map) {
    auto config_arg = utils::get_config_from_map(config_map);
    GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    return m_pimpl->generate(inputs, config, utils::get_streamer_from_map(config_map));
}

ov::genai::GenerationConfig ov::genai::LLMPipeline::get_generation_config() const {
    return m_pimpl->get_generation_config();
}

ov::genai::Tokenizer ov::genai::LLMPipeline::get_tokenizer() {
    return m_pimpl->get_tokenizer();
}

void ov::genai::LLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void ov::genai::LLMPipeline::finish_chat() {
    m_pimpl->finish_chat();
}

void ov::genai::LLMPipeline::set_generation_config(const GenerationConfig& config) {
    m_pimpl->set_generation_config(config);
}

ov::genai::LLMPipeline::~LLMPipeline() = default;

} // namespace genai
} // namespace ov
