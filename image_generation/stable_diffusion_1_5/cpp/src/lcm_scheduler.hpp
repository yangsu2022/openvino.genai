#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>

// TODO: delete after randn_tensor
std::vector<float> read_vector_from_txt(std::string& file_name) {
    std::ifstream input_data(file_name, std::ifstream::in);
    std::istream_iterator<float> start(input_data), end;
    std::vector<float> res(start, end);
    return res;
}

// https://gist.github.com/lorenzoriano/5414671
template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N - 1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

// linspace and floor
std::vector<int64_t> get_inf_indices(float start, float end, uint32_t num, bool endpoint=false) {
    std::vector<int64_t> indices;
    if (num != 0) {
        if (num == 1)
            indices.push_back(static_cast<int64_t>(start));
        else {
            if (endpoint)
                --num;

            float delta = (end - start) / num;
            for(uint32_t i = 0; i < num; i++)
                indices.push_back(static_cast<int64_t>(start + delta * i));

            if (endpoint)
                indices.push_back(static_cast<int64_t>(end));
        }
    }
    return indices;
}

class LCMScheduler {
    public:
    // config
    int num_train_timesteps_config;
    int original_inference_steps_config;
    std::string prediction_type_config;
    float timestep_scaling_config;

    std::vector<int64_t> timesteps;

    int64_t init_noise_sigma = 1.0;

    // construct
    LCMScheduler(int num_train_timesteps = 1000,
                 float beta_start = 0.00085,
                 float beta_end = 0.012,
                 std::string beta_schedule = "scaled_linear",
                 std::vector<float> trained_betas = {},
                 int original_inference_steps = 50,
                 bool set_alpha_to_one = true,
                 int steps_offset = 0,
                 std::string prediction_type = "epsilon",
                 std::string timestep_spacing = "leading",
                 float timestep_scaling = 10,
                 bool rescale_betas_zero_snr = false):
                 original_inference_steps_config(original_inference_steps),
                 num_train_timesteps_config(num_train_timesteps),
                 prediction_type_config(prediction_type),
                 timestep_scaling_config(timestep_scaling) {
        
        std::string _predictionType = prediction_type;
        auto Derivatives = std::vector<std::vector<float>>{};
        auto Timesteps = std::vector<int>();

        auto alphas = std::vector<float>();
        auto betas = std::vector<float>();
        if (!trained_betas.empty()) {
            auto betas = trained_betas;
        } else if (beta_schedule == "linear") {
            for (int32_t i = 0; i < num_train_timesteps; i++) {
                betas.push_back(beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1));
            }
        } else if (beta_schedule == "scaled_linear") {
            float start = sqrt(beta_start);
            float end = sqrt(beta_end);
            std::vector<float> temp = linspace(start, end, num_train_timesteps);
            for (float b : temp) {
                betas.push_back(b * b);
            }
        } else {
            std::cout << " beta_schedule must be one of 'linear' or 'scaled_linear' " << std::endl;
        }
        for (float b : betas) {
            alphas.push_back(1 - b);
        }
        for (int32_t i = 1; i <= (int)alphas.size(); i++) {
            float alpha_cumprod =
                std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
            alphas_cumprod.push_back(alpha_cumprod);
        }

        final_alpha_cumprod = set_alpha_to_one ? 1 : alphas_cumprod[0];
    }  

    void set_timesteps(int64_t num_inference_steps,
                       int64_t original_inference_steps = 50,
                       float strength = 1) {
        // LCM Timesteps Setting
        int64_t k = num_train_timesteps_config / original_inference_steps;

        int64_t origin_timesteps_size = original_inference_steps * strength;
        std::vector<int64_t> lcm_origin_timesteps(origin_timesteps_size);
        std::iota(lcm_origin_timesteps.begin(), lcm_origin_timesteps.end(), 1);
        std::transform(lcm_origin_timesteps.begin(), lcm_origin_timesteps.end(), lcm_origin_timesteps.begin(), [&k](auto& x) {
            return x * k - 1;
        });

        int64_t skipping_step = origin_timesteps_size / num_inference_steps;
        assert(skipping_step >= 1 && "The combination of `original_steps x strength` is smaller than `num_inference_steps`");

        this->num_inference_steps = num_inference_steps;
        // LCM Inference Steps Schedule
        std::reverse(lcm_origin_timesteps.begin(),lcm_origin_timesteps.end());

        // v1. based on master branch: https://github.com/huggingface/diffusers/blame/2a7f43a73bda387385a47a15d7b6fe9be9c65eb2/src/diffusers/schedulers/scheduling_lcm.py#L387 
        std::vector<int64_t> inference_indices = get_inf_indices(0, origin_timesteps_size, num_inference_steps);
        for (int64_t i : inference_indices){
            timesteps.push_back(lcm_origin_timesteps[i]);
        }

        // // v2. based on diffusers==0.23.1 - remove after debug:
        // std::vector<float> temp;
        // for(int64_t i = 0; i < static_cast<int64_t>(lcm_origin_timesteps.size()); i+=skipping_step)
        //     temp.push_back(lcm_origin_timesteps[i]);
        // for(int64_t i = 0; i < num_inference_steps; i++)
        //     timesteps.push_back(temp[i]);

    }

    // Predict the sample from the previous timestep by reversing the SDE.
    std::tuple<std::vector<float>, std::vector<float>>
    step_func(const std::vector<float>& model_output,
                                    int64_t timestep, // timesteps[i]
                                    int64_t step_index, // i
                                    const std::vector<float>& sample) {

        // 1. get previous step value
        int64_t prev_step_index = step_index + 1;
        int64_t prev_timestep = prev_step_index < static_cast<int64_t>(timesteps.size()) ? timesteps[prev_step_index] : timestep;

        // 2. compute alphas, betas
        float alpha_prod_t = alphas_cumprod[timestep];
        float alpha_prod_t_prev = (prev_timestep >= 0) ? alphas_cumprod[prev_timestep] : final_alpha_cumprod;
        float alpha_prod_t_sqrt = std::sqrt(alpha_prod_t);
        float alpha_prod_t_prev_sqrt = std::sqrt(alpha_prod_t_prev);
        float beta_prod_t_sqrt = std::sqrt(1 - alpha_prod_t);
        float beta_prod_t_prev_sqrt = std::sqrt(1 - alpha_prod_t_prev);

        std::cout <<"beta_prod_t_sqrt:"<< beta_prod_t_sqrt << " " << beta_prod_t_prev_sqrt << std::endl;

        // 3. Get scalings for boundary conditions
        // get_scalings_for_boundary_condition_discrete(...)
        float scaled_timestep = timestep * timestep_scaling_config;
        float c_skip = std::pow(sigma_data, 2) / (std::pow(scaled_timestep, 2) + std::pow(sigma_data, 2));
        float c_out = scaled_timestep / std::sqrt((std::pow(scaled_timestep, 2) + std::pow(sigma_data, 2)));

        // 4. Compute the predicted original sample x_0 based on the model parameterization
        // "epsilon" by default
        std::vector<float> predicted_original_sample(sample.size());
        // beta_prod_t.sqrt() * model_output
        std::transform(model_output.begin(), model_output.end(), predicted_original_sample.begin(),
                       std::bind(std::multiplies<float>(), std::placeholders::_1, beta_prod_t_sqrt));

        // sample - beta_prod_t.sqrt() * model_output
        std::transform(sample.begin(), sample.end(), predicted_original_sample.begin(), 
                       predicted_original_sample.begin(), std::minus<float>());

        // predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        std::transform(predicted_original_sample.begin(), predicted_original_sample.end(), predicted_original_sample.begin(), 
                        std::bind(std::divides<float>(), std::placeholders::_1, alpha_prod_t_sqrt));

        // TODO: 5. Clip or threshold "predicted x_0" - False for Python sample by default

        // 6. Denoise model output using boundary conditions
        // c_out * predicted_original_sample
        std::transform(predicted_original_sample.begin(), predicted_original_sample.end(), predicted_original_sample.begin(),
                       std::bind(std::multiplies<float>(), std::placeholders::_1, c_out));
        std::vector<float> denoised(sample.size());
        // c_skip * sample
        std::transform(sample.begin(), sample.end(), denoised.begin(),
                       std::bind(std::multiplies<float>(), std::placeholders::_1, c_skip));
        // denoised = c_out * predicted_original_sample + c_skip * sample
        std::transform(predicted_original_sample.begin(), predicted_original_sample.end(), denoised.begin(),
                       denoised.begin(), std::plus<float>());
        
        // 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        // Noise is not used on the final timestep of the timestep schedule.
        // This also means that noise is not used for one-step sampling.
        std::vector<float> prev_sample = denoised;
        if (step_index != num_inference_steps - 1) {
            // TODO: std::vector<float> noise = randn_tensor(model_output.size());
            std::string noise_file = "../scripts/noise_" + std::to_string(step_index) + ".txt";
            // read from file - for debug
            std::vector<float> noise = read_vector_from_txt(noise_file);

            // beta_prod_t_prev.sqrt() * noise
            std::transform(noise.begin(), noise.end(), noise.begin(),
                           std::bind(std::multiplies<float>(), std::placeholders::_1, beta_prod_t_prev_sqrt));
            // alpha_prod_t_prev.sqrt() * denoised
            std::transform(prev_sample.begin(), prev_sample.end(), prev_sample.begin(),
                           std::bind(std::multiplies<float>(), std::placeholders::_1, alpha_prod_t_prev_sqrt));
            // prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
            std::transform(noise.begin(), noise.end(), prev_sample.begin(), prev_sample.begin(), std::plus<float>());
        }

        return std::make_tuple(prev_sample, denoised);
    }
    

private:
    std::vector<float> alphas_cumprod;
    float final_alpha_cumprod;
    float sigma_data = 0.5; // Default: 0.5
    int64_t num_inference_steps = 0;
};
