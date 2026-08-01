/*
  USE_MACA: Parse the configuration parameters of the PYTORCH_ENABLE_SAME_RAND_CONF environment variable

  (1) Example of how to use environment variables(use for H20):
      export PYTORCH_ENABLE_SAME_RAND_CONF= multiprocessor_count:78,maxthreads_per_multiprocessor:2048

  (2) Parameter Description
      multiprocessor_count: Number of SMs on the current chip architecture;
      maxthreads_per_multiprocessor: Maximum number of threads that can run on an SM in the current chip architecture

  (3) Note:Both parameters default to -1, the environment variables must be set for both of them together to take effect.
*/
#include <ATen/native/cuda/CudaRandConfig.h>

constexpr size_t multiProcessorNum = -1;
constexpr size_t maxThreads = -1;

CUDARandConfig::CUDARandConfig()
    : m_multi_processor_count(multiProcessorNum),
      m_maxthreads_per_multiprocessor(maxThreads),
      m_last_rand_settings("") {}

void CUDARandConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
    std::vector<char> buf;

    size_t env_length = strlen(env);
    for (size_t i = 0; i < env_length; i++) {
        if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
            if (!buf.empty()) {
                config.emplace_back(buf.begin(), buf.end());
                buf.clear();
            }
            config.emplace_back(1, env[i]);
        } else if (env[i] != ' ') {
            buf.emplace_back(static_cast<char>(env[i]));
        }
    }

    if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
    }
}

void CUDARandConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
    TORCH_CHECK(
        i < config.size() && config[i] == std::string(1, c),
        "Error parsing CUDARandConfig settings, expected ",
        c,
        "");
}

// Parse the sm count parameter passed in via the environment variable
size_t CUDARandConfig::parseMultiProcessorCount(
    const std::vector<std::string>& config,
    size_t i) {
    consumeToken(config, ++i, ':');

    if (++i < config.size()) {
        size_t val1 = stoi(config[i]);
        m_multi_processor_count = val1;
    } else {
        TORCH_CHECK(false, "Error, expecting m_multi_processor_count value", "");
    }

    return i;
}

// Parse the maximum thread per sm parameter passed in via the environment variable
size_t CUDARandConfig::parseMaxthreadsPerMultiProcessor(
    const std::vector<std::string>& config,
    size_t i) {
    consumeToken(config, ++i, ':');

    if (++i < config.size()) {
        size_t val1 = stoi(config[i]);
        m_maxthreads_per_multiprocessor = val1;
    } else {
        TORCH_CHECK(false, "Error, expecting m_multi_processor_count value", "");
    }

    return i;
}

void CUDARandConfig::parseArgs(const char* env) {
    if (env == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_last_rand_settings_mutex);
        m_last_rand_settings = env;
    }

    std::vector<std::string> config;
    lexArgs(env, config);

    for (size_t i = 0; i < config.size(); i++) {
        std::string_view config_item_view(config[i]);
        if (config_item_view == "multiprocessor_count") {
            i = parseMultiProcessorCount(config, i);
        } else if (config_item_view == "maxthreads_per_multiprocessor") {
            i = parseMaxthreadsPerMultiProcessor(config, i);
        } else {
            TORCH_CHECK(
                false, "Unrecognized CUDARandConfig option: ", config_item_view);
        }

        if (i + 1 < config.size()) {
            consumeToken(config, ++i, ',');
        }
    }
}