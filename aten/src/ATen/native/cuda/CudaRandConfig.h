#pragma once
/*
  USE_MACA: Parse the configuration parameters of the PYTORCH_ENABLE_SAME_RAND_CONF environment variable

  (1) Example of how to use environment variables(use for H20):
      export PYTORCH_ENABLE_SAME_RAND_CONF= multiprocessor_count:78,maxthreads_per_multiprocessor:2048

  (2) Parameter Description
      multiprocessor_count: Number of SMs on the current chip architecture;
      maxthreads_per_multiprocessor: Maximum number of threads that can run on an SM in the current chip architecture

  (3) Note:Both parameters default to -1, the environment variables must be set for both of them together to take effect.
*/
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include <cstring>

// Environment config parser
class C10_CUDA_API CUDARandConfig {
public:
    static size_t multi_processor_count() {
        return instance().m_multi_processor_count;
    }

    static size_t maxthreads_per_multiprocessor() {
        return instance().m_maxthreads_per_multiprocessor;
    }

    static std::string last_rand_settings() {
        std::lock_guard<std::mutex> lock(
            instance().m_last_rand_settings_mutex);
        return instance().m_last_rand_settings;
    }

    static CUDARandConfig& instance() {
        static CUDARandConfig* s_instance = ([]() {
            auto inst = new CUDARandConfig();
            const char* env = getenv("PYTORCH_ENABLE_SAME_RAND_CONF");
            inst->parseArgs(env);
            return inst;
        })();
        return *s_instance;
    }

    void parseArgs(const char* env);

private:
    CUDARandConfig();

    static void lexArgs(const char* env, std::vector<std::string>& config);
    static void consumeToken(
        const std::vector<std::string>& config,
        size_t i,
        const char c);
    size_t parseMultiProcessorCount(
        const std::vector<std::string>& config,
        size_t i);

    size_t parseMaxthreadsPerMultiProcessor(
        const std::vector<std::string>& config,
        size_t i);

    std::atomic<size_t> m_multi_processor_count;
    std::atomic<size_t> m_maxthreads_per_multiprocessor;
    std::string m_last_rand_settings;
    std::mutex m_last_rand_settings_mutex;
};