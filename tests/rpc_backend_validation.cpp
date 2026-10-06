#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static double elapsed_us(const std::chrono::steady_clock::time_point & start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char ** argv) {
    const char * endpoint = argc > 1 ? argv[1] : "mkvsock:1:5000";
    ggml_backend_load_all();

    ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoint);
    if (reg == nullptr) {
        std::fprintf(stderr, "RPC registration failed for %s\n", endpoint);
        return 1;
    }
    ggml_backend_register(reg);

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("RPC0");
    if (dev == nullptr) {
        std::fprintf(stderr, "RPC0 device was not registered\n");
        return 1;
    }
    size_t free_mem = 0;
    size_t total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    std::printf("device=%s description=%s free=%zu total=%zu\n",
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), free_mem, total_mem);

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) {
        std::fprintf(stderr, "RPC backend init failed\n");
        return 1;
    }

    const size_t sizes[] = { 1024, 8192, 65536, 1 << 20, 16 << 20 };
    for (size_t bytes : sizes) {
        const int64_t n = (int64_t) ((bytes + sizeof(float) - 1) / sizeof(float));
        ggml_init_params params = {
            ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false), nullptr, true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            std::fprintf(stderr, "ggml context allocation failed for %zu bytes\n", bytes);
            return 1;
        }
        ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
        ggml_tensor * output = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
        ggml_tensor * sum = ggml_add(ctx.get(), input, output);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!buffer) {
            std::fprintf(stderr, "remote allocation failed for %zu bytes\n", bytes);
            return 1;
        }

        std::vector<float> source((size_t) n);
        std::vector<float> result((size_t) n, 0.0f);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = (float) (i % 251) * 0.25f;
        }

        auto start = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(input, source.data(), 0, bytes);
        ggml_backend_tensor_set(output, source.data(), 0, bytes);
        ggml_backend_synchronize(backend.get());
        double write_us = elapsed_us(start);

        start = std::chrono::steady_clock::now();
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, sum);
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "remote graph compute failed for %zu bytes\n", bytes);
            return 1;
        }
        ggml_backend_synchronize(backend.get());
        double compute_us = elapsed_us(start);

        start = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(sum, result.data(), 0, bytes);
        ggml_backend_synchronize(backend.get());
        double read_us = elapsed_us(start);

        bool ok = true;
        for (size_t i = 0; i < source.size(); ++i) {
            if (result[i] != source[i] * 2.0f) {
                ok = false;
                break;
            }
        }
        std::printf("bytes=%zu write_us=%.1f compute_us=%.1f read_us=%.1f result=%s\n",
                    bytes, write_us, compute_us, read_us, ok ? "ok" : "FAIL");
        if (!ok) {
            return 1;
        }
    }
    std::printf("RPC validation passed\n");
    return 0;
}
