#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    constexpr size_t bytes = 374341632; // exact Q8_0 output.weight allocation
    constexpr int warmups = 5;
    constexpr int samples = 50;

    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server("mkvsock:1:5002");
    if (!reg) return 1;
    ggml_backend_register(reg);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("RPC0");
    if (!dev) return 2;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) return 3;

    ggml_init_params params = {
        ggml_tensor_overhead() * 6 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) return 4;
    ggml_tensor * source = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, bytes);
    ggml_tensor * destination = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, bytes);
    ggml_tensor * copy = ggml_cpy(ctx.get(), source, destination);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) return 5;

    std::vector<uint8_t> host(bytes);
    for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<uint8_t>(i % 251);
    ggml_backend_tensor_set(source, host.data(), 0, bytes);
    ggml_backend_synchronize(backend.get());
    host.clear();
    host.shrink_to_fit();

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, copy);
    std::vector<double> times;
    times.reserve(samples);
    uint8_t observed = 0;
    for (int i = -warmups; i < samples; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 6;
        // Reading the final byte makes each sample wait for the full GPU copy.
        ggml_backend_tensor_get(destination, &observed, bytes - 1, 1);
        ggml_backend_synchronize(backend.get());
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (i >= 0) times.push_back(elapsed_ms);
    }
    std::vector<uint8_t> full_result(bytes);
    ggml_backend_tensor_get(destination, full_result.data(), 0, bytes);
    ggml_backend_synchronize(backend.get());
    bool correct = observed == static_cast<uint8_t>((bytes - 1) % 251);
    for (size_t i = 0; i < bytes; ++i) {
        if (full_result[i] != static_cast<uint8_t>(i % 251)) {
            correct = false;
            break;
        }
    }
    std::sort(times.begin(), times.end());
    double sum = 0.0;
    for (double value : times) sum += value;
    const double median = times[times.size() / 2];
    std::printf("bytes=%zu observed=%u expected=%u correct=%s\n", bytes, unsigned(observed),
                unsigned((bytes - 1) % 251), correct ? "yes" : "no");
    std::printf("D2D copy+RPC ms: median=%.3f p95=%.3f mean=%.3f min=%.3f max=%.3f n=%zu\n",
                median, times[times.size() * 95 / 100], sum / times.size(),
                times.front(), times.back(), times.size());
    std::printf("effective aggregate read+write bandwidth=%.3f GiB/s (includes RPC overhead)\n",
                (2.0 * bytes / (1024.0 * 1024.0 * 1024.0)) / (median / 1000.0));
    return correct ? 0 : 7;
}
