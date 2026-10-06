#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

int main(int argc, char ** argv) {
    // Dimensions are GGUF tensor axes: input width then output rows.
    const int64_t hidden = argc > 4 ? std::atoll(argv[4]) : 2688;
    const int64_t vocab = argc > 5 ? std::atoll(argv[5]) : 131072;
    constexpr size_t block_len = 32;
    constexpr size_t block_bytes = 34;
    const int samples = argc > 3 ? std::atoi(argv[3]) : 50;
    if (samples < 10 || samples > 10000) return 10;
    constexpr int warmups = 5;
    const int columns = argc > 1 ? std::atoi(argv[1]) : 1;
    const bool fp16_weights = argc > 2 && std::strcmp(argv[2], "f16") == 0;
    if (columns < 1 || columns > 9 || hidden < 32 || hidden % 32 || vocab < 1) {
        std::fprintf(stderr, "Usage: probe M[1..9] [q8|f16] [samples>=10] [input_width multiple of 32] [output_rows]\n");
        return 9;
    }

    if (ggml_blck_size(GGML_TYPE_Q8_0) != block_len ||
        ggml_type_size(GGML_TYPE_Q8_0) != block_bytes) {
        std::fprintf(stderr, "Unexpected Q8_0 block layout\n");
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server("mkvsock:1:5002");
    if (!reg) return 2;
    ggml_backend_register(reg);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("RPC0");
    if (!dev) return 3;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) return 4;

    ggml_init_params params = {
        ggml_tensor_overhead() * 6 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) return 5;
    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(),
                                                fp16_weights ? GGML_TYPE_F16 : GGML_TYPE_Q8_0,
                                                hidden, vocab);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, columns);
    ggml_tensor * result = ggml_mul_mat(ctx.get(), weights, input);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) return 6;

    const size_t blocks = (size_t) (hidden / block_len) * (size_t) vocab;
    std::vector<uint8_t> weight_data(ggml_nbytes(weights));
    if (fp16_weights) {
        for (size_t i = 0; i < weight_data.size(); i += 2) {
            weight_data[i] = 0x00;
            weight_data[i + 1] = 0x3c;
        }
    } else {
        for (size_t i = 0; i < blocks; ++i) {
            weight_data[i * block_bytes] = 0x00;
            weight_data[i * block_bytes + 1] = 0x3c; // IEEE half-precision 1.0
            std::memset(weight_data.data() + i * block_bytes + 2, 1, block_len);
        }
    }
    std::vector<float> x(hidden * columns, columns == 9 ? 0.0f : 1.0f);
    if (columns == 9) std::fill_n(x.begin(), hidden, 1.0f);
    const auto upload_start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(weights, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_synchronize(backend.get());
    const double upload_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - upload_start).count();
    weight_data.clear();
    weight_data.shrink_to_fit();

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, result);
    std::vector<double> times;
    times.reserve(samples);
    float first = 0.0f;
    for (int i = -warmups; i < samples; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 7;
        // A dependent read waits for P40 CUDA execution; queue synchronization
        // alone only waits for RPC submission and undercounts kernel time.
        ggml_backend_tensor_get(result, &first, 0, sizeof(first));
        ggml_backend_synchronize(backend.get());
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (i >= 0) times.push_back(elapsed_ms);
    }
    // Validate every output after timing; the timed path deliberately fetches
    // only one dependent scalar so it measures execution without a 512 KiB read.
    std::vector<float> full_result(vocab * columns);
    ggml_backend_tensor_get(result, full_result.data(), 0, full_result.size() * sizeof(float));
    ggml_backend_synchronize(backend.get());
    bool correct = true;
    double max_error = 0.0;
    for (int c = 0; c < columns; ++c) {
        for (int64_t row = 0; row < vocab; ++row) {
            const float expected = c == 0 || columns != 9 ? float(hidden) : 0.0f;
            const double error = std::fabs(double(full_result[c * vocab + row]) - expected);
            max_error = std::max(max_error, error);
            if (!std::isfinite(error) || error >= (expected == 0.0f ? 0.01 : 0.3)) correct = false;
        }
    }
    std::sort(times.begin(), times.end());
    double sum = 0.0;
    for (double value : times) sum += value;
    std::printf("shape=%lldx%lld input_columns=%d weight_type=%s weights_bytes=%zu upload_s=%.3f result=%.3f expected=%lld max_error=%.3f correct=%s\n",
                (long long) hidden, (long long) vocab, columns,
                fp16_weights ? "F16" : "Q8_0", ggml_nbytes(weights), upload_s,
                first, (long long) hidden, max_error, correct ? "yes" : "no");
    std::printf("compute+RPC ms: median=%.3f p95=%.3f mean=%.3f min=%.3f max=%.3f n=%zu\n",
                times[times.size()/2], times[times.size()*95/100], sum/times.size(),
                times.front(), times.back(), times.size());
    return correct ? 0 : 8;
}
