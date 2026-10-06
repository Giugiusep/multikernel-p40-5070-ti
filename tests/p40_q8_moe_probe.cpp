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
#include <vector>

// Reproduces GGML_OP_MUL_MAT_ID's six selected experts with one input token
// per verification row. "same" reuses six experts across rows; "disjoint"
// selects six new experts per row. Both store only selected experts rather than
// all 128, so the full model's residency/cache behavior is not covered.
int main(int argc, char ** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 1;
    const int64_t width = argc > 2 ? std::atoll(argv[2]) : 2688;
    const int64_t rows = argc > 3 ? std::atoll(argv[3]) : 1856;
    const int samples = argc > 4 ? std::atoi(argv[4]) : 50;
    const bool disjoint = argc > 5 && std::strcmp(argv[5], "disjoint") == 0;
    const int experts = disjoint ? 6 * m : 6;
    if (m < 1 || m > 8 || width < 32 || width % 32 || rows < 1 || samples < 10) return 9;

    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_rpc_add_server("mkvsock:1:5002");
    if (!reg) return 2;
    ggml_backend_register(reg);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("RPC0");
    if (!dev) return 3;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) return 4;

    ggml_init_params params = {ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false), nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) return 5;
    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q8_0, width, rows, experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, 1, m);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 6, m);
    ggml_tensor * result = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) return 6;

    std::vector<uint8_t> q8(ggml_nbytes(weights));
    for (size_t i = 0; i < q8.size() / 34; ++i) {
        q8[i * 34] = 0;
        q8[i * 34 + 1] = 0x3c;
        std::memset(q8.data() + i * 34 + 2, 1, 32);
    }
    std::vector<float> x(width * m, 1.0f);
    std::vector<int32_t> expert_ids(6 * m);
    for (int col = 0; col < m; ++col) {
        for (int expert = 0; expert < 6; ++expert) expert_ids[col * 6 + expert] = disjoint ? col * 6 + expert : expert;
    }
    const auto upload_start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(weights, q8.data(), 0, q8.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(ids, expert_ids.data(), 0, expert_ids.size() * sizeof(int32_t));
    ggml_backend_synchronize(backend.get());
    const double upload_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_start).count();
    q8.clear();
    q8.shrink_to_fit();

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, result);
    std::vector<double> times;
    float first = 0.0f;
    for (int i = -5; i < samples; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 7;
        ggml_backend_tensor_get(result, &first, 0, sizeof(first));
        ggml_backend_synchronize(backend.get());
        if (i >= 0) times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::vector<float> full(rows * 6 * m);
    ggml_backend_tensor_get(result, full.data(), 0, full.size() * sizeof(float));
    ggml_backend_synchronize(backend.get());
    double max_error = 0.0;
    bool correct = true;
    for (float value : full) {
        const double error = std::fabs(double(value) - double(width));
        max_error = std::max(max_error, error);
        if (!std::isfinite(error) || error > 0.5) correct = false;
    }
    std::sort(times.begin(), times.end());
    double sum = 0.0;
    for (double value : times) sum += value;
    std::printf("shape=%lldx%lldx%d input_columns=%d routing=%s weights_bytes=%zu upload_s=%.3f result=%.3f expected=%lld max_error=%.3f correct=%s\n",
                (long long) width, (long long) rows, experts, m, disjoint ? "disjoint" : "same", ggml_nbytes(weights), upload_s,
                first, (long long) width, max_error, correct ? "yes" : "no");
    std::printf("compute+RPC ms: median=%.3f p95=%.3f mean=%.3f n=%zu\n",
                times[times.size() / 2], times[times.size() * 95 / 100], sum / times.size(), times.size());
    return correct ? 0 : 8;
}
