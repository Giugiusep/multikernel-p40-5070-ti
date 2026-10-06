#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); return 1; } } while (0)

__global__ void read_only(const uint4 * __restrict__ data, uint32_t * __restrict__ sums, size_t words) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    uint32_t sum = 0;
    for (size_t i = tid; i < words; i += stride) {
        const uint4 v = data[i];
        sum += v.x + v.y + v.z + v.w;
    }
    __shared__ uint32_t partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] = partial[0];
}

int main(int argc, char ** argv) {
    constexpr size_t bytes = 512ULL * 1024 * 1024;
    constexpr int threads = 256;
    constexpr int warmups = 5;
    constexpr int samples = 50;
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 512;
    if (blocks < 32 || blocks > 4096) return 2;
    uint4 * input = nullptr;
    uint32_t * output = nullptr;
    CHECK(cudaMalloc(&input, bytes));
    CHECK(cudaMalloc(&output, blocks * sizeof(uint32_t)));
    CHECK(cudaMemset(input, 1, bytes));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int i = -warmups; i < samples; ++i) {
        CHECK(cudaEventRecord(start));
        read_only<<<blocks, threads>>>(input, output, bytes / sizeof(uint4));
        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (i >= 0) times.push_back(ms);
    }
    std::vector<uint32_t> sums(blocks);
    CHECK(cudaMemcpy(sums.data(), output, blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t observed = 0;
    for (uint32_t value : sums) observed += value;
    const uint32_t expected = static_cast<uint32_t>((bytes / sizeof(uint32_t)) * uint64_t{0x01010101});
    std::sort(times.begin(), times.end());
    const float median = times[times.size() / 2];
    const float p95 = times[times.size() * 95 / 100];
    std::printf("bytes=%zu blocks=%d threads=%d expected=%u observed=%u correct=%s\n", bytes, blocks, threads, expected, observed, expected == observed ? "yes" : "no");
    std::printf("read-only median=%.3f ms p95=%.3f ms bandwidth=%.3f GiB/s\n", median, p95, (bytes / double{1ULL << 30}) / (median / 1000.0));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(input));
    CHECK(cudaFree(output));
    return expected == observed ? 0 : 3;
}
