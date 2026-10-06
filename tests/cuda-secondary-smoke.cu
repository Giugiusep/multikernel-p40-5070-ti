#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_OK(call)                                                         \
    do {                                                                      \
        cudaError_t status_ = (call);                                         \
        if (status_ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__,        \
                         __LINE__, #call, cudaGetErrorString(status_));        \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void vector_add(const int *a, const int *b, int *out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}

static int verify(const int *a, const int *b, const int *out, size_t n,
                  const char *phase) {
    for (size_t i = 0; i < n; ++i) {
        int expected = a[i] + b[i];
        if (out[i] != expected) {
            std::fprintf(stderr,
                         "%s mismatch at %zu: got %d, expected %d\n",
                         phase, i, out[i], expected);
            return 1;
        }
    }
    return 0;
}

int main() {
    constexpr size_t count = 1u << 22;
    constexpr size_t bytes = count * sizeof(int);
    constexpr int repetitions = 25;
    int device_count = 0;
    CUDA_OK(cudaGetDeviceCount(&device_count));
    if (device_count != 1) {
        std::fprintf(stderr, "expected exactly one CUDA GPU, found %d\n",
                     device_count);
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s, compute %d.%d, global memory %zu bytes\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem);

    int *a = nullptr, *b = nullptr, *out = nullptr;
    int *da = nullptr, *db = nullptr, *dout = nullptr;
    CUDA_OK(cudaMallocHost(&a, bytes));
    CUDA_OK(cudaMallocHost(&b, bytes));
    CUDA_OK(cudaMallocHost(&out, bytes));
    CUDA_OK(cudaMalloc(&da, bytes));
    CUDA_OK(cudaMalloc(&db, bytes));
    CUDA_OK(cudaMalloc(&dout, bytes));

    for (size_t i = 0; i < count; ++i) {
        a[i] = static_cast<int>((i * 17u + 3u) % 1000003u) - 500001;
        b[i] = static_cast<int>((i * 29u + 11u) % 999983u) - 499991;
    }

    for (int rep = 0; rep < repetitions; ++rep) {
        std::memset(out, 0xa5, bytes);
        CUDA_OK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
        vector_add<<<(count + 255) / 256, 256>>>(da, db, dout, count);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost));
        if (verify(a, b, out, count, "pinned H2D/kernel/D2H"))
            return 1;
    }

    CUDA_OK(cudaFree(dout));
    CUDA_OK(cudaFree(db));
    CUDA_OK(cudaFree(da));
    CUDA_OK(cudaFreeHost(out));
    CUDA_OK(cudaFreeHost(b));
    CUDA_OK(cudaFreeHost(a));
    CUDA_OK(cudaDeviceReset());
    std::printf("PASS: %d exact pinned-memory H2D/kernel/D2H repetitions, "
                "%zu integers each\n", repetitions, count);
    return 0;
}
