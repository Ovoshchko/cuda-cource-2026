#include "kernels.h"

#include <cstddef>
#include <cuda_runtime.h>

template <AccessMode Mode> __device__ std::size_t selectIndex() {
    if constexpr (Mode == AccessMode::Coalesced) {
        return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    } else {
        return static_cast<std::size_t>(threadIdx.x) * gridDim.x + blockIdx.x;
    }
}

template <AccessMode Mode>
__global__ void add(const float *A, const float *B, float *C, const std::size_t n) {
    const std::size_t i = selectIndex<Mode>();

    if (i >= n) {
        return;
    }

    C[i] = A[i] + B[i];
}

template <AccessMode Mode>
__global__ void saxpy(const float *x, float *y, float alpha, std::size_t n) {
    const std::size_t i = selectIndex<Mode>();

    if (i >= n) {
        return;
    }

    y[i] = alpha * x[i] + y[i];
}

template <AccessMode Mode> __global__ void relu(const float *x, float *y, std::size_t n) {
    const std::size_t i = selectIndex<Mode>();

    if (i >= n) {
        return;
    }

    y[i] = fmaxf(0.0f, x[i]);
}

cudaError_t launch_add(AccessMode mode, unsigned int blocks, unsigned int threads, const float *A,
                       const float *B, float *C, std::size_t n) {
    if (mode == AccessMode::Coalesced) {
        add<AccessMode::Coalesced><<<blocks, threads>>>(A, B, C, n);
    } else {
        add<AccessMode::NonCoalesced><<<blocks, threads>>>(A, B, C, n);
    }

    return cudaGetLastError();
}

cudaError_t launch_saxpy(AccessMode mode, unsigned int blocks, unsigned int threads, const float *x,
                         float *y, float alpha, std::size_t n) {
    if (mode == AccessMode::Coalesced) {
        saxpy<AccessMode::Coalesced><<<blocks, threads>>>(x, y, alpha, n);
    } else {
        saxpy<AccessMode::NonCoalesced><<<blocks, threads>>>(x, y, alpha, n);
    }

    return cudaGetLastError();
}

cudaError_t launch_relu(AccessMode mode, unsigned int blocks, unsigned int threads, const float *x,
                        float *y, std::size_t n) {
    if (mode == AccessMode::Coalesced) {
        relu<AccessMode::Coalesced><<<blocks, threads>>>(x, y, n);
    } else {
        relu<AccessMode::NonCoalesced><<<blocks, threads>>>(x, y, n);
    }

    return cudaGetLastError();
}
