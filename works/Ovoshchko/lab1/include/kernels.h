#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

enum class AccessMode { Coalesced, NonCoalesced };

cudaError_t launch_add(AccessMode mode, unsigned int blocks, unsigned int threads, const float *A,
                       const float *B, float *C, std::size_t n);
cudaError_t launch_saxpy(AccessMode mode, unsigned int blocks, unsigned int threads, const float *x,
                         float *y, float alpha, std::size_t n);
cudaError_t launch_relu(AccessMode mode, unsigned int blocks, unsigned int threads, const float *x,
                        float *y, std::size_t n);
