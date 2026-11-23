#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

inline void check_cuda(cudaError_t status, const char *call) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(call) ::check_cuda((call), #call)
