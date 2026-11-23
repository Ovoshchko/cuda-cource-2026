cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES
    75
    CACHE STRING "CUDA architectures")
project(submission LANGUAGES CXX CUDA)

add_executable(submission src/main.cu src/kernels.cu)
add_executable(benchmark src/benchmark.cu src/kernels.cu)

foreach(target IN ITEMS submission benchmark)
  target_include_directories(${target} PRIVATE include)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
endforeach()
