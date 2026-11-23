#include "cuda_check.h"
#include "kernels.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace {

enum class Operation { Add, Saxpy, Relu };
constexpr std::array operations{Operation::Add, Operation::Saxpy, Operation::Relu};
constexpr std::array block_sizes{128u, 256u, 512u};
constexpr std::array validation_sizes{1u, 255u, 256u, 257u};
constexpr unsigned int validation_threads = 256;
constexpr std::size_t default_count = 1u << 22;
constexpr unsigned int seed = 2026;
constexpr float alpha = 2.0f;
constexpr float tolerance = 1e-5f;
constexpr int warmups = 2;
constexpr int repeats = 9;

struct Options {
    std::vector<std::size_t> sizes;
    std::vector<AccessMode> modes{AccessMode::Coalesced, AccessMode::NonCoalesced};
    bool check = false;
};

using Buffer = std::unique_ptr<float, decltype(&cudaFree)>;

Buffer allocate_device(std::size_t bytes) {
    void *data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, bytes));
    return Buffer(static_cast<float *>(data), cudaFree);
}

Buffer allocate_host(std::size_t bytes) {
    void *data = nullptr;
    CUDA_CHECK(cudaMallocHost(&data, bytes));
    return Buffer(static_cast<float *>(data), cudaFreeHost);
}

auto create_event() {
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreate(&event));
    return std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)>(
        event, cudaEventDestroy);
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

struct Buffers {
    std::size_t count;
    std::size_t bytes;
    float allocation_ms;
    Buffer x;
    Buffer y;
    Buffer result;
    Buffer d_x;
    Buffer d_y;
    Buffer d_result;
};

Buffers prepare_buffers(std::size_t count) {
    const std::size_t bytes = count * sizeof(float);
    const auto alloc_start = create_event();
    const auto alloc_stop = create_event();
    CUDA_CHECK(cudaEventRecord(alloc_start.get()));
    CUDA_CHECK(cudaEventSynchronize(alloc_start.get()));
    Buffers buffers{count,
                    bytes,
                    0.0f,
                    allocate_host(bytes),
                    allocate_host(bytes),
                    allocate_host(bytes),
                    allocate_device(bytes),
                    allocate_device(bytes),
                    allocate_device(bytes)};
    CUDA_CHECK(cudaEventRecord(alloc_stop.get()));
    CUDA_CHECK(cudaEventSynchronize(alloc_stop.get()));
    buffers.allocation_ms = elapsed_ms(alloc_start.get(), alloc_stop.get());

    std::mt19937 random(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::generate_n(buffers.x.get(), count, [&] { return dist(random); });
    std::generate_n(buffers.y.get(), count, [&] { return dist(random); });
    buffers.x.get()[0] = 0.0f;
    buffers.y.get()[0] = 0.0f;
    if (count >= 3) {
        buffers.x.get()[1] = -0.5f;
        buffers.x.get()[2] = 0.5f;
        buffers.y.get()[1] = 0.25f;
        buffers.y.get()[2] = -0.25f;
    }
    return buffers;
}

std::vector<float> reference(Operation op, const Buffers &buffers) {
    std::vector<float> expected(buffers.count);
    const float *x = buffers.x.get();
    const float *y = buffers.y.get();
    switch (op) {
    case Operation::Add:
        std::transform(x, x + buffers.count, y, expected.begin(), std::plus<float>{});
        break;
    case Operation::Saxpy:
        std::transform(x, x + buffers.count, y, expected.begin(),
                       [](float x, float y) { return std::fma(alpha, x, y); });
        break;
    case Operation::Relu:
        std::transform(x, x + buffers.count, expected.begin(),
                       [](float x) { return std::fmax(0.0f, x); });
        break;
    }
    return expected;
}

struct Accuracy {
    float max_error = 0.0f;
    std::size_t mismatches = 0;
};

Accuracy compare(const std::vector<float> &expected, const float *result) {
    Accuracy accuracy;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        float error = std::fabs(expected[i] - result[i]);
        if (!std::isfinite(error)) {
            error = std::numeric_limits<float>::infinity();
        }
        accuracy.max_error = std::max(accuracy.max_error, error);
        accuracy.mismatches += error > tolerance;
    }
    return accuracy;
}

void upload_inputs(Operation op, const Buffers &buffers) {
    CUDA_CHECK(
        cudaMemcpyAsync(buffers.d_x.get(), buffers.x.get(), buffers.bytes, cudaMemcpyHostToDevice));
    if (op != Operation::Relu) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.d_y.get(), buffers.y.get(), buffers.bytes,
                                   cudaMemcpyHostToDevice));
    }
}

void launch(Operation op, AccessMode mode, unsigned int threads, const Buffers &buffers) {
    const auto blocks = static_cast<unsigned int>(1 + (buffers.count - 1) / threads);
    switch (op) {
    case Operation::Add:
        CUDA_CHECK(launch_add(mode, blocks, threads, buffers.d_x.get(), buffers.d_y.get(),
                              buffers.d_result.get(), buffers.count));
        break;
    case Operation::Saxpy:
        CUDA_CHECK(launch_saxpy(mode, blocks, threads, buffers.d_x.get(), buffers.d_y.get(), alpha,
                                buffers.count));
        break;
    case Operation::Relu:
        CUDA_CHECK(launch_relu(mode, blocks, threads, buffers.d_x.get(), buffers.d_result.get(),
                               buffers.count));
        break;
    }
}

void download_result(Operation op, const Buffers &buffers) {
    const float *result = op == Operation::Saxpy ? buffers.d_y.get() : buffers.d_result.get();
    CUDA_CHECK(
        cudaMemcpyAsync(buffers.result.get(), result, buffers.bytes, cudaMemcpyDeviceToHost));
}

struct Timing {
    float h2d;
    float kernel;
    float d2h;
    float total;
};

float median(std::array<float, repeats> &samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

Timing measure(Operation op, AccessMode mode, unsigned int threads, const Buffers &buffers) {
    const auto start = create_event();
    const auto after_h2d = create_event();
    const auto after_kernel = create_event();
    const auto after_d2h = create_event();
    std::array<float, repeats> h2d{}, kernel{}, d2h{}, total{};

    for (int pass = 0; pass < warmups + repeats; ++pass) {
        CUDA_CHECK(cudaEventRecord(start.get()));
        upload_inputs(op, buffers);
        CUDA_CHECK(cudaEventRecord(after_h2d.get()));
        launch(op, mode, threads, buffers);
        CUDA_CHECK(cudaEventRecord(after_kernel.get()));
        download_result(op, buffers);
        CUDA_CHECK(cudaEventRecord(after_d2h.get()));
        CUDA_CHECK(cudaEventSynchronize(after_d2h.get()));

        if (pass >= warmups) {
            const int sample = pass - warmups;
            h2d[sample] = elapsed_ms(start.get(), after_h2d.get());
            kernel[sample] = elapsed_ms(after_h2d.get(), after_kernel.get());
            d2h[sample] = elapsed_ms(after_kernel.get(), after_d2h.get());
            total[sample] = elapsed_ms(start.get(), after_d2h.get());
        }
    }
    return {median(h2d), median(kernel), median(d2h), median(total)};
}

const char *name(Operation op) {
    switch (op) {
    case Operation::Add:
        return "Add";
    case Operation::Saxpy:
        return "SAXPY";
    case Operation::Relu:
        return "ReLU";
    }
    return "";
}

const char *name(AccessMode mode) {
    return mode == AccessMode::Coalesced ? "coalesced" : "noncoalesced";
}

bool validate(const Options &options) {
    std::cout << "\nCorrectness | " << validation_threads
              << " threads per block | absolute tolerance " << tolerance << '\n'
              << "Operation  Mode                N    Max abs error   Mismatches  Status\n"
              << "-----------------------------------------------------------------------\n";
    bool passed = true;
    for (const auto n : options.sizes) {
        const auto buffers = prepare_buffers(n);
        for (const auto op : operations) {
            const auto expected = reference(op, buffers);
            for (const auto mode : options.modes) {
                upload_inputs(op, buffers);
                launch(op, mode, validation_threads, buffers);
                download_result(op, buffers);
                CUDA_CHECK(cudaStreamSynchronize(nullptr));

                const auto accuracy = compare(expected, buffers.result.get());
                const bool ok = accuracy.mismatches == 0;
                passed &= ok;
                std::cout << std::left << std::setw(11) << name(op) << std::setw(14) << name(mode)
                          << std::right << std::setw(6) << n << std::scientific
                          << std::setprecision(3) << std::setw(17) << accuracy.max_error
                          << std::setw(13) << accuracy.mismatches << "  " << (ok ? "PASS" : "FAIL")
                          << '\n';
            }
        }
    }
    return passed;
}

void benchmark(std::size_t count, const std::vector<AccessMode> &modes) {
    const auto buffers = prepare_buffers(count);
    std::cout << "\nTiming | " << count << " elements | " << warmups << " warm-ups | median of "
              << repeats << " runs\n"
              << "Host and device allocation (once): " << std::fixed << std::setprecision(3)
              << buffers.allocation_ms << " ms\n";
    for (const auto mode : modes) {
        std::cout << '\n'
                  << name(mode) << '\n'
                  << std::left << std::setw(11) << "Operation" << std::right << std::setw(7)
                  << "Threads" << std::setw(11) << "H2D ms" << std::setw(12) << "Kernel ms"
                  << std::setw(10) << "D2H ms" << std::setw(11) << "Total ms" << '\n'
                  << "--------------------------------------------------------------\n";
        for (const auto op : operations) {
            for (const auto threads : block_sizes) {
                const auto timing = measure(op, mode, threads, buffers);
                std::cout << std::left << std::setw(11) << name(op) << std::right << std::setw(7)
                          << threads << std::setw(11) << timing.h2d << std::setw(12)
                          << timing.kernel << std::setw(10) << timing.d2h << std::setw(11)
                          << timing.total << '\n';
            }
        }
    }
}

std::size_t parse_count(std::string_view value) {
    std::size_t count = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), count);
    if (error != std::errc{} || end != value.data() + value.size() || count == 0 ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("Expected a positive vector size, got: " + std::string(value));
    }
    return count;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--check") {
            options.check = true;
        } else if (argument == "--mode") {
            const std::string_view mode = ++i < argc ? argv[i] : "";
            if (mode == "coalesced") {
                options.modes = {AccessMode::Coalesced};
            } else if (mode == "noncoalesced") {
                options.modes = {AccessMode::NonCoalesced};
            } else {
                throw std::invalid_argument("--mode must be coalesced or noncoalesced");
            }
        } else if (argument.substr(0, 2) == "--") {
            throw std::invalid_argument("Unknown argument: " + std::string(argument));
        } else {
            options.sizes.push_back(parse_count(argument));
        }
    }
    if (options.sizes.empty()) {
        if (options.check) {
            options.sizes.assign(validation_sizes.begin(), validation_sizes.end());
        } else {
            options.sizes.push_back(default_count);
        }
    }
    return options;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: benchmark [--check] [--mode coalesced|noncoalesced] [SIZE ...]\n"
                     "By default measures both modes over "
                  << default_count << " elements.\n"
                  << "--check compares with the CPU reference (default sizes: 1, 255, 256, 257).\n";
        return 0;
    }

    try {
        const Options options = parse_options(argc, argv);
        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        const auto largest = *std::max_element(options.sizes.begin(), options.sizes.end());
        const auto min_threads = options.check ? validation_threads : block_sizes.front();
        if (1 + (largest - 1) / min_threads > static_cast<std::size_t>(properties.maxGridSize[0])) {
            throw std::invalid_argument("ELEMENTS exceeds the GPU grid limit");
        }
        std::cout << "Lab 1 | " << properties.name << " | seed " << seed << " | SAXPY alpha "
                  << alpha << '\n';

        if (options.check) {
            const bool passed = validate(options);
            std::cout << "\nCorrectness: " << (passed ? "PASS" : "FAIL") << '\n';
            return passed ? 0 : 1;
        }
        for (const auto count : options.sizes) {
            benchmark(count, options.modes);
        }
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
