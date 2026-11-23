#include "cuda_check.h"
#include "kernels.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct Options {
    std::string_view input;
    std::string_view output;
    AccessMode mode = AccessMode::Coalesced;
};

struct InputVectors {
    float alpha = 0.0f;
    std::vector<float> x;
    std::vector<float> y;
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag != "--input" && flag != "--output" && flag != "--mode") {
            throw std::invalid_argument("Unknown argument: " + std::string(flag));
        }
        if (++i == argc) {
            throw std::invalid_argument("Missing value for " + std::string(flag));
        }

        const std::string_view value = argv[i];
        if (value.empty() || value.substr(0, 2) == "--") {
            throw std::invalid_argument("Missing value for " + std::string(flag));
        }
        if (flag == "--input") {
            options.input = value;
        } else if (flag == "--output") {
            options.output = value;
        } else if (value == "coalesced") {
            options.mode = AccessMode::Coalesced;
        } else if (value == "noncoalesced") {
            options.mode = AccessMode::NonCoalesced;
        } else {
            throw std::invalid_argument("--mode must be coalesced or noncoalesced");
        }
    }
    if (options.input.empty() || options.output.empty()) {
        throw std::invalid_argument("--input and --output are required. See --help.");
    }
    return options;
}

void print_help() {
    std::cout << "Usage: submission --input PATH --output PATH [--mode coalesced|noncoalesced]\n"
                 "Runs Add, SAXPY, and ReLU in the selected mode (default: coalesced).\n"
                 "Input: N alpha followed by exactly N pairs x y, separated by whitespace. N must "
                 "be positive.\n"
                 "Output: x+y, alpha*x+y, and max(0,x), separated by spaces, one line per pair.\n";
}

template <typename Number> bool read_number(std::istream &file, Number &value) {
    std::string token;
    if (!(file >> token)) {
        return false;
    }
    std::string_view text = token;
    if (text.front() == '+') {
        text.remove_prefix(1);
        if (text.empty() || text.front() == '-') {
            return false;
        }
    }
    const char *end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

InputVectors read_input(std::string_view path) {
    std::ifstream file{std::string(path)};
    if (!file) {
        throw std::invalid_argument("Cannot open input file: " + std::string(path));
    }

    InputVectors input;
    std::size_t n = 0;
    if (!read_number(file, n) || n == 0 || !read_number(file, input.alpha) ||
        !std::isfinite(input.alpha)) {
        throw std::invalid_argument(
            "Expected a positive N and a finite alpha at the start of input");
    }
    for (std::size_t i = 0; i < n; ++i) {
        float x = 0.0f;
        float y = 0.0f;
        if (!read_number(file, x) || !read_number(file, y) || !std::isfinite(x) ||
            !std::isfinite(y)) {
            throw std::invalid_argument("Expected two finite floats for pair " +
                                        std::to_string(i + 1));
        }
        input.x.push_back(x);
        input.y.push_back(y);
    }
    file >> std::ws;
    if (file.bad()) {
        throw std::invalid_argument("Cannot read input file: " + std::string(path));
    }
    if (!file.eof()) {
        throw std::invalid_argument("Unexpected data after N input pairs");
    }
    return input;
}

auto allocate_device(std::size_t bytes) {
    void *data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, bytes));
    return std::unique_ptr<float, decltype(&cudaFree)>(static_cast<float *>(data), cudaFree);
}

void write_results(std::string_view path, const std::vector<float> &add,
                   const std::vector<float> &saxpy, const std::vector<float> &relu) {
    std::ofstream file{std::string(path)};
    if (!file) {
        throw std::runtime_error("Cannot open output file: " + std::string(path));
    }

    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < add.size(); ++i) {
        file << add[i] << ' ' << saxpy[i] << ' ' << relu[i] << '\n';
    }
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot write output file: " + std::string(path));
    }
}

void run_operations(const Options &options, const InputVectors &input) {
    constexpr unsigned int threads = 256;
    const std::size_t n = input.x.size();
    const std::size_t bytes = n * sizeof(float);
    const auto blocks = static_cast<unsigned int>(1 + (n - 1) / threads);

    const auto d_x = allocate_device(bytes);
    const auto d_y = allocate_device(bytes);
    const auto d_add = allocate_device(bytes);
    const auto d_relu = allocate_device(bytes);

    CUDA_CHECK(cudaMemcpy(d_x.get(), input.x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y.get(), input.y.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(launch_add(options.mode, blocks, threads, d_x.get(), d_y.get(), d_add.get(), n));
    CUDA_CHECK(launch_saxpy(options.mode, blocks, threads, d_x.get(), d_y.get(), input.alpha, n));
    CUDA_CHECK(launch_relu(options.mode, blocks, threads, d_x.get(), d_relu.get(), n));
    std::vector<float> result_add(n);
    std::vector<float> result_saxpy(n);
    std::vector<float> result_relu(n);

    CUDA_CHECK(cudaMemcpy(result_add.data(), d_add.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_saxpy.data(), d_y.get(), bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_relu.data(), d_relu.get(), bytes, cudaMemcpyDeviceToHost));

    write_results(options.output, result_add, result_saxpy, result_relu);
    const char *mode = options.mode == AccessMode::Coalesced ? "coalesced" : "noncoalesced";
    std::cout << "Add, SAXPY, ReLU | " << mode << " | " << n << " elements\n"
              << "Results: " << options.output << '\n';
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_help();
        return 0;
    }

    try {
        const Options options = parse_options(argc, argv);
        const InputVectors input = read_input(options.input);
        run_operations(options, input);
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
