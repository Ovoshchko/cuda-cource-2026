#include "cuda_check.h"
#include "kernels.h"

#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string_view input;
    std::string_view output;
    AccessMode mode = AccessMode::Coalesced;
};

struct InputVectors {
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
    std::cout
        << "Usage: submission --input PATH --output PATH [--mode coalesced|noncoalesced]\n"
           "Runs Add, SAXPY, and ReLU in the selected mode (default: coalesced).\n"
           "Input: a text file with two floats x y per line. Blank lines are ignored.\n"
           "Output: a TSV file with columns index, add, saxpy, and relu (index starts at 0).\n";
}

InputVectors read_input(std::string_view path) {
    std::ifstream file{std::string(path)};
    if (!file) {
        throw std::invalid_argument("Cannot open input file: " + std::string(path));
    }

    InputVectors input;
    std::size_t line_number = 0;
    for (std::string line; std::getline(file, line);) {
        ++line_number;
        std::istringstream row(line);
        if ((row >> std::ws).eof()) {
            continue;
        }

        float x = 0.0f;
        float y = 0.0f;
        if (!(row >> x >> y) || !(row >> std::ws).eof()) {
            throw std::invalid_argument("Expected two floats on input line " +
                                        std::to_string(line_number));
        }
        input.x.push_back(x);
        input.y.push_back(y);
    }
    if (file.bad()) {
        throw std::invalid_argument("Cannot read input file: " + std::string(path));
    }
    if (input.x.empty()) {
        throw std::invalid_argument("Input must contain at least one pair of floats");
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

    file << "index\tadd\tsaxpy\trelu\n"
         << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (std::size_t i = 0; i < add.size(); ++i) {
        file << i << '\t' << add[i] << '\t' << saxpy[i] << '\t' << relu[i] << '\n';
    }
    file.close();
    if (!file) {
        throw std::runtime_error("Cannot write output file: " + std::string(path));
    }
}

void run_operations(const Options &options, const InputVectors &input) {
    constexpr unsigned int threads = 256;
    constexpr float alpha = 2.0f;
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
    CUDA_CHECK(launch_saxpy(options.mode, blocks, threads, d_x.get(), d_y.get(), alpha, n));
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
