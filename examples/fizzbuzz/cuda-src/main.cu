#include <iostream>
#include "gen/examples/fizzbuzz/mg-src/fizzbuzz-cuda.cuh"

int main(int argc, char **argv) {
    --argc;
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <number-to-fizzbuzz-test>"
                  << std::endl;
        return 1;
    }

    // TODO: add a base build path to compiler args to avoid having so many
    // nested useless folders when compiling from somewhere else.
    examples::fizzbuzz::mg_src::fizzbuzz_cuda::MyFizzBuzzProgram P;

    std::cout << "Fizz my buzz: " << P.doFizzBuzz(atoi(argv[1])) << std::endl;
}
