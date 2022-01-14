#include <iostream>
#include "gen/examples/moa/testing/mg-src/Sum.hpp"

int main(int argc, char **argv) {
    --argc;
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <number-to-sum-test>"
                  << std::endl;
        return 1;
    }

    // TODO: add a base build path to compiler args to avoid having so many
    // nested useless folders when compiling from somewhere else.
    examples::moa::testing::mg_src::Sum::SumProgram P;

    std::cout << "Sum: " << P.doSum(atoi(argv[1]), atoi(argv[2])) << std::endl;
}