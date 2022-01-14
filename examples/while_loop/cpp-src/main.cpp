#include <iostream>
#include "gen/examples/while_loop/mg-src/while_loop-cpp.hpp"

int main(int argc, char **argv) {
    --argc;
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <nb bits> <start> <bound>"
                  << std::endl;
        return 1;
    }

    int nb_bits = atoi(argv[1]);

    // TODO: add a base build path to compiler args to avoid having so many
    // nested useless folders when compiling from somewhere else.
    examples::while_loop::mg_src::while_loop_cpp::IterationProgram P;

    int32_utils::Int32 iterator32, bound32;
    int16_utils::Int16 iterator16, bound16;

    if (nb_bits == 32) {
        iterator32 = atoi(argv[2]);
        bound32 = atoi(argv[3]);
        P.repeat(iterator32, bound32);
        std::cout << "32-bit iterator: " << iterator32 << std::endl;
    }
    else if (nb_bits == 16) {
        iterator16 = atoi(argv[2]);
        bound16 = atoi(argv[3]);
        P.repeat(iterator16, bound16);
        std::cout << "16-bit iterator: " << iterator16 << std::endl;
    }
    else {
        std::cout << "<nb bits> must be 16 or 32" << std::endl;
    }
}
