#include "gen/examples/containers/mg-src/containers.hpp"
#include <iostream>

using examples::containers::mg_src::containers::NestedPairProgram;

int main(int argc, char **argv) {
    --argc;
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <int16> <int32> <int16> <int32>"
                  << std::endl;
        return 1;
    }

    NestedPairProgram P;
    NestedPairProgram::Int16 e00, e10;
    NestedPairProgram::Int32 e01, e11;

    e00 = atoi(argv[1]);
    e01 = atoi(argv[2]);
    e10 = atoi(argv[3]);
    e11 = atoi(argv[4]);
    
    auto outer_pair = P.make_pair(P.make_pair(e00, e01),
                                  P.make_pair(e10, e11));

    std::cout << "Elements are: " << std::endl;
    std::cout << "("
              << P.first(P.first(outer_pair))
              << ", "
              << P.second(P.first(outer_pair))
              << ")," << std::endl
              << "("
              << P.first(P.second(outer_pair))
              << ", "
              << P.second(P.second(outer_pair))
              << ")" << std::endl;
    return 0;
}
