#include <iostream>
#include "gen/examples/mmult/mg-src/mmult.hpp"

int main(int argc, char **argv) {
    --argc;
    if (argc < 1) {
        std::cerr << "Usage: " << argv[0] << " <row a> <col a> <row b> <col b>"
                  << std::endl;
        return 1;
    }

    // TODO: add a base build path to compiler args to avoid having so many
    // nested useless folders when compiling from somewhere else.
    examples::mmult::mg_src::mmult::MatrixProgram P;

    if (atoi(argv[2]) == atoi(argv[3])) {

        examples::mmult::mg_src::mmult::MatrixProgram::Matrix a = P.rand_matrix(atoi(argv[1]), atoi(argv[2]),10);
        examples::mmult::mg_src::mmult::MatrixProgram::Matrix b = P.rand_matrix(atoi(argv[3]), atoi(argv[4]),10);
        P.printMatrix(a);
        std::cout << "x" << std::endl;
        P.printMatrix(b);
        std::cout << "=" << std::endl;
        P.printMatrix(P.mmult(
        P.rand_matrix(atoi(argv[1]), atoi(argv[2]),10),
        P.rand_matrix(atoi(argv[3]), atoi(argv[3]),10)));
    }
    else {
        std::cout << "Cannot multiply matrices, need shapes m x n and n x p" << std::endl;
    }




}