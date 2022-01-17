#include "gen/examples/mmult/mg-src/mmult.hpp"


namespace examples {
namespace mmult {
namespace mg_src {
namespace mmult {
    int32_utils MatrixProgram::__int32_utils;

    MatrixProgram::Matrix MatrixProgram::_mmult::operator()(const MatrixProgram::Matrix& a, const MatrixProgram::Matrix& b) {
        return __matrix.mmult(a, b);
    };


    MatrixProgram::_mmult MatrixProgram::mmult;

    void MatrixProgram::_printMatrix::operator()(const MatrixProgram::Matrix& a) {
        return __matrix.printMatrix(a);
    };


    MatrixProgram::_printMatrix MatrixProgram::printMatrix;

    MatrixProgram::Matrix MatrixProgram::_create_matrix::operator()(const MatrixProgram::Size& row, const MatrixProgram::Size& col) {
        return __matrix.create_matrix(row, col);
    };


    MatrixProgram::_create_matrix MatrixProgram::create_matrix;

    MatrixProgram::Matrix MatrixProgram::_rand_matrix::operator()(const MatrixProgram::Size& row, const MatrixProgram::Size& col, const MatrixProgram::Size& upper_bound) {
        return __matrix.rand_matrix(row, col, upper_bound);
    };


    MatrixProgram::_rand_matrix MatrixProgram::rand_matrix;

    MatrixProgram::Matrix MatrixProgram::_zeros::operator()(const MatrixProgram::Size& row, const MatrixProgram::Size& col) {
        return __matrix.zeros(row, col);
    };


    MatrixProgram::_zeros MatrixProgram::zeros;

    matrix<MatrixProgram::Int32> MatrixProgram::__matrix;

    MatrixProgram::Int32 MatrixProgram::_add::operator()(const MatrixProgram::Int32& a, const MatrixProgram::Int32& b) {
        return __int32_utils.add(a, b);
    };


    MatrixProgram::_add MatrixProgram::add;

    bool MatrixProgram::_isLowerThan::operator()(const MatrixProgram::Int32& a, const MatrixProgram::Int32& b) {
        return __int32_utils.isLowerThan(a, b);
    };


    MatrixProgram::_isLowerThan MatrixProgram::isLowerThan;

    MatrixProgram::Int32 MatrixProgram::_mult::operator()(const MatrixProgram::Int32& a, const MatrixProgram::Int32& b) {
        return __int32_utils.mult(a, b);
    };


    MatrixProgram::_mult MatrixProgram::mult;

    MatrixProgram::Int32 MatrixProgram::_one::operator()() {
        return __int32_utils.one();
    };


    MatrixProgram::_one MatrixProgram::one;

    MatrixProgram::Int32 MatrixProgram::_zero::operator()() {
        return __int32_utils.zero();
    };


    MatrixProgram::_zero MatrixProgram::zero;

} // examples
} // mmult
} // mg_src
} // mmult