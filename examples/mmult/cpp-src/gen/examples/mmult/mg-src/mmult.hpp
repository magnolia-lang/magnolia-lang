#include "base.hpp"
#include <cassert>


namespace examples {
namespace mmult {
namespace mg_src {
namespace mmult {
struct MatrixProgram {
private:
    static int32_utils __int32_utils;
public:
    typedef int32_utils::Int32 Int32;
    typedef matrix<MatrixProgram::Int32>::Matrix Matrix;
    struct _mmult {
        MatrixProgram::Matrix operator()(const MatrixProgram::Matrix& a, const MatrixProgram::Matrix& b);
    };

    static MatrixProgram::_mmult mmult;
    struct _printMatrix {
        void operator()(const MatrixProgram::Matrix& a);
    };

    static MatrixProgram::_printMatrix printMatrix;
    typedef matrix<MatrixProgram::Int32>::Size Size;
    struct _create_matrix {
        MatrixProgram::Matrix operator()(const MatrixProgram::Size& row, const MatrixProgram::Size& col);
    };

    static MatrixProgram::_create_matrix create_matrix;
    struct _rand_matrix {
        MatrixProgram::Matrix operator()(const MatrixProgram::Size& row, const MatrixProgram::Size& col, const MatrixProgram::Size& upper_bound);
    };

    static MatrixProgram::_rand_matrix rand_matrix;
    struct _zeros {
        MatrixProgram::Matrix operator()(const MatrixProgram::Size& row, const MatrixProgram::Size& col);
    };

    static MatrixProgram::_zeros zeros;
private:
    static matrix<MatrixProgram::Int32> __matrix;
public:
    struct _add {
        MatrixProgram::Int32 operator()(const MatrixProgram::Int32& a, const MatrixProgram::Int32& b);
    };

    static MatrixProgram::_add add;
    struct _isLowerThan {
        bool operator()(const MatrixProgram::Int32& a, const MatrixProgram::Int32& b);
    };

    static MatrixProgram::_isLowerThan isLowerThan;
    struct _mult {
        MatrixProgram::Int32 operator()(const MatrixProgram::Int32& a, const MatrixProgram::Int32& b);
    };

    static MatrixProgram::_mult mult;
    struct _one {
        MatrixProgram::Int32 operator()();
    };

    static MatrixProgram::_one one;
    struct _zero {
        MatrixProgram::Int32 operator()();
    };

    static MatrixProgram::_zero zero;
};
} // examples
} // mmult
} // mg_src
} // mmult