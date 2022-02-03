#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct ArrayProgram {
private:
    static int32_utils __int32_utils;
public:
    typedef int32_utils::Int32 Int32;
    typedef matrix<ArrayProgram::Int32>::Matrix Matrix;
    struct _cat {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2) {
            ArrayProgram::Size x_dim = ArrayProgram::access_shape(m1, ArrayProgram::integerToSize(ArrayProgram::zero()));
            ArrayProgram::Size y_dim;
            if (ArrayProgram::equals(ArrayProgram::sizeToInteger(ArrayProgram::size(ArrayProgram::shape(m2))), ArrayProgram::one()))
            {
                y_dim = ArrayProgram::integerToSize(ArrayProgram::add(ArrayProgram::sizeToInteger(ArrayProgram::access_shape(m1, ArrayProgram::integerToSize(ArrayProgram::one()))), ArrayProgram::one()));
            }
            else
            {
                y_dim = ArrayProgram::integerToSize(ArrayProgram::add(ArrayProgram::sizeToInteger(ArrayProgram::access_shape(m1, ArrayProgram::integerToSize(ArrayProgram::one()))), ArrayProgram::sizeToInteger(ArrayProgram::access_shape(m2, ArrayProgram::integerToSize(ArrayProgram::one())))));
            }
            ArrayProgram::Matrix res = ArrayProgram::zeros(x_dim, y_dim);
            ArrayProgram::Int32 c1 = ArrayProgram::zero();
            ArrayProgram::Int32 c2 = ArrayProgram::zero();
            ArrayProgram::doCat(m1, m2, res, c1, c2);
            return res;
        };
    };

    static ArrayProgram::_cat cat;
    struct _matmult {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2) {
            ArrayProgram::Size x_dim = ArrayProgram::access_shape(m1, ArrayProgram::integerToSize(ArrayProgram::zero()));
            ArrayProgram::Size y_dim = ArrayProgram::access_shape(m2, ArrayProgram::integerToSize(ArrayProgram::one()));
            ArrayProgram::Matrix res = ArrayProgram::zeros(x_dim, y_dim);
            ArrayProgram::Int32 c1 = ArrayProgram::zero();
            ArrayProgram::Int32 c2 = ArrayProgram::zero();
            ArrayProgram::doMatmult(m1, ArrayProgram::transpose(m2), res, c1, c2);
            return res;
        };
    };

    static ArrayProgram::_matmult matmult;
    struct _print_matrix {
        inline void operator()(const ArrayProgram::Matrix& m) {
            return __matrix.print_matrix(m);
        };
    };

    static ArrayProgram::_print_matrix print_matrix;
    struct _print_shape {
        inline void operator()(const ArrayProgram::Matrix& m) {
            return __matrix.print_shape(m);
        };
    };

    static ArrayProgram::_print_shape print_shape;
    struct _test_matrix {
        inline ArrayProgram::Matrix operator()() {
            return __matrix.test_matrix();
        };
    };

    static ArrayProgram::_test_matrix test_matrix;
    struct _test_matrix2 {
        inline ArrayProgram::Matrix operator()() {
            return __matrix.test_matrix2();
        };
    };

    static ArrayProgram::_test_matrix2 test_matrix2;
    struct _test_vector {
        inline ArrayProgram::Matrix operator()() {
            return __matrix.test_vector();
        };
    };

    static ArrayProgram::_test_vector test_vector;
    struct _transpose {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Matrix& m) {
            return __matrix.transpose(m);
        };
    };

    static ArrayProgram::_transpose transpose;
    typedef matrix<ArrayProgram::Int32>::Shape Shape;
    struct _shape {
        inline ArrayProgram::Shape operator()(const ArrayProgram::Matrix& m) {
            return __matrix.shape(m);
        };
    };

    static ArrayProgram::_shape shape;
    typedef matrix<ArrayProgram::Int32>::Size Size;
    struct _access_shape {
        inline ArrayProgram::Size operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Size& s) {
            return __matrix.access_shape(m, s);
        };
    };

    static ArrayProgram::_access_shape access_shape;
    struct _create_matrix {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Size& i, const ArrayProgram::Size& j) {
            return __matrix.create_matrix(i, j);
        };
    };

    static ArrayProgram::_create_matrix create_matrix;
    struct _size {
        inline ArrayProgram::Size operator()(const ArrayProgram::Shape& s) {
            return __matrix.size(s);
        };
        inline ArrayProgram::Size operator()(const ArrayProgram::Matrix& m) {
            return __matrix.size(m);
        };
    };

    static ArrayProgram::_size size;
    struct _zeros {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Size& i, const ArrayProgram::Size& j) {
            return __matrix.zeros(i, j);
        };
    };

    static ArrayProgram::_zeros zeros;
private:
    static matrix<ArrayProgram::Int32> __matrix;
public:
    struct _add {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.add(a, b);
        };
    };

    static ArrayProgram::_add add;
    struct _cPadl {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Int32& i) {
            ArrayProgram::Matrix slice = ArrayProgram::get(m, ArrayProgram::single_I(i));
            ArrayProgram::Matrix pRes = ArrayProgram::transpose(ArrayProgram::cat(slice, ArrayProgram::transpose(m)));
            return pRes;
        };
    };

    static ArrayProgram::_cPadl cPadl;
    struct _cPadr {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Int32& i) {
            ArrayProgram::Matrix slice = ArrayProgram::get(m, ArrayProgram::single_I(i));
            ArrayProgram::Matrix pRes = ArrayProgram::transpose(ArrayProgram::cat(ArrayProgram::transpose(m), slice));
            return pRes;
        };
    };

    static ArrayProgram::_cPadr cPadr;
    struct _catBody {
        inline void operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, ArrayProgram::Matrix& res, ArrayProgram::Int32& c1, ArrayProgram::Int32& c2) {
            ArrayProgram::Int32 m1_row_bound = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(m1, ArrayProgram::integerToSize(ArrayProgram::one())));
            ArrayProgram::Int32 res_row_bound = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(res, ArrayProgram::integerToSize(ArrayProgram::one())));
            if (ArrayProgram::isLowerThan(c2, m1_row_bound))
            {
                ArrayProgram::Index ix = ArrayProgram::create_index(c1, c2);
                ArrayProgram::set(res, c1, c2, ArrayProgram::unwrap_scalar(ArrayProgram::get(m1, ix)));
                c2 = ArrayProgram::add(c2, ArrayProgram::one());
            }
            else
            {
                ArrayProgram::Index ix;
                if (ArrayProgram::equals(ArrayProgram::sizeToInteger(ArrayProgram::size(ArrayProgram::shape(m2))), ArrayProgram::one()))
                {
                    ix = ArrayProgram::single_I(ArrayProgram::sub(c2, m1_row_bound));
                    ArrayProgram::set(res, c1, c2, ArrayProgram::unwrap_scalar(ArrayProgram::get(m2, ix)));
                }
                else
                {
                    ix = ArrayProgram::create_index(c1, ArrayProgram::sub(c2, m1_row_bound));
                    ArrayProgram::set(res, c1, c2, ArrayProgram::unwrap_scalar(ArrayProgram::get(m2, ix)));
                }
                if (ArrayProgram::isLowerThan(c2, ArrayProgram::sub(res_row_bound, ArrayProgram::one())))
                {
                    c2 = ArrayProgram::add(c2, ArrayProgram::one());
                }
                else
                {
                    c1 = ArrayProgram::add(c1, ArrayProgram::one());
                    c2 = ArrayProgram::zero();
                }
            }
        };
    };

    static ArrayProgram::_catBody catBody;
    struct _catUpperBound {
        inline bool operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Matrix& res, const ArrayProgram::Int32& c1, const ArrayProgram::Int32& c2) {
            ArrayProgram::Int32 row_upper = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(res, ArrayProgram::integerToSize(ArrayProgram::zero())));
            return ArrayProgram::isLowerThan(c1, row_upper);
        };
    };

private:
    static while_loop2_3<ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::_catBody, ArrayProgram::_catUpperBound> __while_loop2_3;
public:
    static ArrayProgram::_catUpperBound catUpperBound;
    struct _doCat {
        inline void operator()(const ArrayProgram::Matrix& context1, const ArrayProgram::Matrix& context2, ArrayProgram::Matrix& state1, ArrayProgram::Int32& state2, ArrayProgram::Int32& state3) {
            return __while_loop2_3.repeat(context1, context2, state1, state2, state3);
        };
    };

    static ArrayProgram::_doCat doCat;
    struct _doMatmult {
        inline void operator()(const ArrayProgram::Matrix& context1, const ArrayProgram::Matrix& context2, ArrayProgram::Matrix& state1, ArrayProgram::Int32& state2, ArrayProgram::Int32& state3) {
            return __while_loop2_30.repeat(context1, context2, state1, state2, state3);
        };
    };

    static ArrayProgram::_doMatmult doMatmult;
    struct _equals {
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.equals(a, b);
        };
    };

    static ArrayProgram::_equals equals;
    struct _integerToSize {
        inline ArrayProgram::Size operator()(const ArrayProgram::Int32& i) {
            return __matrix.integerToSize(i);
        };
    };

    static ArrayProgram::_integerToSize integerToSize;
    struct _isLowerThan {
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.isLowerThan(a, b);
        };
    };

    static ArrayProgram::_isLowerThan isLowerThan;
    struct _iterMatMult {
        inline void operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, ArrayProgram::Matrix& resM, ArrayProgram::Int32& i, ArrayProgram::Int32& k) {
            ArrayProgram::Matrix slice1 = ArrayProgram::get(m1, ArrayProgram::single_I(i));
            ArrayProgram::Matrix slice2 = ArrayProgram::get(m2, ArrayProgram::single_I(k));
            ArrayProgram::Int32 i_dim = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(resM, ArrayProgram::integerToSize(ArrayProgram::zero())));
            ArrayProgram::Int32 k_dim = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(resM, ArrayProgram::integerToSize(ArrayProgram::one())));
            ArrayProgram::Matrix result_slice = ArrayProgram::zeros(ArrayProgram::integerToSize(ArrayProgram::one()), ArrayProgram::integerToSize(i_dim));
            ArrayProgram::Int32 vecmultC = ArrayProgram::zero();
            ArrayProgram::vecmult(slice1, slice2, result_slice, vecmultC);
            ArrayProgram::Int32 reduced = ArrayProgram::zero();
            ArrayProgram::Int32 mapsumC = ArrayProgram::zero();
            ArrayProgram::mapsum(result_slice, reduced, mapsumC);
            ArrayProgram::set(resM, i, k, reduced);
            if (ArrayProgram::isLowerThan(ArrayProgram::add(k, ArrayProgram::one()), k_dim))
            {
                k = ArrayProgram::add(k, ArrayProgram::one());
            }
            else
            {
                i = ArrayProgram::add(i, ArrayProgram::one());
                k = ArrayProgram::zero();
            }
        };
    };

    static ArrayProgram::_iterMatMult iterMatMult;
    struct _mapsum {
        inline void operator()(const ArrayProgram::Matrix& context1, ArrayProgram::Int32& state1, ArrayProgram::Int32& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_mapsum mapsum;
    struct _mult {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.mult(a, b);
        };
    };

    static ArrayProgram::_mult mult;
    struct _mult_elementwise {
        inline void operator()(const ArrayProgram::Matrix& a, const ArrayProgram::Matrix& b, ArrayProgram::Matrix& res, ArrayProgram::Int32& counter) {
            ArrayProgram::Index current_index = ArrayProgram::single_I(counter);
            ArrayProgram::Int32 new_value = ArrayProgram::mult(ArrayProgram::unwrap_scalar(ArrayProgram::get(a, current_index)), ArrayProgram::unwrap_scalar(ArrayProgram::get(b, current_index)));
            ArrayProgram::set(res, ArrayProgram::zero(), counter, new_value);
            counter = ArrayProgram::add(counter, ArrayProgram::one());
        };
    };

    static ArrayProgram::_mult_elementwise mult_elementwise;
    struct _one {
        inline ArrayProgram::Int32 operator()() {
            return __int32_utils.one();
        };
    };

    static ArrayProgram::_one one;
    struct _print_number {
        inline void operator()(const ArrayProgram::Int32& e) {
            return __matrix.print_number(e);
        };
    };

    static ArrayProgram::_print_number print_number;
    struct _set {
        inline void operator()(ArrayProgram::Matrix& m, const ArrayProgram::Int32& i, const ArrayProgram::Int32& j, const ArrayProgram::Int32& e) {
            return __matrix.set(m, i, j, e);
        };
    };

    static ArrayProgram::_set set;
    struct _sizeToInteger {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Size& s) {
            return __matrix.sizeToInteger(s);
        };
    };

    static ArrayProgram::_sizeToInteger sizeToInteger;
    struct _sub {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.sub(a, b);
        };
    };

    static ArrayProgram::_sub sub;
    struct _sum_vector {
        inline void operator()(const ArrayProgram::Matrix& a, ArrayProgram::Int32& res, ArrayProgram::Int32& counter) {
            ArrayProgram::Index current_index = ArrayProgram::single_I(counter);
            res = ArrayProgram::add(res, ArrayProgram::unwrap_scalar(ArrayProgram::get(a, current_index)));
            counter = ArrayProgram::add(counter, ArrayProgram::one());
        };
    };

    static ArrayProgram::_sum_vector sum_vector;
    struct _unwrap_scalar {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Matrix& m) {
            return __matrix.unwrap_scalar(m);
        };
    };

    static ArrayProgram::_unwrap_scalar unwrap_scalar;
    struct _upperBoundMatMult {
        inline bool operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Matrix& res, const ArrayProgram::Int32& i, const ArrayProgram::Int32& k) {
            ArrayProgram::Int32 i_dim = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(res, ArrayProgram::integerToSize(ArrayProgram::zero())));
            ArrayProgram::Int32 k_dim = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(res, ArrayProgram::integerToSize(ArrayProgram::one())));
            return (ArrayProgram::isLowerThan(i, i_dim)) && (ArrayProgram::isLowerThan(k, k_dim));
        };
    };

private:
    static while_loop2_3<ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::_iterMatMult, ArrayProgram::_upperBoundMatMult> __while_loop2_30;
public:
    static ArrayProgram::_upperBoundMatMult upperBoundMatMult;
    struct _upperBoundMulElem {
        inline bool operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Matrix& res, const ArrayProgram::Int32& counter) {
            return ArrayProgram::isLowerThan(counter, ArrayProgram::sizeToInteger(ArrayProgram::size(m1)));
        };
    };

private:
    static while_loop2_2<ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::_mult_elementwise, ArrayProgram::_upperBoundMulElem> __while_loop2_2;
public:
    static ArrayProgram::_upperBoundMulElem upperBoundMulElem;
    struct _upperBoundSum {
        inline bool operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Int32& res, const ArrayProgram::Int32& counter) {
            return ArrayProgram::isLowerThan(counter, ArrayProgram::sizeToInteger(ArrayProgram::size(m)));
        };
    };

private:
    static while_loop1_2<ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::_sum_vector, ArrayProgram::_upperBoundSum> __while_loop1_2;
public:
    static ArrayProgram::_upperBoundSum upperBoundSum;
    struct _vecmult {
        inline void operator()(const ArrayProgram::Matrix& context1, const ArrayProgram::Matrix& context2, ArrayProgram::Matrix& state1, ArrayProgram::Int32& state2) {
            return __while_loop2_2.repeat(context1, context2, state1, state2);
        };
    };

    static ArrayProgram::_vecmult vecmult;
    struct _zero {
        inline ArrayProgram::Int32 operator()() {
            return __int32_utils.zero();
        };
    };

    static ArrayProgram::_zero zero;
    typedef matrix<ArrayProgram::Int32>::Index Index;
    struct _create_index {
        inline ArrayProgram::Index operator()(const ArrayProgram::Int32& i, const ArrayProgram::Int32& j) {
            return __matrix.create_index(i, j);
        };
    };

    static ArrayProgram::_create_index create_index;
    struct _get {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Index& i) {
            return __matrix.get(m, i);
        };
    };

    static ArrayProgram::_get get;
    struct _print_index {
        inline void operator()(const ArrayProgram::Index& i) {
            return __matrix.print_index(i);
        };
    };

    static ArrayProgram::_print_index print_index;
    struct _single_I {
        inline ArrayProgram::Index operator()(const ArrayProgram::Int32& i) {
            return __matrix.single_I(i);
        };
    };

    static ArrayProgram::_single_I single_I;
    struct _test_partial_index {
        inline ArrayProgram::Index operator()() {
            return __matrix.test_partial_index();
        };
    };

    static ArrayProgram::_test_partial_index test_partial_index;
    struct _test_total_index {
        inline ArrayProgram::Index operator()() {
            return __matrix.test_total_index();
        };
    };

    static ArrayProgram::_test_total_index test_total_index;
};
} // examples
} // moa
} // mg_src
} // moa_cpp