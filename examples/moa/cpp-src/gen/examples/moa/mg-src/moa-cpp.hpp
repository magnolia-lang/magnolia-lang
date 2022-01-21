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
    struct _doMatMult {
        inline void operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, ArrayProgram::Matrix& resM, ArrayProgram::Int32& i, ArrayProgram::Int32& k) {
            ArrayProgram::Matrix slice1 = ArrayProgram::get(m1, ArrayProgram::create_singleton_index(i));
            ArrayProgram::Matrix slice2 = ArrayProgram::get(m2, ArrayProgram::create_singleton_index(k));
            ArrayProgram::Int32 i_dim = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(resM, ArrayProgram::integerToSize(ArrayProgram::zero())));
            ArrayProgram::Int32 k_dim = ArrayProgram::sizeToInteger(ArrayProgram::access_shape(resM, ArrayProgram::integerToSize(ArrayProgram::one())));
            ArrayProgram::Matrix result_slice = ArrayProgram::zeros(ArrayProgram::integerToSize(ArrayProgram::one()), ArrayProgram::integerToSize(k_dim));
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

    static ArrayProgram::_doMatMult doMatMult;
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
    struct _mapsum {
        inline void operator()(const ArrayProgram::Matrix& context1, ArrayProgram::Int32& state1, ArrayProgram::Int32& state2) {
            return __while_loop1_2.repeat(context1, state1, state2);
        };
    };

    static ArrayProgram::_mapsum mapsum;
    struct _matmult {
        inline void operator()(const ArrayProgram::Matrix& context1, const ArrayProgram::Matrix& context2, ArrayProgram::Matrix& state1, ArrayProgram::Int32& state2, ArrayProgram::Int32& state3) {
            return __while_loop2_3.repeat(context1, context2, state1, state2, state3);
        };
    };

    static ArrayProgram::_matmult matmult;
    struct _mult {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.mult(a, b);
        };
    };

    static ArrayProgram::_mult mult;
    struct _mult_elementwise {
        inline void operator()(const ArrayProgram::Matrix& a, const ArrayProgram::Matrix& b, ArrayProgram::Matrix& res, ArrayProgram::Int32& counter) {
            ArrayProgram::Index current_index = ArrayProgram::create_singleton_index(counter);
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
        inline void operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Int32& i, const ArrayProgram::Int32& j, const ArrayProgram::Int32& e) {
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
    struct _sum_vector {
        inline void operator()(const ArrayProgram::Matrix& a, ArrayProgram::Int32& res, ArrayProgram::Int32& counter) {
            ArrayProgram::Index current_index = ArrayProgram::create_singleton_index(counter);
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
    static while_loop2_3<ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::_doMatMult, ArrayProgram::_upperBoundMatMult> __while_loop2_3;
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
    struct _create_singleton_index {
        inline ArrayProgram::Index operator()(const ArrayProgram::Int32& i) {
            return __matrix.create_singleton_index(i);
        };
    };

    static ArrayProgram::_create_singleton_index create_singleton_index;
    struct _get {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Index& i) {
            return __matrix.get(m, i);
        };
    };

    static ArrayProgram::_get get;
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