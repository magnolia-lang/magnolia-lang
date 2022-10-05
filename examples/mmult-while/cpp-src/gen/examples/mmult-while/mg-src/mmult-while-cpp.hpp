#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace mmult_while {
namespace mg_src {
namespace mmult_while_cpp {
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
    typedef matrix<ArrayProgram::Int32>::Shape Shape;
    typedef matrix<ArrayProgram::Int32>::Size Size;
    struct _create_matrix {
        inline ArrayProgram::Matrix operator()(const ArrayProgram::Size& i, const ArrayProgram::Size& j) {
            return __matrix.create_matrix(i, j);
        };
    };

    static ArrayProgram::_create_matrix create_matrix;
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
    struct _innerLoop {
        inline void operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Int32& innerBound, const ArrayProgram::Int32& outerCounter, const ArrayProgram::Int32& middleCounter, ArrayProgram::Int32& innerCounter, ArrayProgram::Matrix& mres) {
            ArrayProgram::Index x = ArrayProgram::integerToIndex(outerCounter);
            ArrayProgram::Index y = ArrayProgram::integerToIndex(middleCounter);
            ArrayProgram::Index z = ArrayProgram::integerToIndex(innerCounter);
            ArrayProgram::Int32 m1_elem = ArrayProgram::get(m1, x, z);
            ArrayProgram::Int32 m2_elem = ArrayProgram::get(m2, z, y);
            ArrayProgram::set(mres, x, y, ArrayProgram::add(ArrayProgram::get(mres, x, y), ArrayProgram::mult(m1_elem, m2_elem)));
            innerCounter = ArrayProgram::add(innerCounter, ArrayProgram::one());
        };
    };

    static ArrayProgram::_innerLoop innerLoop;
    struct _isLowerThan {
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.isLowerThan(a, b);
        };
    };

    static ArrayProgram::_isLowerThan isLowerThan;
    struct _isLowerThanInner {
        inline bool operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Int32& innerBound, const ArrayProgram::Int32& b, const ArrayProgram::Int32& c, const ArrayProgram::Int32& innerCounter, const ArrayProgram::Matrix& mres) {
            return ArrayProgram::isLowerThan(innerCounter, innerBound);
        };
    };

private:
    static while_loop5_2<ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Matrix, ArrayProgram::_innerLoop, ArrayProgram::_isLowerThanInner> __while_loop5_2;
public:
    static ArrayProgram::_isLowerThanInner isLowerThanInner;
    struct _isLowerThanMid {
        inline bool operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Int32& middleBound, const ArrayProgram::Int32& b, const ArrayProgram::Int32& c, const ArrayProgram::Int32& d, const ArrayProgram::Int32& middleCounter, const ArrayProgram::Matrix& mres) {
            return ArrayProgram::isLowerThan(middleCounter, middleBound);
        };
    };

    static ArrayProgram::_isLowerThanMid isLowerThanMid;
    struct _isLowerThanOuter {
        inline bool operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Int32& outerBound, const ArrayProgram::Int32& b, const ArrayProgram::Int32& c, const ArrayProgram::Int32& d, const ArrayProgram::Int32& e, const ArrayProgram::Int32& outerCounter, const ArrayProgram::Matrix& mres) {
            return ArrayProgram::isLowerThan(outerCounter, outerBound);
        };
    };

    static ArrayProgram::_isLowerThanOuter isLowerThanOuter;
    struct _middleLoop {
        inline void operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Int32& middleBound, const ArrayProgram::Int32& innerBound, const ArrayProgram::Int32& innerCounter, const ArrayProgram::Int32& outerCounter, ArrayProgram::Int32& middleCounter, ArrayProgram::Matrix& mres) {
            ArrayProgram::Int32 innerCounter_upd = innerCounter;
            ArrayProgram::repeatInner(m1, m2, innerBound, outerCounter, middleCounter, innerCounter_upd, mres);
            middleCounter = ArrayProgram::add(middleCounter, ArrayProgram::one());
        };
    };

private:
    static while_loop6_2<ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Matrix, ArrayProgram::_middleLoop, ArrayProgram::_isLowerThanMid> __while_loop6_2;
public:
    static ArrayProgram::_middleLoop middleLoop;
    struct _mult {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.mult(a, b);
        };
    };

    static ArrayProgram::_mult mult;
    struct _one {
        inline ArrayProgram::Int32 operator()() {
            return __int32_utils.one();
        };
    };

    static ArrayProgram::_one one;
    struct _outerLoop {
        inline void operator()(const ArrayProgram::Matrix& m1, const ArrayProgram::Matrix& m2, const ArrayProgram::Int32& outerBound, const ArrayProgram::Int32& middleBound, const ArrayProgram::Int32& innerBound, const ArrayProgram::Int32& middleCounter, const ArrayProgram::Int32& innerCounter, ArrayProgram::Int32& outerCounter, ArrayProgram::Matrix& mres) {
            ArrayProgram::Int32 middleCounter_upd = middleCounter;
            ArrayProgram::repeatMiddle(m1, m2, middleBound, innerBound, innerCounter, outerCounter, middleCounter_upd, mres);
            outerCounter = ArrayProgram::add(outerCounter, ArrayProgram::one());
        };
    };

private:
    static while_loop7_2<ArrayProgram::Matrix, ArrayProgram::Matrix, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Int32, ArrayProgram::Matrix, ArrayProgram::_outerLoop, ArrayProgram::_isLowerThanOuter> __while_loop7_2;
public:
    static ArrayProgram::_outerLoop outerLoop;
    struct _print_number {
        inline void operator()(const ArrayProgram::Int32& e) {
            return __matrix.print_number(e);
        };
    };

    static ArrayProgram::_print_number print_number;
    struct _repeat {
        inline void operator()(const ArrayProgram::Matrix& context1, const ArrayProgram::Matrix& context2, const ArrayProgram::Int32& context3, const ArrayProgram::Int32& context4, const ArrayProgram::Int32& context5, const ArrayProgram::Int32& context6, const ArrayProgram::Int32& context7, ArrayProgram::Int32& state1, ArrayProgram::Matrix& state2) {
            return __while_loop7_2.repeat(context1, context2, context3, context4, context5, context6, context7, state1, state2);
        };
    };

    static ArrayProgram::_repeat repeat;
    struct _repeatInner {
        inline void operator()(const ArrayProgram::Matrix& context1, const ArrayProgram::Matrix& context2, const ArrayProgram::Int32& context3, const ArrayProgram::Int32& context4, const ArrayProgram::Int32& context5, ArrayProgram::Int32& state1, ArrayProgram::Matrix& state2) {
            return __while_loop5_2.repeat(context1, context2, context3, context4, context5, state1, state2);
        };
    };

    static ArrayProgram::_repeatInner repeatInner;
    struct _repeatMiddle {
        inline void operator()(const ArrayProgram::Matrix& context1, const ArrayProgram::Matrix& context2, const ArrayProgram::Int32& context3, const ArrayProgram::Int32& context4, const ArrayProgram::Int32& context5, const ArrayProgram::Int32& context6, ArrayProgram::Int32& state1, ArrayProgram::Matrix& state2) {
            return __while_loop6_2.repeat(context1, context2, context3, context4, context5, context6, state1, state2);
        };
    };

    static ArrayProgram::_repeatMiddle repeatMiddle;
    struct _zero {
        inline ArrayProgram::Int32 operator()() {
            return __int32_utils.zero();
        };
    };

    static ArrayProgram::_zero zero;
    typedef matrix<ArrayProgram::Int32>::Index Index;
    struct _get {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Index& i, const ArrayProgram::Index& j) {
            return __matrix.get(m, i, j);
        };
    };

    static ArrayProgram::_get get;
    struct _indexToInteger {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Index& i) {
            return __matrix.indexToInteger(i);
        };
    };

    static ArrayProgram::_indexToInteger indexToInteger;
    struct _integerToIndex {
        inline ArrayProgram::Index operator()(const ArrayProgram::Int32& i) {
            return __matrix.integerToIndex(i);
        };
    };

    static ArrayProgram::_integerToIndex integerToIndex;
    struct _set {
        inline void operator()(const ArrayProgram::Matrix& m, const ArrayProgram::Index& i, const ArrayProgram::Index& j, const ArrayProgram::Int32& e) {
            return __matrix.set(m, i, j, e);
        };
    };

    static ArrayProgram::_set set;
};
} // examples
} // mmult_while
} // mg_src
} // mmult_while_cpp