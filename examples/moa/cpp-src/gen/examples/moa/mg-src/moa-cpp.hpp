#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace moa {
namespace mg_src {
namespace moa_cpp {
struct ArrayProgram {
    struct _zero {
        template <typename T>
        inline T operator()() {
            T o;
            ArrayProgram::zero0(o);
            return o;
        };
    };

    static ArrayProgram::_zero zero;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            ArrayProgram::one0(o);
            return o;
        };
    };

    static ArrayProgram::_one one;
private:
    static int32_utils __int32_utils;
    static float64_utils __float64_utils;
public:
    typedef int32_utils::Int32 Int32;
    typedef array<ArrayProgram::Int32>::Shape Shape;
    struct _print_shape {
        inline void operator()(const ArrayProgram::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static ArrayProgram::_print_shape print_shape;
    typedef array<ArrayProgram::Int32>::UInt32 UInt32;
    struct _create_shape1 {
        inline ArrayProgram::Shape operator()(const ArrayProgram::UInt32& a) {
            return __array.create_shape1(a);
        };
    };

    static ArrayProgram::_create_shape1 create_shape1;
    struct _create_shape2 {
        inline ArrayProgram::Shape operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b) {
            return __array.create_shape2(a, b);
        };
    };

    static ArrayProgram::_create_shape2 create_shape2;
    struct _create_shape3 {
        inline ArrayProgram::Shape operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b, const ArrayProgram::UInt32& c) {
            return __array.create_shape3(a, b, c);
        };
    };

    static ArrayProgram::_create_shape3 create_shape3;
private:
    static array<ArrayProgram::Int32> __array;
    static inline void one0(ArrayProgram::Int32& o) {
        o = __int32_utils.one();
    };
    static inline void zero0(ArrayProgram::Int32& o) {
        o = __int32_utils.zero();
    };
public:
    typedef array<ArrayProgram::Int32>::Index Index;
    struct _create_index1 {
        inline ArrayProgram::Index operator()(const ArrayProgram::UInt32& a) {
            return __array.create_index1(a);
        };
    };

    static ArrayProgram::_create_index1 create_index1;
    struct _create_index2 {
        inline ArrayProgram::Index operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b) {
            return __array.create_index2(a, b);
        };
    };

    static ArrayProgram::_create_index2 create_index2;
    struct _create_index3 {
        inline ArrayProgram::Index operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b, const ArrayProgram::UInt32& c) {
            return __array.create_index3(a, b, c);
        };
    };

    static ArrayProgram::_create_index3 create_index3;
    struct _test_index {
        inline ArrayProgram::Index operator()() {
            return __array.test_index();
        };
    };

    static ArrayProgram::_test_index test_index;
    typedef float64_utils::Float64 Float64;
    struct _add {
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.add(a, b);
        };
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.add(a, b);
        };
    };

    static ArrayProgram::_add add;
    struct _equals {
        inline bool operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.equals(a, b);
        };
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.equals(a, b);
        };
    };

    static ArrayProgram::_equals equals;
    struct _isLowerThan {
        inline bool operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.isLowerThan(a, b);
        };
        inline bool operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.isLowerThan(a, b);
        };
    };

    static ArrayProgram::_isLowerThan isLowerThan;
    struct _mult {
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.mult(a, b);
        };
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.mult(a, b);
        };
    };

    static ArrayProgram::_mult mult;
    struct _sub {
        inline ArrayProgram::Float64 operator()(const ArrayProgram::Float64& a, const ArrayProgram::Float64& b) {
            return __float64_utils.sub(a, b);
        };
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Int32& a, const ArrayProgram::Int32& b) {
            return __int32_utils.sub(a, b);
        };
    };

    static ArrayProgram::_sub sub;
private:
    static inline void one0(ArrayProgram::Float64& o) {
        o = __float64_utils.one();
    };
    static inline void zero0(ArrayProgram::Float64& o) {
        o = __float64_utils.zero();
    };
public:
    typedef array<ArrayProgram::Int32>::Array Array;
    struct _create_array {
        inline ArrayProgram::Array operator()(const ArrayProgram::Shape& sh) {
            return __array.create_array(sh);
        };
    };

    static ArrayProgram::_create_array create_array;
    struct _dim {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Array& a) {
            return __array.dim(a);
        };
    };

    static ArrayProgram::_dim dim;
    struct _get {
        inline ArrayProgram::Array operator()(const ArrayProgram::Array& a, const ArrayProgram::Index& ix) {
            return __array.get(a, ix);
        };
    };

    static ArrayProgram::_get get;
    struct _get_shape_elem {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Array& a, const ArrayProgram::UInt32& i) {
            return __array.get_shape_elem(a, i);
        };
    };

    static ArrayProgram::_get_shape_elem get_shape_elem;
    struct _print_array {
        inline void operator()(const ArrayProgram::Array& a) {
            return __array.print_array(a);
        };
    };

    static ArrayProgram::_print_array print_array;
    struct _set {
        inline void operator()(ArrayProgram::Array& a, const ArrayProgram::Index& ix, const ArrayProgram::Int32& e) {
            return __array.set(a, ix, e);
        };
    };

    static ArrayProgram::_set set;
    struct _shape {
        inline ArrayProgram::Shape operator()(const ArrayProgram::Array& a) {
            return __array.shape(a);
        };
    };

    static ArrayProgram::_shape shape;
    struct _test_array3_2_2 {
        inline ArrayProgram::Array operator()() {
            return __array.test_array3_2_2();
        };
    };

    static ArrayProgram::_test_array3_2_2 test_array3_2_2;
    struct _test_array3_3 {
        inline ArrayProgram::Array operator()() {
            return __array.test_array3_3();
        };
    };

    static ArrayProgram::_test_array3_3 test_array3_3;
    struct _total {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Array& a) {
            return __array.total(a);
        };
    };

    static ArrayProgram::_total total;
    struct _unwrap_scalar {
        inline ArrayProgram::Int32 operator()(const ArrayProgram::Array& a) {
            return __array.unwrap_scalar(a);
        };
    };

    static ArrayProgram::_unwrap_scalar unwrap_scalar;
};
} // examples
} // moa
} // mg_src
} // moa_cpp