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
private:
    static inline void one0(ArrayProgram::Int32& o) {
        o = __int32_utils.one();
    };
    static inline void zero0(ArrayProgram::Int32& o) {
        o = __int32_utils.zero();
    };
public:
    typedef float64_utils::Float64 Float64;
    typedef array<ArrayProgram::Float64>::Shape Shape;
    struct _print_shape {
        inline void operator()(const ArrayProgram::Shape& sh) {
            return __array.print_shape(sh);
        };
    };

    static ArrayProgram::_print_shape print_shape;
    typedef array<ArrayProgram::Float64>::UInt32 UInt32;
    struct _create_shape_3 {
        inline ArrayProgram::Shape operator()(const ArrayProgram::UInt32& a, const ArrayProgram::UInt32& b, const ArrayProgram::UInt32& c) {
            return __array.create_shape_3(a, b, c);
        };
    };

    static ArrayProgram::_create_shape_3 create_shape_3;
private:
    static array<ArrayProgram::Float64> __array;
public:
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
    typedef array<ArrayProgram::Float64>::Array Array;
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
    struct _total {
        inline ArrayProgram::UInt32 operator()(const ArrayProgram::Array& a) {
            return __array.total(a);
        };
    };

    static ArrayProgram::_total total;
};
} // examples
} // moa
} // mg_src
} // moa_cpp