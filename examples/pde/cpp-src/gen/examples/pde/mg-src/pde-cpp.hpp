#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct BasePDEProgram {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            BasePDEProgram::two0(o);
            return o;
        };
    };

    static BasePDEProgram::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            BasePDEProgram::one0(o);
            return o;
        };
    };

    static BasePDEProgram::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<BasePDEProgram::Float>::Index Index;
    typedef array_ops<BasePDEProgram::Float>::Nat Nat;
    typedef array_ops<BasePDEProgram::Float>::Offset Offset;
private:
    static inline void one0(BasePDEProgram::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<BasePDEProgram::Float> __array_ops;
public:
    struct _dt {
        inline BasePDEProgram::Float operator()() {
            return __constants.dt();
        };
    };

    static BasePDEProgram::_dt dt;
    struct _dx {
        inline BasePDEProgram::Float operator()() {
            return __constants.dx();
        };
    };

    static BasePDEProgram::_dx dx;
    struct _nu {
        inline BasePDEProgram::Float operator()() {
            return __constants.nu();
        };
    };

    static BasePDEProgram::_nu nu;
    struct _three {
        inline BasePDEProgram::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static BasePDEProgram::_three three;
    struct _unary_sub {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline BasePDEProgram::Offset operator()(const BasePDEProgram::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static BasePDEProgram::_unary_sub unary_sub;
private:
    static inline void one0(BasePDEProgram::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(BasePDEProgram::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<BasePDEProgram::Float>::Axis Axis;
    struct _rotateIx {
        inline BasePDEProgram::Index operator()(const BasePDEProgram::Index& ix, const BasePDEProgram::Axis& axis, const BasePDEProgram::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static BasePDEProgram::_rotateIx rotateIx;
    struct _zero {
        inline BasePDEProgram::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static BasePDEProgram::_zero zero;
private:
    static inline void one0(BasePDEProgram::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(BasePDEProgram::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<BasePDEProgram::Float>::Array Array;
    struct _binary_add {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static BasePDEProgram::_binary_add binary_add;
    struct _binary_sub {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static BasePDEProgram::_binary_sub binary_sub;
    struct _div {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& num, const BasePDEProgram::Float& den) {
            return __array_ops.div(num, den);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& num, const BasePDEProgram::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static BasePDEProgram::_div div;
    struct _mul {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static BasePDEProgram::_mul mul;
    struct _psi {
        inline BasePDEProgram::Float operator()(const BasePDEProgram::Index& ix, const BasePDEProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static BasePDEProgram::_psi psi;
    struct _rotate {
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& a, const BasePDEProgram::Axis& axis, const BasePDEProgram::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static BasePDEProgram::_rotate rotate;
    struct _step {
        inline void operator()(BasePDEProgram::Array& u0, BasePDEProgram::Array& u1, BasePDEProgram::Array& u2) {
            BasePDEProgram::Array v0 = u0;
            BasePDEProgram::Array v1 = u1;
            BasePDEProgram::Array v2 = u2;
            v0 = BasePDEProgram::substep(v0, u0, u0, u1, u2);
            v1 = BasePDEProgram::substep(v1, u1, u0, u1, u2);
            v2 = BasePDEProgram::substep(v2, u2, u0, u1, u2);
            u0 = BasePDEProgram::substep(u0, v0, u0, u1, u2);
            u1 = BasePDEProgram::substep(u1, v1, u0, u1, u2);
            u2 = BasePDEProgram::substep(u2, v2, u0, u1, u2);
        };
    };

    static BasePDEProgram::_step step;
    struct _substep {
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& u, const BasePDEProgram::Array& v, const BasePDEProgram::Array& u0, const BasePDEProgram::Array& u1, const BasePDEProgram::Array& u2) {
            return BasePDEProgram::binary_add(u, BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::dt(), BasePDEProgram::two.operator()<Float>()), BasePDEProgram::binary_sub(BasePDEProgram::mul(BasePDEProgram::nu(), BasePDEProgram::binary_sub(BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::div(BasePDEProgram::one.operator()<Float>(), BasePDEProgram::dx()), BasePDEProgram::dx()), BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::div(BasePDEProgram::mul(BasePDEProgram::three(), BasePDEProgram::two.operator()<Float>()), BasePDEProgram::dx()), BasePDEProgram::dx()), u0))), BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::div(BasePDEProgram::one.operator()<Float>(), BasePDEProgram::two.operator()<Float>()), BasePDEProgram::dx()), BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u0), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u1)), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u2))))));
        };
    };

    static BasePDEProgram::_substep substep;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram::two0(o);
            return o;
        };
    };

    static PDEProgram::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram::one0(o);
            return o;
        };
    };

    static PDEProgram::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgram::Float>::Index Index;
    typedef array_ops<PDEProgram::Float>::Nat Nat;
    typedef array_ops<PDEProgram::Float>::Offset Offset;
private:
    static inline void one0(PDEProgram::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<PDEProgram::Float> __array_ops;
public:
    struct _dt {
        inline PDEProgram::Float operator()() {
            return __constants.dt();
        };
    };

    static PDEProgram::_dt dt;
    struct _dx {
        inline PDEProgram::Float operator()() {
            return __constants.dx();
        };
    };

    static PDEProgram::_dx dx;
    struct _nu {
        inline PDEProgram::Float operator()() {
            return __constants.nu();
        };
    };

    static PDEProgram::_nu nu;
    struct _three {
        inline PDEProgram::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram::_three three;
    struct _unary_sub {
        inline PDEProgram::Float operator()(const PDEProgram::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgram::Offset operator()(const PDEProgram::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static PDEProgram::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgram::Float>::Axis Axis;
    struct _rotateIx {
        inline PDEProgram::Index operator()(const PDEProgram::Index& ix, const PDEProgram::Axis& axis, const PDEProgram::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static PDEProgram::_rotateIx rotateIx;
    struct _rotateIxPadded {
        inline PDEProgram::Index operator()(const PDEProgram::Index& ix, const PDEProgram::Axis& axis, const PDEProgram::Offset& o) {
            return __specialize_psi_ops_2.rotateIxPadded(ix, axis, o);
        };
    };

    static PDEProgram::_rotateIxPadded rotateIxPadded;
    struct _zero {
        inline PDEProgram::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram::_zero zero;
private:
    static inline void one0(PDEProgram::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgram::Float>::Array Array;
    struct _binary_sub {
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram::Float operator()(const PDEProgram::Float& num, const PDEProgram::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& num, const PDEProgram::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram::_div div;
    struct _mul {
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram::_mul mul;
    struct _refillPadding {
        inline void operator()(PDEProgram::Array& a) {
            return __specialize_psi_ops_2.refillPadding(a);
        };
    };

    static PDEProgram::_refillPadding refillPadding;
    struct _rotate {
        inline PDEProgram::Array operator()(const PDEProgram::Array& a, const PDEProgram::Axis& axis, const PDEProgram::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram::_rotate rotate;
    struct _schedule {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    static PDEProgram::_schedule schedule;
    struct _schedule3DPadded {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2) {
            return __specialize_psi_ops_2.schedule3DPadded(u, v, u0, u1, u2);
        };
    };

    static PDEProgram::_schedule3DPadded schedule3DPadded;
    struct _step {
        inline void operator()(PDEProgram::Array& u0, PDEProgram::Array& u1, PDEProgram::Array& u2) {
            PDEProgram::Array v0 = u0;
            PDEProgram::Array v1 = u1;
            PDEProgram::Array v2 = u2;
            v0 = [&]() {
                PDEProgram::Array result = PDEProgram::schedule3DPadded(v0, u0, u0, u1, u2);
                PDEProgram::refillPadding(result);
                return result;
            }();
            v1 = [&]() {
                PDEProgram::Array result = PDEProgram::schedule3DPadded(v1, u1, u0, u1, u2);
                PDEProgram::refillPadding(result);
                return result;
            }();
            v2 = [&]() {
                PDEProgram::Array result = PDEProgram::schedule3DPadded(v2, u2, u0, u1, u2);
                PDEProgram::refillPadding(result);
                return result;
            }();
            u0 = [&]() {
                PDEProgram::Array result = PDEProgram::schedule3DPadded(u0, v0, u0, u1, u2);
                PDEProgram::refillPadding(result);
                return result;
            }();
            u1 = [&]() {
                PDEProgram::Array result = PDEProgram::schedule3DPadded(u1, v1, u0, u1, u2);
                PDEProgram::refillPadding(result);
                return result;
            }();
            u2 = [&]() {
                PDEProgram::Array result = PDEProgram::schedule3DPadded(u2, v2, u0, u1, u2);
                PDEProgram::refillPadding(result);
                return result;
            }();
        };
    };

    static PDEProgram::_step step;
    struct _substep {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2) {
            return PDEProgram::binary_add(u, PDEProgram::mul(PDEProgram::div(PDEProgram::dt(), PDEProgram::two.operator()<Float>()), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::nu(), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::mul(PDEProgram::three(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::dx()), u0))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u0), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u1)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram::_substep substep;
    struct _substepIx {
        inline PDEProgram::Float operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Index& ix) {
            return PDEProgram::binary_add(PDEProgram::psi(PDEProgram::ix0(ix), PDEProgram::ix1(ix), PDEProgram::ix2(ix), u), PDEProgram::mul(PDEProgram::div(PDEProgram::dt(), PDEProgram::two.operator()<Float>()), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::nu(), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), v), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), v)), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), v)), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), v))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::mul(PDEProgram::three(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::psi(PDEProgram::ix0(ix), PDEProgram::ix1(ix), PDEProgram::ix2(ix), u0)))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), v), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), v)), PDEProgram::psi(PDEProgram::ix0(ix), PDEProgram::ix1(ix), PDEProgram::ix2(ix), u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), v), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), v)), PDEProgram::psi(PDEProgram::ix0(ix), PDEProgram::ix1(ix), PDEProgram::ix2(ix), u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), v), PDEProgram::psi(PDEProgram::ix0(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix1(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::ix2(PDEProgram::rotateIxPadded(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), v)), PDEProgram::psi(PDEProgram::ix0(ix), PDEProgram::ix1(ix), PDEProgram::ix2(ix), u2)))))));
        };
    };

    typedef forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_substepIx>::ScalarIndex ScalarIndex;
    struct _binary_add {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::ScalarIndex& six, const PDEProgram::Offset& o) {
            return __specialize_psi_ops_2.binary_add(six, o);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram::_binary_add binary_add;
    struct _ix0 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __specialize_psi_ops_2.ix0(ix);
        };
    };

    static PDEProgram::_ix0 ix0;
    struct _ix1 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __specialize_psi_ops_2.ix1(ix);
        };
    };

    static PDEProgram::_ix1 ix1;
    struct _ix2 {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::Index& ix) {
            return __specialize_psi_ops_2.ix2(ix);
        };
    };

    static PDEProgram::_ix2 ix2;
    struct _mkIx {
        inline PDEProgram::Index operator()(const PDEProgram::ScalarIndex& a, const PDEProgram::ScalarIndex& b, const PDEProgram::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static PDEProgram::_mkIx mkIx;
    struct _psi {
        inline PDEProgram::Float operator()(const PDEProgram::ScalarIndex& i, const PDEProgram::ScalarIndex& j, const PDEProgram::ScalarIndex& k, const PDEProgram::Array& a) {
            return __specialize_psi_ops_2.psi(i, j, k, a);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Index& ix, const PDEProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram::_psi psi;
private:
    static forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_substepIx> __forall_ops;
public:
    static PDEProgram::_substepIx substepIx;
    struct _substepIx3D {
        inline PDEProgram::Float operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::ScalarIndex& i, const PDEProgram::ScalarIndex& j, const PDEProgram::ScalarIndex& k) {
            return PDEProgram::binary_add(PDEProgram::psi(i, j, k, u), PDEProgram::mul(PDEProgram::div(PDEProgram::dt(), PDEProgram::two.operator()<Float>()), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::nu(), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), j, k, v), PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::one.operator()<Offset>()), j, k, v)), PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), k, v)), PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::one.operator()<Offset>()), k, v)), PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::one.operator()<Offset>()), v))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::mul(PDEProgram::three(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::psi(i, j, k, u0)))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::one.operator()<Offset>()), j, k, v), PDEProgram::psi(PDEProgram::binary_add(i, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), j, k, v)), PDEProgram::psi(i, j, k, u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::one.operator()<Offset>()), k, v), PDEProgram::psi(i, PDEProgram::binary_add(j, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), k, v)), PDEProgram::psi(i, j, k, u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(i, j, PDEProgram::binary_add(k, PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(i, j, k, u2)))))));
        };
    };

    typedef specialize_psi_ops_2<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Offset, PDEProgram::ScalarIndex, PDEProgram::_substepIx3D>::AxisLength AxisLength;
    struct _mod {
        inline PDEProgram::ScalarIndex operator()(const PDEProgram::ScalarIndex& six, const PDEProgram::AxisLength& sc) {
            return __specialize_psi_ops_2.mod(six, sc);
        };
    };

    static PDEProgram::_mod mod;
    struct _shape0 {
        inline PDEProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape0();
        };
    };

    static PDEProgram::_shape0 shape0;
    struct _shape1 {
        inline PDEProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape1();
        };
    };

    static PDEProgram::_shape1 shape1;
    struct _shape2 {
        inline PDEProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape2();
        };
    };

    static PDEProgram::_shape2 shape2;
private:
    static specialize_psi_ops_2<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Offset, PDEProgram::ScalarIndex, PDEProgram::_substepIx3D> __specialize_psi_ops_2;
public:
    static PDEProgram::_substepIx3D substepIx3D;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram0 {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram0::two0(o);
            return o;
        };
    };

    static PDEProgram0::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram0::one0(o);
            return o;
        };
    };

    static PDEProgram0::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgram0::Float>::Index Index;
    typedef array_ops<PDEProgram0::Float>::Nat Nat;
    typedef array_ops<PDEProgram0::Float>::Offset Offset;
private:
    static inline void one0(PDEProgram0::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<PDEProgram0::Float> __array_ops;
public:
    struct _dt {
        inline PDEProgram0::Float operator()() {
            return __constants.dt();
        };
    };

    static PDEProgram0::_dt dt;
    struct _dx {
        inline PDEProgram0::Float operator()() {
            return __constants.dx();
        };
    };

    static PDEProgram0::_dx dx;
    struct _nu {
        inline PDEProgram0::Float operator()() {
            return __constants.nu();
        };
    };

    static PDEProgram0::_nu nu;
    struct _three {
        inline PDEProgram0::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram0::_three three;
    struct _unary_sub {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgram0::Offset operator()(const PDEProgram0::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static PDEProgram0::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram0::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram0::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgram0::Float>::Axis Axis;
    struct _rotateIx {
        inline PDEProgram0::Index operator()(const PDEProgram0::Index& ix, const PDEProgram0::Axis& axis, const PDEProgram0::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static PDEProgram0::_rotateIx rotateIx;
    struct _zero {
        inline PDEProgram0::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram0::_zero zero;
private:
    static inline void one0(PDEProgram0::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram0::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgram0::Float>::Array Array;
    struct _binary_add {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram0::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram0::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& num, const PDEProgram0::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& num, const PDEProgram0::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram0::_div div;
    struct _mul {
        inline PDEProgram0::Float operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Float& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& lhs, const PDEProgram0::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram0::_mul mul;
    struct _psi {
        inline PDEProgram0::Float operator()(const PDEProgram0::Index& ix, const PDEProgram0::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram0::_psi psi;
    struct _rotate {
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& a, const PDEProgram0::Axis& axis, const PDEProgram0::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram0::_rotate rotate;
    struct _step {
        inline void operator()(PDEProgram0::Array& u0, PDEProgram0::Array& u1, PDEProgram0::Array& u2) {
            PDEProgram0::Array v0 = u0;
            PDEProgram0::Array v1 = u1;
            PDEProgram0::Array v2 = u2;
            v0 = PDEProgram0::substep(v0, u0, u0, u1, u2);
            v1 = PDEProgram0::substep(v1, u1, u0, u1, u2);
            v2 = PDEProgram0::substep(v2, u2, u0, u1, u2);
            u0 = PDEProgram0::substep(u0, v0, u0, u1, u2);
            u1 = PDEProgram0::substep(u1, v1, u0, u1, u2);
            u2 = PDEProgram0::substep(u2, v2, u0, u1, u2);
        };
    };

    static PDEProgram0::_step step;
    struct _substep {
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& u, const PDEProgram0::Array& v, const PDEProgram0::Array& u0, const PDEProgram0::Array& u1, const PDEProgram0::Array& u2) {
            return PDEProgram0::binary_add(u, PDEProgram0::mul(PDEProgram0::div(PDEProgram0::dt(), PDEProgram0::two.operator()<Float>()), PDEProgram0::binary_sub(PDEProgram0::mul(PDEProgram0::nu(), PDEProgram0::binary_sub(PDEProgram0::mul(PDEProgram0::div(PDEProgram0::div(PDEProgram0::one.operator()<Float>(), PDEProgram0::dx()), PDEProgram0::dx()), PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()))), PDEProgram0::mul(PDEProgram0::div(PDEProgram0::div(PDEProgram0::mul(PDEProgram0::three(), PDEProgram0::two.operator()<Float>()), PDEProgram0::dx()), PDEProgram0::dx()), u0))), PDEProgram0::mul(PDEProgram0::div(PDEProgram0::div(PDEProgram0::one.operator()<Float>(), PDEProgram0::two.operator()<Float>()), PDEProgram0::dx()), PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u0), PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u1)), PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram0::_substep substep;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram1 {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram1::two0(o);
            return o;
        };
    };

    static PDEProgram1::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram1::one0(o);
            return o;
        };
    };

    static PDEProgram1::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgram1::Float>::Index Index;
    typedef array_ops<PDEProgram1::Float>::Nat Nat;
    typedef array_ops<PDEProgram1::Float>::Offset Offset;
private:
    static inline void one0(PDEProgram1::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<PDEProgram1::Float> __array_ops;
public:
    struct _dt {
        inline PDEProgram1::Float operator()() {
            return __constants.dt();
        };
    };

    static PDEProgram1::_dt dt;
    struct _dx {
        inline PDEProgram1::Float operator()() {
            return __constants.dx();
        };
    };

    static PDEProgram1::_dx dx;
    struct _nu {
        inline PDEProgram1::Float operator()() {
            return __constants.nu();
        };
    };

    static PDEProgram1::_nu nu;
    struct _three {
        inline PDEProgram1::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram1::_three three;
    struct _unary_sub {
        inline PDEProgram1::Offset operator()(const PDEProgram1::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& f) {
            return __array_ops.unary_sub(f);
        };
    };

    static PDEProgram1::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram1::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram1::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgram1::Float>::Axis Axis;
    struct _rotateIx {
        inline PDEProgram1::Index operator()(const PDEProgram1::Index& ix, const PDEProgram1::Axis& axis, const PDEProgram1::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static PDEProgram1::_rotateIx rotateIx;
    struct _zero {
        inline PDEProgram1::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram1::_zero zero;
private:
    static inline void one0(PDEProgram1::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram1::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgram1::Float>::Array Array;
    struct _binary_add {
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram1::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram1::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& num, const PDEProgram1::Array& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& num, const PDEProgram1::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram1::_div div;
    struct _mul {
        inline PDEProgram1::Float operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Float& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& lhs, const PDEProgram1::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram1::_mul mul;
    struct _psi {
        inline PDEProgram1::Float operator()(const PDEProgram1::Index& ix, const PDEProgram1::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram1::_psi psi;
    struct _rotate {
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& a, const PDEProgram1::Axis& axis, const PDEProgram1::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram1::_rotate rotate;
    struct _schedule {
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    static PDEProgram1::_schedule schedule;
    struct _step {
        inline void operator()(PDEProgram1::Array& u0, PDEProgram1::Array& u1, PDEProgram1::Array& u2) {
            PDEProgram1::Array v0 = u0;
            PDEProgram1::Array v1 = u1;
            PDEProgram1::Array v2 = u2;
            v0 = PDEProgram1::schedule(v0, u0, u0, u1, u2);
            v1 = PDEProgram1::schedule(v1, u1, u0, u1, u2);
            v2 = PDEProgram1::schedule(v2, u2, u0, u1, u2);
            u0 = PDEProgram1::schedule(u0, v0, u0, u1, u2);
            u1 = PDEProgram1::schedule(u1, v1, u0, u1, u2);
            u2 = PDEProgram1::schedule(u2, v2, u0, u1, u2);
        };
    };

    static PDEProgram1::_step step;
    struct _substep {
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2) {
            return PDEProgram1::binary_add(u, PDEProgram1::mul(PDEProgram1::div(PDEProgram1::dt(), PDEProgram1::two.operator()<Float>()), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::nu(), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::dx()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::mul(PDEProgram1::three(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::dx()), u0))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u0), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u1)), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram1::_substep substep;
    struct _substepIx {
        inline PDEProgram1::Float operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2, const PDEProgram1::Index& ix) {
            return PDEProgram1::binary_add(PDEProgram1::psi(ix, u), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::dt(), PDEProgram1::two.operator()<Float>()), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::nu(), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::dx()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), v)), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v)), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::mul(PDEProgram1::three(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::dx()), PDEProgram1::psi(ix, u0)))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u0)), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u1))), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotateIx(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgram1::Array, PDEProgram1::Axis, PDEProgram1::Float, PDEProgram1::Index, PDEProgram1::Nat, PDEProgram1::Offset, PDEProgram1::_substepIx>::ScalarIndex ScalarIndex;
    struct _mkIx {
        inline PDEProgram1::Index operator()(const PDEProgram1::ScalarIndex& a, const PDEProgram1::ScalarIndex& b, const PDEProgram1::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static PDEProgram1::_mkIx mkIx;
private:
    static forall_ops<PDEProgram1::Array, PDEProgram1::Axis, PDEProgram1::Float, PDEProgram1::Index, PDEProgram1::Nat, PDEProgram1::Offset, PDEProgram1::_substepIx> __forall_ops;
public:
    static PDEProgram1::_substepIx substepIx;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct SpecializedAndPaddedProgram {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            SpecializedAndPaddedProgram::two0(o);
            return o;
        };
    };

    static SpecializedAndPaddedProgram::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            SpecializedAndPaddedProgram::one0(o);
            return o;
        };
    };

    static SpecializedAndPaddedProgram::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<SpecializedAndPaddedProgram::Float>::Index Index;
    typedef array_ops<SpecializedAndPaddedProgram::Float>::Nat Nat;
    typedef array_ops<SpecializedAndPaddedProgram::Float>::Offset Offset;
private:
    static inline void one0(SpecializedAndPaddedProgram::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<SpecializedAndPaddedProgram::Float> __array_ops;
public:
    struct _dt {
        inline SpecializedAndPaddedProgram::Float operator()() {
            return __constants.dt();
        };
    };

    static SpecializedAndPaddedProgram::_dt dt;
    struct _dx {
        inline SpecializedAndPaddedProgram::Float operator()() {
            return __constants.dx();
        };
    };

    static SpecializedAndPaddedProgram::_dx dx;
    struct _nu {
        inline SpecializedAndPaddedProgram::Float operator()() {
            return __constants.nu();
        };
    };

    static SpecializedAndPaddedProgram::_nu nu;
    struct _three {
        inline SpecializedAndPaddedProgram::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static SpecializedAndPaddedProgram::_three three;
    struct _unary_sub {
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline SpecializedAndPaddedProgram::Offset operator()(const SpecializedAndPaddedProgram::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static SpecializedAndPaddedProgram::_unary_sub unary_sub;
private:
    static inline void one0(SpecializedAndPaddedProgram::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(SpecializedAndPaddedProgram::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<SpecializedAndPaddedProgram::Float>::Axis Axis;
    struct _rotateIx {
        inline SpecializedAndPaddedProgram::Index operator()(const SpecializedAndPaddedProgram::Index& ix, const SpecializedAndPaddedProgram::Axis& axis, const SpecializedAndPaddedProgram::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static SpecializedAndPaddedProgram::_rotateIx rotateIx;
    struct _rotateIxPadded {
        inline SpecializedAndPaddedProgram::Index operator()(const SpecializedAndPaddedProgram::Index& ix, const SpecializedAndPaddedProgram::Axis& axis, const SpecializedAndPaddedProgram::Offset& o) {
            return __specialize_psi_ops_2.rotateIxPadded(ix, axis, o);
        };
    };

    static SpecializedAndPaddedProgram::_rotateIxPadded rotateIxPadded;
    struct _zero {
        inline SpecializedAndPaddedProgram::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static SpecializedAndPaddedProgram::_zero zero;
private:
    static inline void one0(SpecializedAndPaddedProgram::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(SpecializedAndPaddedProgram::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<SpecializedAndPaddedProgram::Float>::Array Array;
    struct _binary_sub {
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Array& lhs, const SpecializedAndPaddedProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Float& lhs, const SpecializedAndPaddedProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Float& lhs, const SpecializedAndPaddedProgram::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static SpecializedAndPaddedProgram::_binary_sub binary_sub;
    struct _div {
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Float& num, const SpecializedAndPaddedProgram::Float& den) {
            return __array_ops.div(num, den);
        };
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Float& num, const SpecializedAndPaddedProgram::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static SpecializedAndPaddedProgram::_div div;
    struct _mul {
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Array& lhs, const SpecializedAndPaddedProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Float& lhs, const SpecializedAndPaddedProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Float& lhs, const SpecializedAndPaddedProgram::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static SpecializedAndPaddedProgram::_mul mul;
    struct _refillPadding {
        inline void operator()(SpecializedAndPaddedProgram::Array& a) {
            return __specialize_psi_ops_2.refillPadding(a);
        };
    };

    static SpecializedAndPaddedProgram::_refillPadding refillPadding;
    struct _rotate {
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Array& a, const SpecializedAndPaddedProgram::Axis& axis, const SpecializedAndPaddedProgram::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static SpecializedAndPaddedProgram::_rotate rotate;
    struct _schedule {
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Array& u, const SpecializedAndPaddedProgram::Array& v, const SpecializedAndPaddedProgram::Array& u0, const SpecializedAndPaddedProgram::Array& u1, const SpecializedAndPaddedProgram::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    static SpecializedAndPaddedProgram::_schedule schedule;
    struct _schedule3DPadded {
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Array& u, const SpecializedAndPaddedProgram::Array& v, const SpecializedAndPaddedProgram::Array& u0, const SpecializedAndPaddedProgram::Array& u1, const SpecializedAndPaddedProgram::Array& u2) {
            return __specialize_psi_ops_2.schedule3DPadded(u, v, u0, u1, u2);
        };
    };

    static SpecializedAndPaddedProgram::_schedule3DPadded schedule3DPadded;
    struct _step {
        inline void operator()(SpecializedAndPaddedProgram::Array& u0, SpecializedAndPaddedProgram::Array& u1, SpecializedAndPaddedProgram::Array& u2) {
            SpecializedAndPaddedProgram::Array v0 = u0;
            SpecializedAndPaddedProgram::Array v1 = u1;
            SpecializedAndPaddedProgram::Array v2 = u2;
            v0 = [&]() {
                SpecializedAndPaddedProgram::Array result = SpecializedAndPaddedProgram::schedule3DPadded(v0, u0, u0, u1, u2);
                SpecializedAndPaddedProgram::refillPadding(result);
                return result;
            }();
            v1 = [&]() {
                SpecializedAndPaddedProgram::Array result = SpecializedAndPaddedProgram::schedule3DPadded(v1, u1, u0, u1, u2);
                SpecializedAndPaddedProgram::refillPadding(result);
                return result;
            }();
            v2 = [&]() {
                SpecializedAndPaddedProgram::Array result = SpecializedAndPaddedProgram::schedule3DPadded(v2, u2, u0, u1, u2);
                SpecializedAndPaddedProgram::refillPadding(result);
                return result;
            }();
            u0 = [&]() {
                SpecializedAndPaddedProgram::Array result = SpecializedAndPaddedProgram::schedule3DPadded(u0, v0, u0, u1, u2);
                SpecializedAndPaddedProgram::refillPadding(result);
                return result;
            }();
            u1 = [&]() {
                SpecializedAndPaddedProgram::Array result = SpecializedAndPaddedProgram::schedule3DPadded(u1, v1, u0, u1, u2);
                SpecializedAndPaddedProgram::refillPadding(result);
                return result;
            }();
            u2 = [&]() {
                SpecializedAndPaddedProgram::Array result = SpecializedAndPaddedProgram::schedule3DPadded(u2, v2, u0, u1, u2);
                SpecializedAndPaddedProgram::refillPadding(result);
                return result;
            }();
        };
    };

    static SpecializedAndPaddedProgram::_step step;
    struct _substep {
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Array& u, const SpecializedAndPaddedProgram::Array& v, const SpecializedAndPaddedProgram::Array& u0, const SpecializedAndPaddedProgram::Array& u1, const SpecializedAndPaddedProgram::Array& u2) {
            return SpecializedAndPaddedProgram::binary_add(u, SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::dt(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::nu(), SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::one.operator()<Float>(), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::three(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::dx()), u0))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::one.operator()<Float>(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>()), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), u0), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>()), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), u1)), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>()), SpecializedAndPaddedProgram::rotate(v, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), u2))))));
        };
    };

    static SpecializedAndPaddedProgram::_substep substep;
    struct _substepIx {
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Array& u, const SpecializedAndPaddedProgram::Array& v, const SpecializedAndPaddedProgram::Array& u0, const SpecializedAndPaddedProgram::Array& u1, const SpecializedAndPaddedProgram::Array& u2, const SpecializedAndPaddedProgram::Index& ix) {
            return SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(ix), SpecializedAndPaddedProgram::ix1(ix), SpecializedAndPaddedProgram::ix2(ix), u), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::dt(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::nu(), SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::one.operator()<Float>(), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), v), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>())), v)), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), v)), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), v)), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), v)), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), v))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::three(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(ix), SpecializedAndPaddedProgram::ix1(ix), SpecializedAndPaddedProgram::ix2(ix), u0)))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::one.operator()<Float>(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::one.operator()<Offset>())), v), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::zero(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), v)), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(ix), SpecializedAndPaddedProgram::ix1(ix), SpecializedAndPaddedProgram::ix2(ix), u0)), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), v), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::one.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), v)), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(ix), SpecializedAndPaddedProgram::ix1(ix), SpecializedAndPaddedProgram::ix2(ix), u1))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::one.operator()<Offset>())), v), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix1(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), SpecializedAndPaddedProgram::ix2(SpecializedAndPaddedProgram::rotateIxPadded(ix, SpecializedAndPaddedProgram::two.operator()<Axis>(), SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>()))), v)), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::ix0(ix), SpecializedAndPaddedProgram::ix1(ix), SpecializedAndPaddedProgram::ix2(ix), u2)))))));
        };
    };

    typedef forall_ops<SpecializedAndPaddedProgram::Array, SpecializedAndPaddedProgram::Axis, SpecializedAndPaddedProgram::Float, SpecializedAndPaddedProgram::Index, SpecializedAndPaddedProgram::Nat, SpecializedAndPaddedProgram::Offset, SpecializedAndPaddedProgram::_substepIx>::ScalarIndex ScalarIndex;
    struct _binary_add {
        inline SpecializedAndPaddedProgram::ScalarIndex operator()(const SpecializedAndPaddedProgram::ScalarIndex& six, const SpecializedAndPaddedProgram::Offset& o) {
            return __specialize_psi_ops_2.binary_add(six, o);
        };
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Array& lhs, const SpecializedAndPaddedProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline SpecializedAndPaddedProgram::Array operator()(const SpecializedAndPaddedProgram::Float& lhs, const SpecializedAndPaddedProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Float& lhs, const SpecializedAndPaddedProgram::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static SpecializedAndPaddedProgram::_binary_add binary_add;
    struct _ix0 {
        inline SpecializedAndPaddedProgram::ScalarIndex operator()(const SpecializedAndPaddedProgram::Index& ix) {
            return __specialize_psi_ops_2.ix0(ix);
        };
    };

    static SpecializedAndPaddedProgram::_ix0 ix0;
    struct _ix1 {
        inline SpecializedAndPaddedProgram::ScalarIndex operator()(const SpecializedAndPaddedProgram::Index& ix) {
            return __specialize_psi_ops_2.ix1(ix);
        };
    };

    static SpecializedAndPaddedProgram::_ix1 ix1;
    struct _ix2 {
        inline SpecializedAndPaddedProgram::ScalarIndex operator()(const SpecializedAndPaddedProgram::Index& ix) {
            return __specialize_psi_ops_2.ix2(ix);
        };
    };

    static SpecializedAndPaddedProgram::_ix2 ix2;
    struct _mkIx {
        inline SpecializedAndPaddedProgram::Index operator()(const SpecializedAndPaddedProgram::ScalarIndex& a, const SpecializedAndPaddedProgram::ScalarIndex& b, const SpecializedAndPaddedProgram::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static SpecializedAndPaddedProgram::_mkIx mkIx;
    struct _psi {
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::ScalarIndex& i, const SpecializedAndPaddedProgram::ScalarIndex& j, const SpecializedAndPaddedProgram::ScalarIndex& k, const SpecializedAndPaddedProgram::Array& a) {
            return __specialize_psi_ops_2.psi(i, j, k, a);
        };
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Index& ix, const SpecializedAndPaddedProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static SpecializedAndPaddedProgram::_psi psi;
private:
    static forall_ops<SpecializedAndPaddedProgram::Array, SpecializedAndPaddedProgram::Axis, SpecializedAndPaddedProgram::Float, SpecializedAndPaddedProgram::Index, SpecializedAndPaddedProgram::Nat, SpecializedAndPaddedProgram::Offset, SpecializedAndPaddedProgram::_substepIx> __forall_ops;
public:
    static SpecializedAndPaddedProgram::_substepIx substepIx;
    struct _substepIx3D {
        inline SpecializedAndPaddedProgram::Float operator()(const SpecializedAndPaddedProgram::Array& u, const SpecializedAndPaddedProgram::Array& v, const SpecializedAndPaddedProgram::Array& u0, const SpecializedAndPaddedProgram::Array& u1, const SpecializedAndPaddedProgram::Array& u2, const SpecializedAndPaddedProgram::ScalarIndex& i, const SpecializedAndPaddedProgram::ScalarIndex& j, const SpecializedAndPaddedProgram::ScalarIndex& k) {
            return SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::psi(i, j, k, u), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::dt(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::nu(), SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::one.operator()<Float>(), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::binary_add(i, SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>())), j, k, v), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::binary_add(i, SpecializedAndPaddedProgram::one.operator()<Offset>()), j, k, v)), SpecializedAndPaddedProgram::psi(i, SpecializedAndPaddedProgram::binary_add(j, SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>())), k, v)), SpecializedAndPaddedProgram::psi(i, SpecializedAndPaddedProgram::binary_add(j, SpecializedAndPaddedProgram::one.operator()<Offset>()), k, v)), SpecializedAndPaddedProgram::psi(i, j, SpecializedAndPaddedProgram::binary_add(k, SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>())), v)), SpecializedAndPaddedProgram::psi(i, j, SpecializedAndPaddedProgram::binary_add(k, SpecializedAndPaddedProgram::one.operator()<Offset>()), v))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::three(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::psi(i, j, k, u0)))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::div(SpecializedAndPaddedProgram::one.operator()<Float>(), SpecializedAndPaddedProgram::two.operator()<Float>()), SpecializedAndPaddedProgram::dx()), SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::binary_add(SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::binary_add(i, SpecializedAndPaddedProgram::one.operator()<Offset>()), j, k, v), SpecializedAndPaddedProgram::psi(SpecializedAndPaddedProgram::binary_add(i, SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>())), j, k, v)), SpecializedAndPaddedProgram::psi(i, j, k, u0)), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::psi(i, SpecializedAndPaddedProgram::binary_add(j, SpecializedAndPaddedProgram::one.operator()<Offset>()), k, v), SpecializedAndPaddedProgram::psi(i, SpecializedAndPaddedProgram::binary_add(j, SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>())), k, v)), SpecializedAndPaddedProgram::psi(i, j, k, u1))), SpecializedAndPaddedProgram::mul(SpecializedAndPaddedProgram::binary_sub(SpecializedAndPaddedProgram::psi(i, j, SpecializedAndPaddedProgram::binary_add(k, SpecializedAndPaddedProgram::one.operator()<Offset>()), v), SpecializedAndPaddedProgram::psi(i, j, SpecializedAndPaddedProgram::binary_add(k, SpecializedAndPaddedProgram::unary_sub(SpecializedAndPaddedProgram::one.operator()<Offset>())), v)), SpecializedAndPaddedProgram::psi(i, j, k, u2)))))));
        };
    };

    typedef specialize_psi_ops_2<SpecializedAndPaddedProgram::Array, SpecializedAndPaddedProgram::Axis, SpecializedAndPaddedProgram::Float, SpecializedAndPaddedProgram::Index, SpecializedAndPaddedProgram::Offset, SpecializedAndPaddedProgram::ScalarIndex, SpecializedAndPaddedProgram::_substepIx3D>::AxisLength AxisLength;
    struct _mod {
        inline SpecializedAndPaddedProgram::ScalarIndex operator()(const SpecializedAndPaddedProgram::ScalarIndex& six, const SpecializedAndPaddedProgram::AxisLength& sc) {
            return __specialize_psi_ops_2.mod(six, sc);
        };
    };

    static SpecializedAndPaddedProgram::_mod mod;
    struct _shape0 {
        inline SpecializedAndPaddedProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape0();
        };
    };

    static SpecializedAndPaddedProgram::_shape0 shape0;
    struct _shape1 {
        inline SpecializedAndPaddedProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape1();
        };
    };

    static SpecializedAndPaddedProgram::_shape1 shape1;
    struct _shape2 {
        inline SpecializedAndPaddedProgram::AxisLength operator()() {
            return __specialize_psi_ops_2.shape2();
        };
    };

    static SpecializedAndPaddedProgram::_shape2 shape2;
private:
    static specialize_psi_ops_2<SpecializedAndPaddedProgram::Array, SpecializedAndPaddedProgram::Axis, SpecializedAndPaddedProgram::Float, SpecializedAndPaddedProgram::Index, SpecializedAndPaddedProgram::Offset, SpecializedAndPaddedProgram::ScalarIndex, SpecializedAndPaddedProgram::_substepIx3D> __specialize_psi_ops_2;
public:
    static SpecializedAndPaddedProgram::_substepIx3D substepIx3D;
};
} // examples
} // pde
} // mg_src
} // pde_cpp