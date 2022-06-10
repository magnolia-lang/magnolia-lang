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
        inline PDEProgram::Offset operator()(const PDEProgram::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& f) {
            return __array_ops.unary_sub(f);
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
    struct _binary_add {
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram::Array operator()(const PDEProgram::Float& num, const PDEProgram::Array& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram::Float operator()(const PDEProgram::Float& num, const PDEProgram::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram::_div div;
    struct _mul {
        inline PDEProgram::Float operator()(const PDEProgram::Float& lhs, const PDEProgram::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Float& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram::Array operator()(const PDEProgram::Array& lhs, const PDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram::_mul mul;
    struct _psi {
        inline PDEProgram::Float operator()(const PDEProgram::Index& ix, const PDEProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram::_psi psi;
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
    struct _step {
        inline void operator()(PDEProgram::Array& u0, PDEProgram::Array& u1, PDEProgram::Array& u2) {
            PDEProgram::Array v0 = u0;
            PDEProgram::Array v1 = u1;
            PDEProgram::Array v2 = u2;
            v0 = PDEProgram::schedule(v0, u0, u0, u1, u2);
            v1 = PDEProgram::schedule(v1, u1, u0, u1, u2);
            v2 = PDEProgram::schedule(v2, u2, u0, u1, u2);
            u0 = PDEProgram::schedule(u0, v0, u0, u1, u2);
            u1 = PDEProgram::schedule(u1, v1, u0, u1, u2);
            u2 = PDEProgram::schedule(u2, v2, u0, u1, u2);
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
            return PDEProgram::binary_add(PDEProgram::psi(ix, u), PDEProgram::mul(PDEProgram::div(PDEProgram::dt(), PDEProgram::two.operator()<Float>()), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::nu(), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), v)), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v)), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::mul(PDEProgram::three(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::psi(ix, u0)))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotateIx(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_substepIx>::ScalarIndex ScalarIndex;
    struct _mkIx {
        inline PDEProgram::Index operator()(const PDEProgram::ScalarIndex& a, const PDEProgram::ScalarIndex& b, const PDEProgram::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static PDEProgram::_mkIx mkIx;
private:
    static forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_substepIx> __forall_ops;
public:
    static PDEProgram::_substepIx substepIx;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram3D {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram3D::two0(o);
            return o;
        };
    };

    static PDEProgram3D::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram3D::one0(o);
            return o;
        };
    };

    static PDEProgram3D::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgram3D::Float>::Index Index;
    typedef array_ops<PDEProgram3D::Float>::Nat Nat;
    typedef array_ops<PDEProgram3D::Float>::Offset Offset;
private:
    static inline void one0(PDEProgram3D::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<PDEProgram3D::Float> __array_ops;
public:
    struct _dt {
        inline PDEProgram3D::Float operator()() {
            return __constants.dt();
        };
    };

    static PDEProgram3D::_dt dt;
    struct _dx {
        inline PDEProgram3D::Float operator()() {
            return __constants.dx();
        };
    };

    static PDEProgram3D::_dx dx;
    struct _nu {
        inline PDEProgram3D::Float operator()() {
            return __constants.nu();
        };
    };

    static PDEProgram3D::_nu nu;
    struct _three {
        inline PDEProgram3D::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram3D::_three three;
    struct _unary_sub {
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgram3D::Offset operator()(const PDEProgram3D::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static PDEProgram3D::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram3D::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram3D::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgram3D::Float>::Axis Axis;
    struct _rotateIx {
        inline PDEProgram3D::Index operator()(const PDEProgram3D::Index& ix, const PDEProgram3D::Axis& axis, const PDEProgram3D::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static PDEProgram3D::_rotateIx rotateIx;
    struct _rotateIxPadded {
        inline PDEProgram3D::Index operator()(const PDEProgram3D::Index& ix, const PDEProgram3D::Axis& axis, const PDEProgram3D::Offset& o) {
            return __specialize_psi_ops_2.rotateIxPadded(ix, axis, o);
        };
    };

    static PDEProgram3D::_rotateIxPadded rotateIxPadded;
    struct _zero {
        inline PDEProgram3D::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram3D::_zero zero;
private:
    static inline void one0(PDEProgram3D::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram3D::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgram3D::Float>::Array Array;
    struct _binary_sub {
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram3D::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& num, const PDEProgram3D::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& num, const PDEProgram3D::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram3D::_div div;
    struct _mul {
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram3D::_mul mul;
    struct _refillPadding {
        inline void operator()(PDEProgram3D::Array& a) {
            return __specialize_psi_ops_2.refillPadding(a);
        };
    };

    static PDEProgram3D::_refillPadding refillPadding;
    struct _rotate {
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& a, const PDEProgram3D::Axis& axis, const PDEProgram3D::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram3D::_rotate rotate;
    struct _schedule {
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    static PDEProgram3D::_schedule schedule;
    struct _schedule3D {
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2) {
            return __specialize_psi_ops_2.schedule3DPadded(u, v, u0, u1, u2);
        };
    };

    static PDEProgram3D::_schedule3D schedule3D;
    struct _step {
        inline void operator()(PDEProgram3D::Array& u0, PDEProgram3D::Array& u1, PDEProgram3D::Array& u2) {
            PDEProgram3D::Array v0 = u0;
            PDEProgram3D::Array v1 = u1;
            PDEProgram3D::Array v2 = u2;
            v0 = PDEProgram3D::schedule3D(v0, u0, u0, u1, u2);
            v1 = PDEProgram3D::schedule3D(v1, u1, u0, u1, u2);
            v2 = PDEProgram3D::schedule3D(v2, u2, u0, u1, u2);
            u0 = PDEProgram3D::schedule3D(u0, v0, u0, u1, u2);
            u1 = PDEProgram3D::schedule3D(u1, v1, u0, u1, u2);
            u2 = PDEProgram3D::schedule3D(u2, v2, u0, u1, u2);
        };
    };

    static PDEProgram3D::_step step;
    struct _substep {
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2) {
            return PDEProgram3D::binary_add(u, PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::dt(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::binary_sub(PDEProgram3D::mul(PDEProgram3D::nu(), PDEProgram3D::binary_sub(PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::one.operator()<Float>(), PDEProgram3D::dx()), PDEProgram3D::dx()), PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::rotate(v, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::rotate(v, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::rotate(v, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::rotate(v, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::rotate(v, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::rotate(v, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::mul(PDEProgram3D::three(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::dx()), PDEProgram3D::dx()), u0))), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::one.operator()<Float>(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::dx()), PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::rotate(v, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::rotate(v, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), u0), PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::rotate(v, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::rotate(v, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), u1)), PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::rotate(v, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::rotate(v, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram3D::_substep substep;
    struct _substepIx {
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2, const PDEProgram3D::Index& ix) {
            return PDEProgram3D::binary_add(PDEProgram3D::psi(PDEProgram3D::ix0(ix), PDEProgram3D::ix1(ix), PDEProgram3D::ix2(ix), u), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::dt(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::binary_sub(PDEProgram3D::mul(PDEProgram3D::nu(), PDEProgram3D::binary_sub(PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::one.operator()<Float>(), PDEProgram3D::dx()), PDEProgram3D::dx()), PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), v), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>())), v)), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), v)), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), v)), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), v)), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), v))), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::mul(PDEProgram3D::three(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::dx()), PDEProgram3D::dx()), PDEProgram3D::psi(PDEProgram3D::ix0(ix), PDEProgram3D::ix1(ix), PDEProgram3D::ix2(ix), u0)))), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::one.operator()<Float>(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::dx()), PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::one.operator()<Offset>())), v), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::zero(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), v)), PDEProgram3D::psi(PDEProgram3D::ix0(ix), PDEProgram3D::ix1(ix), PDEProgram3D::ix2(ix), u0)), PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), v), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::one.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), v)), PDEProgram3D::psi(PDEProgram3D::ix0(ix), PDEProgram3D::ix1(ix), PDEProgram3D::ix2(ix), u1))), PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::one.operator()<Offset>())), v), PDEProgram3D::psi(PDEProgram3D::ix0(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix1(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), PDEProgram3D::ix2(PDEProgram3D::rotateIx(ix, PDEProgram3D::two.operator()<Axis>(), PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>()))), v)), PDEProgram3D::psi(PDEProgram3D::ix0(ix), PDEProgram3D::ix1(ix), PDEProgram3D::ix2(ix), u2)))))));
        };
    };

    typedef forall_ops<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Nat, PDEProgram3D::Offset, PDEProgram3D::_substepIx>::ScalarIndex ScalarIndex;
    struct _binary_add {
        inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::ScalarIndex& six, const PDEProgram3D::Offset& o) {
            return __specialize_psi_ops_2.binary_add(six, o);
        };
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram3D::_binary_add binary_add;
    struct _ix0 {
        inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::Index& ix) {
            return __specialize_psi_ops_2.ix0(ix);
        };
    };

    static PDEProgram3D::_ix0 ix0;
    struct _ix1 {
        inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::Index& ix) {
            return __specialize_psi_ops_2.ix1(ix);
        };
    };

    static PDEProgram3D::_ix1 ix1;
    struct _ix2 {
        inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::Index& ix) {
            return __specialize_psi_ops_2.ix2(ix);
        };
    };

    static PDEProgram3D::_ix2 ix2;
    struct _mkIx {
        inline PDEProgram3D::Index operator()(const PDEProgram3D::ScalarIndex& a, const PDEProgram3D::ScalarIndex& b, const PDEProgram3D::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static PDEProgram3D::_mkIx mkIx;
    struct _psi {
        inline PDEProgram3D::Float operator()(const PDEProgram3D::ScalarIndex& i, const PDEProgram3D::ScalarIndex& j, const PDEProgram3D::ScalarIndex& k, const PDEProgram3D::Array& a) {
            return __specialize_psi_ops_2.psi(i, j, k, a);
        };
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Index& ix, const PDEProgram3D::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram3D::_psi psi;
private:
    static forall_ops<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Nat, PDEProgram3D::Offset, PDEProgram3D::_substepIx> __forall_ops;
public:
    static PDEProgram3D::_substepIx substepIx;
    struct _substepIx3D {
        inline PDEProgram3D::Float operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2, const PDEProgram3D::ScalarIndex& i, const PDEProgram3D::ScalarIndex& j, const PDEProgram3D::ScalarIndex& k) {
            return PDEProgram3D::binary_add(PDEProgram3D::psi(i, j, k, u), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::dt(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::binary_sub(PDEProgram3D::mul(PDEProgram3D::nu(), PDEProgram3D::binary_sub(PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::one.operator()<Float>(), PDEProgram3D::dx()), PDEProgram3D::dx()), PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::psi(PDEProgram3D::mod(PDEProgram3D::binary_add(i, PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::shape0()), j, k, v), PDEProgram3D::psi(PDEProgram3D::mod(PDEProgram3D::binary_add(i, PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::shape0()), j, k, v)), PDEProgram3D::psi(i, PDEProgram3D::mod(PDEProgram3D::binary_add(j, PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::shape1()), k, v)), PDEProgram3D::psi(i, PDEProgram3D::mod(PDEProgram3D::binary_add(j, PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::shape1()), k, v)), PDEProgram3D::psi(i, j, PDEProgram3D::mod(PDEProgram3D::binary_add(k, PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::shape2()), v)), PDEProgram3D::psi(i, j, PDEProgram3D::mod(PDEProgram3D::binary_add(k, PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::shape2()), v))), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::mul(PDEProgram3D::three(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::dx()), PDEProgram3D::dx()), PDEProgram3D::psi(i, j, k, u0)))), PDEProgram3D::mul(PDEProgram3D::div(PDEProgram3D::div(PDEProgram3D::one.operator()<Float>(), PDEProgram3D::two.operator()<Float>()), PDEProgram3D::dx()), PDEProgram3D::binary_add(PDEProgram3D::binary_add(PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::psi(PDEProgram3D::mod(PDEProgram3D::binary_add(i, PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::shape0()), j, k, v), PDEProgram3D::psi(PDEProgram3D::mod(PDEProgram3D::binary_add(i, PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::shape0()), j, k, v)), PDEProgram3D::psi(i, j, k, u0)), PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::psi(i, PDEProgram3D::mod(PDEProgram3D::binary_add(j, PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::shape1()), k, v), PDEProgram3D::psi(i, PDEProgram3D::mod(PDEProgram3D::binary_add(j, PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::shape1()), k, v)), PDEProgram3D::psi(i, j, k, u1))), PDEProgram3D::mul(PDEProgram3D::binary_sub(PDEProgram3D::psi(i, j, PDEProgram3D::mod(PDEProgram3D::binary_add(k, PDEProgram3D::one.operator()<Offset>()), PDEProgram3D::shape2()), v), PDEProgram3D::psi(i, j, PDEProgram3D::mod(PDEProgram3D::binary_add(k, PDEProgram3D::unary_sub(PDEProgram3D::one.operator()<Offset>())), PDEProgram3D::shape2()), v)), PDEProgram3D::psi(i, j, k, u2)))))));
        };
    };

    typedef specialize_psi_ops_2<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Offset, PDEProgram3D::ScalarIndex, PDEProgram3D::_substepIx3D>::AxisLength AxisLength;
    struct _mod {
        inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::ScalarIndex& six, const PDEProgram3D::AxisLength& sc) {
            return __specialize_psi_ops_2.mod(six, sc);
        };
    };

    static PDEProgram3D::_mod mod;
    struct _shape0 {
        inline PDEProgram3D::AxisLength operator()() {
            return __specialize_psi_ops_2.shape0();
        };
    };

    static PDEProgram3D::_shape0 shape0;
    struct _shape1 {
        inline PDEProgram3D::AxisLength operator()() {
            return __specialize_psi_ops_2.shape1();
        };
    };

    static PDEProgram3D::_shape1 shape1;
    struct _shape2 {
        inline PDEProgram3D::AxisLength operator()() {
            return __specialize_psi_ops_2.shape2();
        };
    };

    static PDEProgram3D::_shape2 shape2;
private:
    static specialize_psi_ops_2<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Offset, PDEProgram3D::ScalarIndex, PDEProgram3D::_substepIx3D> __specialize_psi_ops_2;
public:
    static PDEProgram3D::_substepIx3D substepIx3D;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgram3DPadded {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram3DPadded::two0(o);
            return o;
        };
    };

    static PDEProgram3DPadded::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgram3DPadded::one0(o);
            return o;
        };
    };

    static PDEProgram3DPadded::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgram3DPadded::Float>::Index Index;
    typedef array_ops<PDEProgram3DPadded::Float>::Nat Nat;
    typedef array_ops<PDEProgram3DPadded::Float>::Offset Offset;
private:
    static inline void one0(PDEProgram3DPadded::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<PDEProgram3DPadded::Float> __array_ops;
public:
    struct _dt {
        inline PDEProgram3DPadded::Float operator()() {
            return __constants.dt();
        };
    };

    static PDEProgram3DPadded::_dt dt;
    struct _dx {
        inline PDEProgram3DPadded::Float operator()() {
            return __constants.dx();
        };
    };

    static PDEProgram3DPadded::_dx dx;
    struct _nu {
        inline PDEProgram3DPadded::Float operator()() {
            return __constants.nu();
        };
    };

    static PDEProgram3DPadded::_nu nu;
    struct _three {
        inline PDEProgram3DPadded::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgram3DPadded::_three three;
    struct _unary_sub {
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgram3DPadded::Offset operator()(const PDEProgram3DPadded::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static PDEProgram3DPadded::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgram3DPadded::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgram3DPadded::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgram3DPadded::Float>::Axis Axis;
    struct _rotateIx {
        inline PDEProgram3DPadded::Index operator()(const PDEProgram3DPadded::Index& ix, const PDEProgram3DPadded::Axis& axis, const PDEProgram3DPadded::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static PDEProgram3DPadded::_rotateIx rotateIx;
    struct _rotateIxPadded {
        inline PDEProgram3DPadded::Index operator()(const PDEProgram3DPadded::Index& ix, const PDEProgram3DPadded::Axis& axis, const PDEProgram3DPadded::Offset& o) {
            return __specialize_psi_ops_2.rotateIxPadded(ix, axis, o);
        };
    };

    static PDEProgram3DPadded::_rotateIxPadded rotateIxPadded;
    struct _zero {
        inline PDEProgram3DPadded::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgram3DPadded::_zero zero;
private:
    static inline void one0(PDEProgram3DPadded::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgram3DPadded::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgram3DPadded::Float>::Array Array;
    struct _binary_sub {
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgram3DPadded::_binary_sub binary_sub;
    struct _div {
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& num, const PDEProgram3DPadded::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& num, const PDEProgram3DPadded::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgram3DPadded::_div div;
    struct _mul {
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgram3DPadded::_mul mul;
    struct _refillPadding {
        inline void operator()(PDEProgram3DPadded::Array& a) {
            return __specialize_psi_ops_2.refillPadding(a);
        };
    };

    static PDEProgram3DPadded::_refillPadding refillPadding;
    struct _rotate {
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& a, const PDEProgram3DPadded::Axis& axis, const PDEProgram3DPadded::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgram3DPadded::_rotate rotate;
    struct _schedule {
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    static PDEProgram3DPadded::_schedule schedule;
    struct _schedule3DPadded {
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2) {
            return __specialize_psi_ops_2.schedule3DPadded(u, v, u0, u1, u2);
        };
    };

    static PDEProgram3DPadded::_schedule3DPadded schedule3DPadded;
    struct _step {
        inline void operator()(PDEProgram3DPadded::Array& u0, PDEProgram3DPadded::Array& u1, PDEProgram3DPadded::Array& u2) {
            PDEProgram3DPadded::Array v0 = u0;
            PDEProgram3DPadded::Array v1 = u1;
            PDEProgram3DPadded::Array v2 = u2;
            v0 = [&]() {
                PDEProgram3DPadded::Array result = PDEProgram3DPadded::schedule3DPadded(v0, u0, u0, u1, u2);
                PDEProgram3DPadded::refillPadding(result);
                return result;
            }();
            v1 = [&]() {
                PDEProgram3DPadded::Array result = PDEProgram3DPadded::schedule3DPadded(v1, u1, u0, u1, u2);
                PDEProgram3DPadded::refillPadding(result);
                return result;
            }();
            v2 = [&]() {
                PDEProgram3DPadded::Array result = PDEProgram3DPadded::schedule3DPadded(v2, u2, u0, u1, u2);
                PDEProgram3DPadded::refillPadding(result);
                return result;
            }();
            u0 = [&]() {
                PDEProgram3DPadded::Array result = PDEProgram3DPadded::schedule3DPadded(u0, v0, u0, u1, u2);
                PDEProgram3DPadded::refillPadding(result);
                return result;
            }();
            u1 = [&]() {
                PDEProgram3DPadded::Array result = PDEProgram3DPadded::schedule3DPadded(u1, v1, u0, u1, u2);
                PDEProgram3DPadded::refillPadding(result);
                return result;
            }();
            u2 = [&]() {
                PDEProgram3DPadded::Array result = PDEProgram3DPadded::schedule3DPadded(u2, v2, u0, u1, u2);
                PDEProgram3DPadded::refillPadding(result);
                return result;
            }();
        };
    };

    static PDEProgram3DPadded::_step step;
    struct _substep {
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2) {
            return PDEProgram3DPadded::binary_add(u, PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::dt(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::mul(PDEProgram3DPadded::nu(), PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::one.operator()<Float>(), PDEProgram3DPadded::dx()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::mul(PDEProgram3DPadded::three(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::dx()), u0))), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::one.operator()<Float>(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>()), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), u0), PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>()), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), u1)), PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>()), PDEProgram3DPadded::rotate(v, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram3DPadded::_substep substep;
    struct _substepIx {
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2, const PDEProgram3DPadded::Index& ix) {
            return PDEProgram3DPadded::binary_add(PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(ix), PDEProgram3DPadded::ix1(ix), PDEProgram3DPadded::ix2(ix), u), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::dt(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::mul(PDEProgram3DPadded::nu(), PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::one.operator()<Float>(), PDEProgram3DPadded::dx()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), v), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>())), v)), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), v)), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), v)), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), v)), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), v))), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::mul(PDEProgram3DPadded::three(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(ix), PDEProgram3DPadded::ix1(ix), PDEProgram3DPadded::ix2(ix), u0)))), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::one.operator()<Float>(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::one.operator()<Offset>())), v), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::zero(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), v)), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(ix), PDEProgram3DPadded::ix1(ix), PDEProgram3DPadded::ix2(ix), u0)), PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), v), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::one.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), v)), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(ix), PDEProgram3DPadded::ix1(ix), PDEProgram3DPadded::ix2(ix), u1))), PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::one.operator()<Offset>())), v), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix1(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), PDEProgram3DPadded::ix2(PDEProgram3DPadded::rotateIxPadded(ix, PDEProgram3DPadded::two.operator()<Axis>(), PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>()))), v)), PDEProgram3DPadded::psi(PDEProgram3DPadded::ix0(ix), PDEProgram3DPadded::ix1(ix), PDEProgram3DPadded::ix2(ix), u2)))))));
        };
    };

    typedef forall_ops<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Nat, PDEProgram3DPadded::Offset, PDEProgram3DPadded::_substepIx>::ScalarIndex ScalarIndex;
    struct _binary_add {
        inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::ScalarIndex& six, const PDEProgram3DPadded::Offset& o) {
            return __specialize_psi_ops_2.binary_add(six, o);
        };
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgram3DPadded::_binary_add binary_add;
    struct _ix0 {
        inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::Index& ix) {
            return __specialize_psi_ops_2.ix0(ix);
        };
    };

    static PDEProgram3DPadded::_ix0 ix0;
    struct _ix1 {
        inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::Index& ix) {
            return __specialize_psi_ops_2.ix1(ix);
        };
    };

    static PDEProgram3DPadded::_ix1 ix1;
    struct _ix2 {
        inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::Index& ix) {
            return __specialize_psi_ops_2.ix2(ix);
        };
    };

    static PDEProgram3DPadded::_ix2 ix2;
    struct _mkIx {
        inline PDEProgram3DPadded::Index operator()(const PDEProgram3DPadded::ScalarIndex& a, const PDEProgram3DPadded::ScalarIndex& b, const PDEProgram3DPadded::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static PDEProgram3DPadded::_mkIx mkIx;
    struct _psi {
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::ScalarIndex& i, const PDEProgram3DPadded::ScalarIndex& j, const PDEProgram3DPadded::ScalarIndex& k, const PDEProgram3DPadded::Array& a) {
            return __specialize_psi_ops_2.psi(i, j, k, a);
        };
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Index& ix, const PDEProgram3DPadded::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgram3DPadded::_psi psi;
private:
    static forall_ops<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Nat, PDEProgram3DPadded::Offset, PDEProgram3DPadded::_substepIx> __forall_ops;
public:
    static PDEProgram3DPadded::_substepIx substepIx;
    struct _substepIx3D {
        inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2, const PDEProgram3DPadded::ScalarIndex& i, const PDEProgram3DPadded::ScalarIndex& j, const PDEProgram3DPadded::ScalarIndex& k) {
            return PDEProgram3DPadded::binary_add(PDEProgram3DPadded::psi(i, j, k, u), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::dt(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::mul(PDEProgram3DPadded::nu(), PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::one.operator()<Float>(), PDEProgram3DPadded::dx()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::psi(PDEProgram3DPadded::binary_add(i, PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>())), j, k, v), PDEProgram3DPadded::psi(PDEProgram3DPadded::binary_add(i, PDEProgram3DPadded::one.operator()<Offset>()), j, k, v)), PDEProgram3DPadded::psi(i, PDEProgram3DPadded::binary_add(j, PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>())), k, v)), PDEProgram3DPadded::psi(i, PDEProgram3DPadded::binary_add(j, PDEProgram3DPadded::one.operator()<Offset>()), k, v)), PDEProgram3DPadded::psi(i, j, PDEProgram3DPadded::binary_add(k, PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>())), v)), PDEProgram3DPadded::psi(i, j, PDEProgram3DPadded::binary_add(k, PDEProgram3DPadded::one.operator()<Offset>()), v))), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::mul(PDEProgram3DPadded::three(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::psi(i, j, k, u0)))), PDEProgram3DPadded::mul(PDEProgram3DPadded::div(PDEProgram3DPadded::div(PDEProgram3DPadded::one.operator()<Float>(), PDEProgram3DPadded::two.operator()<Float>()), PDEProgram3DPadded::dx()), PDEProgram3DPadded::binary_add(PDEProgram3DPadded::binary_add(PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::psi(PDEProgram3DPadded::binary_add(i, PDEProgram3DPadded::one.operator()<Offset>()), j, k, v), PDEProgram3DPadded::psi(PDEProgram3DPadded::binary_add(i, PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>())), j, k, v)), PDEProgram3DPadded::psi(i, j, k, u0)), PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::psi(i, PDEProgram3DPadded::binary_add(j, PDEProgram3DPadded::one.operator()<Offset>()), k, v), PDEProgram3DPadded::psi(i, PDEProgram3DPadded::binary_add(j, PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>())), k, v)), PDEProgram3DPadded::psi(i, j, k, u1))), PDEProgram3DPadded::mul(PDEProgram3DPadded::binary_sub(PDEProgram3DPadded::psi(i, j, PDEProgram3DPadded::binary_add(k, PDEProgram3DPadded::one.operator()<Offset>()), v), PDEProgram3DPadded::psi(i, j, PDEProgram3DPadded::binary_add(k, PDEProgram3DPadded::unary_sub(PDEProgram3DPadded::one.operator()<Offset>())), v)), PDEProgram3DPadded::psi(i, j, k, u2)))))));
        };
    };

    typedef specialize_psi_ops_2<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex, PDEProgram3DPadded::_substepIx3D>::AxisLength AxisLength;
    struct _mod {
        inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::ScalarIndex& six, const PDEProgram3DPadded::AxisLength& sc) {
            return __specialize_psi_ops_2.mod(six, sc);
        };
    };

    static PDEProgram3DPadded::_mod mod;
    struct _shape0 {
        inline PDEProgram3DPadded::AxisLength operator()() {
            return __specialize_psi_ops_2.shape0();
        };
    };

    static PDEProgram3DPadded::_shape0 shape0;
    struct _shape1 {
        inline PDEProgram3DPadded::AxisLength operator()() {
            return __specialize_psi_ops_2.shape1();
        };
    };

    static PDEProgram3DPadded::_shape1 shape1;
    struct _shape2 {
        inline PDEProgram3DPadded::AxisLength operator()() {
            return __specialize_psi_ops_2.shape2();
        };
    };

    static PDEProgram3DPadded::_shape2 shape2;
private:
    static specialize_psi_ops_2<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex, PDEProgram3DPadded::_substepIx3D> __specialize_psi_ops_2;
public:
    static PDEProgram3DPadded::_substepIx3D substepIx3D;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgramDNF {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgramDNF::two0(o);
            return o;
        };
    };

    static PDEProgramDNF::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgramDNF::one0(o);
            return o;
        };
    };

    static PDEProgramDNF::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgramDNF::Float>::Index Index;
    typedef array_ops<PDEProgramDNF::Float>::Nat Nat;
    typedef array_ops<PDEProgramDNF::Float>::Offset Offset;
private:
    static inline void one0(PDEProgramDNF::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<PDEProgramDNF::Float> __array_ops;
public:
    struct _dt {
        inline PDEProgramDNF::Float operator()() {
            return __constants.dt();
        };
    };

    static PDEProgramDNF::_dt dt;
    struct _dx {
        inline PDEProgramDNF::Float operator()() {
            return __constants.dx();
        };
    };

    static PDEProgramDNF::_dx dx;
    struct _nu {
        inline PDEProgramDNF::Float operator()() {
            return __constants.nu();
        };
    };

    static PDEProgramDNF::_nu nu;
    struct _three {
        inline PDEProgramDNF::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgramDNF::_three three;
    struct _unary_sub {
        inline PDEProgramDNF::Offset operator()(const PDEProgramDNF::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& f) {
            return __array_ops.unary_sub(f);
        };
    };

    static PDEProgramDNF::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgramDNF::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgramDNF::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgramDNF::Float>::Axis Axis;
    struct _rotateIx {
        inline PDEProgramDNF::Index operator()(const PDEProgramDNF::Index& ix, const PDEProgramDNF::Axis& axis, const PDEProgramDNF::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static PDEProgramDNF::_rotateIx rotateIx;
    struct _zero {
        inline PDEProgramDNF::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgramDNF::_zero zero;
private:
    static inline void one0(PDEProgramDNF::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgramDNF::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgramDNF::Float>::Array Array;
    struct _binary_add {
        inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgramDNF::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgramDNF::_binary_sub binary_sub;
    struct _div {
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& num, const PDEProgramDNF::Array& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& num, const PDEProgramDNF::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgramDNF::_div div;
    struct _mul {
        inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgramDNF::_mul mul;
    struct _psi {
        inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Index& ix, const PDEProgramDNF::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgramDNF::_psi psi;
    struct _rotate {
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& a, const PDEProgramDNF::Axis& axis, const PDEProgramDNF::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgramDNF::_rotate rotate;
    struct _schedule {
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& u, const PDEProgramDNF::Array& v, const PDEProgramDNF::Array& u0, const PDEProgramDNF::Array& u1, const PDEProgramDNF::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    static PDEProgramDNF::_schedule schedule;
    struct _step {
        inline void operator()(PDEProgramDNF::Array& u0, PDEProgramDNF::Array& u1, PDEProgramDNF::Array& u2) {
            PDEProgramDNF::Array v0 = u0;
            PDEProgramDNF::Array v1 = u1;
            PDEProgramDNF::Array v2 = u2;
            v0 = PDEProgramDNF::schedule(v0, u0, u0, u1, u2);
            v1 = PDEProgramDNF::schedule(v1, u1, u0, u1, u2);
            v2 = PDEProgramDNF::schedule(v2, u2, u0, u1, u2);
            u0 = PDEProgramDNF::schedule(u0, v0, u0, u1, u2);
            u1 = PDEProgramDNF::schedule(u1, v1, u0, u1, u2);
            u2 = PDEProgramDNF::schedule(u2, v2, u0, u1, u2);
        };
    };

    static PDEProgramDNF::_step step;
    struct _substep {
        inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& u, const PDEProgramDNF::Array& v, const PDEProgramDNF::Array& u0, const PDEProgramDNF::Array& u1, const PDEProgramDNF::Array& u2) {
            return PDEProgramDNF::binary_add(u, PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::dt(), PDEProgramDNF::two.operator()<Float>()), PDEProgramDNF::binary_sub(PDEProgramDNF::mul(PDEProgramDNF::nu(), PDEProgramDNF::binary_sub(PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::div(PDEProgramDNF::one.operator()<Float>(), PDEProgramDNF::dx()), PDEProgramDNF::dx()), PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::rotate(v, PDEProgramDNF::zero(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>())), PDEProgramDNF::rotate(v, PDEProgramDNF::zero(), PDEProgramDNF::one.operator()<Offset>())), PDEProgramDNF::rotate(v, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>()))), PDEProgramDNF::rotate(v, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>())), PDEProgramDNF::rotate(v, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>()))), PDEProgramDNF::rotate(v, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>()))), PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::div(PDEProgramDNF::mul(PDEProgramDNF::three(), PDEProgramDNF::two.operator()<Float>()), PDEProgramDNF::dx()), PDEProgramDNF::dx()), u0))), PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::div(PDEProgramDNF::one.operator()<Float>(), PDEProgramDNF::two.operator()<Float>()), PDEProgramDNF::dx()), PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::mul(PDEProgramDNF::binary_sub(PDEProgramDNF::rotate(v, PDEProgramDNF::zero(), PDEProgramDNF::one.operator()<Offset>()), PDEProgramDNF::rotate(v, PDEProgramDNF::zero(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>()))), u0), PDEProgramDNF::mul(PDEProgramDNF::binary_sub(PDEProgramDNF::rotate(v, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>()), PDEProgramDNF::rotate(v, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>()))), u1)), PDEProgramDNF::mul(PDEProgramDNF::binary_sub(PDEProgramDNF::rotate(v, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>()), PDEProgramDNF::rotate(v, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgramDNF::_substep substep;
    struct _substepIx {
        inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Array& u, const PDEProgramDNF::Array& v, const PDEProgramDNF::Array& u0, const PDEProgramDNF::Array& u1, const PDEProgramDNF::Array& u2, const PDEProgramDNF::Index& ix) {
            return PDEProgramDNF::binary_add(PDEProgramDNF::psi(ix, u), PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::dt(), PDEProgramDNF::two.operator()<Float>()), PDEProgramDNF::binary_sub(PDEProgramDNF::mul(PDEProgramDNF::nu(), PDEProgramDNF::binary_sub(PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::div(PDEProgramDNF::one.operator()<Float>(), PDEProgramDNF::dx()), PDEProgramDNF::dx()), PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::zero(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>())), v), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::zero(), PDEProgramDNF::one.operator()<Offset>()), v)), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>())), v)), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>()), v)), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>())), v)), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>()), v))), PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::div(PDEProgramDNF::mul(PDEProgramDNF::three(), PDEProgramDNF::two.operator()<Float>()), PDEProgramDNF::dx()), PDEProgramDNF::dx()), PDEProgramDNF::psi(ix, u0)))), PDEProgramDNF::mul(PDEProgramDNF::div(PDEProgramDNF::div(PDEProgramDNF::one.operator()<Float>(), PDEProgramDNF::two.operator()<Float>()), PDEProgramDNF::dx()), PDEProgramDNF::binary_add(PDEProgramDNF::binary_add(PDEProgramDNF::mul(PDEProgramDNF::binary_sub(PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::zero(), PDEProgramDNF::one.operator()<Offset>()), v), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::zero(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>())), v)), PDEProgramDNF::psi(ix, u0)), PDEProgramDNF::mul(PDEProgramDNF::binary_sub(PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>()), v), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::one.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>())), v)), PDEProgramDNF::psi(ix, u1))), PDEProgramDNF::mul(PDEProgramDNF::binary_sub(PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::one.operator()<Offset>()), v), PDEProgramDNF::psi(PDEProgramDNF::rotateIx(ix, PDEProgramDNF::two.operator()<Axis>(), PDEProgramDNF::unary_sub(PDEProgramDNF::one.operator()<Offset>())), v)), PDEProgramDNF::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgramDNF::Array, PDEProgramDNF::Axis, PDEProgramDNF::Float, PDEProgramDNF::Index, PDEProgramDNF::Nat, PDEProgramDNF::Offset, PDEProgramDNF::_substepIx>::ScalarIndex ScalarIndex;
    struct _mkIx {
        inline PDEProgramDNF::Index operator()(const PDEProgramDNF::ScalarIndex& a, const PDEProgramDNF::ScalarIndex& b, const PDEProgramDNF::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static PDEProgramDNF::_mkIx mkIx;
private:
    static forall_ops<PDEProgramDNF::Array, PDEProgramDNF::Axis, PDEProgramDNF::Float, PDEProgramDNF::Index, PDEProgramDNF::Nat, PDEProgramDNF::Offset, PDEProgramDNF::_substepIx> __forall_ops;
public:
    static PDEProgramDNF::_substepIx substepIx;
};
} // examples
} // pde
} // mg_src
} // pde_cpp

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cpp {
struct PDEProgramPadded {
    struct _two {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgramPadded::two0(o);
            return o;
        };
    };

    static PDEProgramPadded::_two two;
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            PDEProgramPadded::one0(o);
            return o;
        };
    };

    static PDEProgramPadded::_one one;
private:
    static constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgramPadded::Float>::Index Index;
    typedef array_ops<PDEProgramPadded::Float>::Nat Nat;
    typedef array_ops<PDEProgramPadded::Float>::Offset Offset;
private:
    static inline void one0(PDEProgramPadded::Offset& o) {
        o = __array_ops.one_offset();
    };
    static array_ops<PDEProgramPadded::Float> __array_ops;
public:
    struct _dt {
        inline PDEProgramPadded::Float operator()() {
            return __constants.dt();
        };
    };

    static PDEProgramPadded::_dt dt;
    struct _dx {
        inline PDEProgramPadded::Float operator()() {
            return __constants.dx();
        };
    };

    static PDEProgramPadded::_dx dx;
    struct _nu {
        inline PDEProgramPadded::Float operator()() {
            return __constants.nu();
        };
    };

    static PDEProgramPadded::_nu nu;
    struct _three {
        inline PDEProgramPadded::Float operator()() {
            return __array_ops.three_float();
        };
    };

    static PDEProgramPadded::_three three;
    struct _unary_sub {
        inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& f) {
            return __array_ops.unary_sub(f);
        };
        inline PDEProgramPadded::Offset operator()(const PDEProgramPadded::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    static PDEProgramPadded::_unary_sub unary_sub;
private:
    static inline void one0(PDEProgramPadded::Float& o) {
        o = __array_ops.one_float();
    };
    static inline void two0(PDEProgramPadded::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgramPadded::Float>::Axis Axis;
    struct _rotateIx {
        inline PDEProgramPadded::Index operator()(const PDEProgramPadded::Index& ix, const PDEProgramPadded::Axis& axis, const PDEProgramPadded::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    static PDEProgramPadded::_rotateIx rotateIx;
    struct _rotateIxPadded {
        inline PDEProgramPadded::Index operator()(const PDEProgramPadded::Index& ix, const PDEProgramPadded::Axis& axis, const PDEProgramPadded::Offset& offset) {
            return __forall_ops.rotateIxPadded(ix, axis, offset);
        };
    };

    static PDEProgramPadded::_rotateIxPadded rotateIxPadded;
    struct _zero {
        inline PDEProgramPadded::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    static PDEProgramPadded::_zero zero;
private:
    static inline void one0(PDEProgramPadded::Axis& o) {
        o = __array_ops.one_axis();
    };
    static inline void two0(PDEProgramPadded::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgramPadded::Float>::Array Array;
    struct _binary_add {
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    static PDEProgramPadded::_binary_add binary_add;
    struct _binary_sub {
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    static PDEProgramPadded::_binary_sub binary_sub;
    struct _div {
        inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& num, const PDEProgramPadded::Float& den) {
            return __array_ops.div(num, den);
        };
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& num, const PDEProgramPadded::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    static PDEProgramPadded::_div div;
    struct _mul {
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    static PDEProgramPadded::_mul mul;
    struct _psi {
        inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Index& ix, const PDEProgramPadded::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    static PDEProgramPadded::_psi psi;
    struct _refillPadding {
        inline void operator()(PDEProgramPadded::Array& a) {
            return __forall_ops.refillPadding(a);
        };
    };

    static PDEProgramPadded::_refillPadding refillPadding;
    struct _rotate {
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& a, const PDEProgramPadded::Axis& axis, const PDEProgramPadded::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    static PDEProgramPadded::_rotate rotate;
    struct _schedule {
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    static PDEProgramPadded::_schedule schedule;
    struct _schedulePadded {
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2) {
            return __forall_ops.schedulePadded(u, v, u0, u1, u2);
        };
    };

    static PDEProgramPadded::_schedulePadded schedulePadded;
    struct _step {
        inline void operator()(PDEProgramPadded::Array& u0, PDEProgramPadded::Array& u1, PDEProgramPadded::Array& u2) {
            PDEProgramPadded::Array v0 = u0;
            PDEProgramPadded::Array v1 = u1;
            PDEProgramPadded::Array v2 = u2;
            v0 = [&]() {
                PDEProgramPadded::Array result = PDEProgramPadded::schedulePadded(v0, u0, u0, u1, u2);
                PDEProgramPadded::refillPadding(result);
                return result;
            }();
            v1 = [&]() {
                PDEProgramPadded::Array result = PDEProgramPadded::schedulePadded(v1, u1, u0, u1, u2);
                PDEProgramPadded::refillPadding(result);
                return result;
            }();
            v2 = [&]() {
                PDEProgramPadded::Array result = PDEProgramPadded::schedulePadded(v2, u2, u0, u1, u2);
                PDEProgramPadded::refillPadding(result);
                return result;
            }();
            u0 = [&]() {
                PDEProgramPadded::Array result = PDEProgramPadded::schedulePadded(u0, v0, u0, u1, u2);
                PDEProgramPadded::refillPadding(result);
                return result;
            }();
            u1 = [&]() {
                PDEProgramPadded::Array result = PDEProgramPadded::schedulePadded(u1, v1, u0, u1, u2);
                PDEProgramPadded::refillPadding(result);
                return result;
            }();
            u2 = [&]() {
                PDEProgramPadded::Array result = PDEProgramPadded::schedulePadded(u2, v2, u0, u1, u2);
                PDEProgramPadded::refillPadding(result);
                return result;
            }();
        };
    };

    static PDEProgramPadded::_step step;
    struct _substep {
        inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2) {
            return PDEProgramPadded::binary_add(u, PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::dt(), PDEProgramPadded::two.operator()<Float>()), PDEProgramPadded::binary_sub(PDEProgramPadded::mul(PDEProgramPadded::nu(), PDEProgramPadded::binary_sub(PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::div(PDEProgramPadded::one.operator()<Float>(), PDEProgramPadded::dx()), PDEProgramPadded::dx()), PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::rotate(v, PDEProgramPadded::zero(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>())), PDEProgramPadded::rotate(v, PDEProgramPadded::zero(), PDEProgramPadded::one.operator()<Offset>())), PDEProgramPadded::rotate(v, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>()))), PDEProgramPadded::rotate(v, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>())), PDEProgramPadded::rotate(v, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>()))), PDEProgramPadded::rotate(v, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>()))), PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::div(PDEProgramPadded::mul(PDEProgramPadded::three(), PDEProgramPadded::two.operator()<Float>()), PDEProgramPadded::dx()), PDEProgramPadded::dx()), u0))), PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::div(PDEProgramPadded::one.operator()<Float>(), PDEProgramPadded::two.operator()<Float>()), PDEProgramPadded::dx()), PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::mul(PDEProgramPadded::binary_sub(PDEProgramPadded::rotate(v, PDEProgramPadded::zero(), PDEProgramPadded::one.operator()<Offset>()), PDEProgramPadded::rotate(v, PDEProgramPadded::zero(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>()))), u0), PDEProgramPadded::mul(PDEProgramPadded::binary_sub(PDEProgramPadded::rotate(v, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>()), PDEProgramPadded::rotate(v, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>()))), u1)), PDEProgramPadded::mul(PDEProgramPadded::binary_sub(PDEProgramPadded::rotate(v, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>()), PDEProgramPadded::rotate(v, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgramPadded::_substep substep;
    struct _substepIx {
        inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2, const PDEProgramPadded::Index& ix) {
            return PDEProgramPadded::binary_add(PDEProgramPadded::psi(ix, u), PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::dt(), PDEProgramPadded::two.operator()<Float>()), PDEProgramPadded::binary_sub(PDEProgramPadded::mul(PDEProgramPadded::nu(), PDEProgramPadded::binary_sub(PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::div(PDEProgramPadded::one.operator()<Float>(), PDEProgramPadded::dx()), PDEProgramPadded::dx()), PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::zero(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>())), v), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::zero(), PDEProgramPadded::one.operator()<Offset>()), v)), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>())), v)), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>()), v)), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>())), v)), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>()), v))), PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::div(PDEProgramPadded::mul(PDEProgramPadded::three(), PDEProgramPadded::two.operator()<Float>()), PDEProgramPadded::dx()), PDEProgramPadded::dx()), PDEProgramPadded::psi(ix, u0)))), PDEProgramPadded::mul(PDEProgramPadded::div(PDEProgramPadded::div(PDEProgramPadded::one.operator()<Float>(), PDEProgramPadded::two.operator()<Float>()), PDEProgramPadded::dx()), PDEProgramPadded::binary_add(PDEProgramPadded::binary_add(PDEProgramPadded::mul(PDEProgramPadded::binary_sub(PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::zero(), PDEProgramPadded::one.operator()<Offset>()), v), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::zero(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>())), v)), PDEProgramPadded::psi(ix, u0)), PDEProgramPadded::mul(PDEProgramPadded::binary_sub(PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>()), v), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::one.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>())), v)), PDEProgramPadded::psi(ix, u1))), PDEProgramPadded::mul(PDEProgramPadded::binary_sub(PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::one.operator()<Offset>()), v), PDEProgramPadded::psi(PDEProgramPadded::rotateIxPadded(ix, PDEProgramPadded::two.operator()<Axis>(), PDEProgramPadded::unary_sub(PDEProgramPadded::one.operator()<Offset>())), v)), PDEProgramPadded::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgramPadded::Array, PDEProgramPadded::Axis, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::Nat, PDEProgramPadded::Offset, PDEProgramPadded::_substepIx>::ScalarIndex ScalarIndex;
    struct _mkIx {
        inline PDEProgramPadded::Index operator()(const PDEProgramPadded::ScalarIndex& a, const PDEProgramPadded::ScalarIndex& b, const PDEProgramPadded::ScalarIndex& c) {
            return __forall_ops.mkIx(a, b, c);
        };
    };

    static PDEProgramPadded::_mkIx mkIx;
private:
    static forall_ops<PDEProgramPadded::Array, PDEProgramPadded::Axis, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::Nat, PDEProgramPadded::Offset, PDEProgramPadded::_substepIx> __forall_ops;
public:
    static PDEProgramPadded::_substepIx substepIx;
};
} // examples
} // pde
} // mg_src
} // pde_cpp