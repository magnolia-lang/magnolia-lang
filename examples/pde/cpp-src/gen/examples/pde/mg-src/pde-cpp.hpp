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
    struct _rotate_ix {
        inline BasePDEProgram::Index operator()(const BasePDEProgram::Index& ix, const BasePDEProgram::Axis& axis, const BasePDEProgram::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static BasePDEProgram::_rotate_ix rotate_ix;
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
    struct _snippet {
        inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& u, const BasePDEProgram::Array& v, const BasePDEProgram::Array& u0, const BasePDEProgram::Array& u1, const BasePDEProgram::Array& u2) {
            return BasePDEProgram::binary_add(u, BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::dt(), BasePDEProgram::two.operator()<Float>()), BasePDEProgram::binary_sub(BasePDEProgram::mul(BasePDEProgram::nu(), BasePDEProgram::binary_sub(BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::div(BasePDEProgram::one.operator()<Float>(), BasePDEProgram::dx()), BasePDEProgram::dx()), BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>())), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()))), BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::div(BasePDEProgram::mul(BasePDEProgram::three(), BasePDEProgram::two.operator()<Float>()), BasePDEProgram::dx()), BasePDEProgram::dx()), u0))), BasePDEProgram::mul(BasePDEProgram::div(BasePDEProgram::div(BasePDEProgram::one.operator()<Float>(), BasePDEProgram::two.operator()<Float>()), BasePDEProgram::dx()), BasePDEProgram::binary_add(BasePDEProgram::binary_add(BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::zero(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u0), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::one.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u1)), BasePDEProgram::mul(BasePDEProgram::binary_sub(BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::one.operator()<Offset>()), BasePDEProgram::rotate(v, BasePDEProgram::two.operator()<Axis>(), BasePDEProgram::unary_sub(BasePDEProgram::one.operator()<Offset>()))), u2))))));
        };
    };

    static BasePDEProgram::_snippet snippet;
    struct _step {
        inline void operator()(BasePDEProgram::Array& u0, BasePDEProgram::Array& u1, BasePDEProgram::Array& u2) {
            BasePDEProgram::Array v0 = u0;
            BasePDEProgram::Array v1 = u1;
            BasePDEProgram::Array v2 = u2;
            v0 = BasePDEProgram::snippet(v0, u0, u0, u1, u2);
            v1 = BasePDEProgram::snippet(v1, u1, u0, u1, u2);
            v2 = BasePDEProgram::snippet(v2, u2, u0, u1, u2);
            u0 = BasePDEProgram::snippet(u0, v0, u0, u1, u2);
            u1 = BasePDEProgram::snippet(u1, v1, u0, u1, u2);
            u2 = BasePDEProgram::snippet(u2, v2, u0, u1, u2);
        };
    };

    static BasePDEProgram::_step step;
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
    struct _rotate_ix {
        inline PDEProgram::Index operator()(const PDEProgram::Index& ix, const PDEProgram::Axis& axis, const PDEProgram::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static PDEProgram::_rotate_ix rotate_ix;
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
    struct _snippet {
        inline PDEProgram::Array operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2) {
            return PDEProgram::binary_add(u, PDEProgram::mul(PDEProgram::div(PDEProgram::dt(), PDEProgram::two.operator()<Float>()), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::nu(), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>())), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::mul(PDEProgram::three(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::dx()), u0))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u0), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u1)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), PDEProgram::rotate(v, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram::_snippet snippet;
    struct _snippet_ix {
        inline PDEProgram::Float operator()(const PDEProgram::Array& u, const PDEProgram::Array& v, const PDEProgram::Array& u0, const PDEProgram::Array& u1, const PDEProgram::Array& u2, const PDEProgram::Index& ix) {
            return PDEProgram::binary_add(PDEProgram::psi(ix, u), PDEProgram::mul(PDEProgram::div(PDEProgram::dt(), PDEProgram::two.operator()<Float>()), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::nu(), PDEProgram::binary_sub(PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::mul(PDEProgram::three(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::dx()), PDEProgram::psi(ix, u0)))), PDEProgram::mul(PDEProgram::div(PDEProgram::div(PDEProgram::one.operator()<Float>(), PDEProgram::two.operator()<Float>()), PDEProgram::dx()), PDEProgram::binary_add(PDEProgram::binary_add(PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::zero(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u0)), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::one.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u1))), PDEProgram::mul(PDEProgram::binary_sub(PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::one.operator()<Offset>()), v), PDEProgram::psi(PDEProgram::rotate_ix(ix, PDEProgram::two.operator()<Axis>(), PDEProgram::unary_sub(PDEProgram::one.operator()<Offset>())), v)), PDEProgram::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _make_ix {
        inline PDEProgram::Index operator()(const PDEProgram::ScalarIndex& a, const PDEProgram::ScalarIndex& b, const PDEProgram::ScalarIndex& c) {
            return __forall_ops.make_ix(a, b, c);
        };
    };

    static PDEProgram::_make_ix make_ix;
private:
    static forall_ops<PDEProgram::Array, PDEProgram::Axis, PDEProgram::Float, PDEProgram::Index, PDEProgram::Nat, PDEProgram::Offset, PDEProgram::_snippet_ix> __forall_ops;
public:
    static PDEProgram::_snippet_ix snippet_ix;
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
    struct _rotate_ix {
        inline PDEProgram0::Index operator()(const PDEProgram0::Index& ix, const PDEProgram0::Axis& axis, const PDEProgram0::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static PDEProgram0::_rotate_ix rotate_ix;
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
    struct _snippet {
        inline PDEProgram0::Array operator()(const PDEProgram0::Array& u, const PDEProgram0::Array& v, const PDEProgram0::Array& u0, const PDEProgram0::Array& u1, const PDEProgram0::Array& u2) {
            return PDEProgram0::binary_add(u, PDEProgram0::mul(PDEProgram0::div(PDEProgram0::dt(), PDEProgram0::two.operator()<Float>()), PDEProgram0::binary_sub(PDEProgram0::mul(PDEProgram0::nu(), PDEProgram0::binary_sub(PDEProgram0::mul(PDEProgram0::div(PDEProgram0::div(PDEProgram0::one.operator()<Float>(), PDEProgram0::dx()), PDEProgram0::dx()), PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::one.operator()<Offset>())), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()))), PDEProgram0::mul(PDEProgram0::div(PDEProgram0::div(PDEProgram0::mul(PDEProgram0::three(), PDEProgram0::two.operator()<Float>()), PDEProgram0::dx()), PDEProgram0::dx()), u0))), PDEProgram0::mul(PDEProgram0::div(PDEProgram0::div(PDEProgram0::one.operator()<Float>(), PDEProgram0::two.operator()<Float>()), PDEProgram0::dx()), PDEProgram0::binary_add(PDEProgram0::binary_add(PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::zero(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u0), PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::one.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u1)), PDEProgram0::mul(PDEProgram0::binary_sub(PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::one.operator()<Offset>()), PDEProgram0::rotate(v, PDEProgram0::two.operator()<Axis>(), PDEProgram0::unary_sub(PDEProgram0::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram0::_snippet snippet;
    struct _step {
        inline void operator()(PDEProgram0::Array& u0, PDEProgram0::Array& u1, PDEProgram0::Array& u2) {
            PDEProgram0::Array v0 = u0;
            PDEProgram0::Array v1 = u1;
            PDEProgram0::Array v2 = u2;
            v0 = PDEProgram0::snippet(v0, u0, u0, u1, u2);
            v1 = PDEProgram0::snippet(v1, u1, u0, u1, u2);
            v2 = PDEProgram0::snippet(v2, u2, u0, u1, u2);
            u0 = PDEProgram0::snippet(u0, v0, u0, u1, u2);
            u1 = PDEProgram0::snippet(u1, v1, u0, u1, u2);
            u2 = PDEProgram0::snippet(u2, v2, u0, u1, u2);
        };
    };

    static PDEProgram0::_step step;
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
    struct _rotate_ix {
        inline PDEProgram1::Index operator()(const PDEProgram1::Index& ix, const PDEProgram1::Axis& axis, const PDEProgram1::Offset& o) {
            return __array_ops.rotate_ix(ix, axis, o);
        };
    };

    static PDEProgram1::_rotate_ix rotate_ix;
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
    struct _snippet {
        inline PDEProgram1::Array operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2) {
            return PDEProgram1::binary_add(u, PDEProgram1::mul(PDEProgram1::div(PDEProgram1::dt(), PDEProgram1::two.operator()<Float>()), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::nu(), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::dx()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>())), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::mul(PDEProgram1::three(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::dx()), u0))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u0), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u1)), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), PDEProgram1::rotate(v, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>()))), u2))))));
        };
    };

    static PDEProgram1::_snippet snippet;
    struct _snippet_ix {
        inline PDEProgram1::Float operator()(const PDEProgram1::Array& u, const PDEProgram1::Array& v, const PDEProgram1::Array& u0, const PDEProgram1::Array& u1, const PDEProgram1::Array& u2, const PDEProgram1::Index& ix) {
            return PDEProgram1::binary_add(PDEProgram1::psi(ix, u), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::dt(), PDEProgram1::two.operator()<Float>()), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::nu(), PDEProgram1::binary_sub(PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::dx()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::mul(PDEProgram1::three(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::dx()), PDEProgram1::psi(ix, u0)))), PDEProgram1::mul(PDEProgram1::div(PDEProgram1::div(PDEProgram1::one.operator()<Float>(), PDEProgram1::two.operator()<Float>()), PDEProgram1::dx()), PDEProgram1::binary_add(PDEProgram1::binary_add(PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::zero(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u0)), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::one.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u1))), PDEProgram1::mul(PDEProgram1::binary_sub(PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::one.operator()<Offset>()), v), PDEProgram1::psi(PDEProgram1::rotate_ix(ix, PDEProgram1::two.operator()<Axis>(), PDEProgram1::unary_sub(PDEProgram1::one.operator()<Offset>())), v)), PDEProgram1::psi(ix, u2)))))));
        };
    };

    typedef forall_ops<PDEProgram1::Array, PDEProgram1::Axis, PDEProgram1::Float, PDEProgram1::Index, PDEProgram1::Nat, PDEProgram1::Offset, PDEProgram1::_snippet_ix>::ScalarIndex ScalarIndex;
    struct _make_ix {
        inline PDEProgram1::Index operator()(const PDEProgram1::ScalarIndex& a, const PDEProgram1::ScalarIndex& b, const PDEProgram1::ScalarIndex& c) {
            return __forall_ops.make_ix(a, b, c);
        };
    };

    static PDEProgram1::_make_ix make_ix;
private:
    static forall_ops<PDEProgram1::Array, PDEProgram1::Axis, PDEProgram1::Float, PDEProgram1::Index, PDEProgram1::Nat, PDEProgram1::Offset, PDEProgram1::_snippet_ix> __forall_ops;
public:
    static PDEProgram1::_snippet_ix snippet_ix;
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
};
} // examples
} // pde
} // mg_src
} // pde_cpp