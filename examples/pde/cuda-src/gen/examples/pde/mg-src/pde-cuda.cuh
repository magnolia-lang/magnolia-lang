#pragma once

#include "base.cuh"
#include <cassert>


namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cuda {
struct BasePDEProgram {
    struct _two {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            BasePDEProgram::two0(o);
            return o;
        };
    };

    BasePDEProgram::_two two;
    struct _one {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            BasePDEProgram::one0(o);
            return o;
        };
    };

    BasePDEProgram::_one one;
private:
    constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<BasePDEProgram::Float>::Index Index;
    typedef array_ops<BasePDEProgram::Float>::Nat Nat;
    typedef array_ops<BasePDEProgram::Float>::Offset Offset;
private:
    __device__ __host__ inline void one0(BasePDEProgram::Offset& o) {
        o = __array_ops.one_offset();
    };
    array_ops<BasePDEProgram::Float> __array_ops;
public:
    struct _dt {
    private:
        constants __constants;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()() {
            return __constants.dt();
        };
    };

    BasePDEProgram::_dt dt;
    struct _dx {
    private:
        constants __constants;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()() {
            return __constants.dx();
        };
    };

    BasePDEProgram::_dx dx;
    struct _nu {
    private:
        constants __constants;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()() {
            return __constants.nu();
        };
    };

    BasePDEProgram::_nu nu;
    struct _three {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()() {
            return __array_ops.three_float();
        };
    };

    BasePDEProgram::_three three;
    struct _unary_sub {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& f) {
            return __array_ops.unary_sub(f);
        };
        __device__ __host__ inline BasePDEProgram::Offset operator()(const BasePDEProgram::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    BasePDEProgram::_unary_sub unary_sub;
private:
    __device__ __host__ inline void one0(BasePDEProgram::Float& o) {
        o = __array_ops.one_float();
    };
    __device__ __host__ inline void two0(BasePDEProgram::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<BasePDEProgram::Float>::Axis Axis;
    struct _rotateIx {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Index operator()(const BasePDEProgram::Index& ix, const BasePDEProgram::Axis& axis, const BasePDEProgram::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    BasePDEProgram::_rotateIx rotateIx;
    struct _zero {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    BasePDEProgram::_zero zero;
private:
    __device__ __host__ inline void one0(BasePDEProgram::Axis& o) {
        o = __array_ops.one_axis();
    };
    __device__ __host__ inline void two0(BasePDEProgram::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<BasePDEProgram::Float>::Array Array;
    struct _binary_add {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    BasePDEProgram::_binary_add binary_add;
    struct _binary_sub {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    BasePDEProgram::_binary_sub binary_sub;
    struct _div {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& num, const BasePDEProgram::Float& den) {
            return __array_ops.div(num, den);
        };
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& num, const BasePDEProgram::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    BasePDEProgram::_div div;
    struct _mul {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Float& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& lhs, const BasePDEProgram::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    BasePDEProgram::_mul mul;
    struct _psi {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Float operator()(const BasePDEProgram::Index& ix, const BasePDEProgram::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    BasePDEProgram::_psi psi;
    struct _rotate {
    private:
        array_ops<BasePDEProgram::Float> __array_ops;
    public:
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& a, const BasePDEProgram::Axis& axis, const BasePDEProgram::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    BasePDEProgram::_rotate rotate;
    struct _substep {
    private:
        BasePDEProgram::_mul mul;
        BasePDEProgram::_binary_add binary_add;
        BasePDEProgram::_div div;
        BasePDEProgram::_binary_sub binary_sub;
        BasePDEProgram::_dt dt;
        BasePDEProgram::_dx dx;
        BasePDEProgram::_nu nu;
        BasePDEProgram::_one one;
        BasePDEProgram::_rotate rotate;
        BasePDEProgram::_three three;
        BasePDEProgram::_two two;
        BasePDEProgram::_unary_sub unary_sub;
        BasePDEProgram::_zero zero;
    public:
        __device__ __host__ inline BasePDEProgram::Array operator()(const BasePDEProgram::Array& u, const BasePDEProgram::Array& v, const BasePDEProgram::Array& u0, const BasePDEProgram::Array& u1, const BasePDEProgram::Array& u2) {
            return binary_add(u, mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(rotate(v, zero(), unary_sub(one.operator()<Offset>())), rotate(v, zero(), one.operator()<Offset>())), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, one.operator()<Axis>(), one.operator()<Offset>())), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, two.operator()<Axis>(), one.operator()<Offset>()))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), u0))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(rotate(v, zero(), one.operator()<Offset>()), rotate(v, zero(), unary_sub(one.operator()<Offset>()))), u0), mul(binary_sub(rotate(v, one.operator()<Axis>(), one.operator()<Offset>()), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u1)), mul(binary_sub(rotate(v, two.operator()<Axis>(), one.operator()<Offset>()), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u2))))));
        };
    };

    struct _step {
    private:
        BasePDEProgram::_substep substep;
    public:
        __device__ __host__ inline void operator()(BasePDEProgram::Array& u0, BasePDEProgram::Array& u1, BasePDEProgram::Array& u2) {
            BasePDEProgram::Array v0 = u0;
            BasePDEProgram::Array v1 = u1;
            BasePDEProgram::Array v2 = u2;
            v0 = substep(v0, u0, u0, u1, u2);
            v1 = substep(v1, u1, u0, u1, u2);
            v2 = substep(v2, u2, u0, u1, u2);
            u0 = substep(u0, v0, u0, u1, u2);
            u1 = substep(u1, v1, u0, u1, u2);
            u2 = substep(u2, v2, u0, u1, u2);
        };
    };

    BasePDEProgram::_step step;
    BasePDEProgram::_substep substep;
};
} // examples
} // pde
} // mg_src
} // pde_cuda

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cuda {
struct PDEProgram3D {
    struct _two {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgram3D::two0(o);
            return o;
        };
    };

    PDEProgram3D::_two two;
    struct _one {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgram3D::one0(o);
            return o;
        };
    };

    PDEProgram3D::_one one;
private:
    constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgram3D::Float>::Index Index;
    typedef scalar_index<PDEProgram3D::Index>::ScalarIndex ScalarIndex;
private:
    scalar_index<PDEProgram3D::Index> __scalar_index;
public:
    struct _ix0 {
    private:
        scalar_index<PDEProgram3D::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::Index& ix) {
            return __scalar_index.ix0(ix);
        };
    };

    PDEProgram3D::_ix0 ix0;
    struct _ix1 {
    private:
        scalar_index<PDEProgram3D::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::Index& ix) {
            return __scalar_index.ix1(ix);
        };
    };

    PDEProgram3D::_ix1 ix1;
    struct _ix2 {
    private:
        scalar_index<PDEProgram3D::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::Index& ix) {
            return __scalar_index.ix2(ix);
        };
    };

    PDEProgram3D::_ix2 ix2;
    struct _mkIx {
    private:
        scalar_index<PDEProgram3D::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3D::Index operator()(const PDEProgram3D::ScalarIndex& a, const PDEProgram3D::ScalarIndex& b, const PDEProgram3D::ScalarIndex& c) {
            return __scalar_index.mkIx(a, b, c);
        };
    };

    PDEProgram3D::_mkIx mkIx;
    typedef array_ops<PDEProgram3D::Float>::Nat Nat;
    typedef array_ops<PDEProgram3D::Float>::Offset Offset;
private:
    axis_length<PDEProgram3D::Offset, PDEProgram3D::ScalarIndex> __axis_length;
    __device__ __host__ inline void one0(PDEProgram3D::Offset& o) {
        o = __array_ops.one_offset();
    };
    array_ops<PDEProgram3D::Float> __array_ops;
public:
    struct _dt {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()() {
            return __constants.dt();
        };
    };

    PDEProgram3D::_dt dt;
    struct _dx {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()() {
            return __constants.dx();
        };
    };

    PDEProgram3D::_dx dx;
    struct _nu {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()() {
            return __constants.nu();
        };
    };

    PDEProgram3D::_nu nu;
    struct _three {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()() {
            return __array_ops.three_float();
        };
    };

    PDEProgram3D::_three three;
    struct _unary_sub {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Offset operator()(const PDEProgram3D::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& f) {
            return __array_ops.unary_sub(f);
        };
    };

    PDEProgram3D::_unary_sub unary_sub;
private:
    __device__ __host__ inline void one0(PDEProgram3D::Float& o) {
        o = __array_ops.one_float();
    };
    __device__ __host__ inline void two0(PDEProgram3D::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef axis_length<PDEProgram3D::Offset, PDEProgram3D::ScalarIndex>::AxisLength AxisLength;
    struct _mod {
    private:
        axis_length<PDEProgram3D::Offset, PDEProgram3D::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::ScalarIndex& six, const PDEProgram3D::AxisLength& sc) {
            return __axis_length.mod(six, sc);
        };
    };

    PDEProgram3D::_mod mod;
    struct _shape0 {
    private:
        axis_length<PDEProgram3D::Offset, PDEProgram3D::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3D::AxisLength operator()() {
            return __axis_length.shape0();
        };
    };

    PDEProgram3D::_shape0 shape0;
    struct _shape1 {
    private:
        axis_length<PDEProgram3D::Offset, PDEProgram3D::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3D::AxisLength operator()() {
            return __axis_length.shape1();
        };
    };

    PDEProgram3D::_shape1 shape1;
    struct _shape2 {
    private:
        axis_length<PDEProgram3D::Offset, PDEProgram3D::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3D::AxisLength operator()() {
            return __axis_length.shape2();
        };
    };

    PDEProgram3D::_shape2 shape2;
    typedef array_ops<PDEProgram3D::Float>::Axis Axis;
    struct _rotateIx {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Index operator()(const PDEProgram3D::Index& ix, const PDEProgram3D::Axis& axis, const PDEProgram3D::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    PDEProgram3D::_rotateIx rotateIx;
    struct _zero {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    PDEProgram3D::_zero zero;
private:
    __device__ __host__ inline void one0(PDEProgram3D::Axis& o) {
        o = __array_ops.one_axis();
    };
    __device__ __host__ inline void two0(PDEProgram3D::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgram3D::Float>::Array Array;
private:
    specialize_base<PDEProgram3D::Array, PDEProgram3D::Float, PDEProgram3D::ScalarIndex> __specialize_base;
public:
    struct _binary_add {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
        axis_length<PDEProgram3D::Offset, PDEProgram3D::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3D::ScalarIndex operator()(const PDEProgram3D::ScalarIndex& six, const PDEProgram3D::Offset& o) {
            return __axis_length.binary_add(six, o);
        };
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    PDEProgram3D::_binary_add binary_add;
    struct _binary_sub {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    PDEProgram3D::_binary_sub binary_sub;
    struct _div {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& num, const PDEProgram3D::Array& den) {
            return __array_ops.div(num, den);
        };
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& num, const PDEProgram3D::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    PDEProgram3D::_div div;
    struct _mul {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Float& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& lhs, const PDEProgram3D::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    PDEProgram3D::_mul mul;
    struct _psi {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
        specialize_base<PDEProgram3D::Array, PDEProgram3D::Float, PDEProgram3D::ScalarIndex> __specialize_base;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::ScalarIndex& i, const PDEProgram3D::ScalarIndex& j, const PDEProgram3D::ScalarIndex& k, const PDEProgram3D::Array& a) {
            return __specialize_base.psi(i, j, k, a);
        };
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Index& ix, const PDEProgram3D::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    PDEProgram3D::_psi psi;
    struct _rotate {
    private:
        array_ops<PDEProgram3D::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& a, const PDEProgram3D::Axis& axis, const PDEProgram3D::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    PDEProgram3D::_rotate rotate;
    struct _substep {
    private:
        PDEProgram3D::_mul mul;
        PDEProgram3D::_binary_add binary_add;
        PDEProgram3D::_div div;
        PDEProgram3D::_binary_sub binary_sub;
        PDEProgram3D::_dt dt;
        PDEProgram3D::_dx dx;
        PDEProgram3D::_nu nu;
        PDEProgram3D::_one one;
        PDEProgram3D::_rotate rotate;
        PDEProgram3D::_three three;
        PDEProgram3D::_two two;
        PDEProgram3D::_unary_sub unary_sub;
        PDEProgram3D::_zero zero;
    public:
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2) {
            return binary_add(u, mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(rotate(v, zero(), unary_sub(one.operator()<Offset>())), rotate(v, zero(), one.operator()<Offset>())), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, one.operator()<Axis>(), one.operator()<Offset>())), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, two.operator()<Axis>(), one.operator()<Offset>()))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), u0))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(rotate(v, zero(), one.operator()<Offset>()), rotate(v, zero(), unary_sub(one.operator()<Offset>()))), u0), mul(binary_sub(rotate(v, one.operator()<Axis>(), one.operator()<Offset>()), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u1)), mul(binary_sub(rotate(v, two.operator()<Axis>(), one.operator()<Offset>()), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u2))))));
        };
    };

    PDEProgram3D::_substep substep;
    struct _substepIx {
    private:
        PDEProgram3D::_mul mul;
        PDEProgram3D::_binary_add binary_add;
        PDEProgram3D::_div div;
        PDEProgram3D::_binary_sub binary_sub;
        PDEProgram3D::_dt dt;
        PDEProgram3D::_dx dx;
        PDEProgram3D::_ix0 ix0;
        PDEProgram3D::_ix1 ix1;
        PDEProgram3D::_ix2 ix2;
        PDEProgram3D::_nu nu;
        PDEProgram3D::_one one;
        PDEProgram3D::_psi psi;
        PDEProgram3D::_rotateIx rotateIx;
        PDEProgram3D::_three three;
        PDEProgram3D::_two two;
        PDEProgram3D::_unary_sub unary_sub;
        PDEProgram3D::_zero zero;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2, const PDEProgram3D::Index& ix) {
            return binary_add(psi(ix0(ix), ix1(ix), ix2(ix), u), mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(psi(ix0(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>()))), ix1(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>()))), ix2(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>()))), v), psi(ix0(rotateIx(ix, zero(), one.operator()<Offset>())), ix1(rotateIx(ix, zero(), one.operator()<Offset>())), ix2(rotateIx(ix, zero(), one.operator()<Offset>())), v)), psi(ix0(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>())), v)), psi(ix0(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>())), v))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), psi(ix0(ix), ix1(ix), ix2(ix), u0)))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(psi(ix0(rotateIx(ix, zero(), one.operator()<Offset>())), ix1(rotateIx(ix, zero(), one.operator()<Offset>())), ix2(rotateIx(ix, zero(), one.operator()<Offset>())), v), psi(ix0(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>()))), ix1(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>()))), ix2(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(ix), ix1(ix), ix2(ix), u0)), mul(binary_sub(psi(ix0(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>())), v), psi(ix0(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(ix), ix1(ix), ix2(ix), u1))), mul(binary_sub(psi(ix0(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>())), v), psi(ix0(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(ix), ix1(ix), ix2(ix), u2)))))));
        };
    };

private:
    forall_ops<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Nat, PDEProgram3D::Offset, PDEProgram3D::_substepIx> __forall_ops;
public:
    struct _schedule {
    private:
        forall_ops<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Nat, PDEProgram3D::Offset, PDEProgram3D::_substepIx> __forall_ops;
    public:
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    PDEProgram3D::_schedule schedule;
    PDEProgram3D::_substepIx substepIx;
    struct _substepIx3D {
    private:
        PDEProgram3D::_mod mod;
        PDEProgram3D::_mul mul;
        PDEProgram3D::_binary_add binary_add;
        PDEProgram3D::_div div;
        PDEProgram3D::_binary_sub binary_sub;
        PDEProgram3D::_dt dt;
        PDEProgram3D::_dx dx;
        PDEProgram3D::_nu nu;
        PDEProgram3D::_one one;
        PDEProgram3D::_psi psi;
        PDEProgram3D::_shape0 shape0;
        PDEProgram3D::_shape1 shape1;
        PDEProgram3D::_shape2 shape2;
        PDEProgram3D::_three three;
        PDEProgram3D::_two two;
        PDEProgram3D::_unary_sub unary_sub;
    public:
        __device__ __host__ inline PDEProgram3D::Float operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2, const PDEProgram3D::ScalarIndex& i, const PDEProgram3D::ScalarIndex& j, const PDEProgram3D::ScalarIndex& k) {
            return binary_add(psi(i, j, k, u), mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(psi(mod(binary_add(i, unary_sub(one.operator()<Offset>())), shape0()), j, k, v), psi(mod(binary_add(i, one.operator()<Offset>()), shape0()), j, k, v)), psi(i, mod(binary_add(j, unary_sub(one.operator()<Offset>())), shape1()), k, v)), psi(i, mod(binary_add(j, one.operator()<Offset>()), shape1()), k, v)), psi(i, j, mod(binary_add(k, unary_sub(one.operator()<Offset>())), shape2()), v)), psi(i, j, mod(binary_add(k, one.operator()<Offset>()), shape2()), v))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), psi(i, j, k, u0)))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(psi(mod(binary_add(i, one.operator()<Offset>()), shape0()), j, k, v), psi(mod(binary_add(i, unary_sub(one.operator()<Offset>())), shape0()), j, k, v)), psi(i, j, k, u0)), mul(binary_sub(psi(i, mod(binary_add(j, one.operator()<Offset>()), shape1()), k, v), psi(i, mod(binary_add(j, unary_sub(one.operator()<Offset>())), shape1()), k, v)), psi(i, j, k, u1))), mul(binary_sub(psi(i, j, mod(binary_add(k, one.operator()<Offset>()), shape2()), v), psi(i, j, mod(binary_add(k, unary_sub(one.operator()<Offset>())), shape2()), v)), psi(i, j, k, u2)))))));
        };
    };

private:
    specialize_psi_ops_2<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Offset, PDEProgram3D::ScalarIndex, PDEProgram3D::_substepIx3D> __specialize_psi_ops_2;
public:
    struct _refillPadding {
    private:
        specialize_psi_ops_2<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Offset, PDEProgram3D::ScalarIndex, PDEProgram3D::_substepIx3D> __specialize_psi_ops_2;
    public:
        __device__ __host__ inline void operator()(PDEProgram3D::Array& a) {
            return __specialize_psi_ops_2.refillPadding(a);
        };
    };

    PDEProgram3D::_refillPadding refillPadding;
    struct _rotateIxPadded {
    private:
        specialize_psi_ops_2<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Offset, PDEProgram3D::ScalarIndex, PDEProgram3D::_substepIx3D> __specialize_psi_ops_2;
    public:
        __device__ __host__ inline PDEProgram3D::Index operator()(const PDEProgram3D::Index& ix, const PDEProgram3D::Axis& axis, const PDEProgram3D::Offset& o) {
            return __specialize_psi_ops_2.rotateIxPadded(ix, axis, o);
        };
    };

    PDEProgram3D::_rotateIxPadded rotateIxPadded;
    struct _schedule3D {
    private:
        specialize_psi_ops_2<PDEProgram3D::Array, PDEProgram3D::Axis, PDEProgram3D::Float, PDEProgram3D::Index, PDEProgram3D::Offset, PDEProgram3D::ScalarIndex, PDEProgram3D::_substepIx3D> __specialize_psi_ops_2;
    public:
        __device__ __host__ inline PDEProgram3D::Array operator()(const PDEProgram3D::Array& u, const PDEProgram3D::Array& v, const PDEProgram3D::Array& u0, const PDEProgram3D::Array& u1, const PDEProgram3D::Array& u2) {
            return __specialize_psi_ops_2.schedule3DPadded(u, v, u0, u1, u2);
        };
    };

    struct _step {
    private:
        PDEProgram3D::_schedule3D schedule3D;
    public:
        __device__ __host__ inline void operator()(PDEProgram3D::Array& u0, PDEProgram3D::Array& u1, PDEProgram3D::Array& u2) {
            PDEProgram3D::Array v0 = u0;
            PDEProgram3D::Array v1 = u1;
            PDEProgram3D::Array v2 = u2;
            v0 = schedule3D(v0, u0, u0, u1, u2);
            v1 = schedule3D(v1, u1, u0, u1, u2);
            v2 = schedule3D(v2, u2, u0, u1, u2);
            u0 = schedule3D(u0, v0, u0, u1, u2);
            u1 = schedule3D(u1, v1, u0, u1, u2);
            u2 = schedule3D(u2, v2, u0, u1, u2);
        };
    };

    PDEProgram3D::_step step;
    PDEProgram3D::_schedule3D schedule3D;
    PDEProgram3D::_substepIx3D substepIx3D;
};
} // examples
} // pde
} // mg_src
} // pde_cuda

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cuda {
struct PDEProgram3DPadded {
    struct _two {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgram3DPadded::two0(o);
            return o;
        };
    };

    PDEProgram3DPadded::_two two;
    struct _one {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgram3DPadded::one0(o);
            return o;
        };
    };

    PDEProgram3DPadded::_one one;
private:
    constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgram3DPadded::Float>::Index Index;
    typedef scalar_index<PDEProgram3DPadded::Index>::ScalarIndex ScalarIndex;
private:
    scalar_index<PDEProgram3DPadded::Index> __scalar_index;
public:
    struct _ix0 {
    private:
        scalar_index<PDEProgram3DPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::Index& ix) {
            return __scalar_index.ix0(ix);
        };
    };

    PDEProgram3DPadded::_ix0 ix0;
    struct _ix1 {
    private:
        scalar_index<PDEProgram3DPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::Index& ix) {
            return __scalar_index.ix1(ix);
        };
    };

    PDEProgram3DPadded::_ix1 ix1;
    struct _ix2 {
    private:
        scalar_index<PDEProgram3DPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::Index& ix) {
            return __scalar_index.ix2(ix);
        };
    };

    PDEProgram3DPadded::_ix2 ix2;
    struct _mkIx {
    private:
        scalar_index<PDEProgram3DPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Index operator()(const PDEProgram3DPadded::ScalarIndex& a, const PDEProgram3DPadded::ScalarIndex& b, const PDEProgram3DPadded::ScalarIndex& c) {
            return __scalar_index.mkIx(a, b, c);
        };
    };

    PDEProgram3DPadded::_mkIx mkIx;
    typedef array_ops<PDEProgram3DPadded::Float>::Nat Nat;
    typedef array_ops<PDEProgram3DPadded::Float>::Offset Offset;
private:
    axis_length<PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex> __axis_length;
    __device__ __host__ inline void one0(PDEProgram3DPadded::Offset& o) {
        o = __array_ops.one_offset();
    };
    array_ops<PDEProgram3DPadded::Float> __array_ops;
public:
    struct _dt {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()() {
            return __constants.dt();
        };
    };

    PDEProgram3DPadded::_dt dt;
    struct _dx {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()() {
            return __constants.dx();
        };
    };

    PDEProgram3DPadded::_dx dx;
    struct _nu {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()() {
            return __constants.nu();
        };
    };

    PDEProgram3DPadded::_nu nu;
    struct _three {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()() {
            return __array_ops.three_float();
        };
    };

    PDEProgram3DPadded::_three three;
    struct _unary_sub {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Offset operator()(const PDEProgram3DPadded::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& f) {
            return __array_ops.unary_sub(f);
        };
    };

    PDEProgram3DPadded::_unary_sub unary_sub;
private:
    __device__ __host__ inline void one0(PDEProgram3DPadded::Float& o) {
        o = __array_ops.one_float();
    };
    __device__ __host__ inline void two0(PDEProgram3DPadded::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef axis_length<PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex>::AxisLength AxisLength;
    struct _mod {
    private:
        axis_length<PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::ScalarIndex& six, const PDEProgram3DPadded::AxisLength& sc) {
            return __axis_length.mod(six, sc);
        };
    };

    PDEProgram3DPadded::_mod mod;
    struct _shape0 {
    private:
        axis_length<PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3DPadded::AxisLength operator()() {
            return __axis_length.shape0();
        };
    };

    PDEProgram3DPadded::_shape0 shape0;
    struct _shape1 {
    private:
        axis_length<PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3DPadded::AxisLength operator()() {
            return __axis_length.shape1();
        };
    };

    PDEProgram3DPadded::_shape1 shape1;
    struct _shape2 {
    private:
        axis_length<PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3DPadded::AxisLength operator()() {
            return __axis_length.shape2();
        };
    };

    PDEProgram3DPadded::_shape2 shape2;
    typedef array_ops<PDEProgram3DPadded::Float>::Axis Axis;
    struct _rotateIx {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Index operator()(const PDEProgram3DPadded::Index& ix, const PDEProgram3DPadded::Axis& axis, const PDEProgram3DPadded::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    PDEProgram3DPadded::_rotateIx rotateIx;
    struct _zero {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    PDEProgram3DPadded::_zero zero;
private:
    __device__ __host__ inline void one0(PDEProgram3DPadded::Axis& o) {
        o = __array_ops.one_axis();
    };
    __device__ __host__ inline void two0(PDEProgram3DPadded::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgram3DPadded::Float>::Array Array;
private:
    specialize_base<PDEProgram3DPadded::Array, PDEProgram3DPadded::Float, PDEProgram3DPadded::ScalarIndex> __specialize_base;
public:
    struct _binary_add {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
        axis_length<PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex> __axis_length;
    public:
        __device__ __host__ inline PDEProgram3DPadded::ScalarIndex operator()(const PDEProgram3DPadded::ScalarIndex& six, const PDEProgram3DPadded::Offset& o) {
            return __axis_length.binary_add(six, o);
        };
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    PDEProgram3DPadded::_binary_add binary_add;
    struct _binary_sub {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    PDEProgram3DPadded::_binary_sub binary_sub;
    struct _div {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& num, const PDEProgram3DPadded::Array& den) {
            return __array_ops.div(num, den);
        };
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& num, const PDEProgram3DPadded::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    PDEProgram3DPadded::_div div;
    struct _mul {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Float& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& lhs, const PDEProgram3DPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    PDEProgram3DPadded::_mul mul;
    struct _psi {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
        specialize_base<PDEProgram3DPadded::Array, PDEProgram3DPadded::Float, PDEProgram3DPadded::ScalarIndex> __specialize_base;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::ScalarIndex& i, const PDEProgram3DPadded::ScalarIndex& j, const PDEProgram3DPadded::ScalarIndex& k, const PDEProgram3DPadded::Array& a) {
            return __specialize_base.psi(i, j, k, a);
        };
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Index& ix, const PDEProgram3DPadded::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    PDEProgram3DPadded::_psi psi;
    struct _rotate {
    private:
        array_ops<PDEProgram3DPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& a, const PDEProgram3DPadded::Axis& axis, const PDEProgram3DPadded::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    PDEProgram3DPadded::_rotate rotate;
    struct _substep {
    private:
        PDEProgram3DPadded::_mul mul;
        PDEProgram3DPadded::_binary_add binary_add;
        PDEProgram3DPadded::_div div;
        PDEProgram3DPadded::_binary_sub binary_sub;
        PDEProgram3DPadded::_dt dt;
        PDEProgram3DPadded::_dx dx;
        PDEProgram3DPadded::_nu nu;
        PDEProgram3DPadded::_one one;
        PDEProgram3DPadded::_rotate rotate;
        PDEProgram3DPadded::_three three;
        PDEProgram3DPadded::_two two;
        PDEProgram3DPadded::_unary_sub unary_sub;
        PDEProgram3DPadded::_zero zero;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2) {
            return binary_add(u, mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(rotate(v, zero(), unary_sub(one.operator()<Offset>())), rotate(v, zero(), one.operator()<Offset>())), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, one.operator()<Axis>(), one.operator()<Offset>())), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, two.operator()<Axis>(), one.operator()<Offset>()))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), u0))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(rotate(v, zero(), one.operator()<Offset>()), rotate(v, zero(), unary_sub(one.operator()<Offset>()))), u0), mul(binary_sub(rotate(v, one.operator()<Axis>(), one.operator()<Offset>()), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u1)), mul(binary_sub(rotate(v, two.operator()<Axis>(), one.operator()<Offset>()), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u2))))));
        };
    };

    PDEProgram3DPadded::_substep substep;
    struct _substepIx3D {
    private:
        PDEProgram3DPadded::_mul mul;
        PDEProgram3DPadded::_binary_add binary_add;
        PDEProgram3DPadded::_div div;
        PDEProgram3DPadded::_binary_sub binary_sub;
        PDEProgram3DPadded::_dt dt;
        PDEProgram3DPadded::_dx dx;
        PDEProgram3DPadded::_nu nu;
        PDEProgram3DPadded::_one one;
        PDEProgram3DPadded::_psi psi;
        PDEProgram3DPadded::_three three;
        PDEProgram3DPadded::_two two;
        PDEProgram3DPadded::_unary_sub unary_sub;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2, const PDEProgram3DPadded::ScalarIndex& i, const PDEProgram3DPadded::ScalarIndex& j, const PDEProgram3DPadded::ScalarIndex& k) {
            return binary_add(psi(i, j, k, u), mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(psi(binary_add(i, unary_sub(one.operator()<Offset>())), j, k, v), psi(binary_add(i, one.operator()<Offset>()), j, k, v)), psi(i, binary_add(j, unary_sub(one.operator()<Offset>())), k, v)), psi(i, binary_add(j, one.operator()<Offset>()), k, v)), psi(i, j, binary_add(k, unary_sub(one.operator()<Offset>())), v)), psi(i, j, binary_add(k, one.operator()<Offset>()), v))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), psi(i, j, k, u0)))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(psi(binary_add(i, one.operator()<Offset>()), j, k, v), psi(binary_add(i, unary_sub(one.operator()<Offset>())), j, k, v)), psi(i, j, k, u0)), mul(binary_sub(psi(i, binary_add(j, one.operator()<Offset>()), k, v), psi(i, binary_add(j, unary_sub(one.operator()<Offset>())), k, v)), psi(i, j, k, u1))), mul(binary_sub(psi(i, j, binary_add(k, one.operator()<Offset>()), v), psi(i, j, binary_add(k, unary_sub(one.operator()<Offset>())), v)), psi(i, j, k, u2)))))));
        };
    };

private:
    specialize_psi_ops_2<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex, PDEProgram3DPadded::_substepIx3D> __specialize_psi_ops_2;
public:
    struct _refillPadding {
    private:
        specialize_psi_ops_2<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex, PDEProgram3DPadded::_substepIx3D> __specialize_psi_ops_2;
    public:
        __device__ __host__ inline void operator()(PDEProgram3DPadded::Array& a) {
            return __specialize_psi_ops_2.refillPadding(a);
        };
    };

    PDEProgram3DPadded::_refillPadding refillPadding;
    struct _rotateIxPadded {
    private:
        specialize_psi_ops_2<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex, PDEProgram3DPadded::_substepIx3D> __specialize_psi_ops_2;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Index operator()(const PDEProgram3DPadded::Index& ix, const PDEProgram3DPadded::Axis& axis, const PDEProgram3DPadded::Offset& o) {
            return __specialize_psi_ops_2.rotateIxPadded(ix, axis, o);
        };
    };

    struct _substepIx {
    private:
        PDEProgram3DPadded::_mul mul;
        PDEProgram3DPadded::_binary_add binary_add;
        PDEProgram3DPadded::_div div;
        PDEProgram3DPadded::_binary_sub binary_sub;
        PDEProgram3DPadded::_dt dt;
        PDEProgram3DPadded::_dx dx;
        PDEProgram3DPadded::_ix0 ix0;
        PDEProgram3DPadded::_ix1 ix1;
        PDEProgram3DPadded::_ix2 ix2;
        PDEProgram3DPadded::_nu nu;
        PDEProgram3DPadded::_one one;
        PDEProgram3DPadded::_psi psi;
        PDEProgram3DPadded::_rotateIxPadded rotateIxPadded;
        PDEProgram3DPadded::_three three;
        PDEProgram3DPadded::_two two;
        PDEProgram3DPadded::_unary_sub unary_sub;
        PDEProgram3DPadded::_zero zero;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Float operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2, const PDEProgram3DPadded::Index& ix) {
            return binary_add(psi(ix0(ix), ix1(ix), ix2(ix), u), mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(psi(ix0(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>()))), ix1(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>()))), ix2(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>()))), v), psi(ix0(rotateIxPadded(ix, zero(), one.operator()<Offset>())), ix1(rotateIxPadded(ix, zero(), one.operator()<Offset>())), ix2(rotateIxPadded(ix, zero(), one.operator()<Offset>())), v)), psi(ix0(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>())), v)), psi(ix0(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>())), v))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), psi(ix0(ix), ix1(ix), ix2(ix), u0)))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(psi(ix0(rotateIxPadded(ix, zero(), one.operator()<Offset>())), ix1(rotateIxPadded(ix, zero(), one.operator()<Offset>())), ix2(rotateIxPadded(ix, zero(), one.operator()<Offset>())), v), psi(ix0(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>()))), ix1(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>()))), ix2(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(ix), ix1(ix), ix2(ix), u0)), mul(binary_sub(psi(ix0(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>())), v), psi(ix0(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(ix), ix1(ix), ix2(ix), u1))), mul(binary_sub(psi(ix0(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix1(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>())), ix2(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>())), v), psi(ix0(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix1(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), ix2(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), v)), psi(ix0(ix), ix1(ix), ix2(ix), u2)))))));
        };
    };

private:
    forall_ops<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Nat, PDEProgram3DPadded::Offset, PDEProgram3DPadded::_substepIx> __forall_ops;
public:
    struct _schedule {
    private:
        forall_ops<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Nat, PDEProgram3DPadded::Offset, PDEProgram3DPadded::_substepIx> __forall_ops;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    PDEProgram3DPadded::_schedule schedule;
    PDEProgram3DPadded::_substepIx substepIx;
    PDEProgram3DPadded::_rotateIxPadded rotateIxPadded;
    struct _schedule3DPadded {
    private:
        specialize_psi_ops_2<PDEProgram3DPadded::Array, PDEProgram3DPadded::Axis, PDEProgram3DPadded::Float, PDEProgram3DPadded::Index, PDEProgram3DPadded::Offset, PDEProgram3DPadded::ScalarIndex, PDEProgram3DPadded::_substepIx3D> __specialize_psi_ops_2;
    public:
        __device__ __host__ inline PDEProgram3DPadded::Array operator()(const PDEProgram3DPadded::Array& u, const PDEProgram3DPadded::Array& v, const PDEProgram3DPadded::Array& u0, const PDEProgram3DPadded::Array& u1, const PDEProgram3DPadded::Array& u2) {
            return __specialize_psi_ops_2.schedule3DPadded(u, v, u0, u1, u2);
        };
    };

    struct _step {
    private:
        PDEProgram3DPadded::_refillPadding refillPadding;
        PDEProgram3DPadded::_schedule3DPadded schedule3DPadded;
    public:
        __device__ __host__ inline void operator()(PDEProgram3DPadded::Array& u0, PDEProgram3DPadded::Array& u1, PDEProgram3DPadded::Array& u2) {
            PDEProgram3DPadded::Array v0 = u0;
            PDEProgram3DPadded::Array v1 = u1;
            PDEProgram3DPadded::Array v2 = u2;
            v0 = [&]() {
                PDEProgram3DPadded::Array result = schedule3DPadded(v0, u0, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            v1 = [&]() {
                PDEProgram3DPadded::Array result = schedule3DPadded(v1, u1, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            v2 = [&]() {
                PDEProgram3DPadded::Array result = schedule3DPadded(v2, u2, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            u0 = [&]() {
                PDEProgram3DPadded::Array result = schedule3DPadded(u0, v0, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            u1 = [&]() {
                PDEProgram3DPadded::Array result = schedule3DPadded(u1, v1, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            u2 = [&]() {
                PDEProgram3DPadded::Array result = schedule3DPadded(u2, v2, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
        };
    };

    PDEProgram3DPadded::_step step;
    PDEProgram3DPadded::_schedule3DPadded schedule3DPadded;
    PDEProgram3DPadded::_substepIx3D substepIx3D;
};
} // examples
} // pde
} // mg_src
} // pde_cuda

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cuda {
struct PDEProgramDNF {
    struct _two {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgramDNF::two0(o);
            return o;
        };
    };

    PDEProgramDNF::_two two;
    struct _one {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgramDNF::one0(o);
            return o;
        };
    };

    PDEProgramDNF::_one one;
private:
    constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgramDNF::Float>::Index Index;
    typedef array_ops<PDEProgramDNF::Float>::Nat Nat;
    typedef array_ops<PDEProgramDNF::Float>::Offset Offset;
private:
    __device__ __host__ inline void one0(PDEProgramDNF::Offset& o) {
        o = __array_ops.one_offset();
    };
    array_ops<PDEProgramDNF::Float> __array_ops;
public:
    struct _dt {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()() {
            return __constants.dt();
        };
    };

    PDEProgramDNF::_dt dt;
    struct _dx {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()() {
            return __constants.dx();
        };
    };

    PDEProgramDNF::_dx dx;
    struct _nu {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()() {
            return __constants.nu();
        };
    };

    PDEProgramDNF::_nu nu;
    struct _three {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()() {
            return __array_ops.three_float();
        };
    };

    PDEProgramDNF::_three three;
    struct _unary_sub {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Offset operator()(const PDEProgramDNF::Offset& o) {
            return __array_ops.unary_sub(o);
        };
        __device__ __host__ inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& f) {
            return __array_ops.unary_sub(f);
        };
    };

    PDEProgramDNF::_unary_sub unary_sub;
private:
    __device__ __host__ inline void one0(PDEProgramDNF::Float& o) {
        o = __array_ops.one_float();
    };
    __device__ __host__ inline void two0(PDEProgramDNF::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgramDNF::Float>::Axis Axis;
    struct _rotateIx {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Index operator()(const PDEProgramDNF::Index& ix, const PDEProgramDNF::Axis& axis, const PDEProgramDNF::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    PDEProgramDNF::_rotateIx rotateIx;
    struct _zero {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    PDEProgramDNF::_zero zero;
private:
    __device__ __host__ inline void one0(PDEProgramDNF::Axis& o) {
        o = __array_ops.one_axis();
    };
    __device__ __host__ inline void two0(PDEProgramDNF::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgramDNF::Float>::Array Array;
    struct _binary_add {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    PDEProgramDNF::_binary_add binary_add;
    struct _binary_sub {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    PDEProgramDNF::_binary_sub binary_sub;
    struct _div {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& num, const PDEProgramDNF::Array& den) {
            return __array_ops.div(num, den);
        };
        __device__ __host__ inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& num, const PDEProgramDNF::Float& den) {
            return __array_ops.div(num, den);
        };
    };

    PDEProgramDNF::_div div;
    struct _mul {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Float& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& lhs, const PDEProgramDNF::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    PDEProgramDNF::_mul mul;
    struct _psi {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Index& ix, const PDEProgramDNF::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    PDEProgramDNF::_psi psi;
    struct _rotate {
    private:
        array_ops<PDEProgramDNF::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& a, const PDEProgramDNF::Axis& axis, const PDEProgramDNF::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    PDEProgramDNF::_rotate rotate;
    struct _substep {
    private:
        PDEProgramDNF::_mul mul;
        PDEProgramDNF::_binary_add binary_add;
        PDEProgramDNF::_div div;
        PDEProgramDNF::_binary_sub binary_sub;
        PDEProgramDNF::_dt dt;
        PDEProgramDNF::_dx dx;
        PDEProgramDNF::_nu nu;
        PDEProgramDNF::_one one;
        PDEProgramDNF::_rotate rotate;
        PDEProgramDNF::_three three;
        PDEProgramDNF::_two two;
        PDEProgramDNF::_unary_sub unary_sub;
        PDEProgramDNF::_zero zero;
    public:
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& u, const PDEProgramDNF::Array& v, const PDEProgramDNF::Array& u0, const PDEProgramDNF::Array& u1, const PDEProgramDNF::Array& u2) {
            return binary_add(u, mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(rotate(v, zero(), unary_sub(one.operator()<Offset>())), rotate(v, zero(), one.operator()<Offset>())), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, one.operator()<Axis>(), one.operator()<Offset>())), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, two.operator()<Axis>(), one.operator()<Offset>()))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), u0))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(rotate(v, zero(), one.operator()<Offset>()), rotate(v, zero(), unary_sub(one.operator()<Offset>()))), u0), mul(binary_sub(rotate(v, one.operator()<Axis>(), one.operator()<Offset>()), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u1)), mul(binary_sub(rotate(v, two.operator()<Axis>(), one.operator()<Offset>()), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u2))))));
        };
    };

    PDEProgramDNF::_substep substep;
    struct _substepIx {
    private:
        PDEProgramDNF::_mul mul;
        PDEProgramDNF::_binary_add binary_add;
        PDEProgramDNF::_div div;
        PDEProgramDNF::_binary_sub binary_sub;
        PDEProgramDNF::_dt dt;
        PDEProgramDNF::_dx dx;
        PDEProgramDNF::_nu nu;
        PDEProgramDNF::_one one;
        PDEProgramDNF::_psi psi;
        PDEProgramDNF::_rotateIx rotateIx;
        PDEProgramDNF::_three three;
        PDEProgramDNF::_two two;
        PDEProgramDNF::_unary_sub unary_sub;
        PDEProgramDNF::_zero zero;
    public:
        __device__ __host__ inline PDEProgramDNF::Float operator()(const PDEProgramDNF::Array& u, const PDEProgramDNF::Array& v, const PDEProgramDNF::Array& u0, const PDEProgramDNF::Array& u1, const PDEProgramDNF::Array& u2, const PDEProgramDNF::Index& ix) {
            return binary_add(psi(ix, u), mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(psi(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>())), v), psi(rotateIx(ix, zero(), one.operator()<Offset>()), v)), psi(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>()), v)), psi(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>()), v))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), psi(ix, u0)))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(psi(rotateIx(ix, zero(), one.operator()<Offset>()), v), psi(rotateIx(ix, zero(), unary_sub(one.operator()<Offset>())), v)), psi(ix, u0)), mul(binary_sub(psi(rotateIx(ix, one.operator()<Axis>(), one.operator()<Offset>()), v), psi(rotateIx(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(ix, u1))), mul(binary_sub(psi(rotateIx(ix, two.operator()<Axis>(), one.operator()<Offset>()), v), psi(rotateIx(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(ix, u2)))))));
        };
    };

private:
    forall_ops<PDEProgramDNF::Array, PDEProgramDNF::Axis, PDEProgramDNF::Float, PDEProgramDNF::Index, PDEProgramDNF::Nat, PDEProgramDNF::Offset, PDEProgramDNF::_substepIx> __forall_ops;
public:
    struct _schedule {
    private:
        forall_ops<PDEProgramDNF::Array, PDEProgramDNF::Axis, PDEProgramDNF::Float, PDEProgramDNF::Index, PDEProgramDNF::Nat, PDEProgramDNF::Offset, PDEProgramDNF::_substepIx> __forall_ops;
    public:
        __device__ __host__ inline PDEProgramDNF::Array operator()(const PDEProgramDNF::Array& u, const PDEProgramDNF::Array& v, const PDEProgramDNF::Array& u0, const PDEProgramDNF::Array& u1, const PDEProgramDNF::Array& u2) {
            return __forall_ops.schedule(u, v, u0, u1, u2);
        };
    };

    struct _step {
    private:
        PDEProgramDNF::_schedule schedule;
    public:
        __device__ __host__ inline void operator()(PDEProgramDNF::Array& u0, PDEProgramDNF::Array& u1, PDEProgramDNF::Array& u2) {
            PDEProgramDNF::Array v0 = u0;
            PDEProgramDNF::Array v1 = u1;
            PDEProgramDNF::Array v2 = u2;
            v0 = schedule(v0, u0, u0, u1, u2);
            v1 = schedule(v1, u1, u0, u1, u2);
            v2 = schedule(v2, u2, u0, u1, u2);
            u0 = schedule(u0, v0, u0, u1, u2);
            u1 = schedule(u1, v1, u0, u1, u2);
            u2 = schedule(u2, v2, u0, u1, u2);
        };
    };

    PDEProgramDNF::_step step;
    PDEProgramDNF::_schedule schedule;
    PDEProgramDNF::_substepIx substepIx;
};
} // examples
} // pde
} // mg_src
} // pde_cuda

namespace examples {
namespace pde {
namespace mg_src {
namespace pde_cuda {
struct PDEProgramPadded {
    struct _two {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgramPadded::two0(o);
            return o;
        };
    };

    PDEProgramPadded::_two two;
    struct _one {
        template <typename T>
        __device__ __host__ inline T operator()() {
            T o;
            PDEProgramPadded::one0(o);
            return o;
        };
    };

    PDEProgramPadded::_one one;
private:
    constants __constants;
public:
    typedef constants::Float Float;
    typedef array_ops<PDEProgramPadded::Float>::Index Index;
    typedef scalar_index<PDEProgramPadded::Index>::ScalarIndex ScalarIndex;
private:
    scalar_index<PDEProgramPadded::Index> __scalar_index;
public:
    struct _ix0 {
    private:
        scalar_index<PDEProgramPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgramPadded::ScalarIndex operator()(const PDEProgramPadded::Index& ix) {
            return __scalar_index.ix0(ix);
        };
    };

    PDEProgramPadded::_ix0 ix0;
    struct _ix1 {
    private:
        scalar_index<PDEProgramPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgramPadded::ScalarIndex operator()(const PDEProgramPadded::Index& ix) {
            return __scalar_index.ix1(ix);
        };
    };

    PDEProgramPadded::_ix1 ix1;
    struct _ix2 {
    private:
        scalar_index<PDEProgramPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgramPadded::ScalarIndex operator()(const PDEProgramPadded::Index& ix) {
            return __scalar_index.ix2(ix);
        };
    };

    PDEProgramPadded::_ix2 ix2;
    struct _mkIx {
    private:
        scalar_index<PDEProgramPadded::Index> __scalar_index;
    public:
        __device__ __host__ inline PDEProgramPadded::Index operator()(const PDEProgramPadded::ScalarIndex& a, const PDEProgramPadded::ScalarIndex& b, const PDEProgramPadded::ScalarIndex& c) {
            return __scalar_index.mkIx(a, b, c);
        };
    };

    PDEProgramPadded::_mkIx mkIx;
    typedef array_ops<PDEProgramPadded::Float>::Nat Nat;
    typedef array_ops<PDEProgramPadded::Float>::Offset Offset;
private:
    __device__ __host__ inline void one0(PDEProgramPadded::Offset& o) {
        o = __array_ops.one_offset();
    };
    array_ops<PDEProgramPadded::Float> __array_ops;
public:
    struct _dt {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()() {
            return __constants.dt();
        };
    };

    PDEProgramPadded::_dt dt;
    struct _dx {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()() {
            return __constants.dx();
        };
    };

    PDEProgramPadded::_dx dx;
    struct _nu {
    private:
        constants __constants;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()() {
            return __constants.nu();
        };
    };

    PDEProgramPadded::_nu nu;
    struct _three {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()() {
            return __array_ops.three_float();
        };
    };

    PDEProgramPadded::_three three;
    struct _unary_sub {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& f) {
            return __array_ops.unary_sub(f);
        };
        __device__ __host__ inline PDEProgramPadded::Offset operator()(const PDEProgramPadded::Offset& o) {
            return __array_ops.unary_sub(o);
        };
    };

    PDEProgramPadded::_unary_sub unary_sub;
private:
    __device__ __host__ inline void one0(PDEProgramPadded::Float& o) {
        o = __array_ops.one_float();
    };
    __device__ __host__ inline void two0(PDEProgramPadded::Float& o) {
        o = __array_ops.two_float();
    };
public:
    typedef array_ops<PDEProgramPadded::Float>::Axis Axis;
    struct _rotateIx {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Index operator()(const PDEProgramPadded::Index& ix, const PDEProgramPadded::Axis& axis, const PDEProgramPadded::Offset& o) {
            return __array_ops.rotateIx(ix, axis, o);
        };
    };

    PDEProgramPadded::_rotateIx rotateIx;
    struct _zero {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Axis operator()() {
            return __array_ops.zero_axis();
        };
    };

    PDEProgramPadded::_zero zero;
private:
    __device__ __host__ inline void one0(PDEProgramPadded::Axis& o) {
        o = __array_ops.one_axis();
    };
    __device__ __host__ inline void two0(PDEProgramPadded::Axis& o) {
        o = __array_ops.two_axis();
    };
public:
    typedef array_ops<PDEProgramPadded::Float>::Array Array;
private:
    forall_ops<PDEProgramPadded::Array, PDEProgramPadded::Axis, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::Nat, PDEProgramPadded::Offset> __forall_ops;
public:
    struct _binary_add {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Float& rhs) {
            return __array_ops.binary_add(lhs, rhs);
        };
    };

    PDEProgramPadded::_binary_add binary_add;
    struct _binary_sub {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Float& rhs) {
            return __array_ops.binary_sub(lhs, rhs);
        };
    };

    PDEProgramPadded::_binary_sub binary_sub;
    struct _div {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& num, const PDEProgramPadded::Float& den) {
            return __array_ops.div(num, den);
        };
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& num, const PDEProgramPadded::Array& den) {
            return __array_ops.div(num, den);
        };
    };

    PDEProgramPadded::_div div;
    struct _mul {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Array& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
        __device__ __host__ inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Float& lhs, const PDEProgramPadded::Float& rhs) {
            return __array_ops.mul(lhs, rhs);
        };
    };

    PDEProgramPadded::_mul mul;
    struct _psi {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Index& ix, const PDEProgramPadded::Array& array) {
            return __array_ops.psi(ix, array);
        };
    };

    PDEProgramPadded::_psi psi;
    struct _refillPadding {
    private:
        forall_ops<PDEProgramPadded::Array, PDEProgramPadded::Axis, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::Nat, PDEProgramPadded::Offset> __forall_ops;
    public:
        __device__ __host__ inline void operator()(PDEProgramPadded::Array& a) {
            return __forall_ops.refillPadding(a);
        };
    };

    PDEProgramPadded::_refillPadding refillPadding;
    struct _rotate {
    private:
        array_ops<PDEProgramPadded::Float> __array_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& a, const PDEProgramPadded::Axis& axis, const PDEProgramPadded::Offset& o) {
            return __array_ops.rotate(a, axis, o);
        };
    };

    PDEProgramPadded::_rotate rotate;
    struct _rotateIxPadded {
    private:
        forall_ops<PDEProgramPadded::Array, PDEProgramPadded::Axis, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::Nat, PDEProgramPadded::Offset> __forall_ops;
    public:
        __device__ __host__ inline PDEProgramPadded::Index operator()(const PDEProgramPadded::Index& ix, const PDEProgramPadded::Axis& axis, const PDEProgramPadded::Offset& offset) {
            return __forall_ops.rotateIxPadded(ix, axis, offset);
        };
    };

    PDEProgramPadded::_rotateIxPadded rotateIxPadded;
    struct _substep {
    private:
        PDEProgramPadded::_mul mul;
        PDEProgramPadded::_binary_add binary_add;
        PDEProgramPadded::_div div;
        PDEProgramPadded::_binary_sub binary_sub;
        PDEProgramPadded::_dt dt;
        PDEProgramPadded::_dx dx;
        PDEProgramPadded::_nu nu;
        PDEProgramPadded::_one one;
        PDEProgramPadded::_rotate rotate;
        PDEProgramPadded::_three three;
        PDEProgramPadded::_two two;
        PDEProgramPadded::_unary_sub unary_sub;
        PDEProgramPadded::_zero zero;
    public:
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2) {
            return binary_add(u, mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(rotate(v, zero(), unary_sub(one.operator()<Offset>())), rotate(v, zero(), one.operator()<Offset>())), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, one.operator()<Axis>(), one.operator()<Offset>())), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), rotate(v, two.operator()<Axis>(), one.operator()<Offset>()))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), u0))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(rotate(v, zero(), one.operator()<Offset>()), rotate(v, zero(), unary_sub(one.operator()<Offset>()))), u0), mul(binary_sub(rotate(v, one.operator()<Axis>(), one.operator()<Offset>()), rotate(v, one.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u1)), mul(binary_sub(rotate(v, two.operator()<Axis>(), one.operator()<Offset>()), rotate(v, two.operator()<Axis>(), unary_sub(one.operator()<Offset>()))), u2))))));
        };
    };

    PDEProgramPadded::_substep substep;
    struct _substepIx {
    private:
        PDEProgramPadded::_mul mul;
        PDEProgramPadded::_binary_add binary_add;
        PDEProgramPadded::_div div;
        PDEProgramPadded::_binary_sub binary_sub;
        PDEProgramPadded::_dt dt;
        PDEProgramPadded::_dx dx;
        PDEProgramPadded::_nu nu;
        PDEProgramPadded::_one one;
        PDEProgramPadded::_psi psi;
        PDEProgramPadded::_rotateIxPadded rotateIxPadded;
        PDEProgramPadded::_three three;
        PDEProgramPadded::_two two;
        PDEProgramPadded::_unary_sub unary_sub;
        PDEProgramPadded::_zero zero;
    public:
        __device__ __host__ inline PDEProgramPadded::Float operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2, const PDEProgramPadded::Index& ix) {
            return binary_add(psi(ix, u), mul(div(dt(), two.operator()<Float>()), binary_sub(mul(nu(), binary_sub(mul(div(div(one.operator()<Float>(), dx()), dx()), binary_add(binary_add(binary_add(binary_add(binary_add(psi(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>())), v), psi(rotateIxPadded(ix, zero(), one.operator()<Offset>()), v)), psi(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>()), v)), psi(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>()), v))), mul(div(div(mul(three(), two.operator()<Float>()), dx()), dx()), psi(ix, u0)))), mul(div(div(one.operator()<Float>(), two.operator()<Float>()), dx()), binary_add(binary_add(mul(binary_sub(psi(rotateIxPadded(ix, zero(), one.operator()<Offset>()), v), psi(rotateIxPadded(ix, zero(), unary_sub(one.operator()<Offset>())), v)), psi(ix, u0)), mul(binary_sub(psi(rotateIxPadded(ix, one.operator()<Axis>(), one.operator()<Offset>()), v), psi(rotateIxPadded(ix, one.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(ix, u1))), mul(binary_sub(psi(rotateIxPadded(ix, two.operator()<Axis>(), one.operator()<Offset>()), v), psi(rotateIxPadded(ix, two.operator()<Axis>(), unary_sub(one.operator()<Offset>())), v)), psi(ix, u2)))))));
        };
    };

private:
    forall_ops<PDEProgramPadded::Array, PDEProgramPadded::Axis, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::Nat, PDEProgramPadded::Offset, PDEProgramPadded::_substepIx> __forall_ops0;
    padded_schedule<PDEProgramPadded::Array, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::_substepIx> __padded_schedule;
public:
    struct _schedule {
    private:
        forall_ops<PDEProgramPadded::Array, PDEProgramPadded::Axis, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::Nat, PDEProgramPadded::Offset, PDEProgramPadded::_substepIx> __forall_ops0;
    public:
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2) {
            return __forall_ops0.schedule(u, v, u0, u1, u2);
        };
    };

    PDEProgramPadded::_schedule schedule;
    struct _schedulePadded {
    private:
        padded_schedule<PDEProgramPadded::Array, PDEProgramPadded::Float, PDEProgramPadded::Index, PDEProgramPadded::_substepIx> __padded_schedule;
    public:
        __device__ __host__ inline PDEProgramPadded::Array operator()(const PDEProgramPadded::Array& u, const PDEProgramPadded::Array& v, const PDEProgramPadded::Array& u0, const PDEProgramPadded::Array& u1, const PDEProgramPadded::Array& u2) {
            return __padded_schedule.schedulePadded(u, v, u0, u1, u2);
        };
    };

    struct _step {
    private:
        PDEProgramPadded::_refillPadding refillPadding;
        PDEProgramPadded::_schedulePadded schedulePadded;
    public:
        __device__ __host__ inline void operator()(PDEProgramPadded::Array& u0, PDEProgramPadded::Array& u1, PDEProgramPadded::Array& u2) {
            PDEProgramPadded::Array v0 = u0;
            PDEProgramPadded::Array v1 = u1;
            PDEProgramPadded::Array v2 = u2;
            v0 = [&]() {
                PDEProgramPadded::Array result = schedulePadded(v0, u0, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            v1 = [&]() {
                PDEProgramPadded::Array result = schedulePadded(v1, u1, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            v2 = [&]() {
                PDEProgramPadded::Array result = schedulePadded(v2, u2, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            u0 = [&]() {
                PDEProgramPadded::Array result = schedulePadded(u0, v0, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            u1 = [&]() {
                PDEProgramPadded::Array result = schedulePadded(u1, v1, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
            u2 = [&]() {
                PDEProgramPadded::Array result = schedulePadded(u2, v2, u0, u1, u2);
                refillPadding(result);
                return result;
            }();
        };
    };

    PDEProgramPadded::_step step;
    PDEProgramPadded::_schedulePadded schedulePadded;
    PDEProgramPadded::_substepIx substepIx;
};
} // examples
} // pde
} // mg_src
} // pde_cuda
