#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

struct array_ops {

    struct Shape {
        std::vector<size_t> components;

        Shape(std::vector<size_t> components) : components(components) {}
        
        size_t index_space_size(void) const {
            size_t total = 1;
            for (auto it = components.begin(); it != components.end(); ++it) {
                total *= *it;
            }
            return total;
        }

        size_t dim(void) const {
            return this->components.size();
        }

        bool operator==(const Shape &other) const {
            return this->components == other.components;
        }
    };

    struct Index {
        std::vector<size_t> value;
        Index(std::vector<size_t> value) : value(value) {}

        size_t dim(void) const {
            return this->value.size();
        }

        inline size_t to_linear(const Shape &shape) const {
            size_t index_space_size = shape.index_space_size();
            size_t linear_ix = 0;

            assert (this->dim() == shape.dim());

            for (auto shape_it = shape.components.begin(), ix_it = this->value.begin();
                 shape_it != shape.components.end() && ix_it != this->value.end();
                 ++shape_it, ++ix_it) {
                index_space_size /= *shape_it;
                linear_ix += *ix_it * index_space_size;
            }

            return linear_ix;
        }

        inline Index to_padded(const std::vector<size_t> &offsets) const {
            std::vector<size_t> out_index;
            for (auto ix_it = this->value.begin(), offset_it = offsets.begin();
                 ix_it != this->value.end() && offset_it != offsets.end();
                 ++ix_it, ++offset_it) {
                out_index.push_back(*ix_it + *offset_it);
            }

            return Index(out_index);
        }

    };

    struct Offset {
        int value;
        Offset(int value) : value(value) {}
        Offset() : value(0) {}
    };

    inline Offset one_offset() {
        return Offset(1);
    }

    struct Axis {
        size_t value;
        Axis(size_t value) : value(value) {}
        Axis() : value(0) {}
    };

    inline Axis zero_axis() {
        return Axis(0);
    }

    inline Axis one_axis() {
        return Axis(1);
    }

    inline Axis two_axis() {
        return Axis(2);
    }

    struct Float {
        float value;
        Float(float value) : value(value) {}
        Float() : value(0) {}
    };

    inline Float one_float() {
        return Float(1);
    }

    inline Float two_float() {
        return Float(2);
    }

    inline Float three_float() {
        return Float(3);
    }

    struct Array {
        private:
            Float *m_content;
            Shape m_shape;

        public:
            Array(const Shape &shape, Float *content) : m_shape(shape) {
                auto array_size = shape.index_space_size();
                this->m_content = new Float[array_size];
                memcpy(this->m_content, content, array_size * sizeof(Float));
            }

            Array(const Shape &shape) : m_shape(shape) {
                auto array_size = shape.index_space_size();
                this->m_content = new Float[array_size];
            }

            Shape shape() const {
                return this->m_shape;
            }

            size_t size() const {
                return this->m_shape.index_space_size();
            }

            Float operator[](const Index &ix) const {
                auto linear_ix = ix.to_linear(this->m_shape);
                return this->m_content[linear_ix];
            }

            Float &operator[](const Index &ix) {
                auto linear_ix = ix.to_linear(this->m_shape);
                return this->m_content[linear_ix];
            }

            Float *unsafe_content() const {
                return this->m_content;
            }

            void rotate(const Axis &in_axis, const Offset &in_offset) {
                auto rotate_aux = [&] (Float *content_ptr, const Axis &axis,
                                       const Offset &offset) {
                   // TODO; 
                };
            }
    };

    struct PaddedArray {
        private:
            Float *m_content;
            Shape m_shape;
            std::vector<std::pair<size_t, size_t>> m_bounds;

        public:
            PaddedArray(const Array &array) : m_shape(array.shape()) {
                auto array_size = this->m_shape.index_space_size();
                this->m_content = new Float[array_size];
                memcpy(this->m_content, array.unsafe_content(), array_size * sizeof(Float));
            }

            Array outer() const {
                return Array(this->m_shape, this->m_content);
            }

    };

    PaddedArray asPadded(const Array &array) { return PaddedArray(array); }

    /* Float ops */
    inline Float unary_sub(Float f) {
        return Float(-f.value);
    }

    inline Float binary_add(Float lhs, Float rhs) {
        return Float(lhs.value + rhs.value);
    }

    inline Float binary_sub(Float lhs, Float rhs) {
        return Float(lhs.value - rhs.value);
    }

    inline Float mul(Float lhs, Float rhs) {
        return Float(lhs.value * rhs.value);
    }

    inline Float div(Float num, Float den) {
        return Float(num.value / den.value);
    }

    /* Scalar-Array ops */
    inline Array binary_add(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return binary_add(lhs, rhs[ix]);
        };
        
        return forall_ix(rhs.shape(), fn); 
    }

    inline Array binary_sub(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return binary_sub(lhs, rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array mul(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return mul(lhs, rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array div(const Float &lhs, const Array &rhs) {
        auto fn = [&](const Index &ix) {
            return div(lhs, rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    /* Array-Array ops */
    inline Array binary_add(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return binary_add(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array binary_sub(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return binary_sub(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array mul(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return mul(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array div(const Array &lhs, const Array &rhs) {
        assert (lhs.shape() == rhs.shape());
        auto fn = [&](const Index &ix) {
            return div(lhs[ix], rhs[ix]);
        };

        return forall_ix(rhs.shape(), fn);
    }

    inline Array forall_ix(const Shape &shape, auto& fn) {
        auto out_array = Array(shape);
        for (size_t linear_ix = 0; linear_ix < shape.index_space_size();
             ++linear_ix) {
            Index ix = from_linear(shape, linear_ix);
            out_array[ix] = fn(ix);
        }
        return out_array;
    }

    inline Array forall_ix_padded(const Shape &out_shape,
                                  std::vector<std::pair<size_t, size_t>> bounds,
                                  auto &fn) {
        std::vector<size_t> offsets;
        for (auto bounds_it = bounds.begin(); bounds_it != bounds.end(); ++bounds_it) {
            offsets.push_back((*bounds_it).first);
        }

        auto out_array = Array(out_shape);
        for (size_t linear_ix = 0; linear_ix < out_shape.index_space_size();
             ++linear_ix) {
            Index ix = from_linear(out_shape, linear_ix);
            out_array[ix] = fn(ix.to_padded(offsets));
        }

        return out_array;
    }

    inline void set(const Index &ix, Array &array, const Float &value) {
        array[ix] = value;
    }

    inline Float psi(const Index &ix, const Array &array) {
        return array[ix];
    }

    Array rotate(const Array &array, const Axis &axis, const Offset &offset) {
       Array result = array;
       result.rotate(axis, offset);
       return result;
    }

    inline Index from_linear(const Shape &shape, size_t linear_ix) {
        size_t index_space_size = shape.index_space_size();
        std::vector<size_t> ix;

        for (auto it = shape.components.begin(); it != shape.components.end(); ++it) {
            index_space_size /= *it;
            ix.push_back(linear_ix / index_space_size);
            linear_ix %= index_space_size;
        }

        return Index(ix);
    }

    /* Offset utils */
    inline Offset unary_sub(const Offset &offset) {
        return Offset(-offset.value);
    }
};

// TODO: bypassing extend mechanism, not great
template <typename _Array, typename _Axis, typename _Float, typename _Index,
          typename _Offset, typename _PaddedArray, class _snippet_ix>
struct forall_ops {
    private:
        array_ops _array_ops;
    public:
    typedef _Array Array;
    typedef _Axis Axis;
    typedef _Float Float;
    typedef _Index Index;
    typedef _Offset Offset;

    static _snippet_ix snippet_ix;

    Array forall_snippet_ix(const Array &u, const Array &v, const Array &u0,
                           const Array &u1, const Array &u2, const Float &c0,
                           const Float &c1, const Float &c2, const Float &c3,
                           const Float &c4) {
        auto fn = [&](const Index &ix) {
            return snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
        };
        return _array_ops.forall_ix(u.shape(), fn);
    }
};


inline array_ops::Array dumpsine(const array_ops::Shape &shape) {
    double step = 0.01;
    double PI = 3.14159265358979323846;
    double amplitude = 10.0;
    double phase = 0.0125;
    double t = 0.0;
    array_ops ops;

    auto fn = [&](const array_ops::Index &ix) {
        auto timestep = step * ix.to_linear(shape);
        return array_ops::Float(amplitude * sin(PI * t + phase));
    };
    return ops.forall_ix(shape, fn);
}
