#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

#include <omp.h>

#define SIDE 512
#define NTILES 4
#define NB_CORES 2

struct array_ops {
  typedef float Float;
  struct Offset { int value; };
  struct Axis { size_t value; };
  typedef size_t Index;
  struct Nat { size_t value; };

  struct Array {
    std::unique_ptr<Float[]> content;
    Array() {
      this->content = std::unique_ptr<Float[]>(new Float[SIDE * SIDE * SIDE]);
    }

    Array(const Array &other) {
      this->content = std::unique_ptr<Float[]>(new Float[SIDE * SIDE * SIDE]);
      memcpy(this->content.get(), other.content.get(),
             SIDE * SIDE * SIDE * sizeof(Float));
    }

    Array &operator=(const Array &other) {
      this->content = std::unique_ptr<Float[]>(new Float[SIDE * SIDE * SIDE]);
      memcpy(this->content.get(), other.content.get(),
             SIDE * SIDE * SIDE * sizeof(Float));
      return *this;
    }

    inline Float operator[](const Index &ix) const {
      return this->content[ix];
    }

    inline Float &operator[](const Index &ix) {
      return this->content[ix];
    }
  };

  inline Float psi(const Index &ix, const Array &array) { return array[ix]; }

  /* Float ops */
  inline Float unary_sub(const Float &f) { return -f; }
  inline Float binary_add(const Float &lhs, const Float &rhs) {
    return lhs + rhs;
  }
  inline Float binary_sub(const Float &lhs, const Float &rhs) {
    return lhs - rhs;
  }
  inline Float mul(const Float &lhs, const Float &rhs) {
    return lhs * rhs;
  }
  inline Float div(const Float &num, const Float &den) {
    return num / den;
  }
  inline Float one_float() { return 1; }
  inline Float two_float() { return 2; }
  inline Float three_float() { return 3; }

  /* Scalar-Array ops */
  inline Array binary_add(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs + rhs[i];
    }
    return out;
  }
  inline Array binary_sub(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs - rhs[i];
    }
    return out;
  }
  inline Array mul(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs * rhs[i];
    }
    return out;
  }
  inline Array div(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs / rhs[i];
    }
    return out;
  }

  /* Array-Array ops */
  inline Array binary_add(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] + rhs[i];
    }
    return out;
  }
  inline Array binary_sub(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] - rhs[i];
    }
    return out;
  }
  inline Array mul(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] * rhs[i];
    }
    return out;
  }
  inline Array div(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      out[i] = lhs[i] / rhs[i];
    }
    return out;
  }

  [[noreturn]] inline Array rotate(const Array &array, const Axis &axis, const Offset &o) {
    throw "rotate not implemented";
    //std::unreachable(); // Always optimize with DNF, do not rotate
  }

  inline Index rotate_ix(const Index &ix, const Axis &axis, const Offset &offset) {
    if (axis.value == 0) {
      return (ix + (offset.value * SIDE * SIDE)) % (SIDE * SIDE * SIDE);
    } else if (axis.value == 1) {
      size_t ix_subarray_base = ix / (SIDE * SIDE);
      size_t ix_in_subarray = (ix + offset.value * SIDE) % (SIDE * SIDE);
      return ix_subarray_base + ix_in_subarray;
    } else if (axis.value == 2) {
      size_t ix_subarray_base = ix / SIDE;
      size_t ix_in_subarray = (ix + offset.value) % SIDE;
      return ix_subarray_base + ix_in_subarray;
    }

    throw "failed at rotating index";
    //std::unreachable();
    return 0;
  }

  inline Axis zero_axis() { return Axis(0); }
  inline Axis one_axis() { return Axis(1); }
  inline Axis two_axis() { return Axis(2); }

  inline Offset one_offset() { return Offset(1); }
  inline Offset unary_sub(const Offset &offset) { return Offset(-offset.value); }
};

template <typename _Array, typename _Axis, typename _Float, typename _Index,
          typename _Nat, typename _Offset, class _snippet_ix>
struct forall_ops {
  typedef _Array Array;
  typedef _Axis Axis;
  typedef _Float Float;
  typedef _Index Index;
  typedef _Nat Nat;
  typedef _Offset Offset;

  _snippet_ix snippet_ix;

  inline Nat nbCores() { return Nat(NB_CORES); }

  inline Array forall_ix_snippet(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Float &c0,
      const Float &c1, const Float &c2, const Float &c3, const Float &c4) {
    Array result;
    std::cout << "in forall_ix_snippet" << std::endl;
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      result[i] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
    }

    std::cout << result[SIDE * SIDE * SIDE - 1] << " "
              << result[0] << std::endl;

    return result;
  }

  inline Array forall_ix_snippet_threaded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Float &c0,
      const Float &c1, const Float &c2, const Float &c3, const Float &c4,
      const Nat &nbThreads) {
    Array result;
    omp_set_num_threads(nbThreads.value);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
      result[i] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
    }
    return result;
  }

  inline Array forall_ix_snippet_tiled(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Float &c0,
      const Float &c1, const Float &c2, const Float &c3, const Float &c4) {
    Array result;

    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t ti = 0; ti < SIDE; ti += SIDE/NTILES) {
      for (size_t tj = 0; tj < SIDE; tj += SIDE/NTILES) {
        for (size_t tk = 0; tk < SIDE; tk += SIDE/NTILES) {
          for (size_t i = ti; i < ti + SIDE/NTILES; ++i) {
            for (size_t j = tj; j < tj + SIDE/NTILES; ++j) {
              for (size_t k = tk; k < tk + SIDE/NTILES; ++k) {
                size_t ix = i * SIDE * SIDE + j * SIDE + k;
                //assert (ix < SIDE * SIDE * SIDE);
                result[ix] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
              }
            }
          }
        }
      }
    }

    return result;
  }
};

inline void dumpsine(array_ops::Array &result) {
  double step = 0.01;
  double PI = 3.14159265358979323846;
  double amplitude = 10.0;
  double phase = 0.0125;
  double t = 0.0;

  for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) {
    result[i] = amplitude * sin(PI * t + phase);
    t += step;
  }
}