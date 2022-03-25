#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

#include <omp.h>

#define S0 512
#define S1 512
#define S2 512
#define PAD0 0
#define PAD1 0
#define PAD2 0

#define PADDED_S0 (S0 + 2 * PAD0)
#define PADDED_S1 (S1 + 2 * PAD1)
#define PADDED_S2 (S2 + 2 * PAD2)

#define TOTAL_PADDED_SIZE (PADDED_S0 * PADDED_S1 * PADDED_S21)

#define NTILES 4
#define NB_CORES 2

struct array_ops {
  typedef float Float;
  struct Offset { int value; };
  struct Axis { size_t value; };
  typedef size_t Index;
  struct Nat { size_t value; };

  struct Array {
    std::shared_ptr<Float[]> content;
    Array() {
      this->content = std::shared_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
    }

    Array(std::shared_ptr<Float[]> content_ptr) {
      this->content = content_ptr;
    }

    Array(const Array &other) {
      this->content = std::shared_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
      memcpy(this->content.get(), other.content.get(),
             TOTAL_PADDED_SIZE * sizeof(Float));
    }

    Array(Array &&other) {
        this->content = std::move(other.content);
    }

    Array &operator=(const Array &other) {
      this->content = std::shared_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
      memcpy(this->content.get(), other.content.get(),
             TOTAL_PADDED_SIZE * sizeof(Float));
      return *this;
    }

    Array &operator=(Array &&other) {
        this->content = std::move(other.content);
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
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs + rhs[i];
    }
    return out;
  }
  inline Array binary_sub(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs - rhs[i];
    }
    return out;
  }
  inline Array mul(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs * rhs[i];
    }
    return out;
  }
  inline Array div(const Float &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs / rhs[i];
    }
    return out;
  }

  /* Array-Array ops */
  inline Array binary_add(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs[i] + rhs[i];
    }
    return out;
  }
  inline Array binary_sub(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs[i] - rhs[i];
    }
    return out;
  }
  inline Array mul(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs[i] * rhs[i];
    }
    return out;
  }
  inline Array div(const Array &lhs, const Array &rhs) {
    Array out;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      out[i] = lhs[i] / rhs[i];
    }
    return out;
  }

  [[noreturn]] inline Array rotate(const Array &array, const Axis &axis, const Offset &o) {
    throw "rotate not implemented";
    //std::unreachable(); // Always optimize with DNF, do not rotate
  }

  inline Index rotate_ix(const Index &ix,
                         const Axis &axis,
                         const Offset &offset) {
    if (axis.value == 0) {
      return (ix + (offset.value * PADDED_S1 * PADDED_S2)) % TOTAL_PADDED_SIZE;
    } else if (axis.value == 1) {
      size_t ix_subarray_base = ix / (PADDED_S1 * PADDED_S2);
      size_t ix_in_subarray = (ix + offset.value * PADDED_S2) % (PADDED_S1 * PADDED_S2);
      return ix_subarray_base + ix_in_subarray;
    } else if (axis.value == 2) {
      size_t ix_subarray_base = ix / PADDED_S2;
      size_t ix_in_subarray = (ix + offset.value) % PADDED_S2;
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
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      result[i] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, i);
    }

    std::cout << result[TOTAL_PADDED_SIZE - 1] << " "
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
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
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
                result[ix] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
              }
            }
          }
        }
      }
    }

    return result;
  }

  /* OF Pad extension */
  struct PaddedArray {
    std::shared_ptr<Float[]> content;
    size_t padding[3];

    PaddedArray(const size_t padding[3]) {
      for (size_t i = 0; i < 3; ++i) {
        this->padding[i] = padding[i];
      }

      size_t total_size = (SIDE + this->padding[0] * 2) *
                          (SIDE + this->padding[1] * 2) *
                          (SIDE + this->padding[2] * 2);
      this->content = std::shared_ptr<Float[]>(new Float[total_size]);
    }

    PaddedArray(const PaddedArray &other) {
      memcpy(this->padding, other.padding, 3 * sizeof(size_t));
      size_t total_size = (SIDE + this->padding[0] * 2) *
                          (SIDE + this->padding[1] * 2) *
                          (SIDE + this->padding[2] * 2);
      this->content = std::shared_ptr<Float[]>(new Float[total_size]);
      memcpy(this->content.get(), other.content.get(),
             total_size * sizeof(Float));
    }

    PaddedArray(PaddedArray &&other) {
      for (size_t i = 0; i < 3; ++i) {
        this->padding[i] = other.padding[i];
      }
      this->content = std::move(other.content);
    }

    PaddedArray(const Array &arr, const size_t padding[3]) {
      for (size_t i = 0; i < 3; ++i) {
        this->padding[i] = padding[i];
      }
      //std::cout << padding[0] << " " << padding[1] << " " << padding[2] << std::endl;

      size_t SIDE_0 = SIDE + padding[0] * 2,
             SIDE_1 = SIDE + padding[1] * 2,
             SIDE_2 = SIDE + padding[2] * 2;

      size_t total_size = SIDE_0 * SIDE_1 * SIDE_2;

      this->content = std::shared_ptr<Float[]>(new Float[total_size]);
      for (size_t i = 0; i < SIDE * SIDE * SIDE; ++i) this->content.get()[i] = 0;
      Float *raw_content = arr.content.get();
      Float *raw_padded_content = this->content.get();

      double begin = omp_get_wtime();
      // Axis 2
      for (size_t i = 0; i < SIDE; ++i) {
          for (size_t j = 0; j < SIDE; ++j) {
              size_t start_offset_padded = (padding[0] + i) * SIDE_1 * SIDE_2 + (padding[1] + j) * SIDE_2;
              size_t start_offset_unpadded = i * SIDE * SIDE + j * SIDE;
              Float *start_padded = raw_padded_content + start_offset_padded;
              Float *start_unpadded = raw_content + start_offset_unpadded;

              memcpy(start_padded, start_unpadded + SIDE - padding[2],
                     padding[2] * sizeof(Float)); // left pad
              memcpy(start_padded + padding[2], start_unpadded,
                     SIDE * sizeof(Float)); // main content
              memcpy(start_padded + padding[2] + SIDE, start_unpadded,
                     padding[2] * sizeof(Float)); // right pad
          }
      }
      double end = omp_get_wtime();

      std::cout << "overhead: " << end - begin << " [s]" << std::endl;

      // Axis 1
      for (size_t i = 0; i < SIDE; i++) {
          // [ ? <content> ? ]
          size_t start_offset = (padding[0] + i) * SIDE_1 * SIDE_2;
          Float *start_padded = raw_padded_content + start_offset;

          memcpy(start_padded,
                 start_padded + SIDE_1 * SIDE_2 - 2 * padding[1] * SIDE_2,
                 padding[1] * SIDE_2 * sizeof(Float)); // left pad
          memcpy(start_padded + SIDE_1 * SIDE_2 - padding[1] * SIDE_2,
                 start_padded + padding[1] * SIDE_2,
                 padding[1] * SIDE_2 * sizeof(Float)); // right pad
      }

      // Axis 0
      memcpy(raw_padded_content,
             raw_padded_content + total_size - 2 * padding[0] * SIDE_1 * SIDE_2,
             padding[0] * SIDE_1 * SIDE_2 * sizeof(Float)); // left pad
      memcpy(raw_padded_content + total_size - padding[0] * SIDE_1 * SIDE_2,
             raw_padded_content + padding[0] * SIDE_1 * SIDE_2,
             padding[0] * SIDE_1 * SIDE_2 * sizeof(Float)); // right pad
    }

    Array inner() const {
      size_t SIDE_0 = SIDE + this->padding[0] * 2,
             SIDE_1 = SIDE + this->padding[1] * 2,
             SIDE_2 = SIDE + this->padding[2] * 2;

      Array result;

      double begin = omp_get_wtime();
      for (size_t i = 0; i < SIDE; ++i) {
        for (size_t j = 0; j < SIDE; ++j) {
          size_t offset = (this->padding[0] + i) * SIDE_1 * SIDE_2 +
                          (this->padding[1] + j) * SIDE_2 +
                          this->padding[2];
          memcpy(result.content.get() + i * SIDE * SIDE + j * SIDE,
                 this->content.get() + offset,
                 SIDE * sizeof(Float));
        }
      }

      double end = omp_get_wtime();

      std::cout << "overhead inner: " << end - begin << " [s]" << std::endl;

      return result;
    }

    Array outer() const {
      return Array(this->content);
    }

    /*PaddedArray &operator=(const Array &in_array) {
      for (size_t i = 0; i < 3; ++i) { this->padding[i] = 0; }
      this->cont
    }*/

    PaddedArray &operator=(const PaddedArray &other) {
      memcpy(this->padding, other.padding, 3 * sizeof(size_t));
      size_t total_size = (SIDE + this->padding[0] * 2) *
                          (SIDE + this->padding[1] * 2) *
                          (SIDE + this->padding[2] * 2);
      this->content = std::shared_ptr<Float[]>(new Float[total_size]);
      memcpy(this->content.get(), other.content.get(),
             total_size * sizeof(Float));
      return *this;
    }

    PaddedArray &operator=(PaddedArray &&other) {
      this->padding = std::move(other.padding);
      this->content = std::move(other.content);
      return *this;
    }

    inline Float operator[](const Index &ix) const {
      return this->content[ix];
    }

    inline Float &operator[](const Index &ix) {
      return this->content[ix];
    }
  };

  struct PaddingAmount { size_t value; };

  PaddedArray cpadlr(const Array &a, const Axis &axis, const PaddingAmount &n) {
    size_t padding[3] = {0, 0, 0};
    padding[axis.value] = n.value;
    return PaddedArray(a, padding);
  }

  Array inner(const PaddedArray &paddedArray) {
    return paddedArray.inner();
  }

  PaddingAmount paddingAmount() {
    return PaddingAmount(1);
  }

  inline PaddedArray forall_ix_snippet_padded(const PaddedArray &pu,
      const PaddedArray &pv, const PaddedArray &pu0, const PaddedArray &pu1,
      const PaddedArray &pu2, const Float &c0, const Float &c1,
      const Float &c2, const Float &c3, const Float &c4) {

    const size_t SIDE_0 = SIDE + pu.padding[0] * 2,
                 SIDE_1 = SIDE + pu.padding[1] * 2,
                 SIDE_2 = SIDE + pu.padding[2] * 2;

    const Array u = pu.outer(), v = pv.outer(), u0 = pu0.outer(),
                u1 = pu1.outer(), u2 = pu2.outer();

    PaddedArray result(pu.padding);

    std::cout << "in forall_ix_snippet_padded" << std::endl;
    for (size_t i = pu.padding[0]; i < pu.padding[0] + SIDE; ++i) {
      for (size_t j = pu.padding[1]; j < pu.padding[1] + SIDE; ++j) {
        for (size_t k = pu.padding[2]; k < pu.padding[2] + SIDE; ++k) {
          size_t ix = i * SIDE_1 * SIDE_2 + j * SIDE_2 + k;
          result[ix] = snippet_ix(u, v, u0, u1, u2, c0, c1, c2, c3, c4, ix);
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

  for (size_t i = 0; i < S0; ++i) {
    for (size_t j = 0; j < S1; ++j) {
      for (size_t k = 0; k < S2; ++k) {
        size_t ix = (PAD0 + i) * PADDED_S1 * PADDED_S2 +
                    (PAD1 + j) * PADDED_S2 +
                    (PAD2 + k);
        result[ix] = amplitude * sin(PI * t + phase);
        t += step;
      }
    }
  }

  result.refill_padding();
}