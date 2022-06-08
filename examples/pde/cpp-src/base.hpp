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

#define PADDED_S0 (S0 + 2 * PAD0)
#define PADDED_S1 (S1 + 2 * PAD1)
#define PADDED_S2 (S2 + 2 * PAD2)

#define TOTAL_PADDED_SIZE (PADDED_S0 * PADDED_S1 * PADDED_S2)

#define NTILES 4
#define NB_CORES 2

#define S_DT 0.00082212448155679772495
#define S_NU 1.0
#define S_DX 1.0

struct constants {
    typedef float Float;
    constexpr Float nu() { return S_NU; }
    constexpr Float dt() { return S_DT; }
    constexpr Float dx() { return S_DX; }
};

template <typename _Float>
struct array_ops {
  typedef _Float Float;
  struct Offset { int value; };
  struct Axis { size_t value; };
  typedef size_t Index;
  struct Nat { size_t value; };

  struct Array {
    std::unique_ptr<Float[]> content;
    Array() {
      this->content = std::unique_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
    }

    Array(const Array &other) {
      this->content = std::unique_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
      memcpy(this->content.get(), other.content.get(),
             TOTAL_PADDED_SIZE * sizeof(Float));
    }

    Array(Array &&other) {
        this->content = std::move(other.content);
    }

    Array &operator=(const Array &other) {
      this->content = std::unique_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
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

    /* OF Pad extension */
    void replenish_padding() {
      Float *raw_content = this->content.get();

      double begin = omp_get_wtime();
      // Axis 2
      if (PAD2 > 0) {
        for (size_t i = 0; i < S0; ++i) {
            for (size_t j = 0; j < S1; ++j) {
                size_t start_offset_padded = (PAD0 + i) * PADDED_S1 * PADDED_S2 +
                                            (PAD1 + j) * PADDED_S2;
                Float *start_padded = raw_content + start_offset_padded;
                // |pad2|s2|pad2|
                // |s2|pad2|pad2|
                // |pad2|pad2|s2|

                memcpy(start_padded, start_padded + S2,
                      PAD2 * sizeof(Float)); // left pad
                memcpy(start_padded + PAD2 + S2, start_padded + PAD2,
                      PAD2 * sizeof(Float)); // right pad
            }
        }
      }
      double end = omp_get_wtime();

      //std::cout << "overhead: " << end - begin << " [s]" << std::endl;

      // Axis 1
      if (PAD1 > 0) {
        for (size_t i = 0; i < S0; i++) {
            // [ ? <content> ? ]
            size_t start_offset = (PAD0 + i) * PADDED_S1 * PADDED_S2;
            Float *start_padded = raw_content + start_offset;

            memcpy(start_padded,
                  start_padded + PADDED_S1 * PADDED_S2 - 2 * PAD1 * PADDED_S2,
                  PAD1 * PADDED_S2 * sizeof(Float)); // left pad
            memcpy(start_padded + PADDED_S1 * PADDED_S2 - PAD1 * PADDED_S2,
                  start_padded + PAD1 * PADDED_S2,
                  PAD1 * PADDED_S2 * sizeof(Float)); // right pad
        }
      }

      // Axis 0
      memcpy(raw_content,
             raw_content + TOTAL_PADDED_SIZE - 2 * PAD0 * PADDED_S1 * PADDED_S2,
             PAD0 * PADDED_S1 * PADDED_S2 * sizeof(Float)); // left pad
      memcpy(raw_content + TOTAL_PADDED_SIZE - PAD0 * PADDED_S1 * PADDED_S2,
             raw_content + PAD0 * PADDED_S1 * PADDED_S2,
             PAD0 * PADDED_S1 * PADDED_S2 * sizeof(Float)); // right pad
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

  inline Array rotate(const Array &array, const Axis &axis, const Offset &o) {
    Array result;

    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      Index ix = rotate_ix(ix, axis, o);
      result[i] = array[ix];
    }

    return result;
    //throw "rotate not implemented";
    //std::unreachable(); // Always optimize with DNF, do not rotate
  }

  inline Index rotate_ix0(const Index &ix, const Offset &offset) {
    if constexpr(PAD0 >= 1) {
      return ix + (offset.value * PADDED_S1 * PADDED_S2);
    } else {

    }
  }

  inline Index rotate_ix(const Index &ix,
                         const Axis &axis,
                         const Offset &offset) {
    if (axis.value == 0) {
      size_t result = (ix + TOTAL_PADDED_SIZE + (offset.value * PADDED_S1 * PADDED_S2)) % TOTAL_PADDED_SIZE;
      return result;
    } else if (axis.value == 1) {
      size_t ix_subarray_base = ix / (PADDED_S1 * PADDED_S2);
      size_t ix_in_subarray = (ix + PADDED_S1 * PADDED_S2 + offset.value * PADDED_S2) % (PADDED_S1 * PADDED_S2);
      return ix_subarray_base * (PADDED_S1 * PADDED_S2) + ix_in_subarray;
    } else if (axis.value == 2) {
      size_t ix_subarray_base = ix / PADDED_S2;
      size_t ix_in_subarray = (ix + PADDED_S2 + offset.value) % PADDED_S2;
      return ix_subarray_base * PADDED_S2 + ix_in_subarray;
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

  inline Array schedule(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2) {
    Array result;
    std::cout << "in schedule" << std::endl;
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      result[i] = snippet_ix(u, v, u0, u1, u2, i);
    }

    //std::cout << "produced: " << result[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << std::endl;

    return result;
  }

  inline Array schedule_threaded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Nat &nbThreads) {
    Array result;
    omp_set_num_threads(nbThreads.value);
    std::cout << "in schedule threaded" << std::endl;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      result[i] = snippet_ix(u, v, u0, u1, u2, i);
    }
    return result;
  }

  inline Array schedule_tiled(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2) {
    Array result;

    #pragma omp parallel for schedule(static) collapse(3)
    for (size_t ti = 0; ti < S0; ti += S0/NTILES) {
      for (size_t tj = 0; tj < S1; tj += S1/NTILES) {
        for (size_t tk = 0; tk < S2; tk += S2/NTILES) {
          for (size_t i = ti; i < ti + S0/NTILES; ++i) {
            for (size_t j = tj; j < tj + S1/NTILES; ++j) {
              for (size_t k = tk; k < tk + S2/NTILES; ++k) {
                size_t ix = i * S1 * S2 + j * S2 + k;
                result[ix] = snippet_ix(u, v, u0, u1, u2, ix);
              }
            }
          }
        }
      }
    }

    return result;
  }

  /* OF Pad extension */
  inline Array schedule_padded(const Array &u,
      const Array &v, const Array &u0, const Array &u1,
      const Array &u2) {

    Array result;

    std::cout << "in schedule_padded" << std::endl;
    for (size_t i = PAD0; i < S0 + PAD0; ++i) {
      for (size_t j = PAD1; j < S1 + PAD1; ++j) {
        for (size_t k = PAD2; k < S2 + PAD2; ++k) {
          size_t ix = i * PADDED_S1 * PADDED_S2 + j * PADDED_S2 + k;
          result[ix] = snippet_ix(u, v, u0, u1, u2, ix);
        }
      }
    }

    //std::cout << "produced: " << result[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << std::endl;
    // exit(1);

    return result;
  }

  // inline Array refill_all_padding(const Array &arr) {
  //   double begin = omp_get_wtime();
  //   Array result(arr);
  //   result.replenish_padding();
  //   double end = omp_get_wtime();
  //   std::cout << "copying: " << end - begin << " [s] elapsed" << std::endl;
  //   return result;
  // }

  inline void refill_all_padding(Array& arr) {
    arr.replenish_padding();
  }

  inline Index rotate_ix_padded(const Index &ix,
                                const Axis &axis,
                                const Offset &offset) {
    if (axis.value == 0) {
      if constexpr(PAD0 >= 1) {
        return ix + (offset.value * PADDED_S1 * PADDED_S2);
      }
      else {
        size_t result = (ix + TOTAL_PADDED_SIZE + (offset.value * PADDED_S1 * PADDED_S2)) % TOTAL_PADDED_SIZE;
        return result;
      }
    } else if (axis.value == 1) {
      if constexpr(PAD1 >= 1) {
        return ix + (offset.value * PADDED_S2);
      }
      else {
        size_t ix_subarray_base = ix / (PADDED_S1 * PADDED_S2);
        size_t ix_in_subarray = (ix + PADDED_S1 * PADDED_S2 + offset.value * PADDED_S2) % (PADDED_S1 * PADDED_S2);
        return ix_subarray_base * (PADDED_S1 * PADDED_S2) + ix_in_subarray;
      }
    } else if (axis.value == 2) {
      if constexpr(PAD2 >= 1) {
        return ix + offset.value;
      }
      else {
        size_t ix_subarray_base = ix / PADDED_S2;
        size_t ix_in_subarray = (ix + PADDED_S2 + offset.value) % PADDED_S2;
        return ix_subarray_base * PADDED_S2 + ix_in_subarray;
      }
    }

    throw "failed at rotating index";
    //std::unreachable();
    return 0;
  }

  /* OF specialize psi extension */

  struct ScalarIndex { size_t value; };

  inline Index make_ix(const ScalarIndex &i, const ScalarIndex &j,
                       const ScalarIndex &k) {
    return i.value * PADDED_S1 * PADDED_S2 + j.value * PADDED_S2 + k.value;
  }

  /* OF Reduce MakeIx Rotate extension */

  struct AxisLength { size_t value; };

  inline ScalarIndex binary_add(const ScalarIndex &six, const Offset &offset) {
    return ScalarIndex(six.value + offset.value);
  }

  inline ScalarIndex mod(const ScalarIndex &six, const AxisLength &sc) {
    return ScalarIndex(six.value % sc.value);
  }

  inline AxisLength shape_0() { return AxisLength(PADDED_S0); }
  inline AxisLength shape_1() { return AxisLength(PADDED_S1); }
  inline AxisLength shape_2() { return AxisLength(PADDED_S2); }

  inline ScalarIndex ix_0(const Index &ix) {
    return ScalarIndex(ix / (PADDED_S1 * PADDED_S2));
  }

  inline ScalarIndex ix_1(const Index &ix) {
    return ScalarIndex((ix % PADDED_S1 * PADDED_S2) / PADDED_S2);
  }

  inline ScalarIndex ix_2(const Index &ix) {
    return ScalarIndex(ix % PADDED_S2);
  }
};

/* OF Specialize Psi extension */
template <typename _Array, typename _Float, typename _Index,
          typename _ScalarIndex, class _snippet_ix_specialized>
struct specialize_psi_ops {
  typedef _Array Array;
  typedef _Float Float;
  typedef _Index Index;
  typedef _ScalarIndex ScalarIndex;

  _snippet_ix_specialized snippet_ix_specialized;

  inline Float psi(const ScalarIndex &i, const ScalarIndex &j,
                   const ScalarIndex &k, const Array &a) {
    return a[i.value * PADDED_S1 * PADDED_S2 + j.value * PADDED_S2 + k.value];
  }

  inline Array schedule_specialized_psi_padded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2) {
    Array result;
    std::cout << "in schedule_specialized_psi_padded" << std::endl;
    for (size_t i = PAD0; i < S0 + PAD0; ++i) {
      for (size_t j = PAD1; j < S1 + PAD1; ++j) {
        for (size_t k = PAD2; k < S2 + PAD2; ++k) {
          size_t ix = i * PADDED_S1 * PADDED_S2 + j * PADDED_S2 + k;
          result[ix] = snippet_ix_specialized(u, v, u0, u1, u2, ScalarIndex(i), ScalarIndex(j), ScalarIndex(k));
        }
      }
    }

    return result;
  }
};



inline void dumpsine(array_ops<float>::Array &result) {
  double step = 0.01;
  double PI = 3.14159265358979323846;
  double amplitude = 10.0;
  double phase = 0.0125;
  double t = 0.0;

  for (size_t i = PAD0; i < S0 + PAD0; ++i) {
    for (size_t j = PAD1; j < S1 + PAD1; ++j) {
      for (size_t k = PAD2; k < S2 + PAD2; ++k) {
        size_t ix = i * PADDED_S1 * PADDED_S2 +
                    j * PADDED_S2 +
                    k;
        result[ix] = amplitude * sin(PI * t + phase);
        t += step;
      }
    }
  }

  // std::cout << result[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2]
  //           << " vs " << result[TOTAL_PADDED_SIZE - PAD0 * PADDED_S1 * PADDED_S2]
  //           << std::endl;

  result.replenish_padding();

  // std::cout << result[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2]
  //           << " vs " << result[TOTAL_PADDED_SIZE - PAD0 * PADDED_S1 * PADDED_S2]
  //           << std::endl;

  // exit(1);

}



/* experimental stuff below */

/* OF Specialize Psi extension */
template <typename _Array, typename _Float, typename _Index, typename _Offset,
          typename _ScalarIndex, class _snippet_ix_specialized>
struct specialize_psi_ops_2 {
  typedef _Array Array;
  typedef _Float Float;
  typedef _Index Index;
  typedef _Offset Offset;
  typedef _ScalarIndex ScalarIndex;

  _snippet_ix_specialized snippet_ix_specialized;

  inline Float psi(const ScalarIndex &i, const ScalarIndex &j,
                   const ScalarIndex &k, const Array &a) {
    return a[i.value * PADDED_S1 * PADDED_S2 + j.value * PADDED_S2 + k.value];
  }

  inline Array schedule_specialized_psi_padded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2) {
    Array result;
    std::cout << "in schedule_specialized_psi_padded" << std::endl;
    for (size_t i = PAD0; i < S0 + PAD0; ++i) {
      for (size_t j = PAD1; j < S1 + PAD1; ++j) {
        for (size_t k = PAD2; k < S2 + PAD2; ++k) {
          size_t ix = i * PADDED_S1 * PADDED_S2 + j * PADDED_S2 + k;
          result[ix] = snippet_ix_specialized(u, v, u0, u1, u2, ScalarIndex(i), ScalarIndex(j), ScalarIndex(k));
        }
      }
    }

    return result;
  }

  /* ExtNeededFns++ */

  struct AxisLength { size_t value; };

  inline ScalarIndex binary_add(const ScalarIndex &six, const Offset &offset) {
    return ScalarIndex(six.value + offset.value);
  }

  inline ScalarIndex mod(const ScalarIndex &six, const AxisLength &sc) {
    return ScalarIndex(six.value % sc.value);
  }

  inline AxisLength shape_0() { return AxisLength(PADDED_S0); }
  inline AxisLength shape_1() { return AxisLength(PADDED_S1); }
  inline AxisLength shape_2() { return AxisLength(PADDED_S2); }

  inline ScalarIndex ix_0(const Index &ix) {
    return ScalarIndex(ix / (PADDED_S1 * PADDED_S2));
  }

  inline ScalarIndex ix_1(const Index &ix) {
    return ScalarIndex((ix % PADDED_S1 * PADDED_S2) / PADDED_S2);
  }

  inline ScalarIndex ix_2(const Index &ix) {
    return ScalarIndex(ix % PADDED_S2);
  }
};






