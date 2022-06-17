#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <omp.h>

#ifndef PAD0
#define PAD0 0
#endif

#ifndef PAD1
#define PAD1 0
#endif

#ifndef PAD2
#define PAD2 0
#endif

#ifndef S0
#define S0 512
#endif

#ifndef S1
#define S1 512
#endif

#ifndef S2
#define S2 512
#endif

#define PADDED_S0 (S0 + 2 * PAD0)
#define PADDED_S1 (S1 + 2 * PAD1)
#define PADDED_S2 (S2 + 2 * PAD2)

#define TOTAL_PADDED_SIZE (PADDED_S0 * PADDED_S1 * PADDED_S2)

#define NTILES 4
#define NB_CORES 2

#define S_DT 0.00082212448155679772495
#define S_NU 1.0
#define S_DX 1.0

// Taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

size_t nbThreadsPerBlock = 1024;
size_t nbBlocks = (TOTAL_PADDED_SIZE / nbThreadsPerBlock) + (TOTAL_PADDED_SIZE % nbThreadsPerBlock > 0 ? 1 : 0);

struct DevicePtrInfo {
  void* ptr;
  size_t size;
  // bool isInUse;
  // DevicePtrInfo(void *ptr, size_t size, bool isInUse) {
  //   this->ptr = ptr;
  //   this->size = size;
  //   this->isInUse = isInUse;
  // }

  DevicePtrInfo(void *ptr, size_t size) {
    this->ptr = ptr;
    this->size = size;
    //this->isInUse = false;
  }
};

struct DeviceAllocator {
  std::vector<DevicePtrInfo> unusedChunks;
  std::vector<DevicePtrInfo> inUseChunks;
  std::vector<std::pair<void*, void*>> wrapperChunks;

  DeviceAllocator() {};
  ~DeviceAllocator() {
    auto freeFn = [](void *ptr) {
      if (ptr != NULL) {
        cudaFree(ptr);
        ptr = NULL;
      }
    };

    for (auto itr = unusedChunks.begin(); itr != unusedChunks.end(); ++itr) {
      freeFn(itr->ptr);
    }

    for (auto itr = inUseChunks.begin(); itr != inUseChunks.end(); ++itr) {
      freeFn(itr->ptr);
    }
  }

  // We could make this a map to have a more efficient lookup, but we do not
  // bother at the moment.
  template <typename T>
  cudaError_t alloc(T** ptr, size_t size) {
    *ptr = NULL;
    cudaError_t returnCode = cudaSuccess;
    for (auto itr = unusedChunks.begin(); itr != unusedChunks.end(); ++itr) {
      if (itr->size == size) {
        //itr->isInUse = true;
        *ptr = (T*)itr->ptr;
        unusedChunks.erase(itr, itr + 1);
        break;
      }
    }

    if (*ptr == NULL) { returnCode = cudaMalloc(ptr, size); }
    inUseChunks.push_back(DevicePtrInfo(*ptr, size));
    //        inUseChunks.push_back(*itr);

    return returnCode;
  }

  // Wrapper has to have a content element
  template <typename Wrapper, typename Content>
  void deviceWrap(Wrapper **result, Content *ptr) {
    *result = NULL;
    // For real stable code, we might want to check that this chunk is actually
    // not in use, or at least make sure to put it in use after this. In
    // practice, this should be fine and not really matter?
    for (auto itr = wrapperChunks.begin(); itr != wrapperChunks.end(); ++itr) {
      if (itr->first == ptr) {
        *result = (Wrapper*)itr->second;
        break;
      }
    }

    if (result == NULL) {
      this->alloc(result, sizeof(Wrapper));
      gpuErrChk(cudaMemcpy(&((*result)->content), &ptr, sizeof(ptr),
                cudaMemcpyHostToDevice));
      inUseChunks.push_back(DevicePtrInfo(*result, sizeof(Wrapper)));
    }
  };

  void free(void *ptr) {
    for (auto itr = inUseChunks.begin(); itr != inUseChunks.end(); ++itr) {
      if (itr->ptr == ptr) {
        unusedChunks.push_back(*itr);
        inUseChunks.erase(itr, itr + 1);
        ptr = NULL;
        break;
      }
    }
  }
};

DeviceAllocator globalAllocator;

struct constants {
    typedef float Float;
    __host__ __device__ constexpr Float nu() { return S_NU; }
    __host__ __device__ constexpr Float dt() { return S_DT; }
    __host__ __device__ constexpr Float dx() { return S_DX; }
};

template <typename _Float>
struct base_types {
  typedef _Float Float;
  struct Offset { int value;
    __host__ __device__ Offset(){}
    __host__ __device__ Offset(const int &v) : value(v) {} };
  struct Axis { size_t value;
    __host__ __device__ Axis(){}
    __host__ __device__ Axis(const size_t &v) : value(v) {} };
  typedef size_t Index;
  struct Nat { size_t value;
    __host__ __device__ Nat(){}
    __host__ __device__ Nat(const size_t &v) : value(v) {} };
  struct ScalarIndex { size_t value;
    __host__ __device__ ScalarIndex(){}
    __host__ __device__ ScalarIndex(const size_t &v) : value(v) {} };

  struct HostArray {
    std::unique_ptr<Float[]> content;
    HostArray() {
      this->content = std::unique_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
    }

    HostArray(const HostArray &other) {
      this->content = std::unique_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
      memcpy(this->content.get(), other.content.get(),
             TOTAL_PADDED_SIZE * sizeof(Float));
    }

    HostArray(HostArray &&other) {
        this->content = std::move(other.content);
    }

    HostArray &operator=(const HostArray &other) {
      this->content = std::unique_ptr<Float[]>(new Float[TOTAL_PADDED_SIZE]);
      memcpy(this->content.get(), other.content.get(),
             TOTAL_PADDED_SIZE * sizeof(Float));
      return *this;
    }

    HostArray &operator=(HostArray &&other) {
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

  struct DeviceArray {
    Float *content;

    __host__ DeviceArray() {
      gpuErrChk(globalAllocator.alloc(&(this->content), TOTAL_PADDED_SIZE * sizeof(Float)));
    }

    __host__ DeviceArray(const DeviceArray &other) {
      gpuErrChk(globalAllocator.alloc(&(this->content), TOTAL_PADDED_SIZE * sizeof(Float)));
      gpuErrChk(cudaMemcpy(this->content, other.content,
                 TOTAL_PADDED_SIZE * sizeof(Float), cudaMemcpyDeviceToDevice));
    }

    __host__ DeviceArray &operator=(const DeviceArray &other) {
      gpuErrChk(cudaMemcpy(this->content, other.content,
                 TOTAL_PADDED_SIZE * sizeof(Float), cudaMemcpyDeviceToDevice));
      return *this;
    }

    __host__ DeviceArray &operator=(const HostArray &host) {
      gpuErrChk(cudaMemcpy(this->content, host.content.get(),
                 TOTAL_PADDED_SIZE * sizeof(Float), cudaMemcpyHostToDevice));
      return *this;
    }

    __host__ ~DeviceArray() {
      if (this->content != NULL) globalAllocator.free(this->content);
    }

    __host__ DeviceArray(DeviceArray &&other) {
      this->content = other.content;
      other.content = NULL;
    }

    __host__ DeviceArray &operator=(DeviceArray &&other) {
      globalAllocator.free(this->content);
      this->content = other.content;
      other.content = NULL;
      return *this;
    }

    __host__ __device__ inline Float operator[](const Index & ix) const {
      return this->content[ix];
    }

    __host__ __device__ inline Float &operator[](const Index & ix) {
      return this->content[ix];
    }

    // TODO: this is probably all broken...
    __host__ void replenish_padding() {

      double begin = omp_get_wtime();

      // static variable allows avoiding cudaMemcpy everytime
      //Array *deviceArr = NULL;

      std::cout << "OK" << std::endl;
      Array *deviceArr;
      globalAllocator.deviceWrap(&deviceArr, this->content);

      std::cout << "wrapped: deviceArr" << std::endl;
      // if (this->onDeviceArr == NULL) {
      //   globalAllocator.alloc(&this->onDeviceArr, sizeof(Array));

      //   gpuErrChk(cudaMemcpy(&(this->onDeviceArr->content), &(this->content),
      //                        sizeof(this->content), cudaMemcpyHostToDevice));
      // }

      replenishPaddingGlobal<<<nbBlocks, nbThreadsPerBlock>>>(deviceArr);

      cudaDeviceSynchronize();
      //globalAllocator.free(deviceArr);

      double end = omp_get_wtime();

      std::cout << "cost: " << end - begin << "[s]" << std::endl;
      return;
    }
  };

  typedef DeviceArray Array;
};

template <class _rotateIx>
__global__ void rotateIxGlobal(
    base_types<constants::Float>::Array *res,
    const base_types<constants::Float>::Array *input,
    const base_types<constants::Float>::Axis *axis,
    const base_types<constants::Float>::Offset *o) {

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  _rotateIx rotateIx;

  if (ix < TOTAL_PADDED_SIZE) {
    res->content[ix] = input->content[rotateIx(ix, *axis, *o)];
  }
}

template <class _binOpIx>
__global__ void binOpIxGlobal(
    base_types<constants::Float>::Array *res,
    const base_types<constants::Float>::Array *lhs,
    const base_types<constants::Float>::Array *rhs) {

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  _binOpIx binOpIx;

  if (ix < TOTAL_PADDED_SIZE) {
    res->content[ix] = binOpIx((*lhs)[ix], (*rhs)[ix]);
  }
}

template <class _binOpIx>
__global__ void binOpIxGlobal(
    base_types<constants::Float>::Array *res,
    const base_types<constants::Float>::Float *lhsFloat,
    const base_types<constants::Float>::Array *rhs) {

  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  _binOpIx binOpIx;

  if (ix < TOTAL_PADDED_SIZE) {
    res->content[ix] = binOpIx(*lhsFloat, (*rhs)[ix]);
  }
}

template <typename _Float>
struct array_ops {
  typedef typename base_types<_Float>::Float Float;
  typedef typename base_types<_Float>::Offset Offset;
  typedef typename base_types<_Float>::Axis Axis;
  typedef typename base_types<_Float>::Index Index;
  typedef typename base_types<_Float>::Nat Nat;
  typedef typename base_types<_Float>::HostArray HostArray;
  typedef typename base_types<_Float>::Array Array;

  __host__ __device__ inline Float psi(const Index & ix,
    const Array & array) {
    return array[ix];
  }

  /* Float ops */
  __host__ __device__ inline Float unary_sub(const Float & f) {
    return -f;
  }

  template <typename _binOpIx>
  Array binOpGlobalWrapper(const Array &lhs, const Array &rhs) {
    Array result;

    Array *result_dev = NULL,
          *lhs_dev = NULL,
          *rhs_dev = NULL;

    // One single globalAllocator.alloc call
    globalAllocator.alloc(&result_dev, 3 * sizeof(Array));

    lhs_dev = result_dev + 1;
    rhs_dev = result_dev + 2;

    const size_t ptrSize = sizeof(result.content);
    const auto htd = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy(&(result_dev->content), &(result.content),
                         ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(lhs_dev->content), &(lhs.content),
                         ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(rhs_dev->content), &(rhs.content),
                         ptrSize, htd));

    binOpIxGlobal<_binOpIx><<<nbBlocks, nbThreadsPerBlock>>>(
      result_dev, lhs_dev, rhs_dev);

    globalAllocator.free(result_dev);

    return result;
  }

  struct _binary_add {
    __host__ __device__ inline Float operator()(const Float & lhs,
      const Float & rhs) {
      return lhs + rhs;
    }
  };

  _binary_add binary_add_ix;

  struct _binary_sub {
    __host__ __device__ inline Float operator()(const Float & lhs,
      const Float & rhs) {
      return lhs - rhs;
    }
  };

  _binary_sub binary_sub_ix;

  struct _mul {
    __host__ __device__ inline Float operator()(const Float & lhs,
      const Float & rhs) {
      return lhs * rhs;
    }
  };

  _mul mul_ix;

  struct _div {
    __host__ __device__ inline Float operator()(const Float & num,
      const Float & den) {
      return num / den;
    }
  };

  _div div_ix;

  __host__ __device__ inline Float one_float() {
    return 1;
  }
  __host__ __device__ inline Float two_float() {
    return 2;
  }
  __host__ __device__ inline Float three_float() {
    return 3;
  }

  __host__ __device__ Float binary_add(const Float &lhs, const Float &rhs) {
    return binary_add_ix(lhs, rhs);
  }

  __host__ __device__ Float binary_sub(const Float &lhs, const Float &rhs) {
    return binary_sub_ix(lhs, rhs);
  }

  __host__ __device__ Float mul(const Float &lhs, const Float &rhs) {
    return mul_ix(lhs, rhs);
  }

  __host__ __device__ Float div(const Float &lhs, const Float &rhs) {
    return div_ix(lhs, rhs);
  }

  __host__ __device__ Array binary_add(const Array &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_binary_add>(lhs, rhs);
  }

  __host__ __device__ Array binary_sub(const Array &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_binary_sub>(lhs, rhs);
  }

  __host__ __device__ Array mul(const Array &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_mul>(lhs, rhs);
  }

  __host__ __device__ Array div(const Array &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_div>(lhs, rhs);
  }

  template <typename _binOpIx>
  Array binOpGlobalWrapper(const Float &lhsFloat, const Array &rhs) {
    Array result;

    Array *result_dev = NULL,
          *rhs_dev = NULL;

    Float *lhsFloat_dev = NULL;

    // One single globalAllocator.alloc call
    globalAllocator.alloc(&result_dev, 2 * sizeof(Array));

    rhs_dev = result_dev + 1;

    globalAllocator.alloc(&lhsFloat_dev, sizeof(Float));

    const size_t ptrSize = sizeof(result.content);
    const auto htd = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy(&(result_dev->content), &(result.content),
                         ptrSize, htd));
    gpuErrChk(cudaMemcpy(lhsFloat_dev, &lhsFloat, sizeof(Float), htd));
    gpuErrChk(cudaMemcpy(&(rhs_dev->content), &(rhs.content),
                         ptrSize, htd));

    binOpIxGlobal<_binOpIx><<<nbBlocks, nbThreadsPerBlock>>>(
      result_dev, lhsFloat_dev, rhs_dev);

    globalAllocator.free(result_dev);
    globalAllocator.free(lhsFloat_dev);

    return result;
  }

  __host__ __device__ Array binary_add(const Float &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_binary_add>(lhs, rhs);
  }

  __host__ __device__ Array binary_sub(const Float &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_binary_sub>(lhs, rhs);
  }

  __host__ __device__ Array mul(const Float &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_mul>(lhs, rhs);
  }

  __host__ __device__ Array div(const Float &lhs, const Array &rhs) {
    return binOpGlobalWrapper<_div>(lhs, rhs);
  }

  struct _rotateIx {
    __host__ __device__ inline Index operator()(const Index &ix,
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

      // TODO: device code does not support exception handling
      //throw "failed at rotating index";
      //std::unreachable();
      return 0;
    }
  };

  _rotateIx rotateIx;

  __host__ inline Array rotate(const Array &input,
                               const Axis &axis,
                               const Offset &offset) {
    Array result;

    Array *result_dev = NULL,
          *input_dev = NULL;

    Axis *axis_dev;
    Offset *offset_dev;

    // One single globalAllocator.alloc call
    globalAllocator.alloc(&result_dev, 2 * sizeof(Array));
    input_dev = result_dev + 1;

    globalAllocator.alloc(&axis_dev, sizeof(Axis));
    globalAllocator.alloc(&offset_dev, sizeof(Offset));

    const size_t ptrSize = sizeof(result.content);
    const auto htd = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy(&(result_dev->content), &(result.content),
                         ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(input_dev->content), &(input.content),
                         ptrSize, htd));
    gpuErrChk(cudaMemcpy(axis_dev, &axis, sizeof(Axis), htd));
    gpuErrChk(cudaMemcpy(offset_dev, &offset, sizeof(Offset), htd));

    rotateIxGlobal<_rotateIx><<<nbBlocks, nbThreadsPerBlock>>>(
      result_dev, input_dev, axis_dev, offset_dev);

    globalAllocator.free(result_dev);
    globalAllocator.free(axis_dev);
    globalAllocator.free(offset_dev);

    return result;
  }

  __host__ __device__ inline Axis zero_axis() { return Axis(0); }
  __host__ __device__ inline Axis one_axis() { return Axis(1); }
  __host__ __device__ inline Axis two_axis() { return Axis(2); }

  __host__ __device__ inline Offset one_offset() { return Offset(1); }
  __host__ __device__ inline Offset unary_sub(const Offset &offset) { return Offset(-offset.value); }
};

// CUDA kernel
template <class _substepIx>
__global__ void substepIxGlobal(array_ops<float>::Array *res,const array_ops<float>::Array *u, const array_ops<float>::Array *v, const array_ops<float>::Array *u0, const array_ops<float>::Array *u1, const array_ops<float>::Array *u2) {
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;

  _substepIx substepIx;

  if (ix < TOTAL_PADDED_SIZE) {
    res->content[ix] = substepIx(*u,*v,*u0,*u1,*u2,ix);
  }
}

template <class _substepIx>
__global__ void substepIxPaddedGlobal(array_ops<float>::Array *res,const array_ops<float>::Array *u, const array_ops<float>::Array *v, const array_ops<float>::Array *u0, const array_ops<float>::Array *u1, const array_ops<float>::Array *u2) {
  size_t ix = (blockIdx.x * blockDim.x + threadIdx.x) + \
              PAD0 * PADDED_S1 * PADDED_S2 + \
              PAD1 * PADDED_S1 + \
              PAD2;

  _substepIx substepIx;

  size_t i = ix / (PADDED_S1 * PADDED_S2),
         j = (ix / PADDED_S2) % PADDED_S1,
         k = ix % PADDED_S2;

  bool isNotPadding = (i >= PAD0 && i < PADDED_S0 - PAD0) &&
                      (j >= PAD1 && j < PADDED_S1 - PAD1) &&
                      (k >= PAD2 && k < PADDED_S2 - PAD2);

  if (ix < TOTAL_PADDED_SIZE) {
    // Hack to avoid divergence: if there is any padding in the array, the
    // 0th index of the array *has* to be padding (since padding is always
    // on both ends of the array). We avoid thread divergence by mapping all
    // padding indices to 0.
    ix = ix * isNotPadding;
    res->content[ix] = substepIx(*u,*v,*u0,*u1,*u2,ix);
  }
}

template <class _substepIx3D>
__global__ void substepIx3DPaddedGlobal(array_ops<float>::Array *res,const array_ops<float>::Array *u, const array_ops<float>::Array *v, const array_ops<float>::Array *u0, const array_ops<float>::Array *u1, const array_ops<float>::Array *u2) {
  size_t ix = (blockIdx.x * blockDim.x + threadIdx.x) + \
              PAD0 * PADDED_S1 * PADDED_S2 + \
              PAD1 * PADDED_S1 + \
              PAD2;

  _substepIx3D substepIx3D;

  size_t i = ix / (PADDED_S1 * PADDED_S2),
         j = (ix / PADDED_S2) % PADDED_S1,
         k = ix % PADDED_S2;

  bool isNotPadding = (i >= PAD0 && i < S0 + PAD0) &&
                      (j >= PAD1 && j < S1 + PAD1) &&
                      (k >= PAD2 && k < S2 + PAD2);

  if (ix < TOTAL_PADDED_SIZE && isNotPadding) {
    base_types<constants::Float>::ScalarIndex si(i), sj(j), sk(k);
    res->content[ix] = substepIx3D(*u, *v, *u0, *u1, *u2, si, sj, sk);
  }
}

__global__ void replenishPaddingGlobal(array_ops<float>::Array *arr) {
  // shape: 2 * (PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2)
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = ix / (PADDED_S1 * PADDED_S2),
         j = (ix / PADDED_S2) % PADDED_S1,
         k = ix % PADDED_S2;

  // Computing the corresponding index within the inner array
  size_t oi = (i < PAD0) ? i + S0 : ((i >= S0 + PAD0) ? i - S0 : i);
  size_t oj = (j < PAD1) ? j + S1 : ((j >= S1 + PAD1) ? j - S1 : j);
  size_t ok = (k < PAD2) ? k + S2 : ((k >= S2 + PAD2) ? k - S2 : k);

  size_t oix = oi * PADDED_S1 * PADDED_S2 + oj * PADDED_S2 + ok;
  arr->content[ix] = arr->content[oix];
}

template <typename _Array, typename _Axis, typename _Float, typename _Index,
          typename _Nat, typename _Offset, class _substepIx>
struct forall_ops {
  typedef _Array Array;
  typedef _Axis Axis;
  typedef _Float Float;
  typedef _Index Index;
  typedef _Nat Nat;
  typedef _Offset Offset;

  inline Nat nbCores() { return Nat(NB_CORES); }

  // TODO: this is meant to always be called on host, which results in a
  // warning. Not sure how to fix that easily at the moment.
  __host__ inline Array schedule(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2) {

    std::cout << "in schedule" << std::endl;

    Array result;

    Array *result_dev = NULL,
          *u_dev = NULL,
          *v_dev = NULL,
          *u0_dev = NULL,
          *u1_dev = NULL,
          *u2_dev = NULL;

    // One single globalAllocator.alloc call
    globalAllocator.alloc(&result_dev, 6 * sizeof(Array));
    u_dev = result_dev + 1;
    v_dev = result_dev + 2;
    u0_dev = result_dev + 3;
    u1_dev = result_dev + 4;
    u2_dev = result_dev + 5;

    const size_t ptrSize = sizeof(result.content);
    const auto htd = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy(&(result_dev->content), &(result.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u_dev->content), &(u.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(v_dev->content), &(v.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u0_dev->content), &(u0.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u1_dev->content), &(u1.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u2_dev->content), &(u2.content), ptrSize, htd));

    substepIxGlobal<_substepIx><<<nbBlocks, nbThreadsPerBlock>>>(result_dev, u_dev, v_dev, u0_dev, u1_dev, u2_dev);

    globalAllocator.free(result_dev);

    return result;
  }

  inline Array schedule_threaded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2, const Nat &nbThreads) {
    Array result;
    omp_set_num_threads(nbThreads.value);
    std::cout << "in schedule threaded" << std::endl;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < TOTAL_PADDED_SIZE; ++i) {
      result[i] = substepIx(u, v, u0, u1, u2, i);
    }
    return result;
  }

  __host__ __device__ inline Array schedule_tiled(const Array &u, const Array &v,
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
                result[ix] = substepIx(u, v, u0, u1, u2, ix);
              }
            }
          }
        }
      }
    }

    return result;
  }

  __host__ __device__ inline void refillPadding(Array& arr) {
    arr.replenish_padding();
  }

  __host__ __device__ inline Index rotateIxPadded(const Index &ix,
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

    //throw "failed at rotating index";
    //std::unreachable();
    return 0;
  }

  /* OF specialize psi extension */

  typedef base_types<constants::Float>::ScalarIndex ScalarIndex;

  __host__ __device__ inline Index mkIx(const ScalarIndex &i, const ScalarIndex &j,
                       const ScalarIndex &k) {
    return i.value * PADDED_S1 * PADDED_S2 + j.value * PADDED_S2 + k.value;
  }

  /* OF Reduce MakeIx Rotate extension */

  struct AxisLength { size_t value; };

  __host__ __device__ inline ScalarIndex binary_add(const ScalarIndex &six, const Offset &offset) {
    return ScalarIndex(six.value + offset.value);
  }

  __host__ __device__ inline ScalarIndex mod(const ScalarIndex &six, const AxisLength &sc) {
    return ScalarIndex(six.value % sc.value);
  }

  __host__ __device__ inline AxisLength shape0() { return AxisLength(PADDED_S0); }
  __host__ __device__ inline AxisLength shape1() { return AxisLength(PADDED_S1); }
  __host__ __device__ inline AxisLength shape2() { return AxisLength(PADDED_S2); }

  __host__ __device__ inline ScalarIndex ix_0(const Index &ix) {
    return ScalarIndex(ix / (PADDED_S1 * PADDED_S2));
  }

  __host__ __device__ inline ScalarIndex ix_1(const Index &ix) {
    return ScalarIndex((ix % PADDED_S1 * PADDED_S2) / PADDED_S2);
  }

  __host__ __device__ inline ScalarIndex ix_2(const Index &ix) {
    return ScalarIndex(ix % PADDED_S2);
  }
};

__host__ __device__ inline void dumpsine(array_ops<float>::HostArray &result) {
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

// CUDA stuff

template<typename _Index>
struct scalar_index {
  typedef _Index Index;

  typedef base_types<constants::Float>::ScalarIndex ScalarIndex;

  __host__ __device__ inline Index mkIx(const ScalarIndex &i, const ScalarIndex &j,
                       const ScalarIndex &k) {
    return i.value * PADDED_S1 * PADDED_S2 + j.value * PADDED_S2 + k.value;
  }
  __host__ __device__ inline ScalarIndex ix0(const array_ops<float>::Index &ix) {
    return ScalarIndex(ix / (PADDED_S1 * PADDED_S2));
  }

  __host__ __device__ inline ScalarIndex ix1(const array_ops<float>::Index &ix) {
    return ScalarIndex((ix % PADDED_S1 * PADDED_S2) / PADDED_S2);
  }

  __host__ __device__ inline ScalarIndex ix2(const array_ops<float>::Index &ix) {
    return ScalarIndex(ix % PADDED_S2);
  }
};

template<typename _Offset, typename _ScalarIndex>
struct axis_length {

  typedef _Offset Offset;
  typedef _ScalarIndex ScalarIndex;

  struct AxisLength { size_t value;
    __host__ __device__ AxisLength(size_t i) {this->value = i;}};

  __host__ __device__ inline AxisLength shape0() { return AxisLength(PADDED_S0); }
  __host__ __device__ inline AxisLength shape1() { return AxisLength(PADDED_S1); }
  __host__ __device__ inline AxisLength shape2() { return AxisLength(PADDED_S2); }

  __host__ __device__ inline ScalarIndex binary_add(const ScalarIndex &six, const Offset &offset) {
    return ScalarIndex(six.value + offset.value);
  }

  __host__ __device__ inline ScalarIndex mod(const ScalarIndex &six, const AxisLength &sc) {
    return ScalarIndex(six.value % sc.value);
  }


};

template< typename _Array, typename _Float, typename _ScalarIndex>
struct specialize_base {

  typedef _Array Array;
  typedef _Float Float;
  typedef _ScalarIndex ScalarIndex;


  __host__ __device__ inline Float psi(const ScalarIndex &i,
    const ScalarIndex &j, const ScalarIndex &k, const Array &a) {
    return a[i.value * PADDED_S1 * PADDED_S2 + j.value * PADDED_S2 + k.value];
    }
};

template<typename _Array, typename _Axis, typename _Index, typename _Offset>
struct padding_extension{
  typedef _Array Array;
  typedef _Axis Axis;
  typedef _Index Index;
  typedef _Offset Offset;

  __host__ __device__ inline void refillPadding(Array& arr) {
    arr.replenish_padding();
  }

  __host__ __device__ inline Index rotateIxPadded(const Index &ix,
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
// device code does not support exception handling
//throw "failed at rotating index";
//std::unreachable();
return 0;
}
};

template<typename _Array, typename _Float, typename _Index, class _substepIx>
struct padded_schedule {

  typedef _Array Array;
  typedef _Index Index;
  typedef _Float Float;

  _substepIx substepIx;

  __host__ inline Array schedulePadded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2) {

    std::cout << "in schedulePadded" << std::endl;

    Array result;

    Array *result_dev = NULL,
          *u_dev = NULL,
          *v_dev = NULL,
          *u0_dev = NULL,
          *u1_dev = NULL,
          *u2_dev = NULL;

    // One single globalAllocator.alloc call
    globalAllocator.alloc(&result_dev, 6 * sizeof(Array));
    u_dev = result_dev + 1;
    v_dev = result_dev + 2;
    u0_dev = result_dev + 3;
    u1_dev = result_dev + 4;
    u2_dev = result_dev + 5;

    const size_t ptrSize = sizeof(result.content);
    const auto htd = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy(&(result_dev->content), &(result.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u_dev->content), &(u.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(v_dev->content), &(v.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u0_dev->content), &(u0.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u1_dev->content), &(u1.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u2_dev->content), &(u2.content), ptrSize, htd));

    substepIxPaddedGlobal<_substepIx><<<nbBlocks, nbThreadsPerBlock>>>(result_dev, u_dev, v_dev, u0_dev, u1_dev, u2_dev);

    globalAllocator.free(result_dev);

    return result;
  }
};

/* experimental stuff below */

/* OF Specialize Psi extension */
template <typename _Array, typename _Axis, typename _Float, typename _Index,
          typename _Offset, typename _ScalarIndex,
          class _substepIx3D>
struct specialize_psi_ops_2 {
  typedef _Array Array;
  typedef _Axis Axis;
  typedef _Float Float;
  typedef _Index Index;
  typedef _Offset Offset;
  typedef _ScalarIndex ScalarIndex;

  _substepIx3D substepIx3D;

  __host__ __device__ inline Float psi(const ScalarIndex &i, const ScalarIndex &j,
                   const ScalarIndex &k, const Array &a) {
    return a[i.value * PADDED_S1 * PADDED_S2 + j.value * PADDED_S2 + k.value];
  }

  __host__ inline Array schedule3DPadded(const Array &u, const Array &v,
      const Array &u0, const Array &u1, const Array &u2) {

    std::cout << "in schedule3DPadded" << std::endl;

    Array result;

    Array *result_dev = NULL,
          *u_dev = NULL,
          *v_dev = NULL,
          *u0_dev = NULL,
          *u1_dev = NULL,
          *u2_dev = NULL;

    // One single globalAllocator.alloc call
    globalAllocator.alloc(&result_dev, 6 * sizeof(Array));
    u_dev = result_dev + 1;
    v_dev = result_dev + 2;
    u0_dev = result_dev + 3;
    u1_dev = result_dev + 4;
    u2_dev = result_dev + 5;

    const size_t ptrSize = sizeof(result.content);
    const auto htd = cudaMemcpyHostToDevice;
    gpuErrChk(cudaMemcpy(&(result_dev->content), &(result.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u_dev->content), &(u.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(v_dev->content), &(v.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u0_dev->content), &(u0.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u1_dev->content), &(u1.content), ptrSize, htd));
    gpuErrChk(cudaMemcpy(&(u2_dev->content), &(u2.content), ptrSize, htd));

    substepIx3DPaddedGlobal<_substepIx3D><<<nbBlocks, nbThreadsPerBlock>>>(result_dev, u_dev, v_dev, u0_dev, u1_dev, u2_dev);

    //std::cout <<  "OK This worked" << std::endl;

    globalAllocator.free(result_dev);

    return result;
  }

  /* ExtNeededFns++ */
  __host__ __device__ inline void refillPadding(Array& arr) {
    arr.replenish_padding();
  }

  __host__ __device__ inline Index rotateIxPadded(const Index &ix,
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
    // device code does not support exception handling
    //throw "failed at rotating index";
    //std::unreachable();
    return 0;
  }
};