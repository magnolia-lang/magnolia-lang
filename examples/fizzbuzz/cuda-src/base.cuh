#include <string>

template <typename _IntLike>
struct fizzbuzz_ops {
    typedef _IntLike IntLike;
    typedef float FizzBuzz;
    __device__ __host__ inline IntLike modulo(const IntLike a, const IntLike modulus) {
        return a % modulus;
    }
    __device__ __host__ inline IntLike five() { return 5; }
    __device__ __host__ inline IntLike three() { return 3; }
    __device__ __host__ inline IntLike zero() { return 0; }
    __device__ __host__ inline FizzBuzz fizz() { return 0xf155; }
    __device__ __host__ inline FizzBuzz buzz() { return 0xb055; }
    __device__ __host__ inline FizzBuzz fizzbuzz() { return 0xf155b055; }
    __device__ __host__ inline FizzBuzz nope() { return 0x90; }
};

struct basic_types {
    typedef int Int32;
};
