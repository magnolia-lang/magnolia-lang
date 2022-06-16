#include <vector>

#include <omp.h>

#include "gen/examples/pde/mg-src/pde-cuda.cuh"
#include "base.cuh"

#ifndef PROGRAM_NAME
#define PROGRAM_NAME PDEProgramDNF
#endif

//typedef array_ops::Shape Shape;
typedef array_ops<float>::Array Array;
typedef array_ops<float>::HostArray HostArray;
typedef array_ops<float>::Index Index;
typedef array_ops<float>::Float Float;
typedef examples::pde::mg_src::pde_cuda::PROGRAM_NAME PROGRAM_NAME;

extern DeviceAllocator globalAllocator;

int main() {
    size_t steps = 50;
    HostArray u0_base, u1_base, u2_base;
    dumpsine(u0_base);
    dumpsine(u1_base);
    dumpsine(u2_base);

    PROGRAM_NAME p;

    Array u0, u1, u2;

    auto copyDataToHost = [] (HostArray &arrHostPtr, const Array &arrDevPtr) {
        auto devPtr = arrDevPtr.content;
        auto hostPtr = arrHostPtr.content.get();
        gpuErrChk(cudaMemcpy(hostPtr, devPtr, sizeof(Float) * TOTAL_PADDED_SIZE, cudaMemcpyDeviceToHost));
    };

    u0 = u0_base;
    u1 = u1_base;
    u2 = u2_base;

    //float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    double begin = omp_get_wtime();

    for (size_t i = 0; i < steps; ++i) {
        p.step(u0, u1, u2);
        std::cout << "step " << i << std::endl;
    }

    double end = omp_get_wtime();

    copyDataToHost(u0_base, u0);
    copyDataToHost(u1_base, u1);
    copyDataToHost(u2_base, u2);

    std::cout
        << u0_base[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2]
        << " "
        << u1_base[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2]
        << " "
        << u2_base[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2]
        << std::endl;

    std::cout << end - begin << "[s] elapsed with sizes ("
              << S0 << ", "
              << S1 << ", "
              << S2 << ") with padding ("
              << PAD0 << ", "
              << PAD1 << ", "
              << PAD2 << ") on GPU for "
              << steps << " steps" << std::endl;
    return 0;
}