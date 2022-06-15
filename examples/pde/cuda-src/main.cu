#include <vector>

#include <omp.h>

#include "gen/examples/pde/mg-src/pde-cuda.cuh"
#include "base.cuh"

//typedef array_ops::Shape Shape;
typedef array_ops<float>::Array Array;
typedef array_ops<float>::Index Index;
typedef array_ops<float>::Float Float;
typedef examples::pde::mg_src::pde_cuda::BasePDEProgram BasePDEProgram;
typedef examples::pde::mg_src::pde_cuda::PDEProgramDNF PDEProgramDNF;

void allocateDeviceMemory(Float* &u0_host_content, 
                          Float* &u1_host_content,    
                          Float* &u2_host_content,
                          Float* &u0_dev_content, 
                          Float* &u1_dev_content, 
                          Float* &u2_dev_content,
                          Array* &u0_dev, Array* &u1_dev, Array* &u2_dev) {

    cudaMalloc((void**)&u0_dev_content, sizeof(Float) * TOTAL_PADDED_SIZE);
    cudaMalloc((void**)&u1_dev_content, sizeof(Float) * TOTAL_PADDED_SIZE);
    cudaMalloc((void**)&u2_dev_content, sizeof(Float) * TOTAL_PADDED_SIZE);

    cudaMalloc((void**)&u0_dev, sizeof(*u0_dev));
    cudaMalloc((void**)&u1_dev, sizeof(*u1_dev));
    cudaMalloc((void**)&u2_dev, sizeof(*u2_dev));
}

void copyDeviceMemory(Float* &u0_host_content, 
                      Float* &u1_host_content,    
                      Float* &u2_host_content,
                      Float* &u0_dev_content, 
                      Float* &u1_dev_content, 
                      Float* &u2_dev_content,
                      Array* &u0_dev, Array* &u1_dev, Array* &u2_dev) {

    cudaMemcpy(u0_dev_content, u0_host_content, sizeof(Float) * TOTAL_PADDED_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(u1_dev_content, u1_host_content, sizeof(Float) * TOTAL_PADDED_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(u2_dev_content, u2_host_content, sizeof(Float) * TOTAL_PADDED_SIZE, cudaMemcpyHostToDevice);

    // Binding pointers with _dev
    cudaMemcpy(&(u0_dev->content), &u0_dev_content, sizeof(u0_dev->content),    cudaMemcpyHostToDevice);
    cudaMemcpy(&(u1_dev->content), &u1_dev_content, sizeof(u1_dev->content),    cudaMemcpyHostToDevice);
    cudaMemcpy(&(u2_dev->content), &u2_dev_content, sizeof(u2_dev->content),    cudaMemcpyHostToDevice);
}

int main() {
    size_t steps = 50;
    Array u0, u1, u2;
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);

    PDEProgramDNF pde_dnf;

    Float *u0_host_content, *u1_host_content, *u2_host_content;
    u0_host_content = u0.content;
    u1_host_content = u1.content;
    u2_host_content = u2.content;

    Float *u0_dev_content, *u1_dev_content, *u2_dev_content;
    Array *u0_dev, *u1_dev, *u2_dev;

    allocateDeviceMemory(u0_host_content, u1_host_content, u2_host_content,
        u0_dev_content, u1_dev_content, u2_dev_content,
        u0_dev, u1_dev, u2_dev);
    copyDeviceMemory(u0_host_content, u1_host_content, u2_host_content,
            u0_dev_content, u1_dev_content, u2_dev_content,
            u0_dev, u1_dev, u2_dev);
    
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (size_t i = 0; i < steps; ++i) {
        pde_dnf.step(*u0_dev, *u1_dev, *u2_dev);//, S_NU, S_DX, S_DT);
        std::cout << u0[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << " "
                  << u1[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << " "
                  << u2[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << std::endl;
    }
    /*
    std::cout << end - begin << "[s] elapsed with sizes ("
              << S0 << ", "
              << S1 << ", "
              << S2 << ") with padding ("
              << PAD0 << ", "
              << PAD1 << ", "
              << PAD2 << ") on "
              << NB_CORES << " threads for "
              << steps << " steps" << std::endl;
              */
}
