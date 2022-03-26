#include <vector>

#include <omp.h>

#include "gen/examples/pde/mg-src/pde-cpp.hpp"
#include "base.hpp"

//typedef array_ops::Shape Shape;
typedef array_ops::Array Array;
typedef array_ops::Index Index;
typedef examples::pde::mg_src::pde_cpp::PDEProgram PDEProgram;

static const double s_dt = 0.00082212448155679772495;
static const double s_nu = 1.0;
static const double s_dx = 1.0;

int main() {
    size_t steps = 50;
    Array u0, u1, u2;
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);

    double begin = omp_get_wtime();

    for (size_t i = 0; i < steps; ++i) {
        PDEProgram::step(u0, u1, u2, s_nu, s_dx, s_dt);
        std::cout << u0[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << " "
                  << u1[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << " "
                  << u2[PAD0 * PADDED_S1 * PADDED_S2 + PAD1 * PADDED_S2 + PAD2] << std::endl;
    }

    double end = omp_get_wtime();

    std::cout << end - begin << "[s] elapsed with sizes ("
              << S0 << ", "
              << S1 << ", "
              << S2 << ") with padding ("
              << PAD0 << ", "
              << PAD1 << ", "
              << PAD2 << ") on "
              << NB_CORES << " threads for "
              << steps << " steps" << std::endl;
}
