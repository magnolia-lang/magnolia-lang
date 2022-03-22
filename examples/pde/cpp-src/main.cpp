#include <vector>

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
    size_t side = SIDE; //256;
    size_t steps = 50;
    //Shape shape = Shape(std::vector<size_t>({ side, side, side }));
    Array u0, u1, u2;
    dumpsine(u0);
    dumpsine(u1);
    dumpsine(u2);
    for (size_t i = 0; i < steps; ++i) {
        PDEProgram::step(u0, u1, u2, s_nu, s_dx, s_dt);
        std::cout << u0[0] << " "
                  << u1[0] << " "
                  << u2[0] << std::endl;
    }
}
