#include <vector>

#include "gen/examples/pde/mg-src/pde-cpp.hpp"
#include "base.hpp"

typedef array_ops::Shape Shape;
typedef array_ops::Array Array;
typedef array_ops::Index Index;
typedef examples::pde::mg_src::pde_cpp::PDEProgram PDEProgram;

static const double s_dt = 0.00082212448155679772495;
static const double s_nu = 1.0;
static const double s_dx = 1.0;

int main() {
    size_t side = 15;
    size_t steps = 30;
    Shape shape = Shape(std::vector<size_t>({ side, side, side }));
    Array u0 = dumpsine(shape), u1 = dumpsine(shape), u2 = dumpsine(shape);
    //for (size_t i = 0; i < 30; ++i) {
        PDEProgram::step(u0, u1, u2, s_nu, s_dx, s_dt);
    //}
}
