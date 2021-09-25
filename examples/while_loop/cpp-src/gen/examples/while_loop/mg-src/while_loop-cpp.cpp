#include "gen/examples/while_loop/mg-src/while_loop-cpp.hpp"


namespace examples {
namespace while_loop {
namespace mg_src {
namespace while_loop_cpp {


    IterationProgram::_one IterationProgram::one;

    int32_utils IterationProgram::__int32_utils;

    int16_utils IterationProgram::__int16_utils;



    IterationProgram::_add IterationProgram::add;



    IterationProgram::_increment IterationProgram::increment;



    while_ops<IterationProgram::Int16, IterationProgram::Int16, IterationProgram::_increment, IterationProgram::_isLowerThan> IterationProgram::__while_ops;

    while_ops<IterationProgram::Int32, IterationProgram::Int32, IterationProgram::_increment, IterationProgram::_isLowerThan> IterationProgram::__while_ops0;

    IterationProgram::_isLowerThan IterationProgram::isLowerThan;



    IterationProgram::_repeat IterationProgram::repeat;

} // examples
} // while_loop
} // mg_src
} // while_loop_cpp