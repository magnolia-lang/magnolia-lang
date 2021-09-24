#include "gen/examples/while_loop/mg-src/while_loop.hpp"


namespace examples {
namespace while_loop {
namespace mg_src {
namespace while_loop {


    IterationProgram::_one IterationProgram::one;

    int32_utils IterationProgram::__int32_utils;

    int16_utils IterationProgram::__int16_utils;

    void IterationProgram::one0(IterationProgram::Int32& o) {
        o = __int32_utils.one();
    };

    IterationProgram::Int32 IterationProgram::_add::operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b) {
        return __int32_utils.add(a, b);
    };

    IterationProgram::Int16 IterationProgram::_add::operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b) {
        return __int16_utils.add(a, b);
    };


    IterationProgram::_add IterationProgram::add;

    void IterationProgram::_increment::operator()(IterationProgram::Int16& counter, const IterationProgram::Int16& bound) {
        counter = IterationProgram::add(counter, IterationProgram::one.operator()<Int16>());
    };

    void IterationProgram::_increment::operator()(IterationProgram::Int32& counter, const IterationProgram::Int32& bound) {
        counter = IterationProgram::add(counter, IterationProgram::one.operator()<Int32>());
    };


    IterationProgram::_increment IterationProgram::increment;

    bool IterationProgram::_isLowerThan::operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b) {
        return __int16_utils.isLowerThan(a, b);
    };

    bool IterationProgram::_isLowerThan::operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b) {
        return __int32_utils.isLowerThan(a, b);
    };


    while_ops<IterationProgram::Int16, IterationProgram::Int16, IterationProgram::_isLowerThan, IterationProgram::_increment> IterationProgram::__while_ops;

    while_ops<IterationProgram::Int32, IterationProgram::Int32, IterationProgram::_isLowerThan, IterationProgram::_increment> IterationProgram::__while_ops0;

    IterationProgram::_isLowerThan IterationProgram::isLowerThan;

    void IterationProgram::_repeat::operator()(IterationProgram::Int16& s, const IterationProgram::Int16& c) {
        return __while_ops.repeat(s, c);
    };

    void IterationProgram::_repeat::operator()(IterationProgram::Int32& s, const IterationProgram::Int32& c) {
        return __while_ops0.repeat(s, c);
    };


    IterationProgram::_repeat IterationProgram::repeat;

    void IterationProgram::one0(IterationProgram::Int16& o) {
        o = __int16_utils.one();
    };

} // examples
} // while_loop
} // mg_src
} // while_loop