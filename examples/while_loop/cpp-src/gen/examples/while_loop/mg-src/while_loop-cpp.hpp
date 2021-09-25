#pragma once

#include "base.hpp"
#include <cassert>


namespace examples {
namespace while_loop {
namespace mg_src {
namespace while_loop_cpp {
struct IterationProgram {
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            IterationProgram::one0(o);
            return o;
        };
    };

    static IterationProgram::_one one;
private:
    static int32_utils __int32_utils;
    static int16_utils __int16_utils;
public:
    typedef int32_utils::Int32 Int32;
private:
    static inline void one0(IterationProgram::Int32& o) {
        o = __int32_utils.one();
    };
public:
    typedef int16_utils::Int16 Int16;
    struct _add {
        inline IterationProgram::Int32 operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b) {
            return __int32_utils.add(a, b);
        };
        inline IterationProgram::Int16 operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b) {
            return __int16_utils.add(a, b);
        };
    };

    static IterationProgram::_add add;
    struct _increment {
        inline void operator()(IterationProgram::Int16& counter, const IterationProgram::Int16& bound) {
            counter = IterationProgram::add(counter, IterationProgram::one.operator()<Int16>());
        };
        inline void operator()(IterationProgram::Int32& counter, const IterationProgram::Int32& bound) {
            counter = IterationProgram::add(counter, IterationProgram::one.operator()<Int32>());
        };
    };

    static IterationProgram::_increment increment;
    struct _isLowerThan {
        inline bool operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b) {
            return __int16_utils.isLowerThan(a, b);
        };
        inline bool operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b) {
            return __int32_utils.isLowerThan(a, b);
        };
    };

private:
    static while_ops<IterationProgram::Int16, IterationProgram::Int16, IterationProgram::_increment, IterationProgram::_isLowerThan> __while_ops;
    static while_ops<IterationProgram::Int32, IterationProgram::Int32, IterationProgram::_increment, IterationProgram::_isLowerThan> __while_ops0;
public:
    static IterationProgram::_isLowerThan isLowerThan;
    struct _repeat {
        inline void operator()(IterationProgram::Int16& s, const IterationProgram::Int16& c) {
            return __while_ops.repeat(s, c);
        };
        inline void operator()(IterationProgram::Int32& s, const IterationProgram::Int32& c) {
            return __while_ops0.repeat(s, c);
        };
    };

    static IterationProgram::_repeat repeat;
private:
    static inline void one0(IterationProgram::Int16& o) {
        o = __int16_utils.one();
    };
};
} // examples
} // while_loop
} // mg_src
} // while_loop_cpp