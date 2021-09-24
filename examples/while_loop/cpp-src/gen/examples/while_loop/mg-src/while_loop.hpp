#include "base.hpp"
#include <cassert>


namespace examples {
namespace while_loop {
namespace mg_src {
namespace while_loop {
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
    static void one0(IterationProgram::Int32& o);
public:
    typedef int16_utils::Int16 Int16;
    struct _add {
        IterationProgram::Int32 operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b);
        IterationProgram::Int16 operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b);
    };

    static IterationProgram::_add add;
    struct _increment {
        void operator()(IterationProgram::Int16& counter, const IterationProgram::Int16& bound);
        void operator()(IterationProgram::Int32& counter, const IterationProgram::Int32& bound);
    };

    static IterationProgram::_increment increment;
    struct _isLowerThan {
        bool operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b);
        bool operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b);
    };

private:
    static while_ops<IterationProgram::Int16, IterationProgram::Int16, IterationProgram::_isLowerThan, IterationProgram::_increment> __while_ops;
    static while_ops<IterationProgram::Int32, IterationProgram::Int32, IterationProgram::_isLowerThan, IterationProgram::_increment> __while_ops0;
public:
    static IterationProgram::_isLowerThan isLowerThan;
    struct _repeat {
        void operator()(IterationProgram::Int16& s, const IterationProgram::Int16& c);
        void operator()(IterationProgram::Int32& s, const IterationProgram::Int32& c);
    };

    static IterationProgram::_repeat repeat;
private:
    static void one0(IterationProgram::Int16& o);
};
} // examples
} // while_loop
} // mg_src
} // while_loop