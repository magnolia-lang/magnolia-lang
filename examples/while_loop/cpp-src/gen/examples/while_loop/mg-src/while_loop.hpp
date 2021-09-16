#include "base.hpp"
#include <cassert>


namespace examples {
namespace while_loop {
namespace mg_src {
namespace while_loop {
struct IterationProgram {
private:
    struct dummy_struct {};
    struct dummy_struct0 {};
    struct dummy_struct1 {};
    struct dummy_struct2 {};
public:
    typedef int16_utils::Int16 Int16;
    typedef int32_utils::Int32 Int32;
    
    struct _add {
        IterationProgram::Int16 operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b);
        IterationProgram::Int32 operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b);
    };

    
    struct _increment {
        void operator()(IterationProgram::Int16& counter, const IterationProgram::Int16& bound);
        void operator()(IterationProgram::Int32& counter, const IterationProgram::Int32& bound);
    };

    
    struct _isLowerThan {
        bool operator()(const IterationProgram::Int16& a, const IterationProgram::Int16& b);
        bool operator()(const IterationProgram::Int32& a, const IterationProgram::Int32& b);
    };

    
    struct _one {
        template <typename T>
        inline T operator()() {
            T o;
            IterationProgram::one0(o);
            return o;
        };
    };

    
    struct _repeat {
        void operator()(IterationProgram::Int16& s, const IterationProgram::Int16& c);
        void operator()(IterationProgram::Int32& s, const IterationProgram::Int32& c);
    };

private:
    static void one0(IterationProgram::Int16& o);
    static void one0(IterationProgram::Int32& o);
    static int16_utils __int16_utils;
    static int32_utils __int32_utils;
public:
    static IterationProgram::_add add;
    static IterationProgram::_increment increment;
    static IterationProgram::_isLowerThan isLowerThan;
    static IterationProgram::_one one;
    static IterationProgram::_repeat repeat;
private:
    static while_ops<IterationProgram::Int16, IterationProgram::Int16, IterationProgram::_isLowerThan, IterationProgram::_increment> __while_ops;
    static while_ops<IterationProgram::Int32, IterationProgram::Int32, IterationProgram::_isLowerThan, IterationProgram::_increment> __while_ops0;
};
} // examples
} // while_loop
} // mg_src
} // while_loop