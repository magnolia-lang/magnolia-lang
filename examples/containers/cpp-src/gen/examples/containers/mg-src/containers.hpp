#include "base.hpp"
#include <cassert>


namespace examples {
namespace containers {
namespace mg_src {
namespace containers {
struct NestedPairProgram {
private:
    static types __types;
public:
    typedef types::Int32 Int32;
    typedef types::Int16 Int16;
private:
    static pair<NestedPairProgram::Int16, NestedPairProgram::Int32> __pair0;
public:
    typedef pair<NestedPairProgram::Int16, NestedPairProgram::Int32>::Pair InnerPair;
    typedef pair<NestedPairProgram::InnerPair, NestedPairProgram::InnerPair>::Pair OuterPair;
private:
    static pair<NestedPairProgram::InnerPair, NestedPairProgram::InnerPair> __pair;
public:
    struct _first {
        NestedPairProgram::InnerPair operator()(const NestedPairProgram::OuterPair& t);
        NestedPairProgram::Int16 operator()(const NestedPairProgram::InnerPair& t);
    };

    static NestedPairProgram::_first first;
    struct _make_pair {
        NestedPairProgram::OuterPair operator()(const NestedPairProgram::InnerPair& a, const NestedPairProgram::InnerPair& b);
        NestedPairProgram::InnerPair operator()(const NestedPairProgram::Int16& a, const NestedPairProgram::Int32& b);
    };

    static NestedPairProgram::_make_pair make_pair;
    struct _second {
        NestedPairProgram::InnerPair operator()(const NestedPairProgram::OuterPair& t);
        NestedPairProgram::Int32 operator()(const NestedPairProgram::InnerPair& t);
    };

    static NestedPairProgram::_second second;
};
} // examples
} // containers
} // mg_src
} // containers