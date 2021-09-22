#include "base.hpp"
#include <cassert>


namespace examples {
namespace containers {
namespace mg_src {
namespace containers {
struct NestedPairProgram {
    struct _first;
    struct _make_pair;
    struct _second;
    typedef types::Int32 Int32;
    typedef types::Int16 Int16;
    typedef pair<Int16, Int32>::Pair InnerPair;
    typedef pair<InnerPair, InnerPair>::Pair OuterPair;
    struct _first {
        NestedPairProgram::Int16 operator()(const NestedPairProgram::InnerPair& t);
        NestedPairProgram::InnerPair operator()(const NestedPairProgram::OuterPair& t);
    };

    struct _make_pair {
        NestedPairProgram::OuterPair operator()(const NestedPairProgram::InnerPair& a, const NestedPairProgram::InnerPair& b);
        NestedPairProgram::InnerPair operator()(const NestedPairProgram::Int16& a, const NestedPairProgram::Int32& b);
    };

    struct _second {
        NestedPairProgram::Int32 operator()(const NestedPairProgram::InnerPair& t);
        NestedPairProgram::InnerPair operator()(const NestedPairProgram::OuterPair& t);
    };

private:
    static types __types;
public:
    static NestedPairProgram::_first first;
    static NestedPairProgram::_make_pair make_pair;
    static NestedPairProgram::_second second;
private:
    static pair<NestedPairProgram::InnerPair, NestedPairProgram::InnerPair> __pair;
    static pair<NestedPairProgram::Int16, NestedPairProgram::Int32> __pair0;
};
} // examples
} // containers
} // mg_src
} // containers