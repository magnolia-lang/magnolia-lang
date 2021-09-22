#include "gen/examples/containers/mg-src/containers.hpp"


namespace examples {
namespace containers {
namespace mg_src {
namespace containers {
    types NestedPairProgram::__types;
    NestedPairProgram::_first NestedPairProgram::first;
    NestedPairProgram::_make_pair NestedPairProgram::make_pair;
    NestedPairProgram::_second NestedPairProgram::second;
    pair<NestedPairProgram::InnerPair, NestedPairProgram::InnerPair> NestedPairProgram::__pair;
    pair<NestedPairProgram::Int16, NestedPairProgram::Int32> NestedPairProgram::__pair0;
    NestedPairProgram::Int16 NestedPairProgram::_first::operator()(const NestedPairProgram::InnerPair& t) {
        return __pair0.first(t);
    };
    NestedPairProgram::InnerPair NestedPairProgram::_first::operator()(const NestedPairProgram::OuterPair& t) {
        return __pair.first(t);
    };
    NestedPairProgram::OuterPair NestedPairProgram::_make_pair::operator()(const NestedPairProgram::InnerPair& a, const NestedPairProgram::InnerPair& b) {
        return __pair.make_pair(a, b);
    };
    NestedPairProgram::InnerPair NestedPairProgram::_make_pair::operator()(const NestedPairProgram::Int16& a, const NestedPairProgram::Int32& b) {
        return __pair0.make_pair(a, b);
    };
    NestedPairProgram::Int32 NestedPairProgram::_second::operator()(const NestedPairProgram::InnerPair& t) {
        return __pair0.second(t);
    };
    NestedPairProgram::InnerPair NestedPairProgram::_second::operator()(const NestedPairProgram::OuterPair& t) {
        return __pair.second(t);
    };

} // examples
} // containers
} // mg_src
} // containers