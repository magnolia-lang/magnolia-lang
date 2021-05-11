#include <iostream>

/* This is the hand translation from the following Magnolia code:
package P;

signature Sig1 = {
    type T1;
    type T2;
    type T3;

    function sig_fun_proto(t1: T1, t2: T2): T2;
    procedure sig_proc_proto(obs t1: T1, upd t2_0: T2, out t2_1: T2);
    predicate sig_pred_proto(t1: T1, t2: T2);
};

implementation Impl1 = {
    require S1;
};

satisfaction Impl1_models_Sig1 = Impl1 models Sig1;

*/

namespace __magnolia {
namespace P {

// The C++ concept is parameterized by a module that satisfies the signature
// requirements defined in the Magnolia code. This module should have inner
// type names T1, T2, and T3. Witnesses for T1 and T2 appear in the requirement
// parameter list.
template <typename SigMod>
concept Sig1 = requires (const SigMod::T1& t1, const SigMod::T2& t2_obs, SigMod::T2& t2_updout) {
    typename SigMod::T3; // no witness in parameter list for T3
    {SigMod().sig_fun_proto(t1, t2_obs)} -> std::same_as<typename SigMod::T2>;
    {SigMod().sig_proc_proto(t1, t2_updout, t2_updout)} -> std::same_as<void>;
    {SigMod().sig_pred_proto(t1, t2_obs)} -> std::same_as<bool>;
};

template <class _T1, class _T2, class _T3> class Impl1 {
public:
    Impl1() {};

    // Associated types
    typedef _T1 T1;
    typedef _T2 T2;
    typedef _T3 T3;

    virtual T2 sig_fun_proto(const T1&, const T2&) const = 0;
    virtual void sig_proc_proto(const T1&, T2&, T2&) const = 0;
    virtual bool sig_pred_proto(const T1&, const T2&) const = 0;
};


template <Sig1 T>
void models_Sig1(T) {}

}}

int main() {
    // To run this satisfaction check, the methods of Impl1 can not be pure
    // virtual. Also, the satisfaction check would only be run for one
    // particular type instantiation.
    //__magnolia::P::models_Sig1(__magnolia::P::Impl1<int, float, double>());
    return 0;
}
