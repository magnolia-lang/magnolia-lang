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

template <class _T1, class _T2, class _T3> class Sig1 {
public:
    Sig1() {};

    // Associated types
    typedef _T1 T1;
    typedef _T2 T2;
    typedef _T3 T3;

    virtual T2 sig_fun_proto(const T1&, const T2&) const = 0;
    virtual void sig_proc_proto(const T1&, T2&, T2&) const = 0;
    virtual bool sig_pred_proto(const T1&, const T2&) const = 0;
};

// When Sig1 is represented as an abstract class, inheritance is enough to
// model the satisfaction relation between Impl1 and Sig1.
template <class _T1, class _T2, class _T3> class Impl1
    : public virtual Sig1<_T1, _T2, _T3>
{};

}}

int main() {
    return 0;
}
