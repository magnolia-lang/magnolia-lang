#include <iostream>

// Imported dependencies, defined in C++ files
double gen() { return 1337.0; }
double f(const int& arg) { return arg * 18.0; }

/* This is a hand translation from the following simple Magnolia code:
package P;

implementation Impl = {
    require type T1;
    type T2;
    function proto_f() : T2;
    external function external_f(e: T1): T2;
    function impl_f(e1: T1, e2: T2): T2 = external_f(e1);
};

program Prog = {
    use Impl[ T1 => int
            , T2 => double
            , external_f => f
            ];

    function proto_f() : double = gen();

    external C++ {
        type int;
        type double;
        function gen() : double;
    }
};

*/

namespace __magnolia {
namespace P {

// For implementations, we generate a class context along with a nested class.
// This may not be necessary, since we define all methods in the nested class.
// The only point of doing that is to carry along the meaning of what types are
// "specified" within the implementation, and which ones are "required". This
// slight difference may not be meaningful enough that it makes sense to carry
// it over to C++.
template <class T2> class Impl_Ctx {
public:
    Impl_Ctx() {};
    template <class T1> class Impl {
    public:
        Impl() {};
        // Both prototypes and externals are made into pure virtual functions
        // in an implementation. This is because both types of functions can
        // be renamed before being actually used in a program. The only
        // difference is that external_f will always be overriden by a call to
        // an external function.
        virtual T2 proto_f() const = 0;
        virtual T2 external_f(const T1& e) const = 0;
        T2 impl_f(const T1& e1, const T2& e2) const { return external_f(e1); }
    };
};

class Prog : private virtual Impl_Ctx<double>::Impl<int> {
private:
    // the implementation of renamed functions is done here; the renaming is
    // performed by adding an inline wrapper around the function (see f,
    // below).
    virtual double external_f(const int& arg) const override {
        using ::f; // use external definition of f, not the local one
        return f(arg);
    }
public: // Exposed API
    double proto_f() const override { return gen(); }
    // Inline renaming of external_f
    inline double f(const int& e) const { return external_f(e); }
    // Explicit 'using' here to refer to the implementation, and for
    // documentation purposes.
    using Impl_Ctx<double>::Impl<int>::impl_f; // derived
};

}}

int main() {
    __magnolia::P::Prog p;
    std::cout << p.f(2) << std::endl;
    std::cout << p.impl_f(3, 4.0) << std::endl;
    std::cout << p.proto_f() << std::endl;
    return 0;
}
