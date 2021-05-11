#include <iostream>

/* The code below is a translation from the following Magnolia code:
package P;

external C++ Ext {
    type Int;
    type Double;
    type String;

    function as_double(e: Int): Double;
    function as_str(e: Int): String;
}

// Prog is a program that requires overloading purely on the return type of
// 'as'.
program Prog = {
   use Ext[ as_double => as, as_str => as ];
};

*/

// Ext is not generated, but instead is implemented by users directly in C++.
class Ext {
public:
    typedef std::string String;
    typedef double Double;
    typedef int Int;

    Double as_double(const Int& e) const { return e; }
    String as_str(const Int& e) const { return std::to_string(e); }
};

/* Note that Ext could be partially generated from Magnolia, leaving holes to
be filled by the user. It is not clear if this is necessary (and anyway, it
would not be the default mode of operation). A generated Ext would look as
follows:

class Ext {
public:
    typedef ?0 String;
    typedef ?1 Double;
    typedef ?2 Int;

    Double as_double(const Int& e) const { ?3 }
    String as_str(const Int& e) const { ?4 }
};

*/

namespace __magnolia {
namespace P {
class Prog : private virtual Ext {
private:
    // To solve the challenge of function overloading purely on return types,
    // we 'mutify' the functions so that the return type is instead the type
    // of one of the function's parameters. We then generate, for each
    // function, one private inline function. All these generated functions
    // must have the same name, which should not already be bound within the
    // namespace.
    //
    // Then, we produce a public templated function (1) which calls the right
    // generated function based on the provided argument type.
    //
    // The downside of this approach is that calling functions that are
    // overloaded solely on their return types looks differently from calling
    // a different function, e.g.
    //
    // P.f<int>() vs P.g(), for f a function overloaded on its return type,
    //                      and g a function that is not overloaded solely on
    //                      its return type in P.
    //
    // The upside, however, is that using this function looks pretty much like
    // it would in Magnolia in ambiguous contexts (f() : int).
    inline void as_localOverloadingResolutionFreeName(const Int& e, Double& o) const {
        o = as_double(e);
    }
    inline void as_localOverloadingResolutionFreeName(const Int& e, String& o) const {
        o = as_str(e);
    };
public:
    typedef Ext::String String;
    typedef Ext::Int Int;
    typedef Ext::Double Double;

    // (1) This is the templated function referenced in the above text.
    template <typename T>
    T as(const Int& e) const {
        T o;
        as_localOverloadingResolutionFreeName(e, o);
        return o;
    }
};
}}

typedef __magnolia::P::Prog::Int Int;
typedef __magnolia::P::Prog::String String;

int main() {
    __magnolia::P::Prog p;
    Int i = 4;
    std::cout << p.as<String>(i) << std::endl;
    return 0;
}
