# Code Generation Design (C++)

This document describes the design of code generation from Magnolia to (for the
moment) C++.

For simplicity's sake, satisfaction claims are mostly omitted from this initial
specification. We focus only on generating code for the basic modular building
blocks of Magnolia, that is to say `signature`s, `concept`s, `implementation`s,
`external`s, and `program`s.

The document starts out by describing how each of these building blocks is
translated, and then moves on to explaining subtleties in how they interact
with each other. The hope in spelling this out in details here is that we will

1. create a valuable resource for understanding how Magnolia works and how its
   constructs can be expressed in other languages;
2. gain some insight into how code generation can be generalized to include
   other interesting targets (*JavaScript, Python, …*).

## Signature and Concepts

Accompanying code for this subsection can be found in
`signature_impl_satisfaction_concept.cpp`, and
`signature_impl_satisfaction_abstract.cpp`.

Fundamentally, a concept is just a signature augmented with some additional
axioms (which one may see as testing, or debugging functions). For the moment,
C++ concepts seem to be equivalent to Magnolia signatures; for now, we ignore
axioms, and bundle signatures and concepts together.

Take the following signature (or concept):

```mag
signature Sig1 = {
    type T1;
    type T2;
    type T3;

    function sig_fun_proto(t1: T1, t2: T2): T2;
    procedure sig_proc_proto(obs t1: T1, upd t2_0: T2, out t2_1: T2);
    predicate sig_pred_proto(t1: T1, t2: T2);
};
```

Assuming we have a class `SigMod` that exposes this signature, here is what it
would look like using concepts:

```cpp
template <typename SigMod>
concept Sig1 = requires (const SigMod::T1& t1, const SigMod::T2& t2_obs, SigMod::T2& t2_updout) {
    typename SigMod::T3; // no witness in parameter list for T3
    {SigMod().sig_fun_proto(t1, t2_obs)} -> std::same_as<typename SigMod::T2>;
    {SigMod().sig_proc_proto(t1, t2_updout, t2_updout)} -> std::same_as<void>;
    {SigMod().sig_pred_proto(t1, t2_obs)} -> std::same_as<bool>;
};
```

To make a signature defined like so useful, one must be able to ensure its API
is implemented by other modules. That means we would like to have the ability to
translate satisfaction claims.

An example follows. Take the following Magnolia code:

```mag
implementation Impl1 = {
    require S1;
};

satisfaction Impl1_models_Sig1 = Impl1 models Sig1;
```

One natural way to translate this code is as follows:

```cpp
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

int main() {
    //models_Sig1(Impl1<int, float, double>());
    return 0;
}
```

However, that Impl1 satisfies Sig1 can not be shown using concepts for two
reasons:

1. C++ can not check that `Impl1` models `Sig1` for every set of parameters
   `<T1, T2, T3>`;
2. `Impl1` can not be directly instantiated, because it contains pure virtual
   methods. This could be fixed by forcing *dirty* default implementations
   (*a generic non-void valid body is `return (retType)0;`*), but this is
   undesirable.

One additional possible disadvantage of using concepts is compatibility: C++
only implemented concepts in C++20. This is likely not a huge issue, since it is
unlikely Magnolia will get a lot (any?) users anyway. Another possible way of
specifying such a signature, which might be more suitable, is to use an
abstract class.

`signature_impl_satisfaction_abstract.cpp` re-implements this same example using
an abstract class instead. The satisfaction relation becomes redundant, as
inheritance ensures the child implementation class exposes the same API as the
parent signature.

Other elements to consider, however:

1. using abstract classes *may* require dynamic dispatch if we do not restrict
   ourselves to actually using programs as the exported API (*if restricting
   ourselves to programs, which are concrete classes, there is no reason a VFT
   would be required*). Not sure if that's such a problem. Concepts, on the
   other hand, require purely compile time checks;
2. there may not be much point in exporting `signature`s directly, since
   programs exported by Magnolia are not generic anymore. This consideration
   may change for exporting concepts however, which adds implemented axioms to
   the mix.

**TODO: develop here what exactly to do with axioms**

## Implementations and Programs

Signatures and concepts form more or less developed specifications.
Implementations allow callables to have bodies, and types to be fixed. However,
they remain abstract in the general case, as they allow keeping external
requirements. Programs, on the other hand, are always completely concrete: they
have no external requirements.

## Challenges

### Overloading on Return Types

Accompanying code for this subsection can be found in
`overloading_on_return_type.cpp`.

Overloading functions solely on their return types is not supported in C++. It
is however something Magnolia supports. There is thus no canonical way to
perform the conversion between Magnolia and C++ in this case. What follows is a
proposal for how to handle this conversion.

Take the following Magnolia code:

```mag
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
```

The crux of the issue is that `Prog` now needs to expose two methods with
prototypes

```cpp
Double as(const Int& e) const; // (1)
String as(const Int& e) const; // (2)
```

While C++ is not able to perform overloading solely on return types, it does
deal with simpler instances of overloading. With mutification, we can produce
two overloaded functions that can be disambiguated:

```cpp
// (1) becomes
void as_mutified(const Int& e, Double& o) const;
// (2) becomes
void as_mutified(const Int& e, String& o) const;
```

The only catch here is that calling `as_mutified` requires one additional
argument in comparison with the original function prototypes for `as`. However,
now that we can disambiguate between these functions, we are able to define:

```cpp
template <typename T>
T as(const Int& e) const {
    T o;
    as_mutified(e, o);
    return o;
}
```

This implementation of `as` has the same prototype as the Magnolia definition of
`as`. The catch is that it is now left as an exercise to the user to
systematically specify which version of `as` should be called through template
instantiation (*e.g. `as<Double>(4)` or `as<String>(5)`*).

See below how the generated code for `program Prog` would look like:

```cpp
class Prog ... {
...

private:
    inline void as_mutified(const Int& e, Double& o) const {
        o = as_double(e);
    }
    inline void as_mutified(const Int& e, String& o) const {
        o = as_str(e);
    };
public:
    template <typename T>
    T as(const Int& e) const {
        T o;
        as_mutified(e, o);
        return o;
    }
...
}
```
