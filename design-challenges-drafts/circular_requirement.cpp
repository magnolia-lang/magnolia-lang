#include <iostream>

template <typename T>
struct Ext1 {
    typedef int X;
    typedef T Y;
};

template <typename T>
struct Ext2 {
    typedef T X;
    typedef double Y;
    X p(X a) { return a; }
    Y p(Y a) { return a; }
};

struct T1 {};

int main() {
    // Problems: circular dependencies between externals of the form
    // external C++ Ext1 { type X; require type Y; } and
    // external C++ Ext2 { require type X; type Y; }.
    // The external class must be fully specialized before it can be referenced.
    // Specializing with random existing types is unsafe, because for example
    // specializing Ext2 with double would yield a broken program.
    // A possible solution is to generate empty structs in our Magnolia
    // namespace, and use them for specialization.
    Ext1<Ext2<T1>::Y> e;
    return 0;
}
