#include <iostream>

// Our program is like:
// program P = {
//   type float;
//   type int;
//   function f(): float;
//   function f(): int;
//   function f(a: int, b: int): int;
// }
// Assumption for this codegen: any external code parameterized by the
// function operator *DOES NOT* use the overloaded declarations.

struct Program {
public:
    template <typename T>
    struct FunctionOp {
    private:
        inline void f(float &f) { f = 18.5; }
        inline void f(int &i) { i = 20; }
    public:
        inline T operator()() {
            T o;
            f(o);
            return o;
        }
        int operator()(int a, int b) {
            return 4;
        }
    };

    template <typename T>
    inline T f() {
        Program::FunctionOp<T> f_ops;
        return f_ops();
    }

    inline int f(int a, int b) {
        Program::FunctionOp<int> f_ops; // use one of the types that is known
                                        // to be a valid template instanciation
        return f_ops(a, b);
    }
};

int main() {
    Program p;
    std::cout << p.f(1, 2) << std::endl;
    std::cout << p.f<float>() << std::endl;
    std::cout << p.f<int>() << std::endl;
    return 0;
}
