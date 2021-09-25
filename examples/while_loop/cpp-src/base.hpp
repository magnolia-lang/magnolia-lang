#include <string>

template <typename _Context, typename _State, class _body, class _cond>
struct while_ops {
    typedef _State State;
    typedef _Context Context;

    _body body;
    _cond cond;

    inline void repeat(State &state, const Context &context) {
        while (while_ops::cond(state, context)) {
            while_ops::body(state, context);
        }
    }
};

struct int16_utils {
    typedef short Int16;

    inline Int16 one() { return 1; }
    inline Int16 add(const Int16 a, const Int16 b) {
        return a + b;
    }
    inline bool isLowerThan(const Int16 a, const Int16 b) {
        return a < b;
    }
};

struct int32_utils {
    typedef int Int32;

    inline Int32 one() { return 1; }
    inline Int32 add(const Int32 a, const Int32 b) {
        return a + b;
    }
    inline bool isLowerThan(const Int32 a, const Int32 b) {
        return a < b;
    }
};
