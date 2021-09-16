#include <string>

template <typename _Context, typename _State, class _cond, class _body>
struct while_ops {
    typedef _State State;
    typedef _Context Context;

    struct fop_cond {
        inline bool operator()(const State &state, const Context &context) {
            return ext_cond(state, context);
        }

        private:
            _cond ext_cond;
    };

    struct fop_body {
        inline void operator()(State &state, const Context &context) {
            ext_body(state, context);
        }

        private:
            _body ext_body;
    };

    inline void repeat(State &state, const Context &context) {
        while (while_ops::cond(state, context)) {
            while_ops::body(state, context);
        }
    }

    private:
        fop_cond cond;
        fop_body body;
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
