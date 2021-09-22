#include <fstream>
#include <iostream>
#include <list>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

// TODO: at the moment, Magnolia externals do not deal with required types in
// externals (i.e. they do not instantiate templated structs). This will be
// fixed soon, but is the reason the API here is not exactly the one described
// in examples/bgl/mg-src/ExternalDataStructures.mg.

struct string {
    typedef std::string String;
};
template <typename _E>
struct queue {
    typedef _E E;
    typedef std::queue<queue::E> Queue;
    inline queue::Queue nil() { return Queue(); }
    inline queue::Queue enqueue(const queue::Queue &q, const queue::E &e) {
        auto _q = q;
        _q.push(e);
        return _q;
    }
    inline queue::Queue dequeue(const queue::Queue &q) {
        auto _q = q;
        _q.pop();
        return _q;
    }
    inline queue::E first(const queue::Queue &q) {
        return q.front();
    }
    inline bool empty(const queue::Queue &q) { return q.empty(); }
};
template <typename _E>
struct hash_set {
    typedef _E E;
    typedef std::set<hash_set::E> HashSet;
    static inline hash_set::HashSet nil() { return HashSet(); }
    inline hash_set::HashSet insert(const hash_set::HashSet &h,
                                    const hash_set::E &e) {
        auto _h = h;
        _h.insert(e);
        return _h;
    }
    inline hash_set::HashSet remove(const hash_set::HashSet &h,
                                    const hash_set::E &e) {
        auto _h = h;
        _h.erase(e);
        return _h;
    }
    inline hash_set::E min(const hash_set::HashSet &h) {
        return *h.begin();
    }
    inline bool member(const hash_set::HashSet &h, const hash_set::E &e) {
        return h.find(e) != h.end();
    }
    inline bool empty(const hash_set::HashSet &h) { return h.empty(); }
};
// TODO: have to template manually and fix the generated code here
template <typename _A, typename _B>
struct pair {
    typedef _A A;
    typedef _B B;
    typedef std::pair<pair::A, pair::B> Pair;

    inline pair::Pair make_pair(const pair::A &a, const pair::B &b) {
        return std::make_pair(a, b);
    }
    inline pair::A first(const pair::Pair &pair) {
        return pair.first;
    }
    inline pair::B second(const pair::Pair &pair) {
        return pair.second;
    }
};
// TODO: have to template manually and fix the generated code here
template <typename _A, typename _B, typename _C>
struct tuple_3 {
    typedef _A A;
    typedef _B B;
    typedef _C C;

    struct Tuple {
        tuple_3<_A, _B, _C>::A first;
        tuple_3<_A, _B, _C>::B second;
        tuple_3<_A, _B, _C>::C third;

        Tuple(const tuple_3<_A, _B, _C>::A &a,
              const tuple_3<_A, _B, _C>::B &b,
              const tuple_3<_A, _B, _C>::C &c)
            : first(a), second(b), third(c) {};
    };

    inline tuple_3::Tuple make_tuple(const tuple_3::A &a,
                                     const tuple_3::B &b,
                                     const tuple_3::C &c) {
        return tuple_3::Tuple(a, b, c);
    }
    inline tuple_3::A first(const tuple_3::Tuple &t) {
        return t.first;
    }
    inline tuple_3::B second(const tuple_3::Tuple &t) {
        return t.second;
    }
    inline tuple_3::C third(const tuple_3::Tuple &t) {
        return t.third;
    }
};
template <typename _E>
struct pprinter {
    typedef _E E;

    // TODO: write everything as function objects so it can be passed around
    struct _pprint {
        void operator()(const E &e) {
            //std::cout << "TODO: fix, check compile" << std::endl;
            std::cout << e << std::endl;
        }
    };

    _pprint pprint;
};
struct unit {
    struct Unit {};
};
// TODO: have to edit generated files to template these
// order: - typenames, alphabetical
//        - classes, alphabetical
template <typename _Context, typename _State, class _cond, class _body>
struct while_loop {
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
        while (while_loop::cond(state, context)) {
            while_loop::body(state, context);
        }
    }

    private:
        fop_cond cond;
        fop_body body;
};

template <typename _Vertex>
struct edge {
    typedef _Vertex Vertex;
    typedef std::pair<_Vertex, _Vertex> Edge;

    inline Vertex source(const Edge& e) { return e.first; }
    inline Vertex target(const Edge& e) { return e.second; }

    inline Edge make_edge(const Vertex &src, const Vertex &tgt) {
        return std::make_pair(src, tgt);
    }
};

template <typename _A, typename _B, typename _CollectionA,
          typename _CollectionB, class _addToCollection,
          class _emptyCollectionB, class _extractOneElement, class _f,
          class _isCollectionEmpty, class _removeFromCollection>
struct map_function {
    typedef _A A;
    typedef _B B;
    typedef _CollectionA CollectionA;
    typedef _CollectionB CollectionB;

    _addToCollection addToCollection;
    _emptyCollectionB emptyCollectionB;
    _extractOneElement extractOneElement;
    _isCollectionEmpty isCollectionEmpty;
    _f f;
    _removeFromCollection removeFromCollection;

    CollectionB map(const CollectionA &ca) {
        CollectionB result = emptyCollectionB();
        CollectionA copy_ca = ca;
        
        while (!isCollectionEmpty(copy_ca)) {
            A element = extractOneElement(copy_ca);
            addToCollection(result, f(element));
            copy_ca = removeFromCollection(copy_ca, element);
        }

        return result;
    }
};

template <typename _Edge, typename _EdgeCollection, typename _Vertex,
          class _addToCollection, class _emptyCollection,
          class _extractOneElement, class _isCollectionEmpty, class _isIn,
          class _make_edge, class _removeFromCollection, class _source,
          class _target>
struct incidence_graph {
    typedef _Vertex Vertex;
    typedef _Edge Edge;
    typedef _EdgeCollection EdgeCollection;
    
    struct Graph {
        EdgeCollection edges;
        Graph(EdgeCollection edges) : edges(edges) {};
    };

    _addToCollection addToCollection;
    _emptyCollection emptyCollection;
    _extractOneElement extractOneElement;
    _isCollectionEmpty isCollectionEmpty;
    _isIn isIn;
    _make_edge make_edge;
    _removeFromCollection removeFromCollection;
    _source source;
    _target target;
    
    EdgeCollection outEdges(const Graph &g, const Vertex &v) {
        EdgeCollection result = emptyCollection(),
                       allEdges = g.edges;
        while (!isCollectionEmpty(allEdges)) {
            Edge lastEdge = extractOneElement(allEdges);
            
            if (source(lastEdge) == v) {
                addToCollection(result, lastEdge);
            }

            allEdges = removeFromCollection(allEdges, lastEdge);
        }
        
        return result;
    }
};

template <typename _Vertex, typename _VertexCollection>
struct adjacency_graph {
    struct Graph {
        std::vector<_Vertex> vertices;
        std::vector<std::pair<_Vertex, _Vertex>> edges;
        Graph(std::vector<_Vertex> vertices, std::vector<std::pair<_Vertex, _Vertex>> edges)
            : vertices(vertices), edges(edges) {};
    };

    typedef _Vertex Vertex;
    typedef _VertexCollection VertexCollection;

    VertexCollection adjacentVertices(const Graph &g, const Vertex &v) {
        typename hash_set<Vertex>::HashSet collection = hash_set<Vertex>::nil(); // TODO: fix
        for (auto it = g.edges.begin(); it != g.edges.end(); ++it) {
            if ((*it).first == v) collection.insert((*it).second);
            if ((*it).second == v) collection.insert((*it).first);
            //std::cout << (*it).first << " " << (*it).second;
            //std::cin;
        }
        return collection;
    }
};
