#include <list>
#include <map>
#include <queue>
#include <tuple>
#include <utility>

// base_types_cpp
struct base_types {
    typedef int Int;
    typedef int Vertex;
    typedef int VertexCount;
};

// color_marker_cpp
struct color_marker {
    enum class Color { White, Gray, Black };

    inline Color white() { return Color::White; }
    inline Color gray() { return Color::Gray; }
    inline Color black() { return Color::Black; }
};

// graph_cpp
template <typename _Vertex>
struct edge {
    typedef _Vertex Vertex;
    typedef std::pair<Vertex, Vertex> Edge;

    inline Vertex src(const Edge &e) { return e.first; }
    inline Vertex tgt(const Edge &e) { return e.second; }
};

template <typename _Edge, typename _Vertex, class src, class tgt>
struct graph {};

template <typename _Edge, typename _EdgeList, typename _Graph, typename _Vertex,
          typename _VertexCount, class _allEdges, class _cons,
          class _emptyEdgeList, class _head, class _isEmpty, class _src,
          class _tail, class _tgt>
struct incidence_graph {
    typedef _Edge Edge;
    typedef _EdgeList EdgeList;
    typedef _Graph Graph;
    typedef _Vertex Vertex;
    typedef _VertexCount VertexCount;

    _allEdges allEdges;
    _cons cons;
    _emptyEdgeList emptyEdgeList;
    _head head;
    _isEmpty isEmpty;
    _src src;
    _tail tail;
    _tgt tgt;

    inline EdgeList outEdges(const Vertex &v, const Graph &g) {
        EdgeList edges = allEdges(g);
        EdgeList result = emptyEdgeList();
        while (!isEmpty(edges)) {
            Edge &current_edge = head(edges);
            if (src(current_edge) == v) {
                result = cons(current_edge, result);
            }
            edges = tail(edges);
        }
        return result;
    }

    inline VertexCount outDegree(const Vertex &v, const Graph &g) {
        // TODO
        return 0;
    }
};

// TODO: vertex_list_graph should really be fully parameterized? It's like
// a Haskell typeclass…
template <typename _Edge, typename _EdgeList, typename _Graph,
          typename _Vertex, typename _VertexCount, typename _VertexList,
          class _allEdges, class _cons, class _emptyEdgeList,
          class _emptyVertexList, class _head, class _isEmpty,
          class _numVertices, class _src, class _tgt, class _vertices>
struct vertex_list_graph {
    typedef _Edge Edge;
    typedef _EdgeList EdgeList;
    typedef _Graph Graph;
    typedef _Vertex Vertex;
    typedef _VertexCount VertexCount;
    typedef _VertexList VertexList;

    _allEdges allEdges;
    _cons cons;
    _emptyEdgeList emptyEdgeList;
    _emptyVertexList emptyVertexList;
    _head head;
    _isEmpty isempty;
    _numVertices numVertices;
    _src src;
    _tgt tgt;
    _vertices vertices;
};

// list_cpp
template <typename _A>
struct list {
    typedef _A A;
    typedef std::list<A> List;

    inline List empty() { return List(); }
    inline List cons(const A &a, const List &l) {
        auto _l = l;
        _l.push_front(a);
        return _l;
    }
    inline A head(const List &l) {
        return l.front();
    }
    inline List tail(const List &l) {
        auto _l = l;
        _l.pop_front();
        return _l;
    }
    inline bool isEmpty(const List &l) {
        return l.empty();
    }
};

// property_map_cpp
template <typename _Key, typename _KeyList, typename _Value, class _cons,
          class _emptyKeyList, class _head, class _isEmpty, class _tail>
struct property_map {
    typedef _Key Key;
    typedef _KeyList KeyList;
    typedef _Value Value;
    typedef std::map<Key, Value> PropertyMap;

    _cons cons;
    _emptyKeyList emptyKeyList;
    _head head;
    _isEmpty isEmpty;
    _tail tail;

    inline Value get(const PropertyMap &pm, const Key &k) {
        return pm[k];
    }

    inline PropertyMap put(const PropertyMap &pm, const Key &k, const Value &v) {
        auto _pm = pm;
        _pm.insert({k, v});
        return _pm;
    }

    inline PropertyMap initMap(const KeyList &kl, const Value &v) {
        auto result = PropertyMap();
        KeyList _kl = kl;

        while (!isEmpty(_kl)) {
            Key &current_key = head(_kl);
            result = put(result, current_key, v);
            _kl = tail(_kl);
        }
        return result;
    }
};


// queue_cpp
template <typename _A>
struct queue {
    typedef _A A;
    typedef std::queue<A> Queue;

    inline bool isEmpty(const Queue &q) { return q.empty(); }

    inline Queue push(const Queue &q, const A &a) {
        auto _q = q;
        _q.push(a);
        return _q;
    }

    inline Queue pop(const Queue &q) {
        auto _q = q;
        _q.pop();
        return _q;
    }

    inline A front(const Queue &q) { return q.front(); }
};

// tuple_cpp
template <typename _A, typename _B>
struct pair {
    typedef _A A;
    typedef _B B;
    typedef std::pair<A, B> Pair;

    Pair makePair(const A &a, const B &b) {
        return std::make_pair(a, b);
    }

    inline A first(const Pair &pair) { return pair.first; }
    inline B second(const Pair &pair) { return pair.second; }
};

template <typename _A, typename _B, typename _C>
struct triplet {
    typedef _A A;
    typedef _B B;
    typedef _C C;
    typedef std::tuple<A, B, C> Triplet;
    
    Triplet makeTriplet(const A &a, const B &b, const C &c) {
        return std::make_tuple(a, b, c);
    }
    
    inline A first(const Triplet &triplet) { return std::get<0>(triplet); }
    inline B second(const Triplet &triplet) { return std::get<1>(triplet); }
    inline C third(const Triplet &triplet) { return std::get<2>(triplet); }
};

// while_loop cpp
template <typename _Context, typename _State, class _cond, class _step>
struct while_loop {
    typedef _State State;
    typedef _Context Context;

    _cond cond;
    _step step;

    inline void repeat(State &state, const Context &context) {
        while (while_loop::cond(state, context)) {
            while_loop::step(state, context);
        }
    }
};
