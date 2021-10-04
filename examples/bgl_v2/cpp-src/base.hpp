#pragma once

#include <list>
#include <map>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <utility>

// base_types_cpp
struct base_types {
    typedef unsigned int Int;
    typedef int Vertex;
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
    inline Edge makeEdge(const Vertex &s, const Vertex &t) {
        return Edge(s, t);
    }
};

template <typename _Edge, typename _EdgeList, typename _Vertex,
          typename _VertexList, class _consEdgeList, class _consVertexList,
          class _emptyEdgeList, class _emptyVertexList, class _headEdgeList,
          class _headVertexList, class _isEmptyEdgeList,
          class _isEmptyVertexList, class _makeEdge, class _src,
          class _tailEdgeList, class _tailVertexList, class _tgt>
struct incidence_and_vertex_list_graph {
    typedef _Edge Edge;
    typedef _EdgeList EdgeList;
    typedef _Vertex Vertex;
    typedef _VertexList VertexList;

    typedef int VertexCount;
    struct Graph {
        std::unordered_set<Vertex> vertices;
        std::list<Edge> edges;

        Graph(const std::list<Edge> &edges) {
            for (auto edge_it = edges.begin(); edge_it != edges.end(); ++edge_it) {
                this->edges.push_back(*edge_it);
                this->vertices.insert(src(*edge_it));
                this->vertices.insert(tgt(*edge_it));
            }
        }

        bool operator==(const Graph &other) {
            return this->edges == other.edges;
        }

        // TODO
    };

    static inline _consEdgeList consEdgeList;
    static inline _consVertexList consVertexList;
    static inline _emptyEdgeList emptyEdgeList;
    static inline _emptyVertexList emptyVertexList;
    static inline _headEdgeList headEdgeList;
    static inline _headVertexList headVertexList;
    static inline _isEmptyEdgeList isEmptyEdgeList;
    static inline _isEmptyVertexList isEmptyVertexList;
    static inline _makeEdge makeEdge;
    static inline _src src;
    static inline _tailEdgeList tailEdgeList;
    static inline _tailVertexList tailVertexList;
    static inline _tgt tgt;

    inline EdgeList outEdges(const Vertex &v, const Graph &g) {
        EdgeList result = emptyEdgeList();
        for (auto edge_it = g.edges.begin(); edge_it != g.edges.end(); ++edge_it) {
            if (src(*edge_it) == v) {
                result = consEdgeList(*edge_it, result);
            }
        }
        return result;
    }

    inline VertexCount outDegree(const Vertex &v, const Graph &g) {
        auto outEdgesList = outEdges(v, g);
        VertexCount result = 0;
        while (!isEmptyEdgeList(outEdgesList)) {
            outEdgesList = tailEdgeList(outEdgesList);
            result += 1;
        }
        return result;
    }

    inline VertexList vertices(const Graph &g) {
        VertexList result = emptyVertexList();
        for (auto vertex_it = g.vertices.begin(); vertex_it != g.vertices.end(); ++vertex_it) {
            result = consVertexList(*vertex_it, result);
        }
        return result;
    }

    inline VertexCount numVertices(const Graph &g) {
        return g.vertices.size();
    }
};

/*template <typename _Edge, typename _EdgeList, typename _Graph, typename _Vertex,
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
*/

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
struct read_write_property_map {
    typedef _Key Key;
    typedef _KeyList KeyList;
    typedef _Value Value;
    typedef std::map<Key, Value> PropertyMap;

    static inline _cons cons;
    static inline _emptyKeyList emptyKeyList;
    static inline _head head;
    static inline _isEmpty isEmpty;
    static inline _tail tail;

    inline Value get(const PropertyMap &pm, const Key &k) {
        return pm.find(k)->second;
    }

    inline PropertyMap put(const PropertyMap &pm, const Key &k, const Value &v) {
        auto _pm = pm;
        _pm[k] = v;
        return _pm;
    }

    inline PropertyMap initMap(const KeyList &kl, const Value &v) {
        auto result = PropertyMap();
        KeyList _kl = kl;

        while (!isEmpty(_kl)) {
            const Key &current_key = head(_kl);
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

    inline Queue empty() {
        return Queue();
    }

    inline Queue push(const A &a, const Queue &q) {
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
