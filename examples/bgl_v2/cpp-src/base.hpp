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

struct base_float_ops {
    typedef float Float;
    inline Float plus(const Float &i1, const Float &i2) { return i1 + i2; }
    inline bool less(const Float &i1, const Float &i2) const { return i1 < i2; }
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
    struct Edge {
        Vertex source;
        Vertex target;

        Edge(const Vertex &source, const Vertex &target)
            : source(source), target(target) {};

        bool operator<(const Edge &other) const {
            if (this->source == other.source) {
                return this->target < other.target;
            }
            else {
                return this->source < other.source ;
            }
        };
        bool operator==(const Edge &other) const = default;
    };

    inline Vertex src(const Edge &e) { return e.source; }
    inline Vertex tgt(const Edge &e) { return e.target; }
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

        bool operator==(const Graph &other) const = default;

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

    inline PropertyMap emptyMap() { return PropertyMap(); }

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

    bool operator==(const read_write_property_map &other) const = default;
};


// queue_cpp
template <typename _A>
struct fifo_queue {
    typedef _A A;
    typedef std::queue<A> FIFOQueue;

    inline bool isEmpty(const FIFOQueue &q) { return q.empty(); }

    inline FIFOQueue empty() {
        return FIFOQueue();
    }

    inline FIFOQueue push(const A &a, const FIFOQueue &q) {
        auto _q = q;
        _q.push(a);
        return _q;
    }

    inline FIFOQueue pop(const FIFOQueue &q) {
        auto _q = q;
        _q.pop();
        return _q;
    }

    inline A front(const FIFOQueue &q) { return q.front(); }
};

// TODO: make actual implementation
template <typename _A, typename _Priority, typename _PriorityMap, class _get>
struct priority_queue {
    typedef _A A;
    typedef _Priority Priority;
    typedef _PriorityMap PriorityMap;

    // A heap
    struct HeapNode {
        Priority prio;
        A element;
        bool operator==(const HeapNode &other) const = default;
    };

    struct Heap {
        PriorityMap priority_map;
        std::vector<HeapNode> heap_nodes;
        Heap(const PriorityMap &priority_map)
                : priority_map(priority_map) {};
        bool operator==(const Heap &other) const = default;

        bool isEmpty() const {
            return this->heap_nodes.empty();
        }

        inline size_t leftChildIndex(const size_t i) { return 2 * i + 1; }
        inline size_t rightChildIndex(const size_t i) { return 2 * i + 2; }
        inline size_t parentIndex(const size_t i) { return (i - 1) / 2; }
        void filterDown(size_t k) {
            while (leftChildIndex(k) < heap_nodes.size()) {
                auto left_idx = leftChildIndex(k),
                                right_idx = rightChildIndex(k);

                auto left_prio = heap_nodes[left_idx].prio,
                     k_prio = heap_nodes[k].prio;
                if (right_idx < heap_nodes.size()) { // two children
                    auto right_prio = heap_nodes[right_idx].prio;

                    if (k_prio <= left_prio && k_prio <= right_prio) {
                        break;
                    }
                    else {
                        auto to_swap_with_k = right_idx;
                        if (left_prio < right_prio) {
                            to_swap_with_k = left_idx;
                        }
                        std::iter_swap(heap_nodes.begin() + to_swap_with_k,
                                       heap_nodes.begin() + k);
                        k = to_swap_with_k;
                    }
                }
                else { // only one (left) child
                    if (k_prio <= left_prio) {
                        break;
                    }
                    else {
                        std::iter_swap(heap_nodes.begin() + left_idx,
                                       heap_nodes.begin() + k);
                        k = left_idx;
                    }
                }
            }
        }

        void filterUp(size_t k) {
            while (k != 0) {
                auto parent_idx = parentIndex(k);
                auto k_prio = heap_nodes[k].prio,
                     parent_prio = heap_nodes[parent_idx].prio;

                if (k_prio < parent_prio) {
                    std::iter_swap(heap_nodes.begin() + parent_idx,
                                   heap_nodes.begin() + k);
                    k = parent_idx;
                }
                else {
                    break;
                }
            }
        }

        void remove(const A &toRemove) {
            if (this->isEmpty()) return;

            for (auto k = 0; k < this->heap_nodes.size(); ++k) {
                if (heap_nodes[k].element == toRemove) {
                    removeAt(k);
                    break;
                }
            }
        }
        void removeAt(size_t k) {
            heap_nodes[k] = heap_nodes.back();
            heap_nodes.pop_back();

            // TODO: Can't we just sequentially filterDown and up?
            if (k == 0 || heap_nodes[parentIndex(k)].prio < heap_nodes[k].prio) {
                filterDown(k);
            }
            else {
                filterUp(k);
            }
        }

        void push(const A &a) {
            _get get;
            // TODO: handle exceptions
            auto a_prio = get(priority_map, a);

            HeapNode a_node;
            a_node.element = a;
            a_node.prio = a_prio;

            heap_nodes.push_back(a_node);
            filterUp(heap_nodes.size() - 1);
        }

        void pop() {
            removeAt(0);
        }

        A front() const {
            // TODO: assert not empty, or default element?
            return heap_nodes[0].element;
        }

        void update(const PriorityMap &pm, const A &a) {
            remove(a);
            this->priority_map = pm;
            push(a);
        }
    };

    typedef Heap PriorityQueue;

    _get get;

    bool isEmpty(const PriorityQueue &pq) const {
        return pq.isEmpty();
    };

    inline PriorityQueue empty(const PriorityMap &pm) { return PriorityQueue(pm); }
    inline PriorityQueue update(const PriorityMap &pm, const A &a, const PriorityQueue &pq) {
        auto _pq = pq;
        _pq.update(pm, a);
        return _pq;
    }

    PriorityQueue push(const A &a, const PriorityQueue &pq) {
        auto _pq = pq;
        _pq.push(a);
        return _pq;
    }

    PriorityQueue pop(const PriorityQueue &pq) {
        auto _pq = pq;
        _pq.pop();
        return _pq;
    }

    A front(const PriorityQueue &pq) {
        return pq.front();
    }
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