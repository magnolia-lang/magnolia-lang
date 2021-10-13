#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/two_bit_color_map.hpp>

#include <condition_variable>
#include <chrono>
#include <iostream>

#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>
#include <stdlib.h>

#define DEBUG 1
#ifdef DEBUG

#define BENCHD(name, op) std::chrono::steady_clock::time_point time_begin = std::chrono::steady_clock::now(); op; std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now(); std::cout << "Time difference (" << name << ") = " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() << "[ns]" << std::endl;
#define BENCH(name, op) time_begin = std::chrono::steady_clock::now(); op; time_end = std::chrono::steady_clock::now(); std::cout << "Time difference (" << name << ") = " << std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count() << "[ns]" << std::endl;

#else

#define BENCHD(name, op) op;
#define BENCH(name, op) op;

#endif

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
/*
template <typename _Vertex, typename _VertexDescriptor>
struct edge {
    typedef _Vertex Vertex;
    typedef _VertexDescriptor VertexDescriptor;

    typedef boost::adjacency_list< boost::listS, boost::listS, boost::directedS >::edge_descriptor EdgeDescriptor;

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

    inline const VertexDescriptor &src(const EdgeDescriptor &e) { return boost::source(e); }
    inline const VertexDescriptor &tgt(const EdgeDescriptor &e) { return boost::target(e); }
    inline const Vertex &src(const Edge &e) { return e.source; }
    inline const Vertex &tgt(const Edge &e) { return e.target; }
    inline Edge makeEdge(const Vertex &s, const Vertex &t) {
        return Edge(s, t);
    }
};*/

template <typename _Vertex>
    struct incidence_and_vertex_list_graph {
        typedef _Vertex Vertex;
        typedef std::pair<Vertex, Vertex> Edge;

        typedef boost::adjacency_list< boost::vecS, boost::vecS, boost::directedS > Graph;
        typedef Graph::vertex_descriptor VertexDescriptor;
        typedef Graph::edge_descriptor EdgeDescriptor;

        typedef Graph::out_edge_iterator EdgeIterator;
        typedef Graph::vertex_iterator VertexIterator;

        typedef std::list<EdgeDescriptor> EdgeList;
        typedef std::list<VertexDescriptor> VertexList;

        typedef int VertexCount;

        /*
        static inline _consEdgeList consEdgeList;
        static inline _consVertexList consVertexList;
        static inline _edgeIterBegin edgeIterBegin;
        static inline _edgeIterEnd edgeIterEnd;
        static inline _edgeIterNext edgeIterNext;
        static inline _edgeIterUnpack edgeIterUnpack;
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
        static inline _vertexIterBegin vertexIterBegin;
        static inline _vertexIterEnd vertexIterEnd;
        static inline _vertexIterNext vertexIterNext;
        static inline _vertexIterUnpack vertexIterUnpack;
        */

        inline VertexDescriptor toVertexDescriptor(const Vertex &v, const Graph &g) {
            return boost::vertex(v, g);
        }

        inline VertexDescriptor src(const EdgeDescriptor &ed, const Graph &g) {
            return boost::source(ed, g);
        }

        inline VertexDescriptor tgt(const EdgeDescriptor &ed, const Graph &g) {
            return boost::target(ed, g);
        }

        inline Edge makeEdge(const Vertex &v1, const Vertex& v2) {
            return std::make_pair(v1, v2);
        }

        inline void edgeIterNext(EdgeIterator &it) { ++it; }
        inline EdgeDescriptor edgeIterUnpack(const EdgeIterator &it) { return *it; }

        inline void vertexIterNext(VertexIterator &it) { ++it; }
        inline VertexDescriptor vertexIterUnpack(const VertexIterator &it) { return *it; }

        inline void outEdges(const VertexDescriptor &v, const Graph &g, EdgeIterator &it_begin, EdgeIterator &it_end) {
            boost::tie(it_begin, it_end) = boost::out_edges(v, g);
        }

        inline VertexCount outDegree(const VertexDescriptor &v, const Graph &g) {
            /*auto outEdgesList = outEdges(v, g);
            VertexCount result = 0;
            while (!isEmptyEdgeList(outEdgesList)) {
                tailEdgeList(outEdgesList);
                result += 1;
            }
            return result;*/
            return 0;
        }

        inline void vertices(const Graph &g, VertexIterator &it_begin, VertexIterator &it_end) {
            boost::tie(it_begin, it_end) = boost::vertices(g);
        }

        inline VertexCount numVertices(const Graph &g) {
            auto vs = boost::vertices(g);
            return std::distance(vs.first, vs.second);
            //.size();
        }
    };

    // list_cpp
    template <typename _A>
    struct list {
        typedef _A A;
        typedef std::list<A> List;

        inline List empty() { return List(); }
        inline void cons(const A &a, List &l) { l.push_front(a); }
        
        inline const A& head(const List &l) {
            return l.front();
        }
        inline void tail(List &l) { l.pop_front(); }
        
        inline bool isEmpty(const List &l) {
            return l.empty();
        }
    };

    // vector_cpp
    template <typename _A>
    struct vector {
        typedef _A A;
        typedef std::vector<A> Vector;

        inline Vector empty() { return Vector(); }
        inline void pushBack(const A &a, Vector &v) { v.push_back(a); }
    };


    template <typename _A>
    struct thread_safe_vector {
        typedef _A A;
        struct Vector {
            std::vector<A> m_vec;
            private:
            mutable std::mutex m_mutex;

            public:
            Vector() : m_vec(), m_mutex() {};
            Vector(const Vector &source) : m_vec(source.m_vec), m_mutex() {};

            void push_back(const A &a) {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_vec.push_back(a);
            }

            // TODO: add iterators to vector
        };

        inline Vector empty() { return Vector(); }
        inline void pushBack(const A &a, Vector &v) { v.push_back(a); }
    };

    // list_cpp
    template <typename _A>
    struct iterable_list {
        typedef _A A;
        typedef std::list<A> List;
        typedef typename std::list<A>::const_iterator ListIterator;

        inline List empty() { return List(); }
    inline void cons(const A &a, List &l) { l.push_front(a); }

    inline const A& head(const List &l) {
        return l.front();
    }
    inline void tail(List &l) { l.pop_front(); }

    inline bool isEmpty(const List &l) {
        return l.empty();
    }

    inline ListIterator iterBegin(const List &l) { return l.begin(); }
    constexpr inline void iterNext(ListIterator &it) { ++it; }
    inline ListIterator iterEnd(const List &l) { return l.end(); }
    inline const A &iterUnpack(const ListIterator &it) { return *it; }
};

namespace utils {
template <typename Iterator, class iterNext>
size_t distance(Iterator first, Iterator last) {
    size_t difference = 0;
    while (first != last) {
        ++difference;
        iterNext(first);
    }
    return difference;
}
}

template <typename _A>
struct thread_safe_iterable_list {
    typedef _A A;
    typedef typename std::list<A>::const_iterator ListIterator;

    struct List {
        private:
            std::list<A> m_list;
            mutable std::mutex m_mutex;
            std::condition_variable cond;

        public:
            List() : m_list(), m_mutex() {};

            inline bool empty() const { return m_list.empty(); }

            inline const A& front() const {
                std::unique_lock<std::mutex> lock(m_mutex);
                cond.wait(lock, [&]{ return !m_list.empty(); });

                return this->m_list.front(); // not safe
            }

            inline void push_front(const A &a) {
                std::unique_lock<std::mutex> lock(m_mutex);
                this->m_list.push_front(a);
            }

            inline void pop_front() {
                std::unique_lock<std::mutex> lock(m_mutex);
                cond.wait(lock, [&]{ return !m_list.empty(); });

                this->m_list.pop_front();
            }

            inline ListIterator begin() const {
                return this->m_list.begin();
            }

            inline ListIterator end() const {
                return this->m_list.end();
            }
    };

    inline List empty() { return List(); }
    inline void cons(const A &a, List &l) { l.push_front(a); }

    inline const A& head(const List &l) {
        return l.front();
    }
    inline void tail(List &l) { l.pop_front(); }

    inline bool isEmpty(const List &l) {
        return l.empty();
    }

    inline ListIterator iterBegin(const List &l) { return l.begin(); }
    inline void iterNext(ListIterator &it) { ++it; }
    inline ListIterator iterEnd(const List &l) { return l.end(); }
    inline const A &iterUnpack(const ListIterator &it) { return *it; }
};

// property_map_cpp
template <typename _Key, typename _KeyList, typename _KeyListIterator,
          typename _Value, class _cons, class _emptyKeyList, class _head,
          class _isEmpty, class _iterBegin, class _iterEnd, class _iterNext,
          class _iterUnpack, class _tail>
struct read_write_property_map {
    typedef _Key Key;
    typedef _KeyList KeyList;
    typedef _KeyListIterator KeyListIterator;
    typedef _Value Value;
    typedef std::map<Key, Value> PropertyMap;

    static inline _cons cons;
    static inline _emptyKeyList emptyKeyList;
    static inline _head head;
    static inline _isEmpty isEmpty;
    static inline _iterBegin iterBegin;
    static inline _iterEnd iterEnd;
    static inline _iterNext iterNext;
    static inline _iterUnpack iterUnpack;
    static inline _tail tail;

    inline PropertyMap emptyMap() { return PropertyMap(); }

    inline const Value &get(const PropertyMap &pm, const Key &k) {
        return pm.find(k)->second;
    }

    inline void put(PropertyMap &pm, const Key &k, const Value &v) {
        pm[k] = v;
    }

    inline PropertyMap initMap(const KeyList &kl, const Value &v) {
        auto result = PropertyMap();
        auto begin = iterBegin(kl), end = iterEnd(kl);
        BENCHD("initMap",
        for (auto k_it = begin; k_it != end; ++k_it) {
            put(result, iterUnpack(k_it), v);
        })

        return result;
    }

    bool operator==(const read_write_property_map &other) const = default;
};



template <typename _Key, typename _KeyListIterator,
          class _iterNext, class _iterUnpack>
struct two_bit_color_map {
    typedef _Key Key;
    typedef _KeyListIterator KeyListIterator;

    typedef typename boost::two_bit_color_type Color;

    inline Color white() { return boost::two_bit_white; }
    inline Color gray() { return boost::two_bit_gray; }
    inline Color black() { return boost::two_bit_black; }

    // TODO: pass graph
    private:
    typedef typename boost::vec_adj_list_vertex_id_map<boost::no_property, unsigned long> IndexMap;
    typedef typename boost::property_traits<IndexMap> Traits;
    
    public:
    typedef typename boost::two_bit_color_map<IndexMap>
            ColorPropertyMap;

    _iterNext iterNext;
    _iterUnpack iterUnpack;

    // TODO: fix
    inline void put(const ColorPropertyMap &cm, const typename Traits::key_type &key, const Color v) {
        typename Traits::value_type k = boost::get(cm.index, key);
        size_t byte_num = k / 4;
        size_t bit_position = ((k % 4) * 2);
        cm.data.get()[byte_num] = (unsigned char)((cm.data.get()[byte_num] & ~(3 << bit_position))
                | (v << bit_position));
    }

    inline const Color get(const ColorPropertyMap &cm, const Key &key) {
        typename Traits::value_type k = boost::get(cm.index, key);
        size_t byte_num = k / 4;
        size_t bit_position = ((k % 4) * 2);
        return Color((cm.data[byte_num] >> bit_position) & 3);
    }

    inline ColorPropertyMap initMap(KeyListIterator kl_it,
                                    KeyListIterator kl_end,
                                    const Color &v) {
        size_t distance = ([this](KeyListIterator first, KeyListIterator last) {
                size_t difference = 0;
                while (first != last) {
                    ++difference;
                    iterNext(first);
                }
                return difference;
            })(kl_it, kl_end);
        auto result =
            boost::make_two_bit_color_map<IndexMap>(
                distance,
                IndexMap());
        for (; kl_it != kl_end; iterNext(kl_it)) {
            put(result, iterUnpack(kl_it), v);
        }

        return result;
    }
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

    inline void push(const A &a, FIFOQueue &q) { q.push(a); }

    inline void pop(FIFOQueue &q) { q.pop(); }

    inline const A& front(const FIFOQueue &q) { return q.front(); }
};


template <typename _A>
struct thread_safe_fifo_queue {
    typedef _A A;
     struct FIFOQueue {
        std::queue<A> m_queue;
        private:
        mutable std::mutex m_mutex;
        std::condition_variable cond;

        public:
        FIFOQueue() : m_queue(), m_mutex(), cond() {}
        FIFOQueue(const FIFOQueue &source) : m_mutex(), cond() {
            this->m_queue = source.m_queue;
        }

        bool operator==(const FIFOQueue &other) const {
            return this->m_queue == other.m_queue;
        }

        bool empty() const {
            return this->m_queue.empty();
        }

        void push(const A &a) {
            std::unique_lock<std::mutex> lock(m_mutex);
            this->m_queue.push(a);
        }

        void pop() {
            std::unique_lock<std::mutex> lock(m_mutex);
            cond.wait(lock, [&]{ return !m_queue.empty(); });
            this->m_queue.pop();
        }

        const A &front() const {
            std::unique_lock<std::mutex> lock(m_mutex);
            //cond.wait(lock, [&]{ return !m_queue.empty(); });
            return this->m_queue.front();
        }

    };

    inline bool isEmpty(const FIFOQueue &q) { return q.empty(); }

    inline FIFOQueue empty() {
        return FIFOQueue();
    }

    inline void push(const A &a, FIFOQueue &q) { q.push(a); }

    inline void pop(FIFOQueue &q) { q.pop(); }

    inline const A& front(const FIFOQueue &q) { return q.front(); }
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

    inline void push(const A &a, PriorityQueue &pq) { pq.push(a); }

    inline void pop(PriorityQueue &pq) { pq.pop(); }

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

    inline const A &first(const Pair &pair) { return pair.first; }
    inline const B &second(const Pair &pair) { return pair.second; }
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


template <typename _Context1, typename _Context2, typename _Iterator,
          typename _State1, typename _State2, typename _State3,
          class _iterNext, class _step>
struct for_iterator_loop3_2 {
    typedef _Context1 Context1;
    typedef _Context2 Context2;
    typedef _Iterator Iterator;
    typedef _State1 State1;
    typedef _State2 State2;
    typedef _State3 State3;

    _iterNext iterNext;
    _step step;

    inline __attribute__((always_inline)) void forLoopRepeat(Iterator &itr,
                                                             const Iterator &end_itr,
                                                             State1 &s1,
                                                             State2 &s2,
                                                             State3 &s3,
                                                             const Context1 &c1,
                                                             const Context2 &c2) {
        for (; itr != end_itr; iterNext(itr)) {
            step(itr, end_itr, s1, s2, s3, c1, c2);
        }
    }
};

template <typename _Context1, typename _Context2, typename _Iterator,
          typename _State1, typename _State2, typename _State3,
          class _iterNext, class _step>
struct for_parallel_iterator_loop3_2 {
    typedef _Context1 Context1;
    typedef _Context2 Context2;
    typedef _Iterator Iterator;
    typedef _State1 State1;
    typedef _State2 State2;
    typedef _State3 State3;

    _iterNext iterNext;
    _step step;

    inline __attribute__((always_inline)) void forLoopRepeat(Iterator &itr,
                                                             const Iterator &end_itr,
                                                             State1 &s1,
                                                             State2 &s2,
                                                             State3 &s3,
                                                             const Context1 &c1,
                                                             const Context2 &c2) {
        #pragma omp parallel for
        for (; itr != end_itr; iterNext(itr)) {
            step(itr, end_itr, s1, s2, s3, c1, c2);
        }
    }
};

// while_loop cpp
template <typename _Context, typename _State, class _cond, class _step>
struct while_loop {
    typedef _State State;
    typedef _Context Context;

    _cond cond;
    _step step;

    inline __attribute__((always_inline)) void repeat(State &state, const Context &context) {
        while (while_loop::cond(state, context)) {
            while_loop::step(state, context);
        }
    }
};


template <typename _Context, typename _State1, typename _State2, 
          typename _State3, class _cond, class _step>
struct while_loop3 {
    typedef _State1 State1;
    typedef _State2 State2;
    typedef _State3 State3;
    typedef _Context Context;

    _cond cond;
    _step step;

    inline __attribute__((always_inline)) void repeat(State1 &state1, State2 &state2, State3 &state3,
                       const Context &context) {
        //BENCHD("outer loop repeat",
        while (while_loop3::cond(state1, state2, state3, context)) {
            //BENCHD("outer loop step",
            while_loop3::step(state1, state2, state3, context);
            //)
        }//)
    }
};


template <typename _Context1, typename _Context2, typename _Context3, typename _State1,
          typename _State2, typename _State3, typename _State4,
          class _cond, class _step>
struct while_loop4_3 {
    typedef _State1 State1;
    typedef _State2 State2;
    typedef _State3 State3;
    typedef _State4 State4;
    typedef _Context1 Context1;
    typedef _Context2 Context2;
    typedef _Context3 Context3;

    _cond cond;
    _step step;

    inline __attribute__((always_inline)) void repeat(State1 &state1, State2 &state2, State3 &state3,
                       State4 &state4, const Context1 &context1,
                       const Context2 &context2, const Context3 &context3) {
        while (cond(state1, state2, state3, state4, context1, context2, context3)) {
            step(state1, state2, state3, state4, context1, context2, context3);
        }
    }
};
