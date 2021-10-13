#pragma once

#include <chrono>
#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl_v2/mg-src/bgl_v2-cpp.hpp"
#include "testcase.hpp"

/*
namespace bfsParallel {
// BFS test
using examples::bgl_v2::mg_src::bgl_v2_cpp::CppBFSParallelTestVisitor;
typedef CppBFSParallelTestVisitor::Graph Graph;
typedef CppBFSParallelTestVisitor::Vertex Vertex;

auto makeEdge = CppBFSParallelTestVisitor::makeEdge;
auto emptyVertexList = CppBFSParallelTestVisitor::emptyVertexList;
auto emptyVertexVector = CppBFSParallelTestVisitor::emptyVertexVector;
auto head = CppBFSParallelTestVisitor::head;
auto tail = CppBFSParallelTestVisitor::tail;

inline std::list<CppBFSParallelTestVisitor::Edge> testEdges() {
    std::list<CppBFSParallelTestVisitor::Edge> edges;
    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(0, 4));
    edges.push_back(makeEdge(4, 5));
    edges.push_back(makeEdge(3, 6));

    return edges;
}

inline std::list<CppBFSParallelTestVisitor::Edge> lotsOfEdges() {
    auto test_case = gen_test_case();
    std::list<CppBFSParallelTestVisitor::Edge> edges;

    for (auto pair_it = test_case.second.begin();
         pair_it != test_case.second.end();
         ++pair_it) {
        edges.push_back(makeEdge(pair_it->first, pair_it->second));
    }

    return edges;
}

inline void bfsParallelTest() {
    std::cout << "BFS parallel test:" << std::endl;
    Graph g(testEdges(), 7);
    Vertex start = 0;

    auto bfsResult =
        CppBFSParallelTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());

    std::cout << std::endl;
}

inline void bfsParallelPerfTest() {
    std::cout << "BFS parallel perf test:" << std::endl;
    Graph g(lotsOfEdges(), NB_TEST_VERTICES);
    std::cout << head(g.vertices) << std::endl;
    Vertex start = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto bfsResult =
        CppBFSParallelTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}
}
*/
