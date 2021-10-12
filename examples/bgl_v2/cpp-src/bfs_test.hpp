#pragma once

#include <chrono>
#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl_v2/mg-src/bgl_v2-cpp.hpp"
#include "testcase.hpp"

namespace bfs {
// BFS test
using examples::bgl_v2::mg_src::bgl_v2_cpp::CppBFSTestVisitor;
typedef CppBFSTestVisitor::Graph Graph;
typedef CppBFSTestVisitor::Vertex Vertex;

auto makeEdge = CppBFSTestVisitor::makeEdge;
auto emptyVertexList = CppBFSTestVisitor::emptyVertexList;
auto emptyVertexVector = CppBFSTestVisitor::emptyVertexVector;
auto head = CppBFSTestVisitor::head;
auto tail = CppBFSTestVisitor::tail;

inline std::list<CppBFSTestVisitor::Edge> testEdges() {
    std::list<CppBFSTestVisitor::Edge> edges;
    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(0, 4));
    edges.push_back(makeEdge(4, 5));
    edges.push_back(makeEdge(3, 6));

    return edges;
}

inline std::list<CppBFSTestVisitor::Edge> lotsOfEdges() {
    auto test_case = gen_test_case();
    std::list<CppBFSTestVisitor::Edge> edges;

    for (auto pair_it = test_case.second.begin();
         pair_it != test_case.second.end();
         ++pair_it) {
        edges.push_back(makeEdge(pair_it->first, pair_it->second));
    }

    return edges;
}

inline void bfsTest() {
    std::cout << "BFS test:" << std::endl;
    Graph g(testEdges(), 7);
    Vertex start = 0;

    auto bfsResult =
        CppBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());

    for (auto vit = bfsResult.begin(); vit != bfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;
}

inline void bfsPerfTest() {
    std::cout << "BFS perf test:" << std::endl;
    Graph g(lotsOfEdges(), NB_TEST_VERTICES);
    std::cout << head(g.vertices) << std::endl;
    Vertex start = 0;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto bfsResult =
        CppBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    /*for (auto vit = bfsResult.begin(); vit != bfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;*/

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}


}
