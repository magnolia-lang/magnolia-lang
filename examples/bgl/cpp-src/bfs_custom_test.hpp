#pragma once

#include <chrono>
#include <iostream>

#include "base.hpp"
#include "gen/examples/bgl/mg-src/bgl-cpp.hpp"
#include "testcase.hpp"

namespace bfs_custom {
// BFS test
using examples::bgl::mg_src::bgl_cpp::CppCustomGraphTypeBFSTestVisitor;
typedef CppCustomGraphTypeBFSTestVisitor::Graph Graph;
typedef CppCustomGraphTypeBFSTestVisitor::Vertex Vertex;

auto emptyVertexVector = CppCustomGraphTypeBFSTestVisitor::emptyVertexVector;
auto makeEdge = CppCustomGraphTypeBFSTestVisitor::makeEdge;

inline std::list<CppCustomGraphTypeBFSTestVisitor::Edge> testEdges() {
    std::list<CppCustomGraphTypeBFSTestVisitor::Edge> edges;
    edges.push_back(makeEdge(0, 1));
    edges.push_back(makeEdge(1, 2));
    edges.push_back(makeEdge(1, 3));
    edges.push_back(makeEdge(3, 4));
    edges.push_back(makeEdge(0, 4));
    edges.push_back(makeEdge(4, 5));
    edges.push_back(makeEdge(3, 6));

    return edges;
}

inline std::list<CppCustomGraphTypeBFSTestVisitor::Edge> lotsOfEdges() {
    auto test_case = gen_test_case();
    std::list<CppCustomGraphTypeBFSTestVisitor::Edge> edges;

    for (auto pair_it = test_case.second.begin();
         pair_it != test_case.second.end();
         ++pair_it) {
        edges.push_back(makeEdge(pair_it->first, pair_it->second));
    }

    return edges;
}

inline void bfsTest() {
    std::cout << "BFS with custom graph type test:" << std::endl;

    auto edges = testEdges();

    Graph g(edges, 7);
    Vertex start = Vertex(0);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto bfsResult =
        CppCustomGraphTypeBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    for (auto vit = bfsResult.begin(); vit != bfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}

inline void bfsPerfTest() {
    std::cout << "BFS with custom graph type perf test:" << std::endl;
    auto edges = lotsOfEdges();
    Graph g(edges, NB_TEST_VERTICES);
    Vertex start = Vertex(0);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto bfsResult =
        CppCustomGraphTypeBFSTestVisitor::breadthFirstSearch(g, start, emptyVertexVector());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    for (auto vit = bfsResult.begin(); vit != bfsResult.end(); ++vit) {
        std::cout << *vit << " ";
    }

    std::cout << std::endl;

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}


}
