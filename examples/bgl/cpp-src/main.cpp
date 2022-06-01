#include <iostream>

#include "bellman_ford_test.hpp"
#include "bfs_custom_test.hpp"
#include "bfs_test.hpp"
#include "bfs_parallel_test.hpp"
#include "bgl_test.hpp"
#include "dfs_test.hpp"
#include "dijkstra_test.hpp"
#include "johnson_test.hpp"
#include "prim_test.hpp"

#if BENCH_ALL
#define BFS 1
#define DFS 1
#define DIJKSTRA 1
#define BELLMANFORD 1
#define PRIM 1
#define JOHNSON 1
#endif

void bfsPerfTest() {
    bfs::bfsPerfTest();
    bfs_custom::bfsPerfTest();
    bgl::bfsPerfTest();
}

void dfsPerfTest() {
    //dfs::dfsPerfTest();
    //bgl::dfsPerfTest();
}

void dijkstraPerfTest() {
    dijkstra::dijkstraPerfTest();
    bgl::dijkstraPerfTest();
}

int main() {
#if BENCHMARK
#if BFS
    bfsPerfTest();
#endif
#if DFS
    //dfsPerfTest();
#endif
#if DIJKSTRA
    //dijkstraPerfTest();
#endif
#if BELLMANFORD
#endif
#if PRIM
#endif
#if JOHNSON
#endif
#else
    bfs::bfsTest();
    dfs::dfsTest();
    bfs_custom::bfsTest();
    dijkstra::dijkstraTest();
    bellmanFord::bellmanFordTest();
    prim::primTest();
    johnson::johnsonTest();
#endif
    return 0;
}
