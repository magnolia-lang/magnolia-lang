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

int main() {
    bfs::bfsTest();
    dfs::dfsTest();
    bfs_custom::bfsTest();
    dijkstra::dijkstraTest();
    bellmanFord::bellmanFordTest();
    prim::primTest();
    johnson::johnsonTest();
    bfs::bfsPerfTest();
    bfs_custom::bfsPerfTest();
    //bfsParallel::bfsParallelPerfTest();
    bgl::testBgl();
    return 0;
}
