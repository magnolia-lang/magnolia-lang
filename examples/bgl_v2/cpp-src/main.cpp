#include <iostream>

#include "bfs_test.hpp"
#include "bgl_test.hpp"
#include "dijkstra_test.hpp"

int main() {
    //bfs::bfsTest();
    //dfs::dfsTest();
    //dijkstra::dijkstraTest();
    bfs::bfsPerfTest();
    bgl::testBgl();
    return 0;
}
