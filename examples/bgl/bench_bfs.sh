#!/bin/bash

RUNS=10
AVG_EXEC_TIME_BFS=0
AVG_EXEC_TIME_BFS_CUSTOM=0
AVG_EXEC_TIME_BGL_BFS=0

NB_TEST_VERTICES=1000000
NB_TEST_EDGES=20000000

make DEFINES="-DBENCHMARK=1 -DBFS=1 -DNB_TEST_VERTICES=$NB_TEST_VERTICES -DNB_TEST_EDGES=$NB_TEST_EDGES"

for i in `seq 1 $RUNS`; do
    result=`bin/bgl.bin | grep -i Time`
    runtime_bfs=`echo $result | cut -d'[' -f1 | cut -d'=' -f2 | cut -d' ' -f2`
    runtime_bfs_custom=`echo $result | cut -d'[' -f2 | cut -d'=' -f2 | cut -d' ' -f2`
    runtime_bgl_bfs=`echo $result | cut -d'[' -f3 | cut -d'=' -f2 | cut -d' ' -f2`
    echo $result
    AVG_EXEC_TIME_BFS=`echo "$AVG_EXEC_TIME_BFS + $runtime_bfs/$RUNS" | bc -l`
    AVG_EXEC_TIME_BFS_CUSTOM=`echo "$AVG_EXEC_TIME_BFS_CUSTOM + $runtime_bfs_custom/$RUNS" | bc -l`
    AVG_EXEC_TIME_BGL_BFS=`echo "$AVG_EXEC_TIME_BGL_BFS + $runtime_bgl_bfs/$RUNS" | bc -l`
done;

echo "Parameters: $NB_TEST_VERTICES vertices, $NB_TEST_EDGES edges"

echo "Average running time for $RUNS runs of the Magnolia BFS: $AVG_EXEC_TIME_BFS[ms]"
echo "Average running time for $RUNS runs of the Magnolia Custom BFS: $AVG_EXEC_TIME_BFS_CUSTOM[ms]"
echo "Average running time for $RUNS runs of the C++ BGL BFS: $AVG_EXEC_TIME_BGL_BFS[ms]"
