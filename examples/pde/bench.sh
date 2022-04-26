#!/bin/bash

RUNS=10
AVG_EXEC_TIME=0
for i in `seq 1 $RUNS`; do
    result=`bin/pde.bin | grep -i elapsed`
    runtime=`echo $result | cut -d'[' -f1`
    echo $result
    rest=`echo $result | cut -d '[' -f2`
    AVG_EXEC_TIME=`echo "$AVG_EXEC_TIME + $runtime/$RUNS" | bc -l`
done;

echo "Average running time for $RUNS runs: $AVG_EXEC_TIME[$rest"
