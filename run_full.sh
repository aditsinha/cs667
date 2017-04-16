#!/bin/bash

./full_yao 5555 -- config.txt xaa.csv > /tmp/p1_log &
./full_yao 5555 localhost config.txt xab.csv > /tmp/p2_log

echo "Party 1 Log"
cat /tmp/p1_log

echo ""
echo "Party 2 Log"
cat /tmp/p2_log
