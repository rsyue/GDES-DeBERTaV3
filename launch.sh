#!/usr/bin/bash

source ~/sft/.venv/bin/activate

nohup python deberta.py > /tmp/stdout.tmp 2> /tmp/stderr.tmp &
PID=$!
mv /tmp/stdout.tmp "${PID}.out"
mv /tmp/stderr.tmp "${PID}.err"
echo "Process with PID ${PID} started. Output in ${PID}.out and ${PID}.err"
