#!/usr/bin/bash

# source ~/sft/.venv/bin/activate

nohup python deberta.py -bs=8 -ld=0.3 -wd=0.1 -lr=2e-5 -g=0.9 -ep=2 --fp16 > /tmp/stdout.tmp 2> /tmp/stderr.tmp &
PID=$!
mv /tmp/stdout.tmp "${PID}.out"
mv /tmp/stderr.tmp "${PID}.err"
echo "Process with PID ${PID} started. Output in ${PID}.out and ${PID}.err"
