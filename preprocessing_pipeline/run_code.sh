#!/bin/bash

for i in {0..17}
do
    printf $i; printf " : "; python3 decoding.py --n-test-case $i
done