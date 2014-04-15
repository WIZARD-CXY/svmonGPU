#!/bin/bash
make clean
make
cp bin/linux/release/multisvm .
bsub -q gpu < multisvm.lsf
