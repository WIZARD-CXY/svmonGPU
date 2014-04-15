
################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= multisvm
# Cuda source files (compiled with cudacc)
CUFILES		:= multisvm.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 

################################################################################
# Rules and targets
#emu=1
USEDRVAPI     := 1
USECUBLAS       := 1
include common.mk
