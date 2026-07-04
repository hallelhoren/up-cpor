cmake_minimum_required(VERSION 3.15)
project(CPOR_Native_Core)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to legacy C source code
set(FF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../FF-v2.3")
include_directories(${FF_DIR})

# 1. Compile FF engine into a static library
# We manually list the C files to ensure order and presence
set(FF_SOURCES 
    ${FF_DIR}/ff_api.c ${FF_DIR}/search.c ${FF_DIR}/relax.c 
    ${FF_DIR}/parse.c ${FF_DIR}/inst_easy.c ${FF_DIR}/inst_hard.c 
    ${FF_DIR}/inst_final.c ${FF_DIR}/inst_pre.c ${FF_DIR}/orderings.c 
    ${FF_DIR}/output.c ${FF_DIR}/memory.c
)

add_library(ff_engine STATIC ${FF_SOURCES})
# FF is legacy C code, allow some warnings
target_compile_options(ff_engine PRIVATE -Wno-everything) 

# 2. Compile your C++ core, including a "shims" file for missing legacy symbols
add_library(cpor_core SHARED 
    native_bridge.cpp FFSolver.cpp SDRSampler.cpp 
    SDRPlanner.cpp BeliefState.cpp ff_shims.cpp
)

# 3. Link everything together
target_link_libraries(cpor_core PUBLIC ff_engine z3)
target_compile_options(cpor_core PRIVATE -O3 -march=native -ffast-math)