cmake_minimum_required(VERSION 3.15)
project(CPOR_Native_Core C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../FF-v2.3")
include_directories(${FF_DIR})

# Legacy FF Source Files (Untouched)
set(FF_SOURCES 
    ${FF_DIR}/ff_api.c ${FF_DIR}/search.c ${FF_DIR}/relax.c 
    ${FF_DIR}/parse.c ${FF_DIR}/inst_easy.c ${FF_DIR}/inst_hard.c 
    ${FF_DIR}/inst_final.c ${FF_DIR}/inst_pre.c ${FF_DIR}/orderings.c 
    ${FF_DIR}/output.c ${FF_DIR}/memory.c
)

# Create FF Engine Library
add_library(ff_engine STATIC ${FF_SOURCES})
set_property(TARGET ff_engine PROPERTY POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(${FF_SOURCES} PROPERTIES COMPILE_FLAGS "-w -fcommon") 

# Create your C++ Library (Now including ff_globals.c!)
add_library(cpor_core SHARED 
    native_bridge.cpp CPORSolver.cpp FFSolver.cpp SDRSampler.cpp 
    SDRPlanner.cpp BeliefState.cpp ff_globals.c
)

target_link_libraries(cpor_core PUBLIC ff_engine z3)
target_compile_options(cpor_core PRIVATE -O3 -march=native -ffast-math)

# Tests
add_executable(test_state tests/test_state.cpp)
target_link_libraries(test_state PRIVATE cpor_core z3)

add_executable(test_evaluator tests/test_evaluator.cpp)
target_link_libraries(test_evaluator PRIVATE cpor_core z3)

add_executable(test_solver tests/test_solver.cpp)
target_link_libraries(test_solver PRIVATE cpor_core z3)

add_executable(test_integration tests/test_integration.cpp)
target_link_libraries(test_integration PRIVATE cpor_core z3)

add_executable(test_sdr_full_lifecycle tests/test_sdr_full_lifecycle.cpp)
target_link_libraries(test_sdr_full_lifecycle PRIVATE cpor_core z3)