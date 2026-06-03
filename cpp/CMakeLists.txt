cmake_minimum_required(VERSION 3.15)
project(CPOR_Native_Core)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile as a standard C-compatible Shared Library (.so)
add_library(cpor_core SHARED native_bridge.cpp)

# Aggressive optimization flags for high-performance computing
target_compile_options(cpor_core PRIVATE -O3 -march=native -ffast-math)

# Add the C++ Test Executable
add_executable(test_state test_state.cpp)
target_compile_options(test_state PRIVATE -O3 -march=native)

# Add the Evaluator Test Executable
add_executable(test_evaluator test_evaluator.cpp)
target_compile_options(test_evaluator PRIVATE -O3 -march=native)

# Add the CPORSolver Test Executable
add_executable(test_solver test_solver.cpp native_bridge.cpp)
target_compile_options(test_solver PRIVATE -O3 -march=native)

# Add the Integration Test Executable (Links against native_bridge.cpp)
add_executable(test_integration test_integration.cpp native_bridge.cpp)
target_compile_options(test_integration PRIVATE -O3 -march=native)