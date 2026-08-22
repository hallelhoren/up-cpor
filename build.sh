#!/bin/bash
set -e

echo "=================================="
echo "1. Building C++ Native Core..."
echo "=================================="
mkdir -p cpp/build
cd cpp/build
cmake ..
make -j"$(nproc)"
# Copy the compiled shared library to the repo root so Python can find it.
cp libcpor_core.so ../../
cd ../..

echo "=================================="
echo "2. Installing Python Package..."
echo "=================================="
pip install -e .
