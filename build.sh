#!/bin/bash

echo "=================================="
echo "1. Building C# CPORLib..."
echo "=================================="
#dotnet build CPORLib/CPORLibSolution.sln

echo "=================================="
echo "2. Building C++ Native Core..."
echo "=================================="
# יצירת תיקיית build זמנית בתוך cpp/
mkdir -p cpp/build
cd cpp/build
# יצירת ההוראות לקימפול והרצת הקומפיילר
cmake ..
make
# העתקת התוצר המקומפל (.so) לתיקיית השורש כדי שהפייתון ימצא אותו בקלות
cp libcpor_core.so ../../
cd ../..

echo "=================================="
echo "3. Installing Python Package..."
echo "=================================="
#pip uninstall -y up-cpor
#pip install -e .