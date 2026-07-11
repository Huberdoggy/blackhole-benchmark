#!/bin/bash

echo "Nuking prior build artifacts..."

if [ -f benchmark.py ]; then
  rm -f ./benchmark.py
fi
if [ -d __pycache__ ]; then
  rm -rf ./__pycache__
fi

echo "Cleaned cache and old demo logic (benchmark.py)"
