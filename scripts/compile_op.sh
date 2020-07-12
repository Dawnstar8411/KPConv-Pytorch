#!/bin/bash

cd ../utils/cpp_wrappers/nearest_neighbors
python3 setup.py install --home="."

cd ../cpp_subsampling
python3 setup.py build_ext --inplace

cd ../cpp_neighbors
python3 setup.py build_ext --inplace


