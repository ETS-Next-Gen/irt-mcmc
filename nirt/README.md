CNTree: Clusters and Neighbors Tree mini-clustering algorithm.

This is part of the Multiscale Optimization (MO) project. Learn more about it and CNTree at
https://share.amazon.com/sites/MultiscaleOpt

# Running the Demo

Generate and plot the clustering:
%run hms/scripts/run_cntree_demo.py --mode=uniform --size=100 --dim=2 --plot

Profile code runtime:
%run hms/scripts/run_cntree_profile.py --size=100 --dim=2


# Running Tests

To run tests, in the current directory:
1. export PYTHONPATH=. if . is not yet in your Python path.
2. Run all unit tests: pytest . 
3. Run all unit tests: pytest . -o python_files='benchmark_*.py' -o python_classes=Benchmark -o python_functions='benchmark_*'
