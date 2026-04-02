# DAPHNE Compiler: Advanced Sparsity Estimation

**Note:** *This branch contains the implementation of the Matrix Non-zero Count (MNC) sketch, developed as part of the Large-scale Data Engineering project at Technische Universität Berlin.*

## Project Overview

Modern Machine Learning relies heavily on sparse matrices, but accurately predicting their memory footprint during complex execution plans remains a significant challenge. To address this, our team integrated the Matrix Non-zero Count (MNC) sketch directly into the DAPHNE compiler infrastructure. 

### Key Technical Contributions
* **Compiler Integration:** Engineered dynamic sparsity inference during the compiling phase utilizing the `InferencePass`.
* **Sketch Propagation:** Implemented probabilistic rounding to accurately predict the output structure of deep, complex matrix product chains.
* **Extended Operations:** Developed sparsity estimation support for Transpose and Element-wise addition/multiplication operations.

### Code Navigation
To facilitate review within the broader DAPHNE codebase, our core C++ contributions are located in the following directories:
* `src/`: Core implementation of the MNC Sketch data structures and collision probability algorithms.
* `scripts/`: Execution plans and benchmarking scripts used for our experiments.
* `test/`: Testing suites designed to verify the accuracy of the sketch propagation logic.

### Benchmark Results
The implementation was benchmarked against 19 real-world datasets within a Dockerized environment. The MNC sketch significantly outperformed naive metadata estimators on structured data, demonstrating lower and more consistent error rates. This allows the system to refine thresholds and effectively reduce its overall memory footprint.

## Original DAPHNE Overview

"The DAPHNE project aims to define and build an open and extensible system infrastructure for integrated data analysis pipelines, including data management and processing, high-performance computing (HPC), and machine learning (ML) training and scoring." (more information on https://daphne-eu.eu/)

In this repository, we develop the whole DAPHNE system with all its components including, but not limited to DaphneDSL, DaphneLib, DaphneIR, the DAPHNE Compiler, and the DAPHNE Run-time. The system will be built up and extended gradually in the course of the project.

### Getting Started
Find information on getting started in the documentation.

### Getting Involved
Read our contribution guidelines.
Have a look at the online documentation.
Browse open issues (e.g. "good first issues") or create a new issue.
