<!--
Copyright 2021 The DAPHNE Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# The DAPHNE System

### Overview

"The DAPHNE project aims to define and build an open and extensible system infrastructure for integrated data analysis pipelines, including data management and processing, high-performance computing (HPC), and machine learning (ML) training and scoring." (more information on https://daphne-eu.eu/)

In this repository, we develop the whole DAPHNE system with all its components including, but not limited to *DaphneDSL*, *DaphneLib*, *DaphneIR*, the *DAPHNE Compiler*, and the *DAPHNE Run-time*.
The system will be built up and extended gradually in the course of the project.

### Getting Started

- Find information on [getting started](/doc/GettingStarted.md) in the documentation.

### Getting Involved

- Read our [contribution guidelines](/CONTRIBUTING.md).

- Have a look at the [online documentation](https://daphne-eu.github.io/daphne/).

- [Browse open issues](https://github.com/daphne-eu/daphne/issues) (e.g. ["good first issues"](https://github.com/daphne-eu/daphne/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)) or [create a new issue](https://github.com/daphne-eu/daphne/issues/new).

MNC Sparsity Estimator Project (Large Scale Data Engineering
This section documents the environment and results for the MNC Sparsity Estimator
1. Experimental Setting
All experiments were conducted using a Docker container to ensure a consistent environment for library linking and reproducibility.
Hardware: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz with 16.0 GB RAM.
OS/Software: Windows 11 Home running through WSL (Windows Subsystem for Linux).
System Type: 64-bit operating system, x64-based processor.
2. Metrics
Dataset: We used real matrices from the SuiteSparse Matrix Collection (e.g., gre__115.mtx) instead of synthetic random data to better reflect real-world distributions.
Metrics: We measured Accuracy Error (the difference between the MNC estimate and actual sparsity) and Decision Correctness.
Variance: Because MNC is a probabilistic sketching method, small variations in results between runs are expected and were managed by using mean values for reporting.
3. Key Results
Decision Logic: Our implementation uses a 0.25 density threshold. It correctly triggers SPARSE storage for workloads measured below this value and DENSE for those above.
Reliability: The prototype passed 2,452 unit tests in the DAPHNE Catch2 suite, confirming it integrates safely with the existing system kernels.

