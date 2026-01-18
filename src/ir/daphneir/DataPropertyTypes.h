/*
 * Copyright 2025 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
/*
 * This header contains custom C++ types used to represent the data properties of DAPHNE data objects (e.g., matrices
 * and frames). These types are used in both, the IR/compiler and the runtime.
 */

namespace mlir::daphne {
    enum class BoolOrUnknown { Unknown = -1, False = 0, True = 1 };
    struct MncSketchType{
        // dimensions
        std::size_t m = 0; // rows
        std::size_t n = 0; // cols

        // core counts
        std::vector<std::uint32_t> hr;   // nnz per row (size m)
        std::vector<std::uint32_t> hc;   // nnz per col (size n)

        // Extended counts (optional, only constructed if maxHr or maxHc > 1)
        // her[i]: nnz in row i that lie in columns with hc == 1
        // hec[j]: nnz in column j that lie in rows with hr == 1
        std::vector<std::uint32_t> her;
        std::vector<std::uint32_t> hec;

        // Summary statistics
        std::uint32_t maxHr = 0;
        std::uint32_t maxHc = 0;
        std::uint32_t nnzRows = 0;       // # n rows with hr > 0
        std::uint32_t nnzCols = 0;       // # n cols with hc > 0
        std::uint32_t rowsEq1 = 0;       // # n rows with hr == 1, 
        std::uint32_t colsEq1 = 0;       // # n cols with hc == 1
        std::uint32_t rowsGtHalf = 0;    // # n rows with hr > n/2
        std::uint32_t colsGtHalf = 0;    // # n cols with hc > m/2
        BoolOrUnknown isDiagonal = BoolOrUnknown::Unknown;         // optional flag if A is (full) diagonal
    };
} // namespace mlir::daphne

// Make it available in the global namespace for backward compatibility if needed
using mlir::daphne::BoolOrUnknown;
using mlir::daphne::MncSketchType;