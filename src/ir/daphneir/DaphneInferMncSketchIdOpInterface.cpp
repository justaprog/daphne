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

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <runtime/local/datastructures/MncSketch.h>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferMncSketchIdOpInterface.cpp.inc>
static thread_local MncSketchRegistry *gMncReg = nullptr;

void setActiveMncSketchRegistry(MncSketchRegistry *reg) { gMncReg = reg; }
void clearActiveMncSketchRegistry() { gMncReg = nullptr; }
MncSketchRegistry *getActiveMncSketchRegistry() { return gMncReg; }
}

#include <vector>
#include <memory>
#include <random>
#include <cstdint>
#include <algorithm>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

static int64_t getMncSketchIdOrUnknownFromType(Value v) {
    Type t = v.getType();
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(t))
        return mt.getMncSketchId();
    return -1;
}
// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

static std::uint64_t effectiveSeed(std::int64_t seed) {
    if (seed >= 0) return static_cast<std::uint64_t>(seed);
    // If you want nondeterminism for seed == -1, use random_device:
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
}

MncSketch sketchFromRand(std::size_t m, std::size_t n, double sparsity, std::int64_t seed) {
    MncSketch s;
    s.m = m; s.n = n;
    s.isDiagonal = false;

    s.hr = std::make_shared<std::vector<std::uint32_t>>(m, 0);
    s.hc = std::make_shared<std::vector<std::uint32_t>>(n, 0);

    // clamp p safely
    double p = std::clamp(sparsity, 0.0, 1.0);

    std::mt19937_64 rng(effectiveSeed(seed));

    // Row/col NNZ as Binomial draws
    // (This approximates independent Bernoulli per entry.)
    std::binomial_distribution<std::uint32_t> rowDist(
        static_cast<unsigned int>(n), p
    );
    std::binomial_distribution<std::uint32_t> colDist(
        static_cast<unsigned int>(m), p
    );

    for (std::size_t i = 0; i < m; i++)
        (*s.hr)[i] = rowDist(rng);

    for (std::size_t j = 0; j < n; j++)
        (*s.hc)[j] = colDist(rng);

    // Summary statistics
    auto &hr = *s.hr;
    auto &hc = *s.hc;

    for (std::size_t i = 0; i < m; i++) {
        s.maxHr = std::max(s.maxHr, hr[i]);
        if (hr[i] > 0) s.nnzRows++;
        if (hr[i] == 1) s.rowsEq1++;
        if (hr[i] > n/2) s.rowsGtHalf++;
    }
    for (std::size_t j = 0; j < n; j++) {
        s.maxHc = std::max(s.maxHc, hc[j]);
        if (hc[j] > 0) s.nnzCols++;
        if (hc[j] == 1) s.colsEq1++;
        if (hc[j] > m/2) s.colsGtHalf++;
    }

    // Extended counts (approximate, because we don't know the exact positions)
    if (s.maxHr > 1 || s.maxHc > 1) {
        s.her = std::make_shared<std::vector<std::uint32_t>>(m, 0);
        s.hec = std::make_shared<std::vector<std::uint32_t>>(n, 0);

        // C1 = number of columns with hc == 1
        std::size_t C1 = 0;
        for (std::size_t j = 0; j < n; j++) if (hc[j] == 1) C1++;

        // R1 = number of rows with hr == 1
        std::size_t R1 = 0;
        for (std::size_t i = 0; i < m; i++) if (hr[i] == 1) R1++;

        double qColsEq1 = (n == 0) ? 0.0 : static_cast<double>(C1) / static_cast<double>(n);
        double qRowsEq1 = (m == 0) ? 0.0 : static_cast<double>(R1) / static_cast<double>(m);

        // her[i] ~ Binomial(hr[i], C1/n)
        for (std::size_t i = 0; i < m; i++) {
            std::binomial_distribution<std::uint32_t> d(hr[i], qColsEq1);
            (*s.her)[i] = d(rng);
        }

        // hec[j] ~ Binomial(hc[j], R1/m)
        for (std::size_t j = 0; j < n; j++) {
            std::binomial_distribution<std::uint32_t> d(hc[j], qRowsEq1);
            (*s.hec)[j] = d(rng);
        }
    }

    return s;
}


std::vector<int64_t> daphne::RandMatrixOp::inferMncSketchId() {
    // we cannot infer the mncSketchId of a random matrix, since it can have any structure
    // rand arg: numRows, numCols, min, max, sparsity, seed
    auto *reg = getActiveMncSketchRegistry();
    if (reg == nullptr) {
        return {-1};
    }
    std::pair numRows = CompilerUtils::isConstant<ssize_t>(getNumRows());
    std::pair numCols = CompilerUtils::isConstant<ssize_t>(getNumCols());
    std::pair sparsity = CompilerUtils::isConstant<double>(getSparsity());
    std::pair seed = CompilerUtils::isConstant<std::int64_t>(getSeed());
    MncSketch hC = sketchFromRand(
        numRows.first ? static_cast<std::size_t>(numRows.second) : 0,
        numCols.first ? static_cast<std::size_t>(numCols.second) : 0,
        sparsity.first ? sparsity.second : 1.0,
        seed.first ? seed.second : -1
    );
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::MatMulOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (!reg)
        return {-1};

    auto lhsTy = llvm::dyn_cast<daphne::MatrixType>(getLhs().getType());
    auto rhsTy = llvm::dyn_cast<daphne::MatrixType>(getRhs().getType());

    auto lhsId = lhsTy.getMncSketchId();
    auto rhsId = rhsTy.getMncSketchId();
    if (lhsId == -1 || rhsId == -1) {
        return {-1};
    }
    const MncSketch *hA = reg->get(lhsId);
    const MncSketch *hB = reg->get(rhsId);
    if (hA == nullptr || hB == nullptr) {
        return {-1};
    }
    MncSketch hC = propagateMM(*hA, *hB);
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::TransposeOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (reg == nullptr) {
        return {-1};
    }
    auto mt = llvm::dyn_cast<daphne::MatrixType>(getArg().getType());
    auto id = mt.getMncSketchId();
    if (id != -1) {
        const MncSketch *hA = reg->get(id);
        if (hA == nullptr) {
            return {-1};
        }
        MncSketch hC = propagateTranspose(*hA);
        return {reg->store(std::move(hC))};
    }
    return {-1};
}

// TODO: readop : some property of the input data, mncsketch (the whole) can be stored as metadata files 
// and can be read by the compiler, just for 

// take data from files metadata and infer mnc sketch if available, otherwise return unknown
/*

std::vector<double> daphne::ReadOp::inferMncSketchID() {
    std::pair<bool, std::string> p = CompilerUtils::isConstant<std::string>(getFileName());
    if (p.first) {
        FileMetaData fmd = MetaDataParser::readMetaData(p.second);
        if (fmd.numNonZeros == -1)
            return {-1.0};
        // TODO: maybe use type shape info instead of file? (would require
        // correct order of optimization passes)
        return {(static_cast<double>(fmd.numNonZeros) / fmd.numRows) / fmd.numCols};
    } else
        return {-1.0};
}
*/


// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<int64_t> daphne::tryInferMncSketchId(Operation *op) {
    if (auto inferMncSketchIdOp = llvm::dyn_cast<daphne::InferMncSketchId>(op))
        // If the operation implements the inference interface, we apply that.
        return inferMncSketchIdOp.inferMncSketchId();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<int64_t> mncSketchIds;
        for (size_t i = 0; i < op->getNumResults(); i++)
            mncSketchIds.push_back(-1);
        return mncSketchIds;
    }
}