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
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/MncSketch.h>
#include <runtime/local/datastructures/MncSketchPropagate.h>
#include <runtime/local/datastructures/MncSketchBuild.h>

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
    return {-1};
}

template <typename VT>
static int64_t buildAndStoreSketch(uint64_t matrixAddr) {
    auto *reg = daphne::getActiveMncSketchRegistry();
    if (!reg)
        return {-1};

    const DenseMatrix<VT> *mat = reinterpret_cast<const DenseMatrix<VT> *>(matrixAddr);
    MncSketch h = buildMncFromDenseMatrix(*mat);
    return static_cast<int64_t>(reg->store(std::move(h)));
}


// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

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
    MncSketch hC = buildMncFromRand(
        numRows.first ? static_cast<std::size_t>(numRows.second) : 0,
        numCols.first ? static_cast<std::size_t>(numCols.second) : 0,
        sparsity.first ? sparsity.second : 1.0,
        seed.first ? seed.second : -1
    );
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::FillOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (reg == nullptr) {
        return {-1};
    }
    std::pair value = CompilerUtils::isConstant<double>(getArg());
    std::pair numRows = CompilerUtils::isConstant<ssize_t>(getNumRows());
    std::pair numCols = CompilerUtils::isConstant<ssize_t>(getNumCols());
    MncSketch hC = buildMncFromFill(
        value.first ? value.second : 0.0,
        numRows.first ? static_cast<std::size_t>(numRows.second) : 0,
        numCols.first ? static_cast<std::size_t>(numCols.second) : 0
    );
    return {reg->store(std::move(hC))};
}

std::vector<int64_t> daphne::SeqOp::inferMncSketchId() {
    auto *reg = getActiveMncSketchRegistry();
    if (reg == nullptr) {
        return {-1};
    }
    std::pair from = CompilerUtils::isConstant<ssize_t>(getFrom());
    std::pair to = CompilerUtils::isConstant<ssize_t>(getTo());
    std::pair inc = CompilerUtils::isConstant<double>(getInc());
    MncSketch hC = buildMncFromSeq(
        from.first ? static_cast<std::size_t>(from.second) : 0,
        to.first ? static_cast<std::size_t>(to.second) : 0,
        inc.first ? inc.second : 1.0
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

std::vector<int64_t> daphne::MatrixConstantOp::inferMncSketchId() {
    auto p = CompilerUtils::isConstant<uint64_t>(getMatrixAddr());
    if (!p.first)
        return {-1};

    const uint64_t matrixAddr = p.second;
    Type vt = CompilerUtils::getValueType(getResult().getType());

    if (vt.isF64())
        return {buildAndStoreSketch<double>(matrixAddr)};
    if (vt.isF32())
        return {buildAndStoreSketch<float>(matrixAddr)};
    if (vt.isSignedInteger(64))
        return {buildAndStoreSketch<int64_t>(matrixAddr)};
    if (vt.isSignedInteger(32))
        return {buildAndStoreSketch<int32_t>(matrixAddr)};
    if (vt.isSignedInteger(8))
        return {buildAndStoreSketch<int8_t>(matrixAddr)};
    if (vt.isUnsignedInteger(64))
        return {buildAndStoreSketch<uint64_t>(matrixAddr)};
    if (vt.isUnsignedInteger(32))
        return {buildAndStoreSketch<uint32_t>(matrixAddr)};
    if (vt.isUnsignedInteger(8))
        return {buildAndStoreSketch<uint8_t>(matrixAddr)};

    return {-1};
}

// elementwise
std::vector<int64_t> daphne::EwAddOp::inferMncSketchId() {
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
    MncSketch hC = propagateAdd(*hA, *hB);
    return {reg->store(std::move(hC))};
}
std::vector<int64_t> daphne::EwSubOp::inferMncSketchId() {
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
    MncSketch hC = propagateAdd(*hA, *hB);
    return {reg->store(std::move(hC))};
}
std::vector<int64_t> daphne::EwMulOp::inferMncSketchId() {
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
    MncSketch hC = propagateMul(*hA, *hB);
    return {reg->store(std::move(hC))};
}
std::vector<int64_t> daphne::EwDivOp::inferMncSketchId() {
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
    MncSketch hC = propagateMul(*hA, *hB);
    return {reg->store(std::move(hC))};
}
// readop : some property of the input data, mncsketch (the whole) can be stored as metadata files 
// and can be read by the compiler, just for 

// take data from files metadata and infer mnc sketch if available, otherwise return unknown
std::vector<int64_t> daphne::ReadOp::inferMncSketchId() {
    std::pair<bool, std::string> p = CompilerUtils::isConstant<std::string>(getFileName());
    if (p.first) {
        FileMetaData fmd = MetaDataParser::readMetaData(p.second);
        if (fmd.mncSketch.has_value()) {
            auto *reg = getActiveMncSketchRegistry();
            if (reg == nullptr) {
                return {-1};
            }
            return {reg->store(fmd.mncSketch.value())};
        }
    }
    return {-1};
}

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