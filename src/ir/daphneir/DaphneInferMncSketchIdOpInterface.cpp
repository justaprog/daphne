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

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferMncSketchIdOpInterface.cpp.inc>
}

#include <vector>

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Inference interface implementations
// ****************************************************************************

std::vector<ssize_t> daphne::TransposeOp::inferMncSketchId() {
    // TransposeOp retains the symmetry of its argument.
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(getArg().getType()))
        return {mt.getMncSketchId()};
    return {1};
}

// ****************************************************************************
// Inference function
// ****************************************************************************

std::vector<ssize_t> daphne::tryInferMncSketchId(Operation *op) {
    if (auto inferMncSketchIdOp = llvm::dyn_cast<daphne::InferMncSketchId>(op))
        // If the operation implements the inference interface, we apply that.
        return inferMncSketchIdOp.inferMncSketchId();
    else {
        // If the operation does not implement the inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<ssize_t> mncSketchIds;
        for (size_t i = 0; i < op->getNumResults(); i++)
            mncSketchIds.push_back(1);
        return mncSketchIds;
    }
}