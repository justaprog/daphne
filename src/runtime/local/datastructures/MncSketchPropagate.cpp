/*
 * Copyright 2021 The DAPHNE Consortium
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
#include <runtime/local/datastructures/MncSketchPropagate.h>
#include <runtime/local/datastructures/MncSketchEstimate.h>

MncSketch propagateTranspose(const MncSketch &hA){
    MncSketch hC;
    // Swap Dimensions
    hC.m = hA.n;
    hC.n = hA.m;

    //Rows become collumns, collums become rows
    hC.hr = hA.hc;
    hC.hc = hA.hr;

    //Extended Vectors swap
    hC.her = hA.hec;
    hC.hec = hA.her;

    //Swap characteristics
    hC.maxHr = hA.maxHc;
    hC.maxHc = hA.maxHr;
    hC.nnzRows = hA.nnzCols;
    hC.nnzCols = hA.nnzRows;
    hC.rowsEq1 = hA.colsEq1;
    hC.colsEq1 = hA.rowsEq1;
    hC.rowsGtHalf = hA.colsGtHalf;
    hC.colsGtHalf = hA.rowsGtHalf;

    // Keeping the diaganal property
    hC.isDiagonal = hA.isDiagonal;
    return hC;
}
void propagateVector(
    const std::vector<std::uint32_t>& input,
    std::vector<std::uint32_t>& output,
    double scale,
    std::uint32_t& outNNZ,    
    std::uint32_t& outMaxVal,  
    std::uint32_t& outEq1,     
    std::uint32_t& outGtHalf,  
    std::size_t halfThreshold, 
    std::mt19937& gen
) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (std::size_t i = 0; i < input.size(); ++i) {
        double val = input[i] * scale;
        std::uint32_t count = static_cast<std::uint32_t>(val);
        // Probabilistic rounding
        if ((val - count) > dis(gen)) {
            count++;
        }
        output[i] = count;
        // Update statistics
        if (count > 0) {
            outNNZ++;
            if (count > outMaxVal) 
                outMaxVal = count;
            if (count == 1) 
                outEq1++;              
            if (count > halfThreshold) 
                outGtHalf++; 
        }
    }
}

bool propagateExact(const MncSketch &hA, const MncSketch &hB, MncSketch &hC) {
    // Case 1: A is diagonal square
    if (hA.isDiagonal && hA.m == hA.n && hA.nnzRows == hA.m) {
        hC = hB;
        return true;
    }
    // Case 2: B is diagonal square
    if (hB.isDiagonal && hB.m == hB.n && hB.nnzCols == hB.n) {
        hC = hA;
        return true;
    }
    return false;
}

MncSketch propagateMM(const MncSketch &hA, const MncSketch &hB) {
    // 1. Try exact propagation first
    MncSketch hC;
    if (propagateExact(hA, hB, hC)) {
        return hC;
    }

    // 2. Prepare approximate propagation
    hC.m = hA.m;
    hC.n = hB.n;
    //Changes (Abdullah)
    hC.hr = std::make_shared<std::vector<std::uint32_t>>(hC.m, 0);
    hC.hc = std::make_shared<std::vector<std::uint32_t>>(hC.n, 0);

    // FIX: Removed thread_local to simplify
    static std::random_device rd;
    static std::mt19937 gen(rd());

    double sparsity = estimateSparsity_product(hA, hB);
    double targetTotalNNZ = sparsity * hC.m * hC.n;

    //Changes (Abdullah)
    std::uint32_t totalRows = std::accumulate(hA.hr->begin(), hA.hr->end(), 0U);
    std::uint32_t totalCols = std::accumulate(hB.hc->begin(), hB.hc->end(), 0U);

    double rowScale = (totalRows > 0) ? (targetTotalNNZ / totalRows) : 0.0;
    double colScale = (totalCols > 0) ? (targetTotalNNZ / totalCols) : 0.0;

    // 3. Propagate Rows
    propagateVector(
        *hA.hr,          // Change (Abdullah)
        *hC.hr,          // Change (Abdullah)
        rowScale,        
        hC.nnzRows,     
        hC.maxHr,        
        hC.rowsEq1,      
        hC.rowsGtHalf,   
        hC.n / 2,        
        gen
    );

    // 4. Propagate Columns
    propagateVector(
        *hB.hc,          // Change (Abdullah)
        *hC.hc,          // Change (Abdullah)
        colScale,        
        hC.nnzCols,      
        hC.maxHc,        
        hC.colsEq1,      
        hC.colsGtHalf,   
        hC.m / 2,        
        gen
    );

    return hC;
}

MncSketch propagateChain(const std::vector<MncSketch> &chain) {
    if (chain.empty()) return MncSketch();
    
    MncSketch currentResult = chain[0];
    for (size_t i = 1; i < chain.size(); ++i) {
        currentResult = propagateMM(currentResult, chain[i]);
    }
    return currentResult;
}

