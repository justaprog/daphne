#pragma once
#include "MncSketch.h" 
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdint> 
#include "MNCSparsityEstimation.h"

// ----------------------------------------------------------------------------
// Helper: Propagate a vector (rows or columns) 
// ----------------------------------------------------------------------------
inline void propagateVector(
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

// ----------------------------------------------------------------------------
// Exact propagation for Trivial cases (Diagonal matrices)
// ----------------------------------------------------------------------------
inline bool propagateExact(const MncSketch &hA, const MncSketch &hB, MncSketch &hC) {
    // Case 1: A is diagonal square
    if (hA.isDiagonal && hA.m == hA.n && hA.nnzRows == hA.m) {
        hC = hB;
        hC.m = hA.m;
        hC.isDiagonal = hB.isDiagonal;
        return true;
    }
    // Case 2: B is diagonal square
    if (hB.isDiagonal && hB.m == hB.n && hB.nnzCols == hB.n) {
        hC = hA;
        hC.n = hB.n;
        hC.isDiagonal = hA.isDiagonal;
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
// Single-step propagation (A * B)
// ----------------------------------------------------------------------------
inline MncSketch propagateMM(const MncSketch &hA, const MncSketch &hB) {
    // 1. Try exact propagation first
    MncSketch hC;
    if (propagateExact(hA, hB, hC)) {
        return hC;
    }

    // 2. Prepare approximate propagation
    hC.m = hA.m;
    hC.n = hB.n;
    hC.hr.resize(hC.m);
    hC.hc.resize(hC.n);

    thread_local std::mt19937 gen(std::random_device{}());

    double sparsity = estimateSparsity(hA, hB);
    double targetTotalNNZ = sparsity * hC.m * hC.n;

    std::uint64_t totalRows = std::accumulate(hA.hr.begin(), hA.hr.end(), 0ULL);
    std::uint64_t totalCols = std::accumulate(hB.hc.begin(), hB.hc.end(), 0ULL);

    double rowScale = (totalRows > 0) ? (targetTotalNNZ / totalRows) : 0.0;
    double colScale = (totalCols > 0) ? (targetTotalNNZ / totalCols) : 0.0;

    // 3. Propagate Rows
    propagateVector(
        hA.hr,           
        hC.hr,           
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
        hB.hc,           
        hC.hc,           
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

// ----------------------------------------------------------------------------
// Chain propagation (multiple sketches)
// ----------------------------------------------------------------------------
inline MncSketch propagateChain(const std::vector<MncSketch> &chain) {
    if (chain.empty()) return MncSketch();
    
    MncSketch currentResult = chain[0];
    for (size_t i = 1; i < chain.size(); ++i) {
        currentResult = propagateMM(currentResult, chain[i]);
    }
    return currentResult;
}