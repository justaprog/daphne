#pragma once
#include "MncSketch.h" 
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdint> 
#include "MNCSparsityEstimation.h"

/**
 * MNC Sparsity Propagation 
 * This module implements the propagation logic for the MNC (Matrix non zero count) sketches
 * With this module we are able to predict the intermediate sparsity and structure of a chain
 * of multiplication of matrices (A*B*C) without having to execute the whole operation, this 
 * enables us to save both time and memory. The Mnc Sparsity Propagation uses the MNC Sparsity estimation
 * to get an inital density, it then scales the row and column histograms accordingly and then applies
 * probablistic rounding(to avoid always rouding down to zero) 
 * 
 * 
 * 
 * 
 * * This module implements propagation logic for MNC (Matrix Non-zero Count) sketches.
 * It allows the system to predict the intermediate sparsity and structural metadata 
 * of a chain of matrix multiplications (A * B * C...) without executing the 
 * complete operations. By operating solely on the sketches, it saves significant 
 * time and memory space during the query optimization phase.
 * * The algorithm uses the MNC sparsity estimator to establish initial density targets,
 * scales the row and column histograms accordingly, and applies probabilistic 
 * rounding via a Mersenne Twister to ensure statistical accuracy across the chain.
 */




/*
 * Helper function that propagates the histogram values (rows or columns) with rounding.
 * The scaled counts are derived from the ratio of output to input NNZ.
 * Probabilistic rounding is applied to prevent always rounding down to zero and 
 * to maintain statistical accuracy.
 *
 * Input: The source Histogram from the input sketch.
 * Output: The target vector to be filled in the new sketch.
 * outNNZ: Number of non-zero entries in the resulting vector.
 */
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
/*
*Handles the base cases where one of the matices is a simple square diagnal matrix i.e Indetity matrix
*For this case we can avoid the scaling and just copy the input sketch of the other matrix i.e non-identiy matrix
*Input : SKetches hA and hB
*Output : Sketch hC as the copy of the matrix B if diagnol condition is met
*Return : True if we have the exact case, otherwise False if we require scaling 
*/
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

/*
*Preforms a single step propogation for 2 the matrices A and B i.e(A*B)
*It first checks if the simple diaganol condition is met if not, then we use
*non trvial method where the MNC Sparisity estimator is used to find the target
*NNZ and then calcualte the scaling factor
*Input : Sketches of hA and hB
*Output : Sketch of hC with the predicted sparsity of the result
*/
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
/*
*This estimates the sparsiy of a chain of matrix multiplications, it works recursively,
*by taking the result sketch of the first last multiplicatio and then using that as an
*input for the next matrix in the chain, for example (A*B*C)
*Input : A vector of MncSketches of the matrix chain
*Output : The final sketch for the multiplication chain
 */


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