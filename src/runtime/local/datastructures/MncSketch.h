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


/*
This MNC Sketch implementatipn is based on the paper: Johanna Sommer, Matthias Boehm,
 Alexandre V. Evfimievski, Berthold Reinwald, Peter J. Haas (2019). 
 "MNC: Structure-Exploiting Sparsity Estimation for Matrix Expressions". 
SIGMOD '19: Proceedings of the 2019 International Conference on Management of Data
*/

#pragma once

#include <iostream>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <numeric>   
#include <random>
#include <algorithm> 

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DenseMatrix.h>

/*
MNC Sketch data structure to capture the structure of a matrix to estimate sparsity
*/
struct MncSketch{
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
    bool isDiagonal;         // optional flag if A is (full) diagonal
};

/**
 * A function to build MNC sketch from a CSRMatrix
 * @param A The input CSRMatrix
 * @return The MNC sketch of A
*/
template<typename VT>
MncSketch buildMncFromCsrMatrix(const CSRMatrix<VT> &A) {
    MncSketch h;
    h.m = A.getNumRows();
    h.n = A.getNumCols();

    h.hr.assign(h.m, 0);
    h.hc.assign(h.n, 0);

    const std::size_t *rowOffsets = A.getRowOffsets();
    const std::size_t *colIdxs    = A.getColIdxs(0);

    // --- 1) per-row nnz (hr) and row stats ---
    for(std::size_t i = 0; i < h.m; ++i) {
        // row i uses indices [rowOffsets[i], rowOffsets[i+1])
        std::size_t s   = rowOffsets[i];
        std::size_t e   = rowOffsets[i+1];
        std::uint32_t cnt = static_cast<std::uint32_t>(e - s);

        h.hr[i] = cnt;

        if(cnt > 0) {
            h.nnzRows++;
            if(cnt == 1)
                h.rowsEq1++;
            if(cnt > h.n / 2)
                h.rowsGtHalf++;
        }
        if(cnt > h.maxHr)
            h.maxHr = cnt;
    }

    // --- 2) per-column nnz (hc) and column stats ---
    // We must iterate all nnz in this *view*:
    std::size_t nnzBegin = rowOffsets[0];
    std::size_t nnzEnd   = rowOffsets[h.m];

    for(std::size_t k = nnzBegin; k < nnzEnd; ++k) {
        std::size_t j = colIdxs[k];
        auto &cnt = h.hc[j];
        if(cnt == 0)
            h.nnzCols++;
        cnt++;
    }

    for(std::size_t j = 0; j < h.n; ++j) {
        auto cnt = h.hc[j];
        if(cnt == 1)
            h.colsEq1++;
        if(cnt > h.m / 2)
            h.colsGtHalf++;
        if(cnt > h.maxHc)
            h.maxHc = cnt;
    }

    // --- 3) isDiagonal ---
    // We call a matrix "diagonal" if it is square and every non-zero lies on i == j.
    // Optimization: if any row/col has >1 nnz, it cannot be diagonal.
    if (h.maxHr > 1 || h.maxHc > 1) {
        h.isDiagonal = false;
    } 
    else if (h.m == h.n && nnzEnd > nnzBegin) {
        bool diag = true;
        for(std::size_t i = 0; i < h.m && diag; ++i) {
            std::size_t s = rowOffsets[i];
            std::size_t e = rowOffsets[i+1];
            for(std::size_t k = s; k < e; ++k) {
                std::size_t j = colIdxs[k];
                if(j != i) {
                    diag = false;
                    break;
                }
            }
        }
        h.isDiagonal = diag;
    } else {
        h.isDiagonal = false;
    }

    // --- 4) extended counts her, hec --- (only if there is something to extend)
    if(h.maxHr > 1 || h.maxHc > 1) {
        h.her.assign(h.m, 0);
        h.hec.assign(h.n, 0);

        // For each nnz at (i,j):
        //  - if hc[j] == 1, it contributes to her[i]
        //  - if hr[i] == 1, it contributes to hec[j]
        for(std::size_t i = 0; i < h.m; ++i) {
            std::size_t s = rowOffsets[i];
            std::size_t e = rowOffsets[i+1];
            for(std::size_t k = s; k < e; ++k) {
                std::size_t j = colIdxs[k];

                if(h.hc[j] == 1)
                    h.her[i]++;

                if(h.hr[i] == 1)
                    h.hec[j]++;
            }
        }
    }

    return h;
}

/**
 * A function to build MNC sketch from a DenseMatrix
 * @param A The input DenseMatrix
 * @return The MNC sketch of A
*/
template<typename VT>
MncSketch buildMncFromDenseMatrix(const DenseMatrix<VT> &A) {
    MncSketch h;
    h.m = A.getNumRows();
    h.n = A.getNumCols();

    h.hr.assign(h.m, 0);
    h.hc.assign(h.n, 0);

    const VT *rowPtr = A.getValues();
    const std::size_t rowSkip = A.getRowSkip();

    // --- 1) compute hr, hc, row stats, and diagonal flag in one dense scan ---
    // Definition: diagonal if square AND every non-zero lies on i==j.
    // Note: the all-zero square matrix is considered diagonal under this definition.
    bool diag = (h.m == h.n);

    for (std::size_t i = 0; i < h.m; ++i) {
        std::uint32_t rowNnz = 0;

        for (std::size_t j = 0; j < h.n; ++j) {
            const VT v = rowPtr[j];
            if (v != static_cast<VT>(0)) {
                rowNnz++;
                // increment column nnz
                // (safe as long as counts fit into uint32_t; if not, switch to uint64_t)
                h.hc[j]++;

                // diagonal check
                if (diag && j != i)
                    diag = false;
            }
        }

        h.hr[i] = rowNnz;

        if (rowNnz > 0) {
            h.nnzRows++;
            if (rowNnz == 1)
                h.rowsEq1++;
            if (rowNnz > h.n / 2)
                h.rowsGtHalf++;
        }
        if (rowNnz > h.maxHr)
            h.maxHr = rowNnz;

        // advance to next row (DenseMatrix might have padding)
        rowPtr += rowSkip;
    }

    h.isDiagonal = diag;

    // --- 2) column stats from hc ---
    for (std::size_t j = 0; j < h.n; ++j) {
        const std::uint32_t cnt = h.hc[j];
        if (cnt > 0)
            h.nnzCols++;
        if (cnt == 1)
            h.colsEq1++;
        if (cnt > h.m / 2)
            h.colsGtHalf++;
        if (cnt > h.maxHc)
            h.maxHc = cnt;
    }

    // --- 3) extended counts her/hec (optional) ---
    if (h.maxHr > 1 || h.maxHc > 1) {
        h.her.assign(h.m, 0);
        h.hec.assign(h.n, 0);

        // Second scan over all entries:
        //  - if value!=0 and hc[j]==1 => her[i]++
        //  - if value!=0 and hr[i]==1 => hec[j]++
        const VT *rowPtr2 = A.getValues();
        for (std::size_t i = 0; i < h.m; ++i) {
            const bool rowIsSingleton = (h.hr[i] == 1);

            for (std::size_t j = 0; j < h.n; ++j) {
                const VT v = rowPtr2[j];
                if (v != static_cast<VT>(0)) {
                    if (h.hc[j] == 1)
                        h.her[i]++;

                    if (rowIsSingleton)
                        h.hec[j]++;
                }
            }
            rowPtr2 += rowSkip;
        }
    }

    return h;
}


// TODO: implement Density Map Estimator
/** This implementation of estimate Sparsity follows the pseudocode from the paper 
"MNC: Structure-Exploiting Sparsity Estimation for
Matrix Expressions" section 3.2 
*/
// Edm returns an estimated DENSITY in [0, 1] over the remaining 'p' cells.
// Caller can do: nnz += Edm(...) * p;
inline double EdmDensity(const std::vector<std::uint32_t>& hcA_res,
                  const std::vector<std::uint32_t>& hrB_res,
                  std::uint32_t p)
{
    if (p == 0) return 0.0;
    // std::cout << "hcA_res size: " << hcA_res.size() << ", hrB_res size: " << hrB_res.size() << ", p: " << p << std::endl;
    if (hcA_res.size() != hrB_res.size())
        throw std::invalid_argument("EdmDensity: vector sizes must match (same inner dimension).");

    // t = total number of candidate contributions ("hits")
    long double t = 0.0L;
    for (std::size_t k = 0; k < hcA_res.size(); ++k) {
        t += static_cast<long double>(hcA_res[k]) * static_cast<long double>(hrB_res[k]);
    }

    if (t <= 0.0L) return 0.0;

    // s = 1 - (1 - 1/p)^t  (use log/exp for numerical stability)
    const long double invP = 1.0L / static_cast<long double>(p);
    const long double logBase = std::log1p(-invP);           // log(1 - 1/p), negative
    const long double emptyProb = std::exp(t * logBase);    // (1 - 1/p)^t
    long double s = 1.0L - emptyProb;

    // Clamp due to floating point rounding
    if (s < 0.0L) s = 0.0L;
    if (s > 1.0L) s = 1.0L;
    return static_cast<double>(s);
}


/**
 * Estimate the sparsity of the product of two matrices given their MNC sketches. 
 * Based on Algorithm 1 from the MNC paper.
 * @param hA MNC sketch of matrix A
 * @param hB MNC sketch of matrix B
 * @return Estimated sparsity of the product A * B
 * 
 */
inline double estimateSparsity_product(const MncSketch &hA, const MncSketch &hB) {
    const std::size_t m = hA.m;
    const std::size_t l = hB.n;

    //Fix use size_t instead of double for avoiding counting errors
    std::size_t exact_nnz = 0;
    double prob_nnz = 0.0;

    // Case 1: Exact count
    if(hA.maxHr <= 1 || hB.maxHc <= 1) {
        for(std::size_t j = 0; j < hA.n; ++j)
            // Multiply as integers (size_t)
            exact_nnz += static_cast<std::size_t>(hA.hc[j]) * static_cast<std::size_t>(hB.hr[j]);
    }

    // Case 2: Extended count
    else if(!hA.her.empty() && !hB.her.empty()) { // Note: I also applied the && fix here 
        
        // Fused (Exact Part 1 + Exact Part 2)
        for(std::size_t k = 0; k < hA.n; ++k) {
            // Term 1: hA^ec * hB^r
            exact_nnz += static_cast<std::size_t>(hA.hec[k]) * static_cast<std::size_t>(hB.hr[k]);

            // Term 2: hB^er * (hA^c - hA^ec)
            // Check to ensure positive result before subtraction
            if (hA.hc[k] > hA.hec[k]) {
                exact_nnz += static_cast<std::size_t>(hB.her[k]) * (static_cast<std::size_t>(hA.hc[k]) - static_cast<std::size_t>(hA.hec[k]));
            }
        }

        // Remaining uncertain cells (Probabilistic part)
        std::size_t p = (hA.nnzRows - hA.rowsEq1) * (hB.nnzCols - hB.colsEq1);

        if(p > 0) {
            std::vector<uint32_t> hcA_res;
            std::vector<uint32_t> hrB_res;
            // Pre-allocate memory for speed
            hcA_res.reserve(hA.n);
            hrB_res.reserve(hB.m);

            for(std::size_t j = 0; j < hA.n; ++j) {
                std::uint32_t hcA_val = static_cast<std::uint32_t>(hA.hc[j]);
                // Safety check to ensure we don't underflow if something is wrong
                std::uint32_t sub = static_cast<std::uint32_t>(hA.hec[j]);
                hcA_res.push_back((hcA_val > sub) ? (hcA_val - sub) : 0);
            }
            for(std::size_t i = 0; i < hB.m; ++i) {
                std::uint32_t hrB_val = static_cast<std::uint32_t>(hB.hr[i]);
                std::uint32_t sub = static_cast<std::uint32_t>(hB.her[i]);
                hrB_res.push_back((hrB_val > sub) ? (hrB_val - sub) : 0);
            }
            double dens = EdmDensity(hcA_res, hrB_res, p);
            prob_nnz += dens * static_cast<double>(p);
        }
    }

    // Case 3: Generic fallback
    else {
        std::size_t p = hA.nnzRows * hB.nnzCols;
        if(p > 0) {
            double dens = EdmDensity(hA.hc, hB.hr, p);
            prob_nnz = dens * static_cast<double>(p);
        }
    }

   // Combine exact and probabilistic counts
   double totalNNZ = static_cast<double>(exact_nnz) + prob_nnz;

   // Lower bound
   std::size_t lower = hA.rowsGtHalf * hB.colsGtHalf;

   if(totalNNZ < static_cast<double>(lower))
       totalNNZ = static_cast<double>(lower); 

   return totalNNZ / static_cast<double>(m * l);
}

// ----------------------------------------------------------------------------
//    MNC Sparsity Propagation
//    This module implements the propagation logic for the MNC (Matrix non zero count) sketches
//    With this module we are able to predict the intermediate sparsity and structure of a chain
//    of multiplication of matrices (ABC) without having to execute the whole operation, this
//    enables us to save both time and memory. The Mnc Sparsity Propagation uses the MNC Sparsity estimation
//    to get an inital density, it then scales the row and column histograms accordingly and then applies
//    probablistic rounding(to avoid always rouding down to zero)
// ----------------------------------------------------------------------------

/**
 * Helper function that propagates the histogram values (rows or columns) with rounding.
 * The scaled counts are derived from the ratio of output to input NNZ.
 * Probabilistic rounding is applied to prevent always rounding down to zero and
 * to maintain statistical accuracy.*
 * @param input The source Histogram from the input sketch.
 * @param output The target vector to be filled in the new sketch.
 * @param outNNZ Number of non-zero entries in the resulting vector.
 * 
*/
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
        return true;
    }
    // Case 2: B is diagonal square
    if (hB.isDiagonal && hB.m == hB.n && hB.nnzCols == hB.n) {
        hC = hA;
        return true;
    }
    return false;
}

/* Single-step propagation (A * B)
Preforms a single step propogation for 2 the matrices A and B i.e(AB)
*It first checks if the simple diaganol condition is met if not, then we use
*non trvial method where the MNC Sparisity estimator is used to find the target
*NNZ and then calcualte the scaling factor
*Input : Sketches of hA and hB
*Output : Sketch of hC with the predicted sparsity of the result
*/
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

    // FIX: Removed thread_local to simplify
    static std::random_device rd;
    static std::mt19937 gen(rd());

    double sparsity = estimateSparsity_product(hA, hB);
    double targetTotalNNZ = sparsity * hC.m * hC.n;

    std::uint32_t totalRows = std::accumulate(hA.hr.begin(), hA.hr.end(), 0U);
    std::uint32_t totalCols = std::accumulate(hB.hc.begin(), hB.hc.end(), 0U);

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
input for the next matrix in the chain, for example (AB*C)
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
