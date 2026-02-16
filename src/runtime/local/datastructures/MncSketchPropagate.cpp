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
static std::uint64_t sumVec(const std::shared_ptr<std::vector<std::uint32_t>> &v) {
    if(!v) return 0;
    return std::accumulate(v->begin(), v->end(), std::uint64_t{0});
}

static std::uint64_t totalNnz(const MncSketch &A) {
    // Prefer hr if valid, else hc, else 0
    if (A.hr && A.hr->size() == A.m) return sumVec(A.hr);
    if (A.hc && A.hc->size() == A.n) return sumVec(A.hc);
    return 0;
}

static void computeSummary(MncSketch &S) {
    S.maxHr = 0; S.maxHc = 0;
    S.nnzRows = 0; S.nnzCols = 0;
    S.rowsEq1 = 0; S.colsEq1 = 0;
    S.rowsGtHalf = 0; S.colsGtHalf = 0;

    if (S.hr && S.hr->size() == S.m) {
        for (std::size_t i = 0; i < S.m; i++) {
            auto v = (*S.hr)[i];
            S.maxHr = std::max(S.maxHr, v);
            if (v > 0) S.nnzRows++;
            if (v == 1) S.rowsEq1++;
            if (v > S.n / 2) S.rowsGtHalf++;
        }
    }
    if (S.hc && S.hc->size() == S.n) {
        for (std::size_t j = 0; j < S.n; j++) {
            auto v = (*S.hc)[j];
            S.maxHc = std::max(S.maxHc, v);
            if (v > 0) S.nnzCols++;
            if (v == 1) S.colsEq1++;
            if (v > S.m / 2) S.colsGtHalf++;
        }
    }
}

MncSketch propagateMncFromReshape(const MncSketch &A, std::size_t outM, std::size_t outN) {
    // Cell count must be preserved.
    const __int128 inCells  = static_cast<__int128>(A.m) * static_cast<__int128>(A.n);
    const __int128 outCells = static_cast<__int128>(outM) * static_cast<__int128>(outN);
    if (inCells != outCells) {
        throw std::runtime_error("reshape: number of cells must be retained");
    }

    MncSketch C;
    C.m = outM;
    C.n = outN;
    C.isDiagonal = (outM == A.m && outN == A.n) ? A.isDiagonal : false;

    // Always allocate core vectors.
    C.hr = std::make_shared<std::vector<std::uint32_t>>(outM, 0);
    C.hc = std::make_shared<std::vector<std::uint32_t>>(outN, 0);

    // Reshape does not preserve extended vectors in general -> do not propagate them.
    C.her.reset();
    C.hec.reset();

    // Handle degenerate shapes
    if (outM == 0 || outN == 0) {
        computeSummary(C);
        return C;
    }

    // Need input vectors for the exact (paper) case.
    const bool haveHrA = A.hr && A.hr->size() == A.m;
    const bool haveHcA = A.hc && A.hc->size() == A.n;

    const std::uint64_t nnz = totalNnz(A);

    // --- Exact row-wise concatenation case from the MNC paper ---
    // They focus on: reshape m×n -> k×l with m mod k = 0 (k = outM),
    // so each output row is a concatenation of g = m/k input rows. :contentReference[oaicite:2]{index=2}
    if (haveHrA && haveHcA && outM > 0 && (A.m % outM == 0)) {
        const std::size_t g = A.m / outM; // input rows per output row
        // For row-wise reshape, l = n * g must hold.
        if (outN == A.n * g) {
            // hr_C: sum every g input rows (exact in this case)
            for (std::size_t r = 0; r < outM; r++) {
                std::uint64_t acc = 0;
                for (std::size_t t = 0; t < g; t++) {
                    acc += (*A.hr)[r * g + t];
                }
                (*C.hr)[r] = static_cast<std::uint32_t>(acc);
            }

            // hc_C: scale+replicate column counts (paper says rep(round(hcA/g), g)).
            // We do an equivalent deterministic split that preserves totals exactly.
            // For each original column j, distribute hcA[j] across g replicated columns.
            for (std::size_t j = 0; j < A.n; j++) {
                const std::uint32_t x = (*A.hc)[j];
                const std::uint32_t base = (g == 0) ? 0 : (x / static_cast<std::uint32_t>(g));
                const std::uint32_t rem  = (g == 0) ? 0 : (x % static_cast<std::uint32_t>(g));
                for (std::size_t t = 0; t < g; t++) {
                    (*C.hc)[t * A.n + j] = base + (t < rem ? 1u : 0u);
                }
            }

            computeSummary(C);
            return C;
        }
    }

    // --- Fallback (best-effort): preserve nnz and spread uniformly ---
    // (This is needed when reshape splits rows, or dimensions don't align nicely.)
    {
        const std::uint64_t baseR = nnz / outM;
        const std::uint64_t remR  = nnz % outM;
        for (std::size_t i = 0; i < outM; i++) {
            (*C.hr)[i] = static_cast<std::uint32_t>(baseR + (i < remR ? 1 : 0));
        }

        const std::uint64_t baseC = nnz / outN;
        const std::uint64_t remC  = nnz % outN;
        for (std::size_t j = 0; j < outN; j++) {
            (*C.hc)[j] = static_cast<std::uint32_t>(baseC + (j < remC ? 1 : 0));
        }

        computeSummary(C);
        return C;
    }
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

/*Element-wise Addition propagation (see section 4.2 from "MNC: Structure-Exploiting Sparsity Estimation for
Matrix Expressions" paper)*/
MncSketch propagateAdd(const MncSketch &A, const MncSketch &B) {
    MncSketch C;
    C.m = A.m;
    C.n = A.n;
    C.hr = std::make_shared<std::vector<uint32_t>>(C.m, 0);
    C.hc = std::make_shared<std::vector<uint32_t>>(C.n, 0);

    double sparsity = estimateSparsity_ElementWiseAddition(A, B);
    double targetNNZ = sparsity * C.m * C.n;

    double totalRowA = std::accumulate(A.hr->begin(), A.hr->end(), 0U);
    double totalRowB = std::accumulate(B.hr->begin(), B.hr->end(), 0U);
    double totalColA = std::accumulate(A.hc->begin(), A.hc->end(), 0U);
    double totalColB = std::accumulate(B.hc->begin(), B.hc->end(), 0U);

    double lambda_r = (totalRowA * totalRowB > 0) ? targetNNZ / (totalRowA * totalRowB) : 0.0;
    double lambda_c = (totalColA * totalColB > 0) ? targetNNZ / (totalColA * totalColB) : 0.0;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    propagateVector(*A.hr, *C.hr, 0.0, C.nnzRows, C.maxHr, C.rowsEq1, C.rowsGtHalf, C.n / 2, gen);
    for (size_t i = 0; i < C.m; ++i) {
        double val = (*A.hr)[i] + (*B.hr)[i] - (*A.hr)[i] * (*B.hr)[i] * lambda_c;
        (*C.hr)[i] = std::floor(val) + ((val - std::floor(val)) > std::uniform_real_distribution<>(0,1)(gen) ? 1 : 0);
    }

    propagateVector(*A.hc, *C.hc, 0.0, C.nnzCols, C.maxHc, C.colsEq1, C.colsGtHalf, C.m / 2, gen);
    for (size_t j = 0; j < C.n; ++j) {
        double val = (*A.hc)[j] + (*B.hc)[j] - (*A.hc)[j] * (*B.hc)[j] * lambda_r;
        (*C.hc)[j] = std::floor(val) + ((val - std::floor(val)) > std::uniform_real_distribution<>(0,1)(gen) ? 1 : 0);
    }

    return C;
}
/*Element-wise Multiplication propagation (see section 4.2 from "MNC: Structure-Exploiting Sparsity Estimation for
Matrix Expressions" paper)*/
MncSketch propagateMul(const MncSketch &A, const MncSketch &B) {
    MncSketch C;
    C.m = A.m;
    C.n = A.n;
    C.hr = std::make_shared<std::vector<uint32_t>>(C.m, 0);
    C.hc = std::make_shared<std::vector<uint32_t>>(C.n, 0);

    double sparsity = estimateSparsity_ElementWiseMultiplication(A, B);
    double targetNNZ = sparsity * C.m * C.n;

    double totalRowA = std::accumulate(A.hr->begin(), A.hr->end(), 0U);
    double totalRowB = std::accumulate(B.hr->begin(), B.hr->end(), 0U);
    double totalColA = std::accumulate(A.hc->begin(), A.hc->end(), 0U);
    double totalColB = std::accumulate(B.hc->begin(), B.hc->end(), 0U);

    double lambda_r = (totalRowA * totalRowB > 0) ? targetNNZ / (totalRowA * totalRowB) : 0.0;
    double lambda_c = (totalColA * totalColB > 0) ? targetNNZ / (totalColA * totalColB) : 0.0;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    propagateVector(*A.hr, *C.hr, 0.0, C.nnzRows, C.maxHr, C.rowsEq1, C.rowsGtHalf, C.n / 2, gen);
    for (size_t i = 0; i < C.m; ++i) {
        double val = (*A.hr)[i] * (*B.hr)[i] * lambda_c;
        (*C.hr)[i] = std::floor(val) + ((val - std::floor(val)) > std::uniform_real_distribution<>(0,1)(gen) ? 1 : 0);
    }

    propagateVector(*A.hc, *C.hc, 0.0, C.nnzCols, C.maxHc, C.colsEq1, C.colsGtHalf, C.m / 2, gen);
    for (size_t j = 0; j < C.n; ++j) {
        double val = (*A.hc)[j] * (*B.hc)[j] * lambda_r;
        (*C.hc)[j] = std::floor(val) + ((val - std::floor(val)) > std::uniform_real_distribution<>(0,1)(gen) ? 1 : 0);
    }

    return C;
}
