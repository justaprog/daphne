
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
#include <runtime/local/datastructures/MncSketchBuild.h>

template<typename VT>
MncSketch buildMncFromCsrMatrix(const CSRMatrix<VT> &A) {
    MncSketch h;
    h.m = A.getNumRows();
    h.n = A.getNumCols();

    // Changes (Abdullah)
    h.hr = std::make_shared<std::vector<std::uint32_t>>(h.m, 0);
    h.hc = std::make_shared<std::vector<std::uint32_t>>(h.n, 0);

    const std::size_t *rowOffsets = A.getRowOffsets();
    const std::size_t *colIdxs    = A.getColIdxs(0);

    // --- 1) per-row nnz (hr) and row stats ---
    for(std::size_t i = 0; i < h.m; ++i) {
        // row i uses indices [rowOffsets[i], rowOffsets[i+1])
        std::size_t s   = rowOffsets[i];
        std::size_t e   = rowOffsets[i+1];
        std::uint32_t cnt = static_cast<std::uint32_t>(e - s);

        // Changes (Abdullah)
        (*h.hr)[i] = cnt;

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
        auto &cnt = (*h.hc)[j];
        if(cnt == 0)
            h.nnzCols++;
        cnt++;
    }

    for(std::size_t j = 0; j < h.n; ++j) {
        auto cnt = (*h.hc)[j];
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
        h.her = std::make_shared<std::vector<std::uint32_t>>(h.m, 0);
        h.hec = std::make_shared<std::vector<std::uint32_t>>(h.n, 0);

        // For each nnz at (i,j):
        //  - if hc[j] == 1, it contributes to her[i]
        //  - if hr[i] == 1, it contributes to hec[j]
        for(std::size_t i = 0; i < h.m; ++i) {
            std::size_t s = rowOffsets[i];
            std::size_t e = rowOffsets[i+1];
            for(std::size_t k = s; k < e; ++k) {
                std::size_t j = colIdxs[k];

                if((*h.hc)[j] == 1)
                    (*h.her)[i]++;

                if((*h.hr)[i] == 1)
                    (*h.hec)[j]++;
            }
        }
    }

    return h;
}

template<typename VT>
MncSketch buildMncFromDenseMatrix(const DenseMatrix<VT> &A) {
    MncSketch h;
    h.m = A.getNumRows();
    h.n = A.getNumCols();

    // Changes (Abdullah)
    h.hr = std::make_shared<std::vector<std::uint32_t>>(h.m, 0);
    h.hc = std::make_shared<std::vector<std::uint32_t>>(h.n, 0);

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
                (*h.hc)[j]++;

                // diagonal check
                if (diag && j != i)
                    diag = false;
            }
        }

        (*h.hr)[i] = rowNnz;

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
        const std::uint32_t cnt = (*h.hc)[j];
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
        h.her = std::make_shared<std::vector<std::uint32_t>>(h.m, 0);
        h.hec = std::make_shared<std::vector<std::uint32_t>>(h.n, 0);

        // Second scan over all entries:
        //  - if value!=0 and hc[j]==1 => her[i]++
        //  - if value!=0 and hr[i]==1 => hec[j]++
        const VT *rowPtr2 = A.getValues();
        for (std::size_t i = 0; i < h.m; ++i) {
            const bool rowIsSingleton = ((*h.hr)[i] == 1);

            for (std::size_t j = 0; j < h.n; ++j) {
                const VT v = rowPtr2[j];
                if (v != static_cast<VT>(0)) {
                    if ((*h.hc)[j] == 1)
                        (*h.her)[i]++;

                    if (rowIsSingleton)
                        (*h.hec)[j]++;
                }
            }
            rowPtr2 += rowSkip;
        }
    }

    return h;
}

static std::uint64_t effectiveSeed(std::int64_t seed) {
    if (seed >= 0) return static_cast<std::uint64_t>(seed);
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ rd();
}

/**
 * 1. RAND: Generates a probabilistic sketch for X = rand(rows, cols, sparsity)
inline MncSketch createMncFromRand(size_t rows, size_t cols, double sparsity) {
    MncSketch h;
    h.m = rows; h.n = cols;
    h.hr = std::make_shared<std::vector<uint32_t>>(rows, 0);
    h.hc = std::make_shared<std::vector<uint32_t>>(cols, 0);
    h.her = std::make_shared<std::vector<uint32_t>>(rows, 0);
    h.hec = std::make_shared<std::vector<uint32_t>>(cols, 0);

    if (sparsity <= 0.0) return h;
    if (sparsity >= 1.0) {
        // Dense logic
        std::fill(h.hr->begin(), h.hr->end(), static_cast<uint32_t>(cols));
        std::fill(h.hc->begin(), h.hc->end(), static_cast<uint32_t>(rows));
        h.nnzRows = rows; h.nnzCols = cols;
        h.maxHr = cols; h.maxHc = rows;
        if (cols == 1) { h.rowsEq1 = rows; std::fill(h.her->begin(), h.her->end(), 1); }
        if (rows == 1) { h.colsEq1 = cols; std::fill(h.hec->begin(), h.hec->end(), 1); }
        h.rowsGtHalf = rows; h.colsGtHalf = cols;
        return h;
    }

    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::binomial_distribution<uint32_t> rowDist(cols, sparsity);
    for(size_t i = 0; i < rows; ++i) {
        uint32_t nnz = rowDist(gen);
        (*h.hr)[i] = nnz;
        if (nnz > 0) {
            h.nnzRows++;
            if (nnz > h.maxHr) h.maxHr = nnz;
            if (nnz == 1) { h.rowsEq1++; (*h.her)[i] = 1; }
            if (nnz > cols/2) h.rowsGtHalf++;
        }
    }

    std::binomial_distribution<uint32_t> colDist(rows, sparsity);
    for(size_t j = 0; j < cols; ++j) {
        uint32_t nnz = colDist(gen);
        (*h.hc)[j] = nnz;
        if (nnz > 0) {
            h.nnzCols++;
            if (nnz > h.maxHc) h.maxHc = nnz;
            if (nnz == 1) { h.colsEq1++; (*h.hec)[j] = 1; }
            if (nnz > rows/2) h.colsGtHalf++;
        }
    }
    h.isDiagonal = false; 
    return h;
}
 */
MncSketch buildMncFromRand(std::size_t m, std::size_t n, double density, std::int64_t seed) {
    MncSketch s;
    s.m = m; s.n = n;
    s.isDiagonal = false;

    s.hr = std::make_shared<std::vector<std::uint32_t>>(m, 0);
    s.hc = std::make_shared<std::vector<std::uint32_t>>(n, 0);

    if (density <= 0.0) return s;
    if (density >= 1.0) {
        s.her = std::make_shared<std::vector<std::uint32_t>>(m, 0);
        s.hec = std::make_shared<std::vector<std::uint32_t>>(n, 0);
        std::fill(s.hr->begin(), s.hr->end(), static_cast<uint32_t>(n));
        std::fill(s.hc->begin(), s.hc->end(), static_cast<uint32_t>(m));
        s.nnzRows = static_cast<uint32_t>(m);
        s.nnzCols = static_cast<uint32_t>(n);
        s.maxHr = static_cast<uint32_t>(n);
        s.maxHc = static_cast<uint32_t>(m);
        if (n == 1) { s.rowsEq1 = static_cast<uint32_t>(m); std::fill(s.her->begin(), s.her->end(), 1); }
        if (m == 1) { s.colsEq1 = static_cast<uint32_t>(n); std::fill(s.hec->begin(), s.hec->end(), 1); }
        s.rowsGtHalf = static_cast<uint32_t>(m);
        s.colsGtHalf = static_cast<uint32_t>(n);
        return s;
    }

    double p = std::clamp(density, 0.0, 1.0);
    std::mt19937_64 rng(effectiveSeed(seed));
    std::binomial_distribution<std::uint32_t> rowDist(static_cast<unsigned int>(n), p);

    // store chosen columns per row so we can compute her exactly later
    std::vector<std::vector<std::uint32_t>> colsPerRow(m);

    // 1) sample row nnz + positions -> induces consistent hc
    for (std::size_t i = 0; i < m; i++) {
        auto k = rowDist(rng);
        (*s.hr)[i] = k;
        colsPerRow[i].reserve(k);

        // sample k distinct columns (simple set; fine for tests)
        std::unordered_set<std::uint32_t> picked;
        picked.reserve(k * 2);

        std::uniform_int_distribution<std::uint32_t> colU(0, static_cast<uint32_t>(n - 1));
        while (picked.size() < k) {
            picked.insert(colU(rng));
        }
        for (auto c : picked) {
            colsPerRow[i].push_back(c);
            (*s.hc)[c] += 1;
        }
    }

    // 2) stats from hr/hc
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

    // 3) exact extended counts from positions
    if (s.maxHr > 1 || s.maxHc > 1) {
        s.her = std::make_shared<std::vector<std::uint32_t>>(m, 0);
        s.hec = std::make_shared<std::vector<std::uint32_t>>(n, 0);

        std::vector<char> colIsSingleton(n, 0);
        for (std::size_t j = 0; j < n; j++) colIsSingleton[j] = (hc[j] == 1);

        // her[i] = count of picked cols that are singleton columns
        for (std::size_t i = 0; i < m; i++) {
            uint32_t cnt = 0;
            for (auto c : colsPerRow[i]) cnt += colIsSingleton[c];
            (*s.her)[i] = cnt;
        }

        // hec[j] = nnz in col j that lie in singleton rows (hr==1)
        for (std::size_t i = 0; i < m; i++) {
            if (hr[i] == 1) {
                // singleton row has exactly one picked col
                (*s.hec)[colsPerRow[i][0]] += 1;
            }
        }
    }

    return s;
}

/**
 * 2. FILL & SEQ Helpers
 */
MncSketch buildMncFromFill(double val, size_t rows, size_t cols) {
    return buildMncFromRand(rows, cols, (val == 0.0) ? 0.0 : 1.0);
}

MncSketch buildMncFromSeq(double start, double end, double step) {
    if (step == 0.0) throw std::runtime_error("Seq step cannot be 0");
    size_t rows = static_cast<size_t>(std::floor((end - start) / step)) + 1;
    MncSketch h; 
    h.m = rows; h.n = 1;
    h.hr = std::make_shared<std::vector<uint32_t>>(rows, 1);
    h.hc = std::make_shared<std::vector<uint32_t>>(1, rows);
    h.her = std::make_shared<std::vector<uint32_t>>(rows, 1);
    h.hec = std::make_shared<std::vector<uint32_t>>(1, 0);
    h.nnzRows = rows; h.nnzCols = 1; h.maxHr = 1; h.maxHc = rows;
    h.rowsEq1 = rows; h.colsEq1 = (rows==1)?1:0;
    return h;
}