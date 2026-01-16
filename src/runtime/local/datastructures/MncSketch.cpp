// MncSketch.cpp
// Implementation of MNC sketch propagation methods

#include <runtime/local/datastructures/MncSketch.h>

#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

// ============================================================================
// Helper Methods
// ============================================================================

std::uint64_t MncSketch::totalCounts(const std::vector<std::uint32_t>& counts) {
    return std::accumulate(counts.begin(), counts.end(), 0ull);
}

std::vector<double> MncSketch::scaleCounts(const std::vector<std::uint32_t>& counts,
                                          double scaleFactor) {
    std::vector<double> scaled(counts.size());
    for (size_t i = 0; i < counts.size(); i++) {
        scaled[i] = static_cast<double>(counts[i]) * scaleFactor;
    }
    return scaled;
}

std::vector<std::uint32_t> MncSketch::probabilisticRound(const std::vector<double>& counts) {
    std::vector<std::uint32_t> rounded(counts.size());
    static std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for (size_t i = 0; i < counts.size(); i++) {
        double integerPart;
        double fractionalPart = std::modf(counts[i], &integerPart);
        
        if (dist(generator) < fractionalPart) {
            rounded[i] = static_cast<std::uint32_t>(integerPart) + 1;
        } else {
            rounded[i] = static_cast<std::uint32_t>(integerPart);
        }
    }
    return rounded;
}

// ============================================================================
// Core Propagation Methods
// ============================================================================

void MncSketch::updateStatistics() {
    // Reset statistics
    maxHr = 0;
    maxHc = 0;
    nnzRows = 0;
    nnzCols = 0;
    rowsEq1 = 0;
    colsEq1 = 0;
    rowsGtHalf = 0;
    colsGtHalf = 0;
    
    // Update row statistics
    for (size_t i = 0; i < m; i++) {
        auto cnt = hr[i];
        if (cnt > 0) {
            nnzRows++;
            if (cnt == 1) rowsEq1++;
            if (cnt > n / 2) rowsGtHalf++;
        }
        if (cnt > maxHr) maxHr = cnt;
    }
    
    // Update column statistics
    for (size_t j = 0; j < n; j++) {
        auto cnt = hc[j];
        if (cnt > 0) {
            nnzCols++;
            if (cnt == 1) colsEq1++;
            if (cnt > m / 2) colsGtHalf++;
        }
        if (cnt > maxHc) maxHc = cnt;
    }
    
    // For propagation, we don't recompute extended counts
    // (they require original matrix data)
    her.clear();
    hec.clear();
    
    // Simple diagonal check (assumes square matrix)
    isDiagonal = (m == n) && (totalCounts(hr) == m) && (totalCounts(hc) == n);
}

MncSketch MncSketch::propagateProduct(const MncSketch& hA,
                                     const MncSketch& hB,
                                     double estimatedSparsity) {
    // Validate dimensions
    if (hA.n != hB.m) {
        // Return empty sketch for invalid input
        return MncSketch{};
    }
    
    MncSketch hC;
    hC.m = hA.m;  // Output rows = rows of A
    hC.n = hB.n;  // Output cols = cols of B
    
    // Total estimated non-zeros in output
    double totalNNZ_est = estimatedSparsity * hC.m * hC.n;
    
    // Scale row counts from A (Equation 11)
    std::uint64_t totalRowA = totalCounts(hA.hr);
    double rowScale = (totalRowA > 0) ? totalNNZ_est / totalRowA : 0.0;
    auto scaledRows = scaleCounts(hA.hr, rowScale);
    
    // Scale column counts from B
    std::uint64_t totalColB = totalCounts(hB.hc);
    double colScale = (totalColB > 0) ? totalNNZ_est / totalColB : 0.0;
    auto scaledCols = scaleCounts(hB.hc, colScale);
    
    // Apply probabilistic rounding
    hC.hr = probabilisticRound(scaledRows);
    hC.hc = probabilisticRound(scaledCols);
    
    // Update statistics for the new sketch
    hC.updateStatistics();
    
    return hC;
}

MncSketch MncSketch::propagateProductExact(const MncSketch& hA,
                                          const MncSketch& hB) {
    // Case 1: B is diagonal and square -> output = A
    if (hB.isDiagonal && hB.m == hB.n && hA.n == hB.m) {
        return hA;
    }
    
    // Case 2: A is diagonal and square -> output = B
    if (hA.isDiagonal && hA.m == hA.n && hA.n == hB.m) {
        return hB;
    }
    
    // No exact propagation possible
    return MncSketch{};
}

MncSketch MncSketch::propagateTranspose(const MncSketch& hA) {
    MncSketch hC;
    hC.m = hA.n;  // Swap dimensions
    hC.n = hA.m;
    
    // Swap row and column counts
    hC.hr = hA.hc;
    hC.hc = hA.hr;
    
    // Swap extended counts if they exist
    if (!hA.her.empty() && !hA.hec.empty()) {
        hC.her = hA.hec;
        hC.hec = hA.her;
    }
    
    // Update statistics
    hC.updateStatistics();
    
    return hC;
}

// ============================================================================
// ESTIMATION METHODS (Algorithm 1)
// ============================================================================

double MncSketch::estimateSparsity(const MncSketch& hA, const MncSketch& hB) {
    // 1. Validation
    if (hA.n != hB.m) return 0.0;
    
    // 2. Exact Case (Theorem 3.1 from Paper)
    if (hA.maxHr <= 1 || hB.maxHc <= 1) {
        double exactNNZ = 0.0;
        for (size_t k = 0; k < hA.n; ++k) {
            exactNNZ += (double)hA.hc[k] * (double)hB.hr[k];
        }
        return exactNNZ / ((double)hA.m * hB.n);
    }
    
    // 3. Fallback Case (Average Case)
    double sA = (double)totalCounts(hA.hr) / (hA.m * hA.n);
    double sB = (double)totalCounts(hB.hr) / (hB.m * hB.n);
    
    double probZero = std::pow(1.0 - (sA * sB), (double)hA.n);
    return 1.0 - probZero;
}