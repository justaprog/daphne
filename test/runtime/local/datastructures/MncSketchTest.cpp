/*
 * Tests for MNC sketch on CSRMatrix
 */

#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <runtime/local/datastructures/MncSketch.h>

#include <tags.h>
#include <catch.hpp>

#include <cstdint>
#include <vector>

// run ./test.sh -nb -d yes [datastructures] after building Daphne to execute this test
TEST_CASE("MNC sketch from CSRMatrix basic", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix:
    // [0 5 0
    //  0 0 3
    //  1 0 0]
    const size_t numRows     = 3;
    const size_t numCols     = 3;
    const size_t numNonZeros = 3;

    CSRMatrix<ValueType> *m =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, /*zero=*/true);

    ValueType *values   = m->getValues();
    size_t   *colIdxs   = m->getColIdxs();
    size_t   *rowOffsets= m->getRowOffsets();

    // Valid CSR: one nnz per row
    // rowOffsets: [0,1,2,3]
    rowOffsets[0] = 0;
    rowOffsets[1] = 1;
    rowOffsets[2] = 2;
    rowOffsets[3] = 3;

    // colIdxs: (0,1)->5, (1,2)->3, (2,0)->1
    colIdxs[0] = 1;
    colIdxs[1] = 2;
    colIdxs[2] = 0;

    values[0] = 5.0;
    values[1] = 3.0;
    values[2] = 1.0;

    MncSketch h = buildMncFromCsr(*m);

    // dimensions
    CHECK(h.m == numRows);
    CHECK(h.n == numCols);

    // row + col nnz
    std::vector<std::uint32_t> expectedHr{1,1,1};
    std::vector<std::uint32_t> expectedHc{1,1,1};
    CHECK(h.hr == expectedHr);
    CHECK(h.hc == expectedHc);

    // summary stats
    CHECK(h.maxHr    == 1);
    CHECK(h.maxHc    == 1);
    CHECK(h.nnzRows  == 3);
    CHECK(h.nnzCols  == 3);
    CHECK(h.rowsEq1  == 3);
    CHECK(h.colsEq1  == 3);
    CHECK(h.rowsGtHalf == 0); // 1 <= n/2 since n=3
    CHECK(h.colsGtHalf == 0); // 1 <= m/2 since m=3

    DataObjectFactory::destroy(m);
}

TEST_CASE("MNC sketch respects CSRMatrix sub-matrix view", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Original 4x3 matrix:
    // row 0: (0,1)
    // row 1: (1,2)
    // row 2: (2,0)
    // row 3: (3,2)  <- we'll slice rows [1,3) = rows 1 and 2
    const size_t numRowsOrig   = 4;
    const size_t numColsOrig   = 3;
    const size_t numNonZeros   = 4;

    CSRMatrix<ValueType> *mOrig =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsOrig, numColsOrig, numNonZeros, /*zero=*/true);

    ValueType *valuesOrig    = mOrig->getValues();
    size_t   *colIdxsOrig    = mOrig->getColIdxs();
    size_t   *rowOffsetsOrig = mOrig->getRowOffsets();

    rowOffsetsOrig[0] = 0;
    rowOffsetsOrig[1] = 1;
    rowOffsetsOrig[2] = 2;
    rowOffsetsOrig[3] = 3;
    rowOffsetsOrig[4] = 4;

    colIdxsOrig[0] = 1; // row 0
    colIdxsOrig[1] = 2; // row 1
    colIdxsOrig[2] = 0; // row 2
    colIdxsOrig[3] = 2; // row 3

    valuesOrig[0] = 1.0;
    valuesOrig[1] = 1.0;
    valuesOrig[2] = 1.0;
    valuesOrig[3] = 1.0;

    // Create sub-matrix with rows [1,3) = rows 1 and 2 of original
    CSRMatrix<ValueType> *mSub = DataObjectFactory::create<CSRMatrix<ValueType>>(mOrig, 1, 3);

    MncSketch hSub = buildMncFromCsr(*mSub);

    // submatrix is 2x3, with nnz rows = 2
    CHECK(hSub.m == 2);
    CHECK(hSub.n == 3);

    // each row in the submatrix has exactly 1 nnz
    std::vector<std::uint32_t> expectedHrSub{1,1};
    CHECK(hSub.hr == expectedHrSub);
    CHECK(hSub.nnzRows == 2);
    CHECK(hSub.rowsEq1 == 2);

    DataObjectFactory::destroy(mSub);
    DataObjectFactory::destroy(mOrig);
}

TEST_CASE("MNC Sketch example from paper", TAG_DATASTRUCTURES) {
    using ValueType = double;
    /* Matrix:
    [0,0,0,0,0,0,0,1,0], 
    [0,1,0,0,1,0,0,0,0], 
    [0,0,0,1,1,1,0,0,0], 
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [1,0,0,1,0,0,0,0,0],
    [0,0,1,0,1,0,1,0,0],
    [0,0,0,0,0,0,0,0,1],
    */
    const size_t numRows     = 9;
    const size_t numCols     = 9;
    const size_t numNonZeros = 14;

    CSRMatrix<ValueType> *m =
        DataObjectFactory::create<CSRMatrix<ValueType>>(numRows, numCols, numNonZeros, /*zero=*/true);

    ValueType *values   = m->getValues();
    size_t   *colIdxs   = m->getColIdxs();
    size_t   *rowOffsets= m->getRowOffsets();

    // Valid CSR: one nnz per row
    // rowOffsets: [ 0  1  3  6  6  7  8 10 13 14]
    rowOffsets[0] = 0;
    rowOffsets[1] = 1;
    rowOffsets[2] = 3;
    rowOffsets[3] = 6;
    rowOffsets[4] = 6;
    rowOffsets[5] = 7;
    rowOffsets[6] = 8;
    rowOffsets[7] = 10;
    rowOffsets[8] = 13;
    rowOffsets[9] = 14;

    // colIdxs: [7 1 4 3 4 5 2 7 0 3 2 4 6 8]
    colIdxs[0] = 7;
    colIdxs[1] = 1;
    colIdxs[2] = 4;
    colIdxs[3] = 3;
    colIdxs[4] = 4;
    colIdxs[5] = 5;
    colIdxs[6] = 2;
    colIdxs[7] = 7;
    colIdxs[8] = 0;
    colIdxs[9] = 3;
    colIdxs[10] = 2;
    colIdxs[11] = 4;
    colIdxs[12] = 6;
    colIdxs[13] = 8;
    
    for (size_t i = 0; i < numNonZeros; ++i)
        values[i] = 1.0;

    MncSketch h = buildMncFromCsr(*m);

    // dimensions
    CHECK(h.m == numRows);
    CHECK(h.n == numCols);

    // row + col nnz
    
    std::vector<std::uint32_t> expectedHr{1,2,3,0,1,1,2,3,1};
    std::vector<std::uint32_t> expectedHc{1,1,2,2,3,1,1,2,1};
    CHECK(h.hr == expectedHr);
    CHECK(h.hc == expectedHc);

    // her and hec
    std::vector<std::uint32_t> expectedHer{0,1,1,0,0,0,1,1,1};
    std::vector<std::uint32_t> expectedHec{0,0,1,0,0,0,0,2,1};
    std::vector<std::uint32_t> notexpectedHec{1,1,1,0,0,0,0,2,1};
    CHECK(h.her == expectedHer);
    CHECK(h.hec == expectedHec);

    DataObjectFactory::destroy(m);
}
// ============================================================================
// PROPAGATION TESTS (Append this to the end of the file)
// ============================================================================

TEST_CASE("MNC sketch propagation - basic product", TAG_DATASTRUCTURES) {
    // Create test sketches manually
    MncSketch hA;
    hA.m = 3;
    hA.n = 3;
    hA.hr = {1, 2, 1};
    hA.hc = {1, 2, 1};
    hA.updateStatistics();
    
    MncSketch hB;
    hB.m = 3;
    hB.n = 3;
    hB.hr = {2, 1, 2};
    hB.hc = {1, 3, 0};
    hB.updateStatistics();
    
    // Test product propagation with estimated sparsity 0.5
    // Total cells = 9. Est NNZ = 4.5.
    MncSketch hC = MncSketch::propagateProduct(hA, hB, 0.5);
    
    REQUIRE(hC.m == 3);
    REQUIRE(hC.n == 3);
    REQUIRE(hC.hr.size() == 3);
    REQUIRE(hC.hc.size() == 3);
    
    // Check totals roughly match expectation (probabilistic rounding handles the 0.5)
    std::uint64_t totalRows = std::accumulate(hC.hr.begin(), hC.hr.end(), 0ull);
    std::uint64_t totalCols = std::accumulate(hC.hc.begin(), hC.hc.end(), 0ull);
    
    // Total should be 4 or 5
    CHECK(totalRows >= 4);
    CHECK(totalRows <= 5);
    CHECK(totalCols >= 4);
    CHECK(totalCols <= 5);
}

TEST_CASE("MNC sketch propagation - transpose", TAG_DATASTRUCTURES) {
    MncSketch hA;
    hA.m = 2;
    hA.n = 3;
    hA.hr = {1, 2};
    hA.hc = {1, 0, 2};
    hA.updateStatistics();
    
    MncSketch hT = MncSketch::propagateTranspose(hA);
    
    REQUIRE(hT.m == 3);       // Rows become cols
    REQUIRE(hT.n == 2);       // Cols become rows
    CHECK(hT.hr == hA.hc);    // Row counts should match original col counts
    CHECK(hT.hc == hA.hr);    // Col counts should match original row counts
}

TEST_CASE("MNC sketch propagation - exact diagonal", TAG_DATASTRUCTURES) {
    // Create a diagonal sketch (Identity-like 3x3)
    MncSketch hDiag;
    hDiag.m = 3;
    hDiag.n = 3;
    hDiag.hr = {1, 1, 1};
    hDiag.hc = {1, 1, 1};
    hDiag.updateStatistics(); // Will detect isDiagonal = true
    
    // Create a regular sketch (3x4)
    MncSketch hReg;
    hReg.m = 3;
    hReg.n = 4;
    hReg.hr = {2, 1, 3};
    hReg.hc = {1, 2, 1, 2};
    hReg.updateStatistics();
    
    // Test exact propagation: Diagonal(3x3) * Regular(3x4) = Regular(3x4)
    MncSketch hResult = MncSketch::propagateProductExact(hDiag, hReg);
    
    // Output should mimic B (hReg) characteristics
    CHECK(hResult.m == hReg.m);
    CHECK(hResult.n == hReg.n);
    CHECK(hResult.hr == hReg.hr);
    CHECK(hResult.hc == hReg.hc);
}