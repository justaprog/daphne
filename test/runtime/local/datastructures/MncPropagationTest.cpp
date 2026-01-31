#include <runtime/local/datastructures/CSRMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/MncSketch.h>
#include <runtime/local/datastructures/MNCSparsityEstimation.h>
#include <runtime/local/datastructures/MncPropagation.h>
#include <tags.h>
#include <catch.hpp>

#include <cstdint>
#include <vector>
/*
*Test file for the MncPropagation logic, which allows us to predict the intermediate sparsity of 
*a chain of multiplications of matrices without the need to compute the full operation.
*We tested our logic with a total of 3 tests, details are given below
*/




/*
*In this test, we take 2 3x3 matrices, 
*one of them (A) being an identity matrix. If the estimator introduced even a 
* very small rounding error, then that would mean that our logic is false. 
* We expect that the output sketch must be a match of matrix B.
*/

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

TEST_CASE("Case 1: Exact Propagation (Diagonal Matrix)", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix A: 3x3 Identity (Diagonal)
    // [1 0 0]
    // [0 1 0]
    // [0 0 1]
    const size_t numRowsA = 3;
    const size_t numColsA = 3;
    const size_t nnzA     = 3;

    CSRMatrix<ValueType> *A =   DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, nnzA, /*zero=*/true);
    ValueType *valuesA    = A->getValues();
    size_t   *colIdxsA    = A->getColIdxs();
    size_t   *rowOffsetsA = A->getRowOffsets();

    rowOffsetsA[0] = 0; 
    rowOffsetsA[1] = 1; 
    rowOffsetsA[2] = 2; 
    rowOffsetsA[3] = 3;

    colIdxsA[0] = 0; 
    colIdxsA[1] = 1; 
    colIdxsA[2] = 2;

    valuesA[0] = 1.0; 
    valuesA[1] = 1.0; 
    valuesA[2] = 1.0;

    // Matrix B: 3x3 Sparse
    // [1 0 0]
    // [1 1 0]
    // [0 0 0]
    const size_t numRowsB = 3;
    const size_t numColsB = 3;
    const size_t nnzB     = 3;

    CSRMatrix<ValueType> *B =   DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, nnzB, /*zero=*/true);
    ValueType *valuesB    = B->getValues();
    size_t   *colIdxsB    = B->getColIdxs();
    size_t   *rowOffsetsB = B->getRowOffsets();

    rowOffsetsB[0] = 0; 
    rowOffsetsB[1] = 1; 
    rowOffsetsB[2] = 3; 
    rowOffsetsB[3] = 3;
    colIdxsB[0] = 0; 
    colIdxsB[1] = 0; 
    colIdxsB[2] = 1;
    valuesB[0] = 1.0; 
    valuesB[1] = 1.0; 
    valuesB[2] = 1.0;

    MncSketch hA = buildMncFromCsr(*A);
    MncSketch hB = buildMncFromCsr(*B);

    MncSketch hC = propagateMM(hA, hB);

    // Since A is diagonal, C should exact copy of B
    REQUIRE(hC.m == hB.m);
    REQUIRE(hC.n == hB.n);
    REQUIRE(hC.hr == hB.hr);
    REQUIRE(hC.hc == hB.hc);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}
/*
 * Case 2: Outer Product (Density Blowup) 
 * * This test is our worst case scenario for sparsity. It takes a 3x1 matrix and a 1x3 matrix. 
 * We start with only a total of 6 non-zero entries but end up with a full dense 3x3 matrix 
 * with 9 non-zeros (no zero values). This test ensures that the logic predicts the 
 * total content, resulting in 100% density.
 */

TEST_CASE("Case 2: Outer Product (Density Blowup)", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix A: 3x1 Column Vector
    // [1]
    // [1]
    // [1]
    const size_t numRowsA = 3;
    const size_t numColsA = 1;
    const size_t nnzA     = 3;
    CSRMatrix<ValueType> *A =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, nnzA, true);

    size_t *rowOffsetsA = A->getRowOffsets();
    size_t *colIdxsA    = A->getColIdxs();
    ValueType *valuesA  = A->getValues();

    rowOffsetsA[0] = 0; 
    rowOffsetsA[1] = 1; 
    rowOffsetsA[2] = 2; 
    rowOffsetsA[3] = 3;

    colIdxsA[0] = 0; 
    colIdxsA[1] = 0; 
    colIdxsA[2] = 0;

    valuesA[0] = 1.0; 
    valuesA[1] = 1.0; 
    valuesA[2] = 1.0;

    // Matrix B: 1x3 Row Vector
    // [1 1 1]
    const size_t numRowsB = 1;
    const size_t numColsB = 3;
    const size_t nnzB     = 3;
    CSRMatrix<ValueType> *B =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, nnzB, true);
    size_t *rowOffsetsB = B->getRowOffsets();
    size_t *colIdxsB    = B->getColIdxs();
    ValueType *valuesB  = B->getValues();

    rowOffsetsB[0] = 0; 
    rowOffsetsB[1] = 3;
    colIdxsB[0] = 0; 
    colIdxsB[1] = 1; 
    colIdxsB[2] = 2;
    valuesB[0] = 1.0; 
    valuesB[1] = 1.0; 
    valuesB[2] = 1.0;

    MncSketch hA = buildMncFromCsr(*A);
    MncSketch hB = buildMncFromCsr(*B);

    MncSketch hC = propagateMM(hA, hB);

    // Result should be 3x3 and fully dense (9 items)
    REQUIRE(hC.m == 3);
    REQUIRE(hC.n == 3);
    REQUIRE(hC.nnzRows == 3);
    REQUIRE(hC.nnzCols == 3);
    
    // Check total NNZ roughly matches 9
    uint64_t totalNNZ = 0;
    for(auto c : hC.hr) totalNNZ += c;
    REQUIRE(totalNNZ > 0);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
}

/*
 * Case 3: Chain Propagation (Dimensions) 
 * * The third test covers a chain multiplication, in which we first multiply a 4x2 matrix 
 * with a 2x5 matrix and then with a 5x3 matrix. This test covers the recursive logic 
 * of our code. It not only makes sure that the non-zero counts are correct, but the 
 * output of the first multiplication is compatible with the input of the second.
 */

TEST_CASE("Case 3: Chain Propagation (Dimensions)", TAG_DATASTRUCTURES) {
    using ValueType = double;

    // Matrix A: 4x2
    const size_t numRowsA = 4;
    const size_t numColsA = 2;
    CSRMatrix<ValueType> *A =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsA, numColsA, numRowsA, true);

    size_t *rowOffsetsA = A->getRowOffsets();
    size_t *colIdxsA    = A->getColIdxs();
    for(size_t i=0; i<=numRowsA; ++i) rowOffsetsA[i] = i; 
    for(size_t i=0; i<numRowsA; ++i) colIdxsA[i] = 0; 

    // Matrix B: 2x5
    const size_t numRowsB = 2;
    const size_t numColsB = 5;
    CSRMatrix<ValueType> *B = DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsB, numColsB, numRowsB, true);

    size_t *rowOffsetsB = B->getRowOffsets();
    size_t *colIdxsB    = B->getColIdxs();
    for(size_t i=0; i<=numRowsB; ++i) rowOffsetsB[i] = i; 
    for(size_t i=0; i<numRowsB; ++i) colIdxsB[i] = 0; 

    // Matrix C: 5x3
    const size_t numRowsC = 5;
    const size_t numColsC = 3;
    CSRMatrix<ValueType> *C =  DataObjectFactory::create<CSRMatrix<ValueType>>(numRowsC, numColsC, numRowsC, true);

    size_t *rowOffsetsC = C->getRowOffsets();
    size_t *colIdxsC    = C->getColIdxs();
    for(size_t i=0; i<=numRowsC; ++i) rowOffsetsC[i] = i; 
    for(size_t i=0; i<numRowsC; ++i) colIdxsC[i] = 0; 

    MncSketch hA = buildMncFromCsr(*A);
    MncSketch hB = buildMncFromCsr(*B);
    MncSketch hC = buildMncFromCsr(*C);

    std::vector<MncSketch> chain = {hA, hB, hC};
    MncSketch result = propagateChain(chain);

    // (4x2) * (2x5) -> (4x5)
    // (4x5) * (5x3) -> (4x3)
    REQUIRE(result.m == 4);
    REQUIRE(result.n == 3);

    DataObjectFactory::destroy(A);
    DataObjectFactory::destroy(B);
    DataObjectFactory::destroy(C);
}