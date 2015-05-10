/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
 
#if !defined(CUSPARSE_H_)
#define CUSPARSE_H_

#if defined(CUSPARSE_V2_H_)
#error "Only one of cusparse_v2.h or cusparse.h header files can be included in a single source file. Please see the CUSPARSE library User's Guide for more information."
#endif

#ifndef CUSPARSEAPI
#ifdef _WIN32
#define CUSPARSEAPI __stdcall
#else
#define CUSPARSEAPI 
#endif
#endif

#include "driver_types.h"
#include "cuComplex.h"   /* import complex data type */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/* CUSPARSE status type returns */
typedef enum{
    CUSPARSE_STATUS_SUCCESS=0,
    CUSPARSE_STATUS_NOT_INITIALIZED=1,
    CUSPARSE_STATUS_ALLOC_FAILED=2,
    CUSPARSE_STATUS_INVALID_VALUE=3,
    CUSPARSE_STATUS_ARCH_MISMATCH=4,
    CUSPARSE_STATUS_MAPPING_ERROR=5,
    CUSPARSE_STATUS_EXECUTION_FAILED=6,
    CUSPARSE_STATUS_INTERNAL_ERROR=7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8
} cusparseStatus_t;

/* Opaque structure holding CUSPARSE library context */
struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;

/* Opaque structure holding the matrix descriptor */
struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

/* Opaque structure holding the sparse triangular solve information */
struct cusparseSolveAnalysisInfo;
typedef struct cusparseSolveAnalysisInfo *cusparseSolveAnalysisInfo_t;

typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0, 
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,     
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2, 
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3 
} cusparseMatrixType_t;

typedef enum {
    CUSPARSE_FILL_MODE_LOWER = 0, 
    CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0, 
    CUSPARSE_DIAG_TYPE_UNIT = 1
} cusparseDiagType_t; 

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0, 
    CUSPARSE_INDEX_BASE_ONE = 1
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,  
    CUSPARSE_OPERATION_TRANSPOSE = 1,  
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2  
} cusparseOperation_t;

typedef enum {
    CUSPARSE_DIRECTION_ROW = 0,  
    CUSPARSE_DIRECTION_COLUMN = 1  
} cusparseDirection_t;

/* CUSPARSE initialization and managment routines */
cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t *handle);
cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int *version);
cusparseStatus_t CUSPARSEAPI cusparseSetKernelStream(cusparseHandle_t handle, cudaStream_t streamId); 

/* sparse matrix descriptor */
/* When the matrix descriptor is created, its fields are initialized to: 
   CUSPARSE_MATRIX_TYPE_GENERAL
   CUSPARSE_INDEX_BASE_ZERO
   All other fields are uninitialized
*/                                   
cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t *descrA);
cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr(cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(const cusparseMatDescr_t descrA);
 
cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType);
cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);

/* sparse traingular solve */
cusparseStatus_t CUSPARSEAPI cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info);



/* --- Sparse Level 1 routines --- */

/* Description: Addition of a scalar multiple of a sparse vector x  
   and a dense vector y. */ 
cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            float alpha, 
                                            const float *xVal, 
                                            const int *xInd, 
                                            float *y, 
                                            cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            double alpha, 
                                            const double *xVal, 
                                            const int *xInd, 
                                            double *y, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            cuComplex alpha, 
                                            const cuComplex *xVal, 
                                            const int *xInd, 
                                            cuComplex *y, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZaxpyi(cusparseHandle_t handle, 
                                            int nnz, 
                                            cuDoubleComplex alpha, 
                                            const cuDoubleComplex *xVal, 
                                            const int *xInd, 
                                            cuDoubleComplex *y, 
                                            cusparseIndexBase_t idxBase);

/* Description: dot product of a sparse vector x and a dense vector y. */
cusparseStatus_t CUSPARSEAPI cusparseSdoti(cusparseHandle_t handle,  
                                           int nnz, 
                                           const float *xVal, 
                                           const int *xInd, 
                                           const float *y,
                                           float *resultHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseDdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const double *xVal, 
                                           const int *xInd, 
                                           const double *y, 
                                           double *resultHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuComplex *xVal, 
                                           const int *xInd, 
                                           const cuComplex *y, 
                                           cuComplex *resultHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuDoubleComplex *xVal, 
                                           const int *xInd, 
                                           const cuDoubleComplex *y, 
                                           cuDoubleComplex *resultHostPtr,
                                           cusparseIndexBase_t idxBase);

/* Description: dot product of complex conjugate of a sparse vector x
   and a dense vector y. */
cusparseStatus_t CUSPARSEAPI cusparseCdotci(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuComplex *xVal, 
                                            const int *xInd, 
                                            const cuComplex *y, 
                                            cuComplex *resultHostPtr,
                                            cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseZdotci(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuDoubleComplex *xVal, 
                                            const int *xInd, 
                                            const cuDoubleComplex *y, 
                                            cuDoubleComplex *resultHostPtr,
                                            cusparseIndexBase_t idxBase);


/* Description: Gather of non-zero elements from dense vector y into 
   sparse vector x. */
cusparseStatus_t CUSPARSEAPI cusparseSgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const float *y, 
                                           float *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const double *y, 
                                           double *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuComplex *y, 
                                           cuComplex *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZgthr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuDoubleComplex *y, 
                                           cuDoubleComplex *xVal, 
                                           const int *xInd, 
                                           cusparseIndexBase_t idxBase);

/* Description: Gather of non-zero elements from desne vector y into 
   sparse vector x (also replacing these elements in y by zeros). */
cusparseStatus_t CUSPARSEAPI cusparseSgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            float *y, 
                                            float *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            double *y, 
                                            double *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            cuComplex *y, 
                                            cuComplex *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZgthrz(cusparseHandle_t handle, 
                                            int nnz, 
                                            cuDoubleComplex *y, 
                                            cuDoubleComplex *xVal, 
                                            const int *xInd, 
                                            cusparseIndexBase_t idxBase);

/* Description: Scatter of elements of the sparse vector x into 
   dense vector y. */
cusparseStatus_t CUSPARSEAPI cusparseSsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const float *xVal, 
                                           const int *xInd, 
                                           float *y, 
                                           cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const double *xVal, 
                                           const int *xInd, 
                                           double *y, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuComplex *xVal, 
                                           const int *xInd, 
                                           cuComplex *y, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZsctr(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuDoubleComplex *xVal, 
                                           const int *xInd, 
                                           cuDoubleComplex *y, 
                                           cusparseIndexBase_t idxBase);

/* Description: Givens rotation, where c and s are cosine and sine, 
   x and y are sparse and dense vectors, respectively. */
cusparseStatus_t CUSPARSEAPI cusparseSroti(cusparseHandle_t handle, 
                                           int nnz, 
                                           float *xVal, 
                                           const int *xInd, 
                                           float *y, 
                                           float c, 
                                           float s, 
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseDroti(cusparseHandle_t handle, 
                                           int nnz, 
                                           double *xVal, 
                                           const int *xInd, 
                                           double *y, 
                                           double c, 
                                           double s, 
                                           cusparseIndexBase_t idxBase);



/* --- Sparse Level 2 routines --- */

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */ 
cusparseStatus_t CUSPARSEAPI cusparseScsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            float alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const float *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const float *x, 
                                            float beta, 
                                            float *y);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            double alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const double *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const double *x, 
                                            double beta,  
                                            double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            cuComplex alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuComplex *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const cuComplex *x, 
                                            cuComplex beta, 
                                            cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            cuDoubleComplex alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuDoubleComplex *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const cuDoubleComplex *x, 
                                            cuDoubleComplex beta, 
                                            cuDoubleComplex *y);

/* Description: Solution of triangular linear system op(A) * y = alpha * x, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */ 
cusparseStatus_t CUSPARSEAPI cusparseScsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const float *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const double *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuComplex *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuDoubleComplex *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseScsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  float alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const float *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const float *x, 
                                                  float *y);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  double alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const double *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const double *x, 
                                                  double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  cuComplex alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuComplex *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuComplex *x, 
                                                  cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  cuDoubleComplex alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuDoubleComplex *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuDoubleComplex *x, 
                                                  cuDoubleComplex *y);
    

/* --- Sparse Level 3 routines --- */           
 
/* Description: Matrix-matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix, B and C are dense and usually tall matrices. */ 
cusparseStatus_t CUSPARSEAPI cusparseScsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            float alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const float  *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const float *B, 
                                            int ldb, 
                                            float beta, 
                                            float *C, 
                                            int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseDcsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            double alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const double *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const double *B, 
                                            int ldb, 
                                            double beta, 
                                            double *C, 
                                            int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseCcsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            cuComplex alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuComplex  *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const cuComplex *B, 
                                            int ldb, 
                                            cuComplex beta, 
                                            cuComplex *C, 
                                            int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseZcsrmm(cusparseHandle_t handle,
                                            cusparseOperation_t transA, 
                                            int m, 
                                            int n, 
                                            int k,  
                                            cuDoubleComplex alpha,
                                            const cusparseMatDescr_t descrA, 
                                            const cuDoubleComplex  *csrValA, 
                                            const int *csrRowPtrA, 
                                            const int *csrColIndA, 
                                            const cuDoubleComplex *B, 
                                            int ldb, 
                                            cuDoubleComplex beta, 
                                            cuDoubleComplex *C, 
                                            int ldc);
                


/* --- Sparse Format Conversion --- */

/* Description: This routine finds the total number of non-zero elements and 
   the number of non-zero elements per row or column in the dense matrix A. */
cusparseStatus_t CUSPARSEAPI cusparseSnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA, 
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const float *A, 
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseDnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const double *A, 
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseCnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const cuComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseZnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const cuDoubleComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalHostPtr);
                                                                                                        
/* Description: This routine converts a dense matrix to a sparse matrix 
   in the CSR storage format, using the information computed by the 
   nnz routine. */
cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                            
                                                const float *A, 
                                                int lda,
                                                const int *nnzPerRow,                                                 
                                                float *csrValA, 
                                                int *csrRowPtrA, 
                                                int *csrColIndA);
 
cusparseStatus_t CUSPARSEAPI cusparseDdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const double *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                double *csrValA, 
                                                int *csrRowPtrA, 
                                                int *csrColIndA);
    
cusparseStatus_t CUSPARSEAPI cusparseCdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const cuComplex *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                cuComplex *csrValA, 
                                                int *csrRowPtrA, 
                                                int *csrColIndA);
 
cusparseStatus_t CUSPARSEAPI cusparseZdense2csr(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                                    
                                                const cuDoubleComplex *A, 
                                                int lda, 
                                                const int *nnzPerRow,                                                 
                                                cuDoubleComplex *csrValA, 
                                                int *csrRowPtrA, 
                                                int *csrColIndA);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a dense matrix. */
cusparseStatus_t CUSPARSEAPI cusparseScsr2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,  
                                                const float *csrValA, 
                                                const int *csrRowPtrA, 
                                                const int *csrColIndA,
                                                float *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsr2dense(cusparseHandle_t handle, 
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const double *csrValA, 
                                                const int *csrRowPtrA, 
                                                const int *csrColIndA,
                                                double *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseCcsr2dense(cusparseHandle_t handle, 
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuComplex *csrValA, 
                                                const int *csrRowPtrA, 
                                                const int *csrColIndA,
                                                cuComplex *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseZcsr2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuDoubleComplex *csrValA, 
                                                const int *csrRowPtrA, 
                                                const int *csrColIndA,
                                                cuDoubleComplex *A, 
                                                int lda); 
                                 
/* Description: This routine converts a dense matrix to a sparse matrix 
   in the CSC storage format, using the information computed by the 
   nnz routine. */
cusparseStatus_t CUSPARSEAPI cusparseSdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                            
                                                const float *A, 
                                                int lda,
                                                const int *nnzPerCol,                                                 
                                                float *cscValA, 
                                                int *cscRowIndA, 
                                                int *cscColPtrA);
 
cusparseStatus_t CUSPARSEAPI cusparseDdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const double *A, 
                                                int lda,
                                                const int *nnzPerCol,                                                
                                                double *cscValA, 
                                                int *cscRowIndA, 
                                                int *cscColPtrA); 

cusparseStatus_t CUSPARSEAPI cusparseCdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,                                     
                                                const cuComplex *A, 
                                                int lda, 
                                                const int *nnzPerCol,                                                 
                                                cuComplex *cscValA, 
                                                int *cscRowIndA, 
                                                int *cscColPtrA);
    
cusparseStatus_t CUSPARSEAPI cusparseZdense2csc(cusparseHandle_t handle,
                                                int m, 
                                                int n,  
                                                const cusparseMatDescr_t descrA,                                    
                                                const cuDoubleComplex *A, 
                                                int lda, 
                                                const int *nnzPerCol,
                                                cuDoubleComplex *cscValA, 
                                                int *cscRowIndA, 
                                                int *cscColPtrA);

/* Description: This routine converts a sparse matrix in CSC storage format
   to a dense matrix. */
cusparseStatus_t CUSPARSEAPI cusparseScsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA,  
                                                const float *cscValA, 
                                                const int *cscRowIndA, 
                                                const int *cscColPtrA,
                                                float *A, 
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const double *cscValA, 
                                                const int *cscRowIndA, 
                                                const int *cscColPtrA,
                                                double *A, 
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseCcsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuComplex *cscValA, 
                                                const int *cscRowIndA, 
                                                const int *cscColPtrA,
                                                cuComplex *A, 
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseZcsc2dense(cusparseHandle_t handle,
                                                int m, 
                                                int n, 
                                                const cusparseMatDescr_t descrA, 
                                                const cuDoubleComplex *cscValA, 
                                                const int *cscRowIndA, 
                                                const int *cscColPtrA,
                                                cuDoubleComplex *A, 
                                                int lda);
    
/* Description: This routine compresses the indecis of rows or columns.
   It can be interpreted as a conversion from COO to CSR sparse storage
   format. */
cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle,
                                              const int *cooRowInd, 
                                              int nnz, 
                                              int m, 
                                              int *csrRowPtr, 
                                              cusparseIndexBase_t idxBase);
    
/* Description: This routine uncompresses the indecis of rows or columns.
   It can be interpreted as a conversion from CSR to COO sparse storage
   format. */
cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle,
                                              const int *csrRowPtr, 
                                              int nnz, 
                                              int m, 
                                              int *cooRowInd, 
                                              cusparseIndexBase_t idxBase);     
    
/* Description: This routine converts a matrix from CSR to CSC sparse 
   storage format. The resulting matrix can be re-interpreted as a 
   transpose of the original matrix in CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseScsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              const float  *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              float *cscValA, 
                                              int *cscRowIndA, 
                                              int *cscColPtrA, 
                                              int copyValues, 
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseDcsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n,
                                              const double  *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              double *cscValA, 
                                              int *cscRowIndA, 
                                              int *cscColPtrA,
                                              int copyValues, 
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n,
                                              const cuComplex  *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              cuComplex *cscValA, 
                                              int *cscRowIndA, 
                                              int *cscColPtrA, 
                                              int copyValues, 
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csc(cusparseHandle_t handle,
                                              int m, 
                                              int n, 
                                              const cuDoubleComplex *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              cuDoubleComplex *cscValA, 
                                              int *cscRowIndA, 
                                              int *cscColPtrA,
                                              int copyValues, 
                                              cusparseIndexBase_t idxBase);


#if defined(__cplusplus)
}
#endif /* __cplusplus */                         
                         
                         
#endif /* CUSPARSE_H_ */
