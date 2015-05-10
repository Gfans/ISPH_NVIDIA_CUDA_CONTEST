/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
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
 
#if !defined(CUSPARSE_V2_H_)
#define CUSPARSE_V2_H_

#if defined(CUSPARSE_H_) 
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

/* Opaque structure holding the hybrid (HYB) storage information */
struct cusparseHybMat;
typedef struct cusparseHybMat *cusparseHybMat_t;

/* Types definitions */
typedef enum { 
    CUSPARSE_POINTER_MODE_HOST = 0,  
    CUSPARSE_POINTER_MODE_DEVICE = 1        
} cusparsePointerMode_t;

typedef enum { 
    CUSPARSE_ACTION_SYMBOLIC = 0,  
    CUSPARSE_ACTION_NUMERIC = 1        
} cusparseAction_t;

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

typedef enum {
    CUSPARSE_HYB_PARTITION_AUTO = 0,  // automatically decide how to split the data into regular/irregular part
    CUSPARSE_HYB_PARTITION_USER = 1,  // store data into regular part up to a user specified treshhold
    CUSPARSE_HYB_PARTITION_MAX = 2,   // store all data in the regular part
} cusparseHybPartition_t;

/* CUSPARSE initialization and managment routines */
cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t *handle);
cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int *version);
cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId); 

/* CUSPARSE type creation, destruction, set and get routines */
cusparseStatus_t CUSPARSEAPI cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t *mode);
cusparseStatus_t CUSPARSEAPI cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);

/* sparse matrix descriptor */
/* When the matrix descriptor is created, its fields are initialized to: 
   CUSPARSE_MATRIX_TYPE_GENERAL
   CUSPARSE_INDEX_BASE_ZERO
   All other fields are uninitialized
*/                                   
cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t *descrA);
cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr (cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(const cusparseMatDescr_t descrA);
 
cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t  descrA, cusparseDiagType_t diagType);
cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(const cusparseMatDescr_t descrA);

cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);

/* sparse triangular solve */
cusparseStatus_t CUSPARSEAPI cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t *info);
cusparseStatus_t CUSPARSEAPI cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info);
cusparseStatus_t CUSPARSEAPI cusparseGetLevelInfo(cusparseHandle_t handle, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  int *nlevels, 
                                                  int **levelPtr, 
                                                  int **levelInd);

/* hybrid (HYB) format */
cusparseStatus_t CUSPARSEAPI cusparseCreateHybMat(cusparseHybMat_t *hybA);
cusparseStatus_t CUSPARSEAPI cusparseDestroyHybMat(cusparseHybMat_t hybA);


/* --- Sparse Level 1 routines --- */

/* Description: Addition of a scalar multiple of a sparse vector x  
   and a dense vector y. */ 
cusparseStatus_t CUSPARSEAPI cusparseSaxpyi_v2(cusparseHandle_t handle, 
                                               int nnz, 
                                               const float *alpha, 
                                               const float *xVal, 
                                               const int *xInd, 
                                               float *y, 
                                               cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDaxpyi_v2(cusparseHandle_t handle, 
                                               int nnz, 
                                               const double *alpha, 
                                               const double *xVal, 
                                               const int *xInd, 
                                               double *y, 
                                               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCaxpyi_v2(cusparseHandle_t handle, 
                                               int nnz, 
                                               const cuComplex *alpha, 
                                               const cuComplex *xVal, 
                                               const int *xInd, 
                                               cuComplex *y, 
                                               cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZaxpyi_v2(cusparseHandle_t handle, 
                                               int nnz, 
                                               const cuDoubleComplex *alpha, 
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
                                           float *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseDdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const double *xVal, 
                                           const int *xInd, 
                                           const double *y, 
                                           double *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuComplex *xVal, 
                                           const int *xInd, 
                                           const cuComplex *y, 
                                           cuComplex *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZdoti(cusparseHandle_t handle, 
                                           int nnz, 
                                           const cuDoubleComplex *xVal, 
                                           const int *xInd, 
                                           const cuDoubleComplex *y, 
                                           cuDoubleComplex *resultDevHostPtr,
                                           cusparseIndexBase_t idxBase);

/* Description: dot product of complex conjugate of a sparse vector x
   and a dense vector y. */
cusparseStatus_t CUSPARSEAPI cusparseCdotci(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuComplex *xVal, 
                                            const int *xInd, 
                                            const cuComplex *y, 
                                            cuComplex *resultDevHostPtr,
                                            cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseZdotci(cusparseHandle_t handle, 
                                            int nnz, 
                                            const cuDoubleComplex *xVal, 
                                            const int *xInd, 
                                            const cuDoubleComplex *y, 
                                            cuDoubleComplex *resultDevHostPtr,
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
cusparseStatus_t CUSPARSEAPI cusparseSroti_v2(cusparseHandle_t handle, 
                                              int nnz, 
                                              float *xVal, 
                                              const int *xInd, 
                                              float *y, 
                                              const float *c, 
                                              const float *s, 
                                              cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseDroti_v2(cusparseHandle_t handle, 
                                              int nnz, 
                                              double *xVal, 
                                              const int *xInd, 
                                              double *y, 
                                              const double *c, 
                                              const double *s, 
                                              cusparseIndexBase_t idxBase);


/* --- Sparse Level 2 routines --- */

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */ 
cusparseStatus_t CUSPARSEAPI cusparseScsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int nnz,
                                               const float *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const float *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const float *x, 
                                               const float *beta, 
                                               float *y);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int nnz,
                                               const double *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const double *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const double *x, 
                                               const double *beta,  
                                               double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n,
                                               int nnz,
                                               const cuComplex *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const cuComplex *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const cuComplex *x, 
                                               const cuComplex *beta, 
                                               cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmv_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int nnz,
                                               const cuDoubleComplex *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const cuDoubleComplex *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const cuDoubleComplex *x, 
                                               const cuDoubleComplex *beta, 
                                               cuDoubleComplex *y);   

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */    
cusparseStatus_t CUSPARSEAPI cusparseShybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const float *x,
                                            const float *beta,
                                            float *y);

cusparseStatus_t CUSPARSEAPI cusparseDhybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const double *x,
                                            const double *beta,
                                            double *y);

cusparseStatus_t CUSPARSEAPI cusparseChybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const cuComplex *x,
                                            const cuComplex *beta,
                                            cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZhybmv(cusparseHandle_t handle,
                                            cusparseOperation_t transA,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cusparseHybMat_t hybA,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);
    
/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in BSR storage format, x and y are dense vectors. */
cusparseStatus_t CUSPARSEAPI cusparseSbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const float *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const float *x,
                                            const float *beta,
                                            float *y);

cusparseStatus_t CUSPARSEAPI cusparseDbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const double *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const double *x,
                                            const double *beta,
                                            double *y);

cusparseStatus_t CUSPARSEAPI cusparseCbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const cuComplex *x,
                                            const cuComplex *beta,
                                            cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZbsrmv(cusparseHandle_t handle,
                                            cusparseDirection_t dirA,
                                            cusparseOperation_t transA,
                                            int mb,
                                            int nb,
                                            int nnzb,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex *bsrValA,
                                            const int *bsrRowPtrA,
                                            const int *bsrColIndA,
                                            int  blockDim,
                                            const cuDoubleComplex *x,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *y);

/* Description: Matrix-vector multiplication  y = alpha * op(A) * x  + beta * y, 
   where A is a sparse matrix in extended BSR storage format, x and y are dense 
   vectors. */
cusparseStatus_t CUSPARSEAPI cusparseSbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const float *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const float *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const float *x,
                                             const float *beta,
                                             float *y);


cusparseStatus_t CUSPARSEAPI cusparseDbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const double *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const double *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const double *x,
                                             const double *beta,
                                             double *y);
    
cusparseStatus_t CUSPARSEAPI cusparseCbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const cuComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuComplex *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const cuComplex *x,
                                             const cuComplex *beta,
                                             cuComplex *y);


cusparseStatus_t CUSPARSEAPI cusparseZbsrxmv(cusparseHandle_t handle,
                                             cusparseDirection_t dirA,
                                             cusparseOperation_t transA,
                                             int sizeOfMask,
                                             int mb,
                                             int nb,
                                             int nnzb,
                                             const cuDoubleComplex *alpha,
                                             const cusparseMatDescr_t descrA,
                                             const cuDoubleComplex *bsrValA,
                                             const int *bsrMaskPtrA,
                                             const int *bsrRowPtrA,
                                             const int *bsrEndPtrA,
                                             const int *bsrColIndA,
                                             int  blockDim,
                                             const cuDoubleComplex *x,
                                             const cuDoubleComplex *beta,
                                             cuDoubleComplex *y);

/* Description: Solution of triangular linear system op(A) * y = alpha * x, 
   where A is a sparse matrix in CSR storage format, x and y are dense vectors. */     
cusparseStatus_t CUSPARSEAPI cusparseScsrsv_analysis_v2(cusparseHandle_t handle, 
                                                        cusparseOperation_t transA, 
                                                        int m, 
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA, 
                                                        const float *csrValA, 
                                                        const int *csrRowPtrA, 
                                                        const int *csrColIndA, 
                                                        cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv_analysis_v2(cusparseHandle_t handle, 
                                                        cusparseOperation_t transA, 
                                                        int m, 
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA, 
                                                        const double *csrValA, 
                                                        const int *csrRowPtrA, 
                                                        const int *csrColIndA, 
                                                        cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv_analysis_v2(cusparseHandle_t handle, 
                                                        cusparseOperation_t transA, 
                                                        int m, 
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA, 
                                                        const cuComplex *csrValA, 
                                                        const int *csrRowPtrA, 
                                                        const int *csrColIndA, 
                                                        cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv_analysis_v2(cusparseHandle_t handle, 
                                                        cusparseOperation_t transA, 
                                                        int m, 
                                                        int nnz,
                                                        const cusparseMatDescr_t descrA, 
                                                        const cuDoubleComplex *csrValA, 
                                                        const int *csrRowPtrA, 
                                                        const int *csrColIndA, 
                                                        cusparseSolveAnalysisInfo_t info); 


cusparseStatus_t CUSPARSEAPI cusparseScsrsv_solve_v2(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m,
                                                     const float *alpha, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const float *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info, 
                                                     const float *x, 
                                                     float *y);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv_solve_v2(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     const double *alpha, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const double *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info, 
                                                     const double *x, 
                                                     double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv_solve_v2(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     const cuComplex *alpha, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuComplex *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info, 
                                                     const cuComplex *x, 
                                                     cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv_solve_v2(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     const cuDoubleComplex *alpha, 
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuDoubleComplex *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info, 
                                                     const cuDoubleComplex *x, 
                                                     cuDoubleComplex *y);      

/* Description: Solution of triangular linear system op(A) * y = alpha * x, 
   where A is a sparse matrix in HYB storage format, x and y are dense vectors. */
cusparseStatus_t CUSPARSEAPI cusparseShybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDhybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);
    
cusparseStatus_t CUSPARSEAPI cusparseChybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZhybsv_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     const cusparseMatDescr_t descrA, 
                                                     cusparseHybMat_t hybA,
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseShybsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t trans, 
                                                  const float *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const float *x,
                                                  float *y);

cusparseStatus_t CUSPARSEAPI cusparseChybsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t trans,
                                                  const cuComplex *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuComplex *x,
                                                  cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseDhybsv_solve(cusparseHandle_t handle,
                                                  cusparseOperation_t trans,
                                                  const double *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA, 
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const double *x,
                                                  double *y);

cusparseStatus_t CUSPARSEAPI cusparseZhybsv_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t trans,
                                                  const cuDoubleComplex *alpha, 
                                                  const cusparseMatDescr_t descra,
                                                  const cusparseHybMat_t hybA,
                                                  cusparseSolveAnalysisInfo_t info,
                                                  const cuDoubleComplex *x,
                                                  cuDoubleComplex *y);


/* --- Sparse Level 3 routines --- */           
 
/* Description: Matrix-matrix multiplication C = alpha * op(A) * B  + beta * C, 
   where A is a sparse matrix, B and C are dense and usually tall matrices. */                 
cusparseStatus_t CUSPARSEAPI cusparseScsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int k,  
                                               int nnz,
                                               const float *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const float  *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const float *B, 
                                               int ldb, 
                                               const float *beta, 
                                               float *C, 
                                               int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseDcsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int k,  
                                               int nnz,
                                               const double *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const double *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const double *B, 
                                               int ldb, 
                                               const double *beta, 
                                               double *C, 
                                               int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseCcsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int k,  
                                               int nnz,
                                               const cuComplex *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const cuComplex  *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const cuComplex *B, 
                                               int ldb, 
                                               const cuComplex *beta, 
                                               cuComplex *C, 
                                               int ldc);
                     
cusparseStatus_t CUSPARSEAPI cusparseZcsrmm_v2(cusparseHandle_t handle,
                                               cusparseOperation_t transA, 
                                               int m, 
                                               int n, 
                                               int k,  
                                               int nnz,
                                               const cuDoubleComplex *alpha,
                                               const cusparseMatDescr_t descrA, 
                                               const cuDoubleComplex  *csrValA, 
                                               const int *csrRowPtrA, 
                                               const int *csrColIndA, 
                                               const cuDoubleComplex *B, 
                                               int ldb, 
                                               const cuDoubleComplex *beta, 
                                               cuDoubleComplex *C, 
                                               int ldc);    


cusparseStatus_t CUSPARSEAPI cusparseScsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const float *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const float *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const float *B,
                                            int ldb,
                                            const float *beta,
                                            float *C,
                                            int ldc);

cusparseStatus_t CUSPARSEAPI cusparseDcsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const double *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const double *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const double *B,
                                            int ldb,
                                            const double *beta,
                                            double *C,
                                            int ldc);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const cuComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const cuComplex *B,
                                            int ldb,
                                            const cuComplex *beta,
                                            cuComplex *C,
                                            int ldc);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmm2(cusparseHandle_t handle,
                                            cusparseOperation_t transa,
                                            cusparseOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            int nnz,
                                            const cuDoubleComplex *alpha,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex *csrValA,
                                            const int *csrRowPtrA,
                                            const int *csrColIndA,
                                            const cuDoubleComplex *B,
                                            int ldb,
                                            const cuDoubleComplex *beta,
                                            cuDoubleComplex *C,
                                            int ldc);


/* Description: Solution of triangular linear system op(A) * Y = alpha * X, 
   with multiple right-hand-sides, where A is a sparse matrix in CSR storage 
   format, X and Y are dense and usually tall matrices. */
cusparseStatus_t CUSPARSEAPI cusparseScsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const float *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const double *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuComplex *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm_analysis(cusparseHandle_t handle, 
                                                     cusparseOperation_t transA, 
                                                     int m, 
                                                     int nnz,
                                                     const cusparseMatDescr_t descrA, 
                                                     const cuDoubleComplex *csrValA, 
                                                     const int *csrRowPtrA, 
                                                     const int *csrColIndA, 
                                                     cusparseSolveAnalysisInfo_t info); 


cusparseStatus_t CUSPARSEAPI cusparseScsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m,
                                                  int n,
                                                  const float *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const float *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const float *x, 
                                                  int ldx,
                                                  float *y,
                                                  int ldy);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const double *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const double *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const double *x, 
                                                  int ldx,
                                                  double *y,
                                                  int ldy);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const cuComplex *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuComplex *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuComplex *x,
                                                  int ldx,
                                                  cuComplex *y,
                                                  int ldy);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsm_solve(cusparseHandle_t handle, 
                                                  cusparseOperation_t transA, 
                                                  int m, 
                                                  int n,
                                                  const cuDoubleComplex *alpha, 
                                                  const cusparseMatDescr_t descrA, 
                                                  const cuDoubleComplex *csrValA, 
                                                  const int *csrRowPtrA, 
                                                  const int *csrColIndA, 
                                                  cusparseSolveAnalysisInfo_t info, 
                                                  const cuDoubleComplex *x,
                                                  int ldx,
                                                  cuDoubleComplex *y,
                                                  int ldy);                                                                 
                    
/* --- Preconditioners --- */ 

/* Description: Compute the incomplete-LU factorization with 0 fill-in (ILU0)
   based on the information in the opaque structure info that was obtained 
   from the analysis phase (csrsv_analysis). */
cusparseStatus_t CUSPARSEAPI cusparseScsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              float *csrValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              double *csrValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuComplex *csrValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuDoubleComplex *csrValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

/* Description: Compute the incomplete-Cholesky factorization with 0 fill-in (IC0)
   based on the information in the opaque structure info that was obtained 
   from the analysis phase (csrsv_analysis). */
cusparseStatus_t CUSPARSEAPI cusparseScsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA,
                                              float *csrValA_ValM,
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */ 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseDcsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              double *csrValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseCcsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuComplex *csrValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              cusparseSolveAnalysisInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseZcsric0(cusparseHandle_t handle, 
                                              cusparseOperation_t trans, 
                                              int m, 
                                              const cusparseMatDescr_t descrA, 
                                              cuDoubleComplex *csrValA_ValM, 
                                              /* matrix A values are updated inplace 
                                                 to be the preconditioner M values */
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA, 
                                              cusparseSolveAnalysisInfo_t info);


/* Description: Solution of tridiagonal linear system A * B = B, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (dl), main (d) and upper (du) diagonals, and 
   the right-hand-sides B are overwritten with the solution. 
   These routine use pivoting */
cusparseStatus_t cusparseSgtsv(cusparseHandle_t handle,
                               int m,        
                               int n,        
                               const float *dl, 
                               const float  *d,   
                               const float *du, 
                               float *B,    
                               int ldb);
                                 
cusparseStatus_t cusparseDgtsv(cusparseHandle_t handle,
                               int m,        
                               int n,       
                               const double *dl,  
                               const double  *d,   
                               const double *du, 
                               double *B,    
                               int ldb);
                                                                 
cusparseStatus_t cusparseCgtsv(cusparseHandle_t handle,
                               int m,        
                               int n,       
                               const cuComplex *dl, 
                               const cuComplex  *d,  
                               const cuComplex *du, 
                               cuComplex *B,     
                               int ldb);

cusparseStatus_t cusparseZgtsv(cusparseHandle_t handle,
                               int m,        
                               int n,       
                               const cuDoubleComplex *dl,  
                               const cuDoubleComplex  *d,  
                               const cuDoubleComplex *du,
                               cuDoubleComplex *B,     
                               int ldb);
/* Description: Solution of tridiagonal linear system A * B = B, 
   with multiple right-hand-sides. The coefficient matrix A is 
   composed of lower (dl), main (d) and upper (du) diagonals, and 
   the right-hand-sides B are overwritten with the solution. 
   These routines do not use pivoting, using a combination of PCR and CR algorithm */                               
cusparseStatus_t cusparseSgtsv_nopivot(cusparseHandle_t handle,
                               int m,        
                               int n,        
                               const float *dl, 
                               const float  *d,   
                               const float *du, 
                               float *B,    
                               int ldb);
                                 
cusparseStatus_t cusparseDgtsv_nopivot(cusparseHandle_t handle,
                               int m,        
                               int n,       
                               const double *dl,  
                               const double  *d,   
                               const double *du, 
                               double *B,    
                               int ldb);
                                                                 
cusparseStatus_t cusparseCgtsv_nopivot(cusparseHandle_t handle,
                               int m,        
                               int n,       
                               const cuComplex *dl, 
                               const cuComplex  *d,  
                               const cuComplex *du, 
                               cuComplex *B,     
                               int ldb);

cusparseStatus_t cusparseZgtsv_nopivot(cusparseHandle_t handle,
                               int m,        
                               int n,       
                               const cuDoubleComplex *dl,  
                               const cuDoubleComplex  *d,  
                               const cuDoubleComplex *du,
                               cuDoubleComplex *B,     
                               int ldb);                               
                                  
/* Description: Solution of a set of tridiagonal linear systems 
   A * x = x, each with a single right-hand-side. The coefficient 
   matrices A are composed of lower (dl), main (d) and upper (du) 
   diagonals and stored separated by a batchStride, while the 
   right-hand-sides x are also separated by a batchStride. */
cusparseStatus_t cusparseSgtsvStridedBatch(cusparseHandle_t handle,
                                           int m, 
                                           const float *dl,
                                           const float  *d,
                                           const float *du,
                                           float *x,
                                           int batchCount,
                                           int batchStride);
                                        
                                        
cusparseStatus_t cusparseDgtsvStridedBatch(cusparseHandle_t handle,
                                           int m, 
                                           const double *dl,
                                           const double  *d,
                                           const double *du,
                                           double *x,
                                           int batchCount,
                                           int batchStride);
                                        
cusparseStatus_t cusparseCgtsvStridedBatch(cusparseHandle_t handle,
                                           int m, 
                                           const cuComplex *dl,
                                           const cuComplex  *d,
                                           const cuComplex *du,
                                           cuComplex *x,
                                           int batchCount,
                                           int batchStride);
                                        
cusparseStatus_t cusparseZgtsvStridedBatch(cusparseHandle_t handle,
                                           int m, 
                                           const cuDoubleComplex *dl,
                                           const cuDoubleComplex  *d,
                                           const cuDoubleComplex *du,
                                           cuDoubleComplex *x,
                                           int batchCount,
                                           int batchStride);                                        
                                         
/* --- Extra --- */ 

/* Description: This routine computes a sparse matrix that results from 
   multiplication of two sparse matrices. */                                              
cusparseStatus_t CUSPARSEAPI cusparseXcsrgemmNnz(cusparseHandle_t handle,
                                                 cusparseOperation_t transA, 
                                                 cusparseOperation_t transB, 
                                                 int m, 
                                                 int n, 
                                                 int k, 
                                                 const cusparseMatDescr_t descrA,
                                                 const int nnzA,
                                                 const int *csrRowPtrA, 
                                                 const int *csrColIndA,     
                                                 const cusparseMatDescr_t descrB,
                                                 const int nnzB,                                                                                              
                                                 const int *csrRowPtrB, 
                                                 const int *csrColIndB,  
                                                 const cusparseMatDescr_t descrC,                                                
                                                 int *csrRowPtrC, 
                                                 int *nnzTotalDevHostPtr);                                              
                                              
cusparseStatus_t CUSPARSEAPI cusparseScsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              const int nnzA,      
                                              const float *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              const int nnzB,                                                    
                                              const float *csrValB, 
                                              const int *csrRowPtrB, 
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              float *csrValC, 
                                              const int *csrRowPtrC, 
                                              int *csrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,      
                                              const double *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const double *csrValB, 
                                              const int *csrRowPtrB, 
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              double *csrValC, 
                                              const int *csrRowPtrC, 
                                              int *csrColIndC);
                                              
cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,      
                                              const cuComplex *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const cuComplex *csrValB, 
                                              const int *csrRowPtrB, 
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              cuComplex *csrValC, 
                                              const int *csrRowPtrC, 
                                              int *csrColIndC); 
                                              
cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm(cusparseHandle_t handle,
                                              cusparseOperation_t transA, 
                                              cusparseOperation_t transB, 
                                              int m, 
                                              int n, 
                                              int k, 
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,      
                                              const cuDoubleComplex *csrValA, 
                                              const int *csrRowPtrA, 
                                              const int *csrColIndA,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,                                                    
                                              const cuDoubleComplex *csrValB, 
                                              const int *csrRowPtrB, 
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC, 
                                              cuDoubleComplex *csrValC, 
                                              const int *csrRowPtrC, 
                                              int *csrColIndC);

/* Description: This routine computes a sparse matrix that results from 
   addition of two sparse matrices. */
cusparseStatus_t CUSPARSEAPI cusparseXcsrgeamNnz(cusparseHandle_t handle,
                                                 int m,
                                                 int n,
                                                 const cusparseMatDescr_t descrA,
                                                 int nnzA,
                                                 const int *csrRowPtrA,
                                                 const int *csrColIndA,
                                                 const cusparseMatDescr_t descrB,
                                                 int nnzB,
                                                 const int *csrRowPtrB,
                                                 const int *csrColIndB,
                                                 const cusparseMatDescr_t descrC,
                                                 int *csrRowPtrC,
                                                 int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseScsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const float *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const float *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const float *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              float *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const double *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const double *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const double *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              double *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);
    
cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cuComplex *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cuComplex *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuComplex *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);
    
cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cuDoubleComplex *alpha,
                                              const cusparseMatDescr_t descrA,
                                              int nnzA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              const cuDoubleComplex *beta,
                                              const cusparseMatDescr_t descrB,
                                              int nnzB,
                                              const cuDoubleComplex *csrValB,
                                              const int *csrRowPtrB,
                                              const int *csrColIndB,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

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
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseDnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const double *A, 
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseCnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const cuComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseZnnz(cusparseHandle_t handle, 
                                          cusparseDirection_t dirA,  
                                          int m, 
                                          int n, 
                                          const cusparseMatDescr_t  descrA,
                                          const cuDoubleComplex *A,
                                          int lda, 
                                          int *nnzPerRowCol, 
                                          int *nnzTotalDevHostPtr);
                                                                                                        
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
cusparseStatus_t CUSPARSEAPI cusparseScsr2csc_v2(cusparseHandle_t handle,
                                                 int m, 
                                                 int n, 
                                                 int nnz,
                                                 const float  *csrVal, 
                                                 const int *csrRowPtr, 
                                                 const int *csrColInd, 
                                                 float *cscVal, 
                                                 int *cscRowInd, 
                                                 int *cscColPtr, 
                                                 cusparseAction_t copyValues, 
                                                 cusparseIndexBase_t idxBase);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsr2csc_v2(cusparseHandle_t handle,
                                                 int m, 
                                                 int n,
                                                 int nnz,
                                                 const double  *csrVal, 
                                                 const int *csrRowPtr, 
                                                 const int *csrColInd,
                                                 double *cscVal, 
                                                 int *cscRowInd, 
                                                 int *cscColPtr,
                                                 cusparseAction_t copyValues, 
                                                 cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2csc_v2(cusparseHandle_t handle,
                                                 int m, 
                                                 int n,
                                                 int nnz,
                                                 const cuComplex  *csrVal, 
                                                 const int *csrRowPtr, 
                                                 const int *csrColInd,
                                                 cuComplex *cscVal, 
                                                 int *cscRowInd, 
                                                 int *cscColPtr, 
                                                 cusparseAction_t copyValues, 
                                                 cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2csc_v2(cusparseHandle_t handle,
                                                 int m, 
                                                 int n, 
                                                 int nnz,
                                                 const cuDoubleComplex *csrVal, 
                                                 const int *csrRowPtr, 
                                                 const int *csrColInd, 
                                                 cuDoubleComplex *cscVal, 
                                                 int *cscRowInd, 
                                                 int *cscColPtr,
                                                 cusparseAction_t copyValues, 
                                                 cusparseIndexBase_t idxBase);
                                                     
/* Description: This routine converts a dense matrix to a sparse matrix 
   in HYB storage format. */
cusparseStatus_t CUSPARSEAPI cusparseSdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const float *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseDdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const double *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseCdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseZdense2hyb(cusparseHandle_t handle,
                                                int m,
                                                int n,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex *A,
                                                int lda,
                                                const int *nnzPerRow,
                                                cusparseHybMat_t hybA,
                                                int userEllWidth,
                                                cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a dense matrix. */
cusparseStatus_t CUSPARSEAPI cusparseShyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                float *A,
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseDhyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                double *A,
                                                int lda);
    
cusparseStatus_t CUSPARSEAPI cusparseChyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                cuComplex *A,
                                                int lda);

cusparseStatus_t CUSPARSEAPI cusparseZhyb2dense(cusparseHandle_t handle,
                                                const cusparseMatDescr_t descrA,
                                                const cusparseHybMat_t hybA,
                                                cuDoubleComplex *A,
                                                int lda);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a sparse matrix in HYB storage format. */
cusparseStatus_t CUSPARSEAPI cusparseScsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);
    
cusparseStatus_t CUSPARSEAPI cusparseDcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a sparse matrix in CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseShyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              float *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);

cusparseStatus_t CUSPARSEAPI cusparseDhyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              double *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);              

cusparseStatus_t CUSPARSEAPI cusparseChyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuComplex *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);

cusparseStatus_t CUSPARSEAPI cusparseZhyb2csr(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuDoubleComplex *csrValA,
                                              int *csrRowPtrA,
                                              int *csrColIndA);

/* Description: This routine converts a sparse matrix in CSC storage format
   to a sparse matrix in HYB storage format. */
cusparseStatus_t CUSPARSEAPI cusparseScsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseDcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseCcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

cusparseStatus_t CUSPARSEAPI cusparseZcsc2hyb(cusparseHandle_t handle,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *cscValA,
                                              const int *cscRowIndA,
                                              const int *cscColPtrA,
                                              cusparseHybMat_t hybA,
                                              int userEllWidth,
                                              cusparseHybPartition_t partitionType);

/* Description: This routine converts a sparse matrix in HYB storage format
   to a sparse matrix in CSC storage format. */
cusparseStatus_t CUSPARSEAPI cusparseShyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              float *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

cusparseStatus_t CUSPARSEAPI cusparseDhyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              double *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

cusparseStatus_t CUSPARSEAPI cusparseChyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuComplex *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

cusparseStatus_t CUSPARSEAPI cusparseZhyb2csc(cusparseHandle_t handle,
                                              const cusparseMatDescr_t descrA,
                                              const cusparseHybMat_t hybA,
                                              cuDoubleComplex *cscVal,
                                              int *cscRowInd,
                                              int *cscColPtr);

/* Description: This routine converts a sparse matrix in CSR storage format
   to a sparse matrix in BSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(cusparseHandle_t handle,
                                                 cusparseDirection_t dirA,
                                                 int m,
                                                 int n,
                                                 const cusparseMatDescr_t descrA,
                                                 const int *csrRowPtrA,
                                                 const int *csrColIndA,
                                                 int blockDim,
                                                 const cusparseMatDescr_t descrC,
                                                 int *bsrRowPtrC,
                                                 int *nnzTotalDevHostPtr);

cusparseStatus_t CUSPARSEAPI cusparseScsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              float *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              double *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseCcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseZcsr2bsr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int m,
                                              int n,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *bsrValC,
                                              int *bsrRowPtrC,
                                              int *bsrColIndC);

/* Description: This routine converts a sparse matrix in BSR storage format
   to a sparse matrix in CSR storage format. */
cusparseStatus_t CUSPARSEAPI cusparseSbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const float *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              float *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseDbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const double *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int   blockDim,
                                              const cusparseMatDescr_t descrC,
                                              double *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseCbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);

cusparseStatus_t CUSPARSEAPI cusparseZbsr2csr(cusparseHandle_t handle,
                                              cusparseDirection_t dirA,
                                              int mb,
                                              int nb,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *bsrValA,
                                              const int *bsrRowPtrA,
                                              const int *bsrColIndA,
                                              int blockDim,
                                              const cusparseMatDescr_t descrC,
                                              cuDoubleComplex *csrValC,
                                              int *csrRowPtrC,
                                              int *csrColIndC);



/* Define the following symbols for the new API routines to be called without "_v2", 
   in other words, append "_v2" to them in the header file. */
/* Level 1 */
#define cusparseSaxpyi cusparseSaxpyi_v2
#define cusparseDaxpyi cusparseDaxpyi_v2
#define cusparseCaxpyi cusparseCaxpyi_v2
#define cusparseZaxpyi cusparseZaxpyi_v2

#define cusparseSroti cusparseSroti_v2
#define cusparseDroti cusparseDroti_v2

/* Level 2 */
#define cusparseScsrsv_analysis cusparseScsrsv_analysis_v2
#define cusparseDcsrsv_analysis cusparseDcsrsv_analysis_v2
#define cusparseCcsrsv_analysis cusparseCcsrsv_analysis_v2
#define cusparseZcsrsv_analysis cusparseZcsrsv_analysis_v2

#define cusparseScsrsv_solve cusparseScsrsv_solve_v2
#define cusparseDcsrsv_solve cusparseDcsrsv_solve_v2
#define cusparseCcsrsv_solve cusparseCcsrsv_solve_v2
#define cusparseZcsrsv_solve cusparseZcsrsv_solve_v2

#define cusparseScsrmv cusparseScsrmv_v2
#define cusparseDcsrmv cusparseDcsrmv_v2
#define cusparseCcsrmv cusparseCcsrmv_v2
#define cusparseZcsrmv cusparseZcsrmv_v2

/* Level 3 */
#define cusparseScsrmm cusparseScsrmm_v2
#define cusparseDcsrmm cusparseDcsrmm_v2
#define cusparseCcsrmm cusparseCcsrmm_v2
#define cusparseZcsrmm cusparseZcsrmm_v2

/* Format conversion */
#define cusparseScsr2csc cusparseScsr2csc_v2
#define cusparseDcsr2csc cusparseDcsr2csc_v2
#define cusparseCcsr2csc cusparseCcsr2csc_v2
#define cusparseZcsr2csc cusparseZcsr2csc_v2

#if defined(__cplusplus)
}
#endif /* __cplusplus */                         

#endif /* !defined(CUSPARSE_V2_H_) */

