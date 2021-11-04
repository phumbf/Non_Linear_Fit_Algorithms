#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

//Macro for CUDA error handling
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__,__LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ void updateBetaVals(double *bVals, double *bValsNew, int params_per_fit);

__device__ double getNewParams(double *xData,
	double *yData,
	double *bData,
	double *newbData,
	double *residuals,
	int params_per_fit,
	int data_per_fit, double lambda);

__device__ void getResiduals(double *residuals, double *xData, double *yData, double *bData, int data_per_fit);

__device__ double sumSq(double *residuals, int data_per_fit);

__device__ void invertMat(double *mat, int params_per_fit);

__device__ void multiplyMatrices(double *matA,
	int rA,
	int cA,
	double *matB,
	int rB,
	int cB,
	double *matC,
	int rC,
	int cC);

__device__ void fillInner(double *jacobian, double *jac_T, double *inner, double lambda, int params_per_fit, int data_per_fit);

__device__ void transpose(double *jac, double *jac_T, int data_per_fit, int params_per_fit);

__device__ double df_db0(double x, double *bVals);

__device__ double df_db1(double x, double *bVals);

__device__ double df_db2(double x, double *bVals);

__device__ double df_db3(double x, double *bVals);

__device__ void fillJacobian(double *xVals, double *jac, double *bVals, int params_per_fit, int data_per_fit);

__device__ double gaussian(double &x, double *bVals);

__global__ void LM_Kernel(double* xVals, double* yVals, double *bVals, int n_fits, int data_per_fit, int params_per_fit);

void getVals(double *xVals, double *yVals, double *bVals);

#endif
