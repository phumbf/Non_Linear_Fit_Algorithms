#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <math.h>

#include "kernel.cuh"

//Update parameters
__device__ void updateBetaVals(double *bVals, double *bValsNew, int params_per_fit) {
	for (int i = 0; i < params_per_fit; ++i) {
		bVals[i] = bValsNew[i];
	}
}

//Determine new parameters
__device__ double getNewParams(double *xData,
	                           double *yData,
			                   double *bData,
	                           double *newbData,
	                           double *residuals,
	                           int params_per_fit,
	                           int data_per_fit,
						       double lambda) {


		//Reserve appropriate memory
		double *inner = (double*)malloc(sizeof(double)*params_per_fit*params_per_fit);
		double *outer = (double*)malloc(sizeof(double)*params_per_fit*data_per_fit);
		double *add = (double*)malloc(sizeof(double)*params_per_fit);
		double *jacobian = (double*)malloc(sizeof(double)*data_per_fit*params_per_fit);
		double *jac_T = (double*)malloc(sizeof(double)*data_per_fit*params_per_fit);

		//Determine the Jacobian
		fillJacobian(xData, jacobian, bData, params_per_fit, data_per_fit);

		//Determine the Jacobian transpose
		transpose(jacobian, jac_T, data_per_fit, params_per_fit);

		//Determine the Gauss residuals
		for (int i = 0; i < data_per_fit; ++i) {
			residuals[i] = yData[i] - gaussian(xData[i],bData);
		}

		//Determine the inner matrix JT*J + lambda*I
		fillInner(jacobian, jac_T, inner, lambda, params_per_fit, data_per_fit);

		//Invert the inner matrix 
		invertMat(inner,params_per_fit);

		//Determine the new parameters
		multiplyMatrices(inner, params_per_fit, params_per_fit, jac_T, params_per_fit, data_per_fit, outer, params_per_fit, data_per_fit);
		multiplyMatrices(outer, params_per_fit, data_per_fit, residuals, data_per_fit, 1, add, params_per_fit, 1);
		for (int i = 0; i < params_per_fit; ++i) {
			newbData[i]= bData[i] + add[i];
		}

		//Determine the sumsq 
		getResiduals(residuals, xData, yData, newbData, data_per_fit);
		double sumSquared = sumSq(residuals, data_per_fit);

		delete inner;
		delete outer;
		delete add;
		delete jacobian;
		delete jac_T;

		return sumSquared;
}

//Determine the residuals
__device__ void getResiduals(double *residuals, double *xData, double *yData, double *bData, int data_per_fit) {
	
	for (int i = 0; i < data_per_fit; ++i) {
		residuals[i] = yData[i] - gaussian(xData[i],bData);
	}
}

//Determine sum squares
__device__ double sumSq(double *residuals, int data_per_fit) {

	double sum{ 0.0 };
	for (int i = 0; i < data_per_fit; ++i) {
		sum += residuals[i] * residuals[i];
	}

	return sum;
}

//Invert matrix 
__device__ void invertMat(double *mat, int params_per_fit) {

	//Create augmented matrix
	int order = params_per_fit;

	//Dynamically create augmat 
	double **augmat = (double**)malloc(params_per_fit * sizeof(*augmat));
	for (int i = 0; i < params_per_fit; ++i) {
		augmat[i] = (double*)malloc(params_per_fit * sizeof(augmat[0]));
	}

	int count{ 0 };

	//Augment the matrix with the identity matrix of correct order
	for (int row = 0; row < order; ++row) {
		for (int col = 0; col < order; ++col) {

			augmat[row][col] = mat[count];
			++count;

			if (row == col) {
				augmat[row][col+order] = 1;
			}
			else {
				augmat[row][col + order] = 0;
			}
		}
	}

	//Apply Gauss-Jordan elimination to make off-diagonals in left half = 0 
	for (int row = 0; row < order; row++){
		
		if (augmat[row][row] == 0.0) {
			return;
		}

		for (int col = 0; col < order; ++col) {

			if (row != col) {
				double ratio = augmat[col][row] / augmat[row][row];

				for (int count = 0; count < order * 2; ++count) {
					augmat[col][count] = augmat[col][count] - ratio*augmat[row][count];
				}
			}
		}
	}

	//Ensure that principal diagonal (i.e. not augmented identity block) is always 1
	for (int row = 0; row < order; ++row) {
		for (int col = order; col < 2*order; ++col) {
			augmat[row][col] = augmat[row][col] / augmat[row][row];
		}
	}

	count = 0;
	//Set the mat to the inverse mat
	for (int row = 0; row < order; ++row) {
		for (int col = order; col < order * 2; ++col) {
			mat[count] = augmat[row][col];
			++count;
		}
	}

	//free up memory
	for (int i = 0; i < params_per_fit; ++i) {
		delete(augmat[i]);
	}
	delete(augmat);
}

//multiply two matrices together
__device__ void multiplyMatrices(double *matA,
	                             int rA,
	                             int cA,
	                             double *matB,
	                             int rB,
	                             int cB,
	                             double *matC,
	                             int rC,
	                             int cC) {

	for (int i = 0; i < rA; ++i) {
		for (int j = 0; j < cB; ++j) {
			double sum{ 0.0 };
			for (int k = 0; k < rB; ++k) {
				sum = sum + matA[i * cA + k] * matB[k * cB + j];
			}
			matC[i * cC + j] = sum;
		}
	}
}

//Fill the inner matrix JT*J + lambda*I
__device__ void fillInner(double *jacobian, double *jac_T, double *inner, double lambda, int params_per_fit, int data_per_fit) {

	int iter{ 0 };
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 5; ++j) {
			++iter;
		}
	}

	//Multiply JT*J
	multiplyMatrices(jac_T, params_per_fit, data_per_fit, jacobian, data_per_fit, params_per_fit, inner, params_per_fit, params_per_fit);

	//Add in the identity matrix
	int count{ 0 };
	for (int i = 0; i < params_per_fit; ++i) {
		for (int j = 0; j < params_per_fit; ++j) {
			inner[count] += (i == j) ? lambda : 0;
			++count;
		}
	}
}

//Get transpose
__device__ void transpose(double *jac, double *jac_T, int data_per_fit, int params_per_fit) {

	for (int i = 0; i < data_per_fit; ++i) {
		for (int j = 0; j < params_per_fit; ++j) {

			//Original index 
			int index1 = i*params_per_fit + j;

			//New index
			int index2 = j*data_per_fit + i;

			jac_T[index2] = jac[index1];
		}
	}
}

//Evaluate df/db0
__device__ double df_db0(double x, double *bVals){
	return exp(-( ( (x-bVals[1])*(x-bVals[1]) ) / (2*bVals[2]*bVals[2]) ));
}

//Evaluate df/db1
__device__ double df_db1(double x, double *bVals){
	double first = -bVals[0] * exp(-( ( (x-bVals[1])*(x-bVals[1]) ) / (2*bVals[2]*bVals[2]) ));
	double second = -( x-bVals[1] ) / ( bVals[2]*bVals[2] );
	return first*second;
}

//Evaluate df/db2
__device__ double df_db2(double x, double *bVals){
	double first = -bVals[0] * exp(-( ( (x-bVals[1])*(x-bVals[1]) ) / (2*bVals[2]*bVals[2]) ));
	double second = -( (x-bVals[1])*(x-bVals[1]) ) / ( bVals[2]*bVals[2]*bVals[2] );
	return first*second;
}

//Evaluate df/db3
__device__ double df_db3(double x, double *bVals){
	return 1;
}

//Fill the Jacobian 
__device__ void fillJacobian(double *xVals, double *jac, double *bVals, int params_per_fit, int data_per_fit) {

	int count = 0;

	for (int i = 0; i < data_per_fit; ++i) {
		for (int j = 0; j < params_per_fit; ++j) {

			if (j == 0) {
				jac[count] = df_db0(xVals[i], bVals);
			}
			else if (j == 1) {
				jac[count] = df_db1(xVals[i], bVals);
			}
			else if (j == 2) {
				jac[count] = df_db2(xVals[i], bVals);
			}
			else{
				jac[count] = df_db3(xVals[i], bVals);
			}
			++count;
		}
	}
}

//Implementation of a Gaussian function
__device__ double gaussian(double &x, double *bVals) {

	double exponent = exp(-( (x - bVals[1])*(x - bVals[1])) / (2 * bVals[2] * bVals[2]));
	return (bVals[0] * exponent) + bVals[3];
}

__global__ void LM_Kernel (double* xVals, double* yVals, double *bVals, int n_fits, int data_per_fit, int params_per_fit) {

	//Get the thread id.
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Thread id is: %i\n", id);

	//Obtain the slices for the thread
	int data_begin = id*data_per_fit;
	int param_begin = id*params_per_fit;

	double* xData = (double*)malloc(sizeof(double)*data_per_fit);
	double* yData = (double*)malloc(sizeof(double)*data_per_fit);
	double* bData = (double*)malloc(sizeof(double)*params_per_fit);

	//Obtain the data slice
	{
		int j = 0;
		for (int i = data_begin; i < data_begin + data_per_fit; ++i) {
			xData[j] = xVals[i];
			yData[j] = yVals[i];
			++j;
		}
	}

	//Obtain the parameter slice
	{
		int j = 0;
		for (int i = param_begin; i < param_begin + params_per_fit; ++i) {
			bData[j] = bVals[i];
			++j;
		}
	}

	//Determine the initial residual values
	double *residuals = (double*)malloc(sizeof(double)*data_per_fit);
	getResiduals(residuals, xData, yData, bData, data_per_fit);

	//Determine the initial sum square
	double sumSqInit = sumSq(residuals, data_per_fit);

	//Prepare jacobian, lambda, nu values
	double *bValsLambda = (double*)malloc(sizeof(double)*params_per_fit);
	double *bValsLambdaNu = (double*)malloc(sizeof(double)*params_per_fit);

	double lambda{0.5};
	double nu{2.0};

	int noImpCount{ 0 };
	int iterCount{ 0 };

	//Now begin iteration until less than 1% reduction in sumsq value
	while (noImpCount < 3)
	{

		//Lambda = Lambda
		double lambdaSumSq = getNewParams(xData, yData, bData, bValsLambda, residuals, params_per_fit, data_per_fit, lambda);

		//Lambda = Lambda/mu
		double lambdaSumSqNu = getNewParams(xData, yData, bData, bValsLambdaNu, residuals, params_per_fit, data_per_fit, lambda/nu);
		
		//If both sumSqs WORSE than original, then change lambda and try again
		if(lambdaSumSq > sumSqInit && lambdaSumSqNu > sumSqInit){
			lambda *= nu;
			++iterCount;
			continue;
		}

		//If lambdaSumSq is BETTER update beta values, update sumSq
		else if(lambdaSumSq < sumSqInit){
			updateBetaVals(bData, bValsLambda, params_per_fit);
			double percentDec = 100.0*abs(sumSqInit - lambdaSumSq) / sumSqInit;
			if (percentDec < 1) { ++noImpCount; }
			++iterCount;
			sumSqInit = lambdaSumSq;
		}

		//If lambdaSumSqNu is BETTER instead, update beta, sumsq and lambda
		else if(lambdaSumSqNu < sumSqInit){
			updateBetaVals(bData, bValsLambdaNu, params_per_fit);
			double percentDec = 100.0*abs(sumSqInit - lambdaSumSqNu) / sumSqInit;
			if (percentDec < 1) { ++noImpCount; }
			++iterCount;
			sumSqInit = lambdaSumSqNu;
			lambda = lambda/nu;
		}
	}
	
	//Transfer the bVals across
	{
		int j = 0;
		for (int i = param_begin; i < param_begin + params_per_fit; ++i) {
			bVals[i] = bData[j];
			++j;
		}
	}

	delete xData;
	delete yData;
	delete bData;
	delete residuals;
	delete bValsLambda;
	delete bValsLambdaNu;
}

void cpuGauss(double &xVal, double &A0, double &A1, double &A2, double &A3, double &yVal) {
	yVal = A0 * exp(-( (xVal - A1)*(xVal - A1)) / (2 * A2 * A2)) + A3;
}

void generateRandomToyData(double *xVals, double *yVals, double *bVals, int n_fits, int data_per_fit, int params_per_fit) {

	//For random number generation
	std::random_device rd;
	std::mt19937 gen(rd());

	int count = 0;
	double xValsStart = -10.0;

	//Default gaussian parameters
	double A0 = 3.0;
	double A1 = 0.0;
	double A2 = 0.5;
	double A3 = 1.0;

	for (int i = 0; i < n_fits; ++i) {

		std::normal_distribution<double> d(0, 0.3);
		double modA0 = A0 + d(gen)*A0;
		double modA1 = A1 + d(gen)*A1;
		double modA2 = A2 + d(gen)*A2*0.1;
		double modA3 = A3 + d(gen)*A3;
		printf("DEBUG: Actual params are %f, %f, %f, %f\n", modA0, modA1, modA2, modA3);
		double yVal{ 0.0 };

		count = i * data_per_fit;

		for (int j = 0; j < data_per_fit; ++j) {
			xVals[count + j] = xValsStart + j;
			cpuGauss(xVals[count + j], modA0, modA1, modA2, modA3, yVal);
			yVals[count + j] = yVal;
		}
		
		count = i * 4;
		bVals[count] = A0;
		bVals[count+1] = A1;
		bVals[count+2] = A2;
		bVals[count+3] = A3;

		}
}

//Function to generate some toy Gaussian data
void generateToyData(double *xVals, double *yVals, double *bVals, int n_fits, int data_per_fit, int params_per_fit) {

	for (int fit = 0; fit < n_fits; ++fit) {
		int dataCount = fit*data_per_fit;
		int paramCount = fit * params_per_fit;

		xVals[dataCount] = -5;
		xVals[dataCount + 1] = 0;
		xVals[dataCount + 2] = 5;
		xVals[dataCount + 3] = 10;
		xVals[dataCount + 4] = 15;

		yVals[dataCount] = 0.007;
		yVals[dataCount + 1] = 0.097;
		yVals[dataCount + 2] = 0.13;
		yVals[dataCount + 3] = 0.089;
		yVals[dataCount + 4] = 0.005;

		bVals[paramCount] = 10;
		bVals[paramCount + 1] = 1;
		bVals[paramCount + 2] = 3;
		bVals[paramCount + 3] = -0.2;
	}

}

int main() {

	//Increasing heap memory
	size_t *size = new size_t;
	cudaDeviceGetLimit(size, cudaLimitMallocHeapSize);
	std::cout << *size << std::endl;
	size_t newsize = 150000000;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize,newsize);
	cudaDeviceGetLimit(size, cudaLimitMallocHeapSize);
	std::cout << *size << std::endl;

	int n_data_per_fit{ 5 };
	int n_params_per_fit{ 4 };
	int n_fits{ 1000000 };

	//Create arrays on the host
	double* xVals = (double*)malloc(sizeof(double)*n_data_per_fit*n_fits);
	double* yVals = (double*)malloc(sizeof(double)*n_data_per_fit*n_fits);
	double* bVals = (double*)malloc(sizeof(double)*n_params_per_fit*n_fits);

	//Fill the arrays on the host
	generateToyData(xVals, yVals, bVals, n_fits, n_data_per_fit, n_params_per_fit);
	printf("DATA GENERATED\n");

	//Create arrays on the device
	double *d_xVals, *d_yVals, *d_bVals;
	gpuErrchk(cudaMalloc((void**)&d_xVals, sizeof(double)*n_data_per_fit*n_fits));
	gpuErrchk(cudaMalloc((void**)&d_yVals, sizeof(double)*n_data_per_fit*n_fits));
	gpuErrchk(cudaMalloc((void**)&d_bVals, sizeof(double)*n_params_per_fit*n_fits));

	//Copy from host to device
	gpuErrchk(cudaMemcpy(d_xVals, xVals, sizeof(double)*n_data_per_fit*n_fits, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_yVals, yVals, sizeof(double)*n_data_per_fit*n_fits, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_bVals, bVals, sizeof(double)*n_params_per_fit*n_fits, cudaMemcpyHostToDevice));

	//Run the kernel
	int threads_per_block{ 16 };
	int blocks_per_grid = (int)(n_fits / threads_per_block) + 1;
	dim3 grid(blocks_per_grid);
	dim3 blocks(threads_per_block);

	//Run the Kernel 
	LM_Kernel << <grid, blocks >> > (d_xVals, d_yVals, d_bVals, n_fits, n_data_per_fit, n_params_per_fit);
	gpuErrchk(cudaDeviceSynchronize());

	//transfer the memory back to the host from the device
	gpuErrchk(cudaMemcpy(bVals, d_bVals, sizeof(double)*n_params_per_fit*n_fits, cudaMemcpyDeviceToHost));

	//Cleanup
	gpuErrchk(cudaFree(d_xVals));
	gpuErrchk(cudaFree(d_yVals));
	gpuErrchk(cudaFree(d_bVals));
	gpuErrchk(cudaDeviceReset());

	delete xVals;
	delete yVals;
	delete bVals;

	return 0;
}
