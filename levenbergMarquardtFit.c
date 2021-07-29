#include "gaussian.h"
#include "helperFunctions.h"
#include <stdio.h>

int main(){

	//Define some "Gaussian-like data"
	double xVals[5] = {-5,0,5,10,15};
	double yVals[5] = {0.007, 0.097, 0.13, 0.089, 0.005};

	//Define starting params
	//double bVals[4] = {0.1,5.0,0.30,-0.2};
	double bVals[4] = {10,1.0,3,-0.2};

	//Define initial lambda and nu
	double lambda = 0.5;
	double nu = 2;

	//Determine the initial set of residuals and sumSq
	double resVec[5];
	getGaussResiduals(resVec,xVals,yVals,bVals);
	double sumSqOrig = sumSq(resVec);

	//To write out to
	FILE *outfile = fopen("LMFitting.txt","w+");
	fprintf(outfile, "Starting parameters are:\n");
	fprintf(outfile, "b1 = %f\n",bVals[0]);
	fprintf(outfile, "b2 = %f\n",bVals[1]);
	fprintf(outfile, "b3 = %f\n",bVals[2]);
	fprintf(outfile, "b4 = %f\n",bVals[3]);

	//Iterate
	int nIter = 100;
	for(int iter =0; iter<nIter; iter++){

		//First get sumSq from lamba = lambda
		double bValsLambda[4];
		double sumSqLambda = getNewGaussParams(bVals,bValsLambda,xVals,yVals,lambda);

		//Then, get sumSq from lambda = lambda/nu
		double bValsLambdaNu[4];
		double sumSqLambdaNu = getNewGaussParams(bVals,bValsLambdaNu,xVals,yVals,lambda/nu);

		fprintf(outfile, "----Iteration number: %i\n ", iter+1);
		fprintf(outfile, "Sum squares orig is: %f\n", sumSqOrig);
		fprintf(outfile, "Sum squared of lambda is: %f\n", sumSqLambda);
		fprintf(outfile, "Sum squared of lambda/nu is: %f\n", sumSqLambdaNu);
//Print out status
		fprintf(outfile, "Incremented parameters are:\n");
		fprintf(outfile, "b1 = %f\n",bVals[0]);
		fprintf(outfile, "b2 = %f\n",bVals[1]);
		fprintf(outfile, "b3 = %f\n",bVals[2]);
		fprintf(outfile, "b4 = %f\n",bVals[3]);

		//If both sumSqs WORSE than original, then change lambda and try again
		if(sumSqLambda > sumSqOrig && sumSqLambdaNu > sumSqOrig){
			lambda *= nu;
			continue;
		}

		//If sumSqLambda is BETTER update beta values, update sumSq
		else if(sumSqLambda < sumSqOrig){
			updateBetaVals(bVals, bValsLambda);
			sumSqOrig = sumSqLambda;
		}

		//If sumSqLambdaNu is BETTER instead, update beta, sumsq and lambda
		else if(sumSqLambdaNu < sumSqOrig){
			updateBetaVals(bVals, bValsLambdaNu);
			sumSqOrig = sumSqLambdaNu;
			lambda = lambda/nu;
		}

		
	}
		fprintf(outfile, "Finished incrementing\n");
		fclose(outfile);
}
