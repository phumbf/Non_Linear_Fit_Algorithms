/*Quartic regression using the Gauss-Newton method. Currently hard coded to 
fit a quartic with 5 data points, but generalising would not be difficult.

The Jacobian is calculated only once as the Jacobian of a quartic 
is not a function of the parameters themselves (drop out of the derivatives).

The Jacobian is then inverted using the Gauss-Jordan algorithm. Finally, similarly 
to various machine learning models, I have included a "learning rate" parameter, eta, 
to "slow down" the rate of learning. This is more just to experiment as the fit
is capable of converging pretty much straight away with this relatively simplistic
example.
*/

#include "inverseMatrix.h"
#include "quartic.h"
#include <stdio.h>

int main() {

	//Define the data set
	double xVals[5] = { -0.024,-0.014,0,0.009,0.021 };
	double yVals[5] = {500,1500,2000,1750,1400};

	//Define the initial param values
	//I've quite deliberately gone for values which are "way off" to test the ability of the 
	//fit to converge
	double b1 = -12317;
	double b2 = +234234;
	double b3 = +42;
	double b4 = -323423423;
	double b5 = -100000000000000000;
	double bVec[5] = { b1,b2,b3,b4,b5 };

	//Iterate a certain number of times
	int n_iter = 150;
	double sumSquares = 0;
	double resVec[5] = { 0,0,0,0,0 };

	//Get the Jacobian - this only has to be once due to the form of a quartic jacobian
	double jac[5][5];
	fillJacobian(xVals, jac);

	//Inverse the Jacobian
	getInverse(jac);

	//To write out to
	FILE *outfile = fopen("Fitting.txt","w+");
	fprintf(outfile, "Starting parameters are:\n");
	fprintf(outfile, "b1 = %f\n",bVec[0]);
	fprintf(outfile, "b2 = %f\n",bVec[1]);
	fprintf(outfile, "b3 = %f\n",bVec[2]);
	fprintf(outfile, "b4 = %f\n",bVec[3]);
	fprintf(outfile, "b5 = %f\n",bVec[4]);

	for (int iter = 0; iter < n_iter; iter++) {

		//Determine the residual vector 
		getResiduals(resVec, xVals, yVals, bVec);

		//Determine the sumSquares value
		sumSquares = sumSq(resVec);
		
		fprintf(outfile, "----Iteration number: %i\n ", iter+1);
		fprintf(outfile, "Sum squared of res is: %f\n", sumSquares);
		
		//Increment the parameters
		incParams(bVec,jac,resVec);
	
		fprintf(outfile, "Incremented parameters are:\n");
		fprintf(outfile, "b1 = %f\n",bVec[0]);
		fprintf(outfile, "b2 = %f\n",bVec[1]);
		fprintf(outfile, "b3 = %f\n",bVec[2]);
		fprintf(outfile, "b4 = %f\n",bVec[3]);
		fprintf(outfile, "b5 = %f\n",bVec[4]);
	}

		fprintf(outfile, "Finished incrementing\n");
		fclose(outfile);
}