//Quartic implementation

#include "quartic.h"
#include "math.h"

//Evaluate the quartic
double quartic(double x,
	           double bVals[5]){

	return bVals[0] + x*bVals[1] + x*x*bVals[2] + x*x*x*bVals[3] + x*x*x*x*bVals[4];
}

//Fill the jacobian 
void fillQuarticJacobian(double xVals[5], double jac[5][5]){
	for (int row = 0; row < 5; row++) {
		for (int col = 0; col < 5; col++) {
			double xpow = pow(xVals[col],row);
			jac[row][col] = xpow;
		}
	}
}

//Fill the residual vector 
void getQuarticResiduals(double resVec[5], double xVals[5], double yVals[5], double bVals[5]) {
	for (int i = 0; i < 5; i++) {
		resVec[i] = yVals[i] - quartic(xVals[i], bVals);
	}
}

//Increment the parameters using the Gauss-Newton approach where jac = inverse jacobian
void incQuarticParams(double bVals[5], double jac[5][5], double resVec[5]) {

	//For each vector value
	for (int i = 0; i < 5; i++) {

		//jacSum - matrix multiplied by vector result
		double jacSum = 0;
	
		//Summing over the jacobian * vector
		for (int j = 0; j < 5; j++) {
			jacSum += jac[j][i]*resVec[j];
		}

		//Learning rate to slow down the step in parameter space
		//(This was for pedagogical reasons and really is more like
		//the Levenberg-Marqaudt algorithm
		double eta = 1.0;

		bVals[i] = bVals[i] + eta*jacSum;
	}
}
