//Gaussian implementation

#include "gaussian.h"
#include "math.h"
#include "inverseMatrix.h"
#include "helperFunctions.h"

//Evaluate the gaussian
double gaussian(double x, double bVals[4]){
	double exponent = exp(-( ( (x-bVals[1])*(x-bVals[1]) ) / (2*bVals[2]*bVals[2]) ));
	return (bVals[0] * exponent) + bVals[3];
}

//Evaluate df/db0
double df_db0(double x, double bVals[4]){
	return exp(-( ( (x-bVals[1])*(x-bVals[1]) ) / (2*bVals[2]*bVals[2]) ));
}

//Evaluate df/db1
double df_db1(double x, double bVals[4]){
	double first = -bVals[0] * exp(-( ( (x-bVals[1])*(x-bVals[1]) ) / (2*bVals[2]*bVals[2]) ));
	double second = -( x-bVals[1] ) / ( bVals[2]*bVals[2] );
	return first*second;
}

//Evaluate df/db2
double df_db2(double x, double bVals[4]){
	double first = -bVals[0] * exp(-( ( (x-bVals[1])*(x-bVals[1]) ) / (2*bVals[2]*bVals[2]) ));
	double second = -( (x-bVals[1])*(x-bVals[1]) ) / ( bVals[2]*bVals[2]*bVals[2] );
	return first*second;
}

//Evaluate df/db3
double df_db3(double x, double bVals[4]){
	return 1;
}

//Fill the jacobian
void fillGaussJacobian(double xVals[5], double jac[5][4], double bVals[4]){

	for(int row = 0; row < 5; row++){
		for(int col = 0; col < 4; col++){
		
			//Call the correct derivative depending on the column	
			if(col == 0){
				jac[row][col] = df_db0(xVals[row],bVals);
			}
			else if(col == 1){
				jac[row][col] = df_db1(xVals[row],bVals);
			}
			else if(col == 2){
				jac[row][col] = df_db2(xVals[row],bVals);
			}
			else{
				jac[row][col] = df_db3(xVals[row],bVals);
			}
		}
	}
}

//Fill the residual vector 
void getGaussResiduals(double resVec[5], double xVals[5], double yVals[5], double bVals[4]) {
	for (int i = 0; i < 5; i++) {
		resVec[i] = yVals[i] - gaussian(xVals[i], bVals);
	}
}

//Determine JT*J + lambda*I before inversion
void levenberg(double jac[5][4], double leven[4][4], double lambda){


	//Determine the transpose of the Jacobian
	double jacTr[4][5];
	for(int row=0; row<4; row++){
		for(int col=0; col<5; col++){
			jacTr[row][col] = jac[col][row];
		}
	}

	//Determine JT*J + lambda*identity
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			
			leven[i][j] = 0;

			for(int k=0; k<5; k++){
				leven[i][j] += jacTr[i][k] * jac[k][j];
			}

			//Add the identity matrix * lambda factor
			leven[i][j] += (i==j) ? lambda : 0;
		}
	}	
	
}

//Increment the parameters for bValsNew and return the new sumsq
double getNewGaussParams(double bVals[4],
		       double bValsNew[4],
    	               double xVals[5],
    	               double yVals[5],
		       double lambda){

	//Determine the jacobian
	double jac[5][4];
	fillGaussJacobian(xVals,jac,bVals);

	//Determine the transpose of the Jacobian
	double jacTr[4][5];
	for(int row=0; row<4; row++){
		for(int col=0; col<5; col++){
			jacTr[row][col] = jac[col][row];
		}
	}

	//Determine the residuals
	double rVals[5];
	getGaussResiduals(rVals,xVals,yVals,bVals);

	//Calculate the leven matrix
	double leven[4][4];
	levenberg(jac,leven,lambda);

	//Invert the leven matrix
	getInverse(leven);

	//Perform the matrix algebra to determine the new params
	for(int i=0; i<4; i++){
		double jsum=0;
		for(int j=0; j<4; j++){
			double ksum=0;
			for(int k=0; k<5; k++){
				ksum += jacTr[j][k]*rVals[k];
			}
			jsum += leven[i][j]*ksum;
		}	
		bValsNew[i] = bVals[i] + jsum;
	}

	//Determine the new residuals
	getGaussResiduals(rVals,xVals,yVals,bValsNew);

	//Return the new SumSq
	return sumSq(rVals);
}


