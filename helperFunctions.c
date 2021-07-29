#include "helperFunctions.h"

//Determine the loss function
//Currently set to 5 data points
double sumSq(double res[5]) {
	double sum = 0;
	for (int i = 0; i < 5; i++) {
		sum += res[i] * res[i];
	}
	return sum;
}

void updateBetaVals(double bVals[4], double bValsMod[4]) {

	for (int i = 0; i < 4; i++) {
		bVals[i] = bValsMod[i];
	}

}
