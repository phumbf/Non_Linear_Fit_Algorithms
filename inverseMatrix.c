//Determine the inverse of a 5x5 square matrix using the Gauss-Jordan method

#include <stdio.h>
#include "inverseMatrix.h"

void getInverse(double mat[4][4]) {

	//Create augmented matrix
	int order = 4;
	double augmat[4][8];

	int row = 0, column = 0;

	//Augment the matrix with the identity matrix of correct order
	for (int row = 0; row < order; row++) {
		for (int col = 0; col < order; col++) {

			augmat[row][col] = mat[row][col];

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

		for (int col = 0; col < order; col++) {

			if (row != col) {
				double ratio = augmat[col][row] / augmat[row][row];

				for (int count = 0; count < order * 2; count++) {
					augmat[col][count] = augmat[col][count] - ratio*augmat[row][count];
				}
			}
		}
	}

	//Ensure that principal diagonal (i.e. not augmented identity block) is always 1
	for (int row = 0; row < order; row++) {
		for (int col = order; col < 2*order; col++) {
			augmat[row][col] = augmat[row][col] / augmat[row][row];
		}
	}

	//Set the mat to the inverse mat
	for (int row = 0; row < order; row++) {
		for (int col = order; col < order * 2; col++) {
			mat[row][col-order] = augmat[row][col];
		}
	}
}
