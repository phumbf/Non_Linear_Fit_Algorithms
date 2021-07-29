#pragma once

double quartic(double x, double bVals[5]);

void fillQuarticJacobian(double xvals[5], double jac[5][5]);

void getQuarticResiduals(double resVec[5], double xVals[5], double yVals[5], double bVals[5]);

void incQuarticParams(double bVals[5], double jac[5][5], double resVec[5]);
