#pragma once

double quartic(double x, double bVals[5]);

void fillJacobian(double xvals[5], double jac[5][5]);

double sumSq(double res[5]);

void getResiduals(double resVec[5], double xVals[5], double yVals[5], double bVals[5]);

void incParams(double bVals[5], double jac[5][5], double resVec[5]);
