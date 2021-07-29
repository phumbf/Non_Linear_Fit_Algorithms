#pragma once

double gaussian(double x, double bVals[4]);
double df_db0(double x, double bVals[4]);
double df_db1(double x, double bVals[4]);
double df_db2(double x, double bVals[4]);
double df_db3(double x, double bVals[4]);
void fillGaussJacobian(double xVals[5], double jac[5][4], double bVals[4]);
void getGaussResiduals(double resVec[5], double xVals[5], double yVals[5], double bVals[4]);
void levenberg(double jac[5][4], double leven[4][4], double lambda);
double getNewGaussParams(double bVals[4],double bValsNew[4],double xVals[5],double yVals[5],double lambda);
