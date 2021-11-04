1) gaussNewtonFit.c - An example of implementing a quartic regression in C using the Gauss-Newton algorithm. 

In this example a quartic (5 parameters) is fit to 5 data points. This results in a square Jacobian which is quick to invert.
The inversion is performed using the Gauss-Jordan algorithm.
Expanding to n data points would be trivial to implement and the accomoanying non-square Jacobian algebra can be found in the 
Levenberg-Marquadt algorithm in this repo. 

The option to slow down the steps in parameter space via the application of a learning rate parameter is included, similarly to as
is done in other learning models.

2) levenbergMarquardtFit.c - An example of implementing a Gaussian fit using the Levenberg-Marquadt algorithm.
At the moment the example still uses a simple 5 data point fit and iterates a fixed number of times. This could be changed 
to stop iterating once an appropriate objective function value has been reached. 

The Jacobian here is non-square meaning that additional linear algebra is implemented using both the Jacobian and Jacobian transpose

3) kernel.cu - A quick, initial implementation of the Levenberg-Marquadt fit using a GPU to increase speed. This does increase performance (can perform around 100,000 fits in under 10 seconds), but additional work could still improve this dramatically.
