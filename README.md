# Numerical Linear Algebra Library

This repo was created as an extra credit project by Noble Mushtak for his MATH 2331: Linear Algebra class at Northeastern University (USA). The file `numerical.py` contains functions which implement the following features:

 * Vector operations, such as adding, subtracting, and scaling vectors
 * L1-norm, L2-norm, and L∞-norm of vectors
 * Absolute and relative errors of vectors
 * Matrix-vector and matrix-matrix products
 * Row and column operations on matrices
 * Transpose of matrices
 * Inverting and computing the sign of permutations
 * Naive pivoting, simple partial pivoting, scaled partial pivoting, and full pivoting
 * Gaussian elimination
 * Back-substitution
 * Solving systems of equations using elimination
 * LU decomposition
 * Forward-substitution
 * Solving systems of equations using the LU decomposition
 * Finding the image, kernel, and inverse of a matrix given the LU decomposition
 * Determining if a matrix is singular given the LU decomposition
 * Finding the exact determinant of an integer-valued matrix using permutations
 * Finding the determinant of a matrix using LU decomposition
 * QR decomposition using Householder reflections
 * Finding the least-squares solution of a system of equations given the QR decomposition
 * Calculating the best-fit coefficients of a linear regression model
 * Schur decomposition using a naive form of the QR algorithm
 * Finding the eigendecomposition given the Schur decomposition

Finally, the `numerical_test.py` contains several unit tests which ensure that all of these functions work as intended.