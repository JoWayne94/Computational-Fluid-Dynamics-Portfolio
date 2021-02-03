/**
 * @file Thomas Algo & Steepest Descent.cpp
 *
 * Fundamentals of Computational Fluid Dynamics
 *
 * Solution to Coursework 1 Questions
 *
 * Solves Helmholtz equation using Thomas and Steepest Descent algorithms.
 *
 * The Helmholtz matrix is tri-diagonal and symmetric (question 1). We
 * adapt the thomas algorithm and the steepest descent iterative 
 * algorithm to instead use a (symmetric in q1) banded storage, thereby
 * significantly enhancing the efficiency.
 */
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstring>
using namespace std;

#include "cblas.h"

#define FORMAT(vOut, x, IC, u, e) \
        vOut << setw(10) << x << setw(12) << IC << setw(12) << u << setw(12) << e << endl;

/**
 * @brief Populate Helmholtz symmetric banded matrix (upper only) for ques 1
 *
 * @param   nsv     Leading dimension of matrix
 * @param   H       Pointer to matrix storage of size ldh*nsv
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void FillHelmholtzMatrix(int nsv, double* H, double lam, double dx) {
    const int ldh = 2;                  // Diagonal and upper diagonal
    const double oodx2 = 1.0/dx/dx;
    
    H[1] = - lam - 2.0*oodx2;
    for (int i = 1; i < nsv; ++i) {     // index rows
            H[i*ldh    ] = oodx2;
            H[i*ldh + 1] = -lam - 2.0*oodx2;
    }
}

/**
 * @brief Populate Helmholtz banded matrix (upper and lower) for ques 2
 *
 * @param   nsv     Leading dimension of matrix
 * @param   H       Pointer to matrix storage of size 2*nsv
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void FillHelmholtzMatrix2(int nsv, double* H, double lam, double dx) {
    const int ldh = 3;                  // Diagonal, upper and lower diagonal
    const double oodx = 1.0;            // Denominator added in Neumann BC
    
    for (int i = 0; i < nsv - 1; ++i) {     // index rows
            H[i*ldh    ] = -lam;
            H[i*ldh + 1] = 1 + 2.0*lam;
            H[i*ldh + 2] = -lam;
    }
    H[2] = 0.0;
    H[(nsv-1)*ldh    ] = 0.0;
    H[(nsv-1)*ldh + 1] = oodx;
    H[(nsv-1)*ldh + 2] = -oodx;
}

/**
 * @brief Fills the forcing vector with
 *        \f$ f = -(\lambda + m^2 \pi^2) \sin(m \pi x) \f$
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to vector storage of length n
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void FillForcingFunction(int n, double* f, double lam, int m, double dx) {
    const double f_coeff = -(lam + pow(m, 2)*pow(M_PI, 2));
    
    for (int i = 0; i < n; ++i) {
        f[i] = f_coeff*sin(m*M_PI*i*dx);
    }
}

/**
 * @brief Fills the forcing vector with
 *        \f$ f = sum_{m=0}^{m=50} (-1)^m 40/((2m + 1)^2 \pi^2) \sin[(2m + 1) \pi x] \f$
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to vector storage of length n
 * @param   dx      Grid spacing
 */
void FillForcingFunction2(int n, double* f, double dx) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= 50; ++j) {
            f[i] += pow(-1, j)*(40/(pow(2*j + 1, 2)*M_PI*M_PI))*(sin((2*j + 1)*M_PI*i*dx));
        }
    }
}

/**
 * @brief Enforces zero Dirichlet boundary conditions. 
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to forcing term storage of length n
 * @param   u       Pointer to solution vector storage of length n
 * @param   lam     Lambda coefficient
 * @param   dx      Grid spacing
 */
void EnforceBoundaryConditions(int n, double* f, double* u, double lam, double dx, int m) {
    u[0] = sin(0);
    u[n-1] = sin(m*M_PI*(n-1)*dx);
    f[1] -= u[0]/dx/dx;
    f[n-2] -= u[n-1]/dx/dx;
}

/**
 * @brief Enforces Dirichlet and Neumann boundary conditions. 
 *
 * @param   n       Vector dimension
 * @param   f       Pointer to forcing term storage of length n
 * @param   u       Pointer to solution vector storage of length n
 * @param   sig     Sigma coefficient
 * @param   dx      Grid spacing
 * @param   t       Current time step
 */
void EnforceBoundaryConditions2(int n, double* f, double* u, double sig, double dx, double t) {
    u[0] = sin(0);
    double neumann = 0.0; // Initiate Neumann condition value
    for (int j = 0; j <= 50; ++j) {
        neumann += pow(-1, j)*(40/(pow(2*j + 1, 2)*M_PI*M_PI))*(exp(-sig*pow(2*j + 1, 2)*M_PI*M_PI*t))*(2*j + 1)*M_PI*cos((2*j + 1)*M_PI);
    }
    f[1] -= u[0]/dx/dx;
    f[n-1] = neumann*dx; // dx included at RHS of equation for simplicity
}

/**
 * @brief Computes the exact solution for first question with
 *        \f$ u = \sin(m \pi x) \f$
 *
 * @param   n       Vector dimension
 * @param   e       Pointer to solution vector storage of length n
 * @param   dx      Grid spacing
 * @param   m       M coefficient
 */
void ExactSolution(int n, double* e, double dx, int m) {
    for (int i = 0; i < n; ++i) {
        e[i] = sin(m*M_PI*i*dx);
    }
}

/**
 * @brief Computes the exact solution for second question with
 *        \f$ u = sum_{m=0}^{m=50} (-1)^m 40/((2m + 1)^2 \pi^2) \sin[(2m + 1) \pi x] e^{-\sigma (2m + 1)^2 \pi^2 T} \f$
 *
 * @param   n       Vector dimension
 * @param   e       Pointer to solution vector storage of length n
 * @param   dx      Grid spacing
 * @param   T       Solution time
 * @param   sig     Sigma coefficient
 */
void ExactSolution2(int n, double* e, double dx, double T, double sig) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= 50; ++j) {
            e[i] += pow(-1, j)*(40/(pow(2*j + 1, 2)*M_PI*M_PI))*exp(-sig*pow(2*j + 1, 2)*M_PI*M_PI*T)*sin((2*j + 1)*M_PI*i*dx);
        }
    }
}

/**
 * @brief Prints a tri-banded matrix in column-major format.
 *
 * Note that for a symmetric matrix, only the upper diagonals or lower
 * diagonals need to be stored (ldh = 2).
 *
 * @param   nsv     Matrix dimension
 * @param   H       Pointer to matrix storage of size ldh*nsv
 */
void PrintMatrix(int nsv, double* H, int ldh) {
    cout.precision(5);
    for (int i = 0; i < ldh; ++i) {
        for (int j = 0; j < nsv; ++j) {
            cout << setw(7) << H[j*ldh+i] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

/**
 * @brief Prints a vector
 *
 * @param   n       Vector dimension
 * @param   u       Pointer to vector storage of length n
 * @param   dx      Grid spacing
 */
void PrintVector(int n, double* u, double dx) {
    for (int i = 0; i < n; ++i) {
        cout << i*dx << "  " << u[i] << endl;
    }
    cout << endl;
}

/**
 * @brief Computes and outputs the maximum value of the absolute of an input array.
 *
 * @param   n       Vector dimension
 * @param   arr     Pointer to vector storage of length n
 */
double absmax(double* arr, int n) {
    for(int i = 0; i < n; i++) {
        if(arr[i] < 0) arr[i] *= -1;  // make positive
    }
    
    // Initialize maximum element 
    double max = arr[0]; 
  
    // Traverse array elements from second and compare every element with current max  
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) max = arr[i];
    }
  
    return max; 
}

/**
 * @brief Solve the matrix problem \f$ Hu=f \f$ using the thomas algorithm.
 *
 * @param   nsv     Dimension of matrix system
 * @param   H       Pointer to matrix storage of size ldh*nsv containing
 *                  the symmetric banded matrix.
 * @param   f       Pointer to vector of length nsv containing forcing for the
 *                  unknown degrees of freedom.
 * @param   u       Pointer to solution vector of length nsv, for the unknown
 *                  degrees of freedom.
 */
void SolveThomasAlgorithm(int nsv, double* H, double* f, double* u) {
    const int ldh = 2;  // Symmetric matrix, upper and lower diagonals are the same
    double* beta = new double[nsv];
    double* gamma = new double[nsv];
    
    beta[0] = H[1];
    gamma[0] = f[0]/beta[0];
    
    for (int i = 1; i < nsv; ++i) {
        beta[i] = H[i*ldh + 1] - H[i*ldh]*H[(i-1)*ldh]/beta[i-1];
        gamma[i] = (f[i] - H[i*ldh]*gamma[i-1])/beta[i];
    }
    
    u[nsv - 1] = gamma[nsv - 1];
    
    for (int i = nsv - 2; i >= 0; --i) {
        u[i] = gamma[i] - u[i+1]*H[i*ldh + 2]/beta[i];
    }

    delete[] beta;
    delete[] gamma;
}

void SolveThomasAlgorithm2(int nsv, double* H, double* f, double* u) {
    const int ldh = 3; // Non-symmetric matrix, tri-diagonal
    double* beta = new double[nsv];
    double* gamma = new double[nsv];
    
    beta[0] = H[1];
    gamma[0] = f[0]/beta[0];
    
    for (int i = 1; i < nsv; ++i) {
        beta[i] = H[i*ldh + 1] - H[i*ldh + 2]*H[(i-1)*ldh]/beta[i-1];
        gamma[i] = (f[i] - H[i*ldh + 2]*gamma[i-1])/beta[i];
    }
    
    u[nsv - 1] = gamma[nsv - 1];
    
    for (int i = nsv - 1; i >= 0; --i) {
        u[i] = gamma[i] - u[i+1]*H[i*ldh]/beta[i];
    }

    delete[] beta;
    delete[] gamma;
}

/**
 * @brief Solve the matrix problem \f$ Hu=f \f$ using the steepest descent
 * iterative algorithm.
 *
 * @param   nsv     Dimension of matrix system
 * @param   H       Pointer to matrix storage of size ldh*nsv containing
 *                  the symmetric banded matrix.
 * @param   f       Pointer to vector of length nsv containing forcing for the
 *                  unknown degrees of freedom.
 * @param   u       Pointer to solution vector of length nsv, for the unknown
 *                  degrees of freedom.
 */
void SolveSteepestDescent(int nsv, double* H, double* f, double* u) {
    int   ldh = 2;
    double* r = new double[nsv];
    double* t = new double[nsv]; // temp array

    double alpha;
    double eps;
    double tol = 1e-07;

    cblas_dcopy(nsv, f, 1, r, 1);     // r_0 = b (i.e. f)
    cblas_dsbmv(CblasColMajor, CblasUpper, nsv, 1, -1.0, H, ldh, u, 1, 1.0, r, 1);    // r_0 = b - A x_0

    do {
        cblas_dsbmv(CblasColMajor, CblasUpper, nsv, 1, 1.0, H, ldh, r, 1, 0.0, t, 1); // t = A r_k
        alpha = cblas_ddot(nsv, t, 1, r, 1);         // alpha = r_k^T A r_k
        alpha = cblas_ddot(nsv, r, 1, r, 1) / alpha; // alpha_k = r_k^T r_k / r_k^T A r_k

        cblas_daxpy(nsv, alpha, r, 1, u, 1);  // x_{k+1} = x_k + alpha_k r_k
        cblas_dsbmv(CblasColMajor, CblasUpper, nsv, 1, -1.0, H, ldh, u, 1, 1.0, r, 1); // r_k = b - A x_k

        eps = 0;
        for (int i = 0; i < nsv; ++i) {
            eps += pow(r[i], 2);
        }
        eps = sqrt(eps/(nsv + ldh));

        cblas_dcopy(nsv, r, 1, t, 1);
        
    } while (eps < tol); // Set a maximum eps value
    
    cout << "Final Eps: " << eps << endl;
    
    delete[] r;
    delete[] t;
}

/**
 * @brief Solves the Helmholtz problem for question 1 and 2.
 */
int main() {
    /** Code for Question 1. Please comment all if running question 2 */
/****************************************************************************/
//    const double dx  = 0.1;           // Grid-point spacing
//    const int    ldh = 2;             // Leading dimension of H (D + U)
//    const double lam = 1.0;           // Value of Lambda
//    const double L   = 1.0;           // Length of domain
//    const int    n   = (L / dx) + 1;  // Number of grid-points
//    const int    nsv = n - 2;         // Number of unknown DOFs
//    const int    m   = 3;
//
//    double* H = new double[ldh*nsv];// Helmholtz matrix storage
//    double* u = new double[n];      // Solution vector
//    double* f = new double[n];      // Forcing vector
//    double* e = new double[n];      // Exact solution vector
//
//    // Generate the Helmholtz matrix in symmetric banded storage.
//    FillHelmholtzMatrix(nsv, H, lam, dx);
//
//    cout << "Helmholtz matrix (symmetric banded storage): " << endl;
//    PrintMatrix(nsv, H, ldh);
//
//    // Problem 1 with f = -(lambda + m pi^2) sin(m pi x)
//    FillForcingFunction(n, f, lam, m, dx);
//    cblas_dscal(n, 1.0, u, 1);      // u_0 = 0
//    EnforceBoundaryConditions(n, f, u, lam, dx, m);
//
//    // Comment out SolveSteepestDescent if using Thomas algorithm and vice versa
//    // SolveSteepestDescent(nsv, H, f+1, u+1);
//    SolveThomasAlgorithm(nsv, H, f+1, u+1);
//    
//    cout << "Solution: " << endl;
//    PrintVector(n, u, dx);
//
//    cout << "Exact: " << endl;
//    ExactSolution(n, e, dx, m);
//    PrintVector(n, e, dx);
//
//    cblas_daxpy(n, -1.0, u, 1, e, 1);
//    cout << "Max Truncation Error: " << absmax(e, n) << endl;
/****************************************************************************/

    /** Code for Question 2. Please comment all if running question 1 */
/****************************************************************************/
    const double dx  = 0.002;         // Grid-point spacing
    const int    ldh = 3;             // Leading dimension of H (D + U + L)
    const double sig = 4.0;           // Value of Sigma
    const double L   = 1.0;           // Length of domain
    const int    n   = (L / dx) + 1;  // Number of grid-points
    const int    nsv = n - 1;         // Number of unknown DOFs
    const double T   = 0.05;          // Value of Time
    const double dt  = 2e-5;          // Time step
    const int    n_t = T/dt;          // Number of time steps
    const double lam = sig*dt/(dx*dx);// Gain Factor

    double* H = new double[ldh*nsv];// Helmholtz matrix storage
    double* u = new double[n];      // Solution vector
    double* f = new double[n];      // Forcing vector
    double* e = new double[n];      // Exact solution vector

    // Generate the Helmholtz matrix in banded storage.
    FillHelmholtzMatrix2(nsv, H, lam, dx);

    cout << "Helmholtz matrix (banded storage): " << endl;
    PrintMatrix(nsv, H, ldh);

    // Problem 2 with f = u @ time t = n
    FillForcingFunction2(n, f, dx);
    cblas_dscal(n, 1.0, u, 1);       // u_0 = 0
    
    // Vector purely for outputting Initial Condition values
    double* IC = new double[n];      // Initial Conditions
    cblas_dcopy(n, f, 1, IC, 1);     // IC = f
    
    for (int i = 1; i <= n_t; ++i) {
        EnforceBoundaryConditions2(n, f, u, sig, dx, i*dt);
        SolveThomasAlgorithm2(nsv, H, f+1, u+1);
        cblas_dcopy(n, u, 1, f, 1);     // f = u^n+1 (i.e. u)
    }
    
    cout << "Solution: " << endl;
    PrintVector(n, u, dx);

    cout << "Exact: " << endl;
    ExactSolution2(n, e, dx, T, sig);
    PrintVector(n, e, dx);
    
    // Vector purely for outputting Exact Solution values
    double* EX = new double[n];      // Exact solution
    cblas_dcopy(n, e, 1, EX, 1);     // EX = e

    // e = e - u, to compute difference between Exact and Numerical solution
    cblas_daxpy(n, -1.0, u, 1, e, 1);
    cout << "Max Truncation Error: " << absmax(e, n) << endl;
    
    // Code to output values to Output.txt
    ofstream vOut("Output.txt", ios::out | ios::trunc);
    /** Write Initial Condition, Numerical and Exact solution values */
    FORMAT(vOut, "x", "IC", "u", "e");
    vOut.precision(6);
    double x;
    for (int i = 0; i < n; ++i) {
        x = i*dx;
        FORMAT(vOut, x, IC[i], u[i], EX[i]);
    }
    vOut.close();

    delete[] IC;
    delete[] EX;
/****************************************************************************/

    // Clean up
    delete[] H;
    delete[] u;
    delete[] f;
    delete[] e;
}
