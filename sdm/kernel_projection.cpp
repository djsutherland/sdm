/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Redistribution and use in source and binary forms, with or without          *
 * modification, are permitted provided that the following conditions are met: *
 *                                                                             *
 *     * Redistributions of source code must retain the above copyright        *
 *       notice, this list of conditions and the following disclaimer.         *
 *                                                                             *
 *     * Redistributions in binary form must reproduce the above copyright     *
 *       notice, this list of conditions and the following disclaimer in the   *
 *       documentation and/or other materials provided with the distribution.  *
 *                                                                             *
 *     * Neither the name of Carnegie Mellon University nor the                *
 *       names of the contributors may be used to endorse or promote products  *
 *       derived from this software without specific prior written permission. *
 *                                                                             *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   *
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         *
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        *
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  *
 * POSSIBILITY OF SUCH DAMAGE.                                                 *
 ******************************************************************************/
#include "sdm/basics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <boost/format.hpp>

#include "sdm/kernel_projection.hpp"

using boost::format;

namespace sdm {

////////////////////////////////////////////////////////////////////////////////
// General matrix math helpers

// symmetrize a matrix: m = (m + m') / 2
void symmetrize(double* matrix, size_t n) {
    // loop over half of matrix
    for (size_t i = 1; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            matrix[i + j*n] = matrix[j + i*n] =
                (matrix[i + j*n] + matrix[j + i*n]) / 2.;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Eigenvalues / eigenvectors helpers

// LAPACK: computes the eigenvalues/vectors of a square symmetric dense matrix
extern "C" void dsyev_(char *jobz, char *uplo, int *n, double *a,
        int *lda, double *w, double *work, int *lwork, int *info);

/* Takes an n x n symmetric matrix and computes its eigendecomposition.
 *
 * Eigenvalues go in the n-element arre vals (in ascending order).
 *
 * Orthonormal eigenvectors go in the columns of the n^2-element array vecs,
 * in column-major order.
 *
 * jobz determines whether eigenvalues are actually calculated, but use the
 * overload below if you don't want them.
 *
 * Throws std::domain_error if dsyev fails.
 */
void eig(double* matrix, int n, double* vals, double* vecs, char jobz='V') {
    char uplo = 'U'; // indicate that upper-triangular part of matrix is present
    int info; // indicates whether the call was successful
    int lwork = 3*n - 1; // size of the work array
    double *work = new double[lwork]; // the work array

    // copy matrix into vecs, since dsyev_ is in-place
    std::copy(matrix, matrix + n*n, vecs);

    dsyev_(&jobz, &uplo, &n, vecs, &n, vals, work, &lwork, &info);

    delete[] work;

    if (info < 0) {
        throw std::domain_error(
            (format("problem with dsyev argument %d") % (-info)).str());
    } else if (info > 0) {
        throw std::domain_error("dsyev: failed to converge");
    }
}

/* Takes an n x n symmetric matrix and stores its eigenvalues in the n-element
 * array vals, in ascending order.
 */
void eig(double* matrix, size_t n, double* vals) {
    double* vecs = new double[n*n]; // allocate memory for dsyev to destroy
    eig(matrix, n, vals, vecs, 'N');
    delete[] vecs;
}

////////////////////////////////////////////////////////////////////////////////
// Spectral reconstruction

// BLAS: scales a vector by a constant
extern "C" void dscal_(int *n, double *alpha, double *x, int *incx);

// BLAS: computes generalized matrix product  C := alpha * A * B + beta * C
extern "C" void dgemm_(char *transa, char *transb,
       int *m, int *n, int *k,
       double *alpha, double *a, int *lda, void *b, int *ldb, 
       double *beta, void *c, int *ldc);

/* Given an array of eigenvalues and corresponding orthonormal eigenvectors
 * (in column-major format), calculates the reconstructed matrix, optionally
 * throwing away any negative eigenvalues (which projects to the nearest
 * positive semidefinite matrix).
 */
void spectral_reconstruction(int n, double *eigvals, double *eigvecs,
        double *matrix, bool nonnegative_only = false)
{
    // We're calculating  V * max(diag(D), 0) * V'

    int step = 1;

    // First do the left-hand side: V * max(dig(D), 0)
    // Need to scale each column of V by the corresponding eigenvector
    double *leftside = new double[n * n];
    std::copy(eigvecs, eigvecs + n*n, leftside);

    for (size_t j = 0; j < n; j++) {
        double v = eigvals[j];
        if (nonnegative_only && v < 0)
            v = 0.;

        dscal_(&n, &v, eigvecs + j*n, &step);
    }

    // Now do (V * max(diag(D), 0)) * V'
    char no = 'N';
    char trans = 'T';
    double one = 1.;
    double zero = 0.;
    int size = (int) n;
    dgemm_(&no, &trans, &size, &size, &size,
            &one, leftside, &size, eigvecs, &size,
            &zero, matrix, &size);

    delete[] leftside;
}


////////////////////////////////////////////////////////////////////////////////

/* Takes an n x n matrix stored as a flat array, symmetrizes it, and projects
 * in-place to the nearest positive semidefinite matrix.
 */
void project_to_psd(double* matrix, size_t n) {
    double* eigvals = new double[n];
    double* eigvecs = new double[n*n];

    symmetrize(matrix, n);

    try {
        eig(matrix, n, eigvals, eigvecs);

        // are any of our eigenvalues actually negative?
        // (note that we get the values back in ascending order)
        if (eigvals[0] < 0) {
            spectral_reconstruction(n, eigvals, eigvecs, matrix, true);
            symmetrize(matrix, n);
        }

    } catch (...) { delete[] eigvals; delete[] eigvecs; throw; }
    delete[] eigvals; delete[] eigvecs;
}


/* Takes an n x n matrix, stored as a flat array, symmetrizes it, and projects
 * it in-place to the nearest positive semidefinite matrix with unit diagonal.
 */
void project_to_kernel(double* matrix, size_t n, size_t maxsteps, double tol) {
    // The alternating projection method (algorithm 3.3) of 
    //   Nicholas J. Higham, 2002.
    //   Computing the Nearest Correlation Matrix: a Problem from Finance.
    //   IMA Journal of Numerical Analysis, pages 329-343.

    double* eigvals = new double[n];
    double* eigvecs = new double[n*n];

    // Dykstra's correction
    double *S = new double[n*n];
    std::fill(S, S+n*n, 0);

    // R is the corrected form of matrix
    double *R = new double[n*n];

    // X is projected to be symmetric PSD
    double *X = new double[n*n];
    std::copy(matrix, matrix + n*n, X);

    // used to check for x's convergence
    double *prev_X = new double[n*n];


    try {
        symmetrize(matrix, n);

        for (size_t iter = 0; iter < maxsteps; iter++) {
            // remember the previous x
            std::copy(X, X + n*n, prev_X);

            // R is the matrix minus the correction
            for (size_t i = 0; i < n*n; i++)
                R[i] = matrix[i] - S[i];

            // X is projection of R to symmetric PSD matrix
            eig(R, n, eigvals, eigvecs);
            spectral_reconstruction(n, eigvals, eigvecs, X, true);
            symmetrize(X, n);

            // new correction is the difference between X and R
            for (size_t i = 0; i < n*n; i++)
                S[i] = X[i] - R[i];

            // new matrix is X with unit diagonal
            std::copy(X, X + n*n, matrix);
            for (size_t i = 0; i < n; i++)
                matrix[i + i*n] = 1.;

            // have we converged?
            if (iter > 0) {
                double biggest_x = 0;
                double biggest_change = 0;
                for (size_t i = 0; i < n*n; i++) {
                    double mag = std::abs(X[i]);
                    if (mag > biggest_x)
                        biggest_x = mag;

                    double change = std::abs(X[i] - prev_X[i]);
                    if (change > biggest_change)
                        biggest_change = change;
                }

                if (biggest_change <= 1e-7 * biggest_x) {
                    break;
                }
            }
        }

        // make sure we don't have any too-negative eigenvalues
        eig(matrix, n, eigvals);
        if (eigvals[0] < tol) {
            throw std::domain_error(
                    (format("Failed to project to kernel matrix: min eig %g")
                     % eigvals[0]).str());
        }

    } catch (...) { // fake a finally block
        delete[] eigvals; delete[] eigvecs;
        delete[] S; delete[] R;
        delete[] X; delete[] prev_X;
        throw;
    }
    delete[] eigvals; delete[] eigvecs;
    delete[] S; delete[] R;
    delete[] X; delete[] prev_X;
}

} // end namespace
