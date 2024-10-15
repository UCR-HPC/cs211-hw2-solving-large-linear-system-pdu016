#ifndef __MY_C__
#define __MY_C__

#include "include.h"
#include <stdio.h>
#include <stdlib.h>

int mydgetrf(double *A, int *ipiv, int n)
{
    for (int i = 0; i < n - 1; i++) {
        // Pivoting: find the row with the maximum element in the current column
        int maxind = i;
        double max = abs(A[i * n + i]);
        for (int t = i + 1; t < n; t++) {
            if (abs(A[t * n + i]) > max) {
                maxind = t;
                max = abs(A[t * n + i]);
            }
        }

        // Check for singularity
        if (max == 0) {
            printf("LU factorization failed: coefficient matrix is singular.\n");
            return 0;
        }

        // Swap rows if needed
        if (maxind != i) {
            int temp = ipiv[i];
            ipiv[i] = ipiv[maxind];
            ipiv[maxind] = temp;

            for (int j = 0; j < n; j++) {
                double tmp = A[i * n + j];
                A[i * n + j] = A[maxind * n + j];
                A[maxind * n + j] = tmp;
            }
        }

        // LU factorization
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] /= A[i * n + i];
            for (int k = i + 1; k < n; k++) {
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }
    return 1;
}

void mydtrsv(char UPLO, double *A, double *B, int n, int *ipiv)
{
    if (UPLO == 'L') {
        // Forward substitution to solve L * y = B
        for (int i = 0; i < n; i++) {
            int piv = ipiv[i];
            double temp = B[piv];
            B[piv] = B[i];
            B[i] = temp;

            for (int j = 0; j < i; j++) {
                B[i] -= A[i * n + j] * B[j];
            }
        }
    } else if (UPLO == 'U') {
        // Backward substitution to solve U * x = y
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                B[i] -= A[i * n + j] * B[j];
            }
            B[i] /= A[i * n + i];
        }
    }
}

void my_f(double *A, double *B, int n)
{
    int *ipiv = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        ipiv[i] = i;

    if (mydgetrf(A, ipiv, n) == 0) {
        printf("LU factorization failed: coefficient matrix is singular.\n");
        free(ipiv);
        return;
    }

    mydtrsv('L', A, B, n, ipiv);
    mydtrsv('U', A, B, n, ipiv);

    free(ipiv);
}

#endif