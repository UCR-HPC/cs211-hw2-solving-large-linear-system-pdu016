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

int generate_non_singular_matrix(double *A, int n) {
    int retry_count = 0;
    int max_retries = 10;
    while (retry_count < max_retries) {
        // Generate random matrix
        for (int i = 0; i < n * n; i++) {
            A[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n; // Increase diagonal elements to reduce singularity chances
        }

        // Test if matrix is singular during LU factorization
        int *ipiv = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            ipiv[i] = i;
        }
        if (mydgetrf(A, ipiv, n) != 0) {
            free(ipiv);
            return 1; // Successfully generated a non-singular matrix
        }
        free(ipiv);
        retry_count++;
    }
    return 0; // Failed to generate a non-singular matrix after several tries
}

void performance_test()
{
    int sizes[] = {1000, 2000, 3000, 4000, 5000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        double *A = (double *)malloc(n * n * sizeof(double));
        double *B = (double *)malloc(n * sizeof(double));

        if (!generate_non_singular_matrix(A, n)) {
            printf("Failed to generate a non-singular matrix for size %d after multiple attempts.\n", n);
            free(A);
            free(B);
            continue;
        }

        for (int j = 0; j < n; j++) {
            B[j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }

        struct timeval start, end;
        gettimeofday(&start, NULL);
        my_f(A, B, n);
        gettimeofday(&end, NULL);

        double time_taken = (end.tv_sec - start.tv_sec) + 1e-6 * (end.tv_usec - start.tv_usec);
        double gflops = (2.0 * n * n * n / 3.0) / (time_taken * 1e9);

        printf("Matrix size: %d, Time taken: %.2f seconds, Performance: %.2f Gflops\n", n, time_taken, gflops);

        free(A);
        free(B);
    }
}

#endif