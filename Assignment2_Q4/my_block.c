#ifndef __MY_BLOCK_C__
#define __MY_BLOCK_C__

#include "include.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

void mydgemm(double *A, double *B, double *C, int n, int b, int row_offset, int col_offset)
{
    // mydgemm could also remain as void mydgemm(double *A,double *B,int n,int bid,int b)
    // I just save me some work with  int row_offset = bid * b; int col_offset = bid * b;
    // Implement matrix multiplication for block
    // C = A * B, where A is of size (n - row_offset) x b and B is of size b x (n - col_offset)
    for (int i = row_offset; i < n; i++) {
        for (int j = col_offset; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < b; k++) {
                sum += A[i * n + (col_offset + k)] * B[(row_offset + k) * n + j];
            }
            C[i * n + j] -= sum;
        }
    }
}

int mydgetrf_block(double *A, int *ipiv, int n)
{
    int b;
    FILE *pad_file = fopen("pad.txt", "r");
    if (pad_file == NULL) {
        printf("Error: pad.txt file not found.\n");
        return 0;
    }
    fscanf(pad_file, "%d", &b);
    fclose(pad_file);

    for (int ib = 0; ib < n; ib += b) {
        int end = (ib + b < n) ? ib + b : n;

        // Step 1: Factorize A(ib:end, ib:end) using mydgetrf (unblocked LU)
        for (int i = ib; i < end; i++) {
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
            for (int j = i + 1; j < end; j++) {
                A[j * n + i] /= A[i * n + i];
                for (int k = i + 1; k < end; k++) {
                    A[j * n + k] -= A[j * n + i] * A[i * n + k];
                }
            }
        }

        // Step 2: Update trailing matrix using mydgemm
        if (end < n) {
            for (int i = end; i < n; i++) {
                for (int j = ib; j < end; j++) {
                    A[i * n + j] /= A[j * n + j];
                }
            }
            mydgemm(A, A, A, n, b, end, end);
        }
    }
    return 1;
}

void my_block_f(double *A, double *B, int n)
{
    int *ipiv = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        ipiv[i] = i;
    }
    if (mydgetrf_block(A, ipiv, n) == 0) {
        printf("LU factoration failed: coefficient matrix is singular.\n");
        free(ipiv);
        return;
    }
    mydtrsv('L', A, B, n, ipiv);
    mydtrsv('U', A, B, n, ipiv);
    free(ipiv);
}

// Complete test function to test my_block_f with different block sizes from pad.txt
void test_my_block_f()
{
    int sizes[] = {1000, 2000, 3000, 4000, 5000};  // Same sizes as in Q2 for performance comparison
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        double *A = (double *)malloc(n * n * sizeof(double));
        double *B = (double *)malloc(n * sizeof(double));

        // Initialize matrix A and vector B with random values
        srand(time(NULL));
        for (int j = 0; j < n * n; j++) {
            A[j] = ((double)rand() / RAND_MAX) * 10;  // Random values between 0 and 10
        }
        for (int j = 0; j < n; j++) {
            B[j] = ((double)rand() / RAND_MAX) * 10;  // Random values between 0 and 10
        }

        // Measure the time taken for LU decomposition with blocking
        struct timeval start, end;
        gettimeofday(&start, NULL);
        my_block_f(A, B, n);
        gettimeofday(&end, NULL);

        double time_taken = (end.tv_sec - start.tv_sec) + 1e-6 * (end.tv_usec - start.tv_usec);
        double gflops = (2.0 / 3.0) * n * n * n / (time_taken * 1e9);
        printf("Matrix size: %d, Time taken: %.6f seconds, Performance: %.2f Gflops\n", n, time_taken, gflops);

        free(A);
        free(B);
    }
}

#endif
