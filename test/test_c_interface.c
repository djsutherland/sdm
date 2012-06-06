#include <sdm/sdm_c.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#define num_data 7
#define dim 2

void print_progress(size_t n) {
    printf("NN searches left: %zu\n", n);
}

int main() {
    int i;
    float ** data = (float **) malloc(num_data * sizeof(float *));
    for (i = 0; i < num_data; i++)
        data[i] = malloc(dim * 3 * sizeof(float));

    data[0][0] =  0.;   data[0][1] = 0.1 ;
    data[0][2] =  0.01; data[0][3] = 0.89;
    data[0][2] = -0.2;  data[0][5] = 0.95;

    data[1][0] =  0.05; data[1][1] = 0.2 ;
    data[1][2] =  0.02; data[1][3] = 0.90;
    data[1][4] = -0.3;  data[1][5] = 0.96;

    data[2][0] =  4. ;  data[2][1] = 0.2 ;
    data[2][2] =  5. ;  data[2][3] = 0.96;
    data[2][4] =  4.6;  data[2][5] = 0.99;

    data[3][0] =  5. ;  data[3][1] = 0.1 ;
    data[3][2] =  4. ;  data[3][3] = 0.95;
    data[3][4] =  4.5;  data[3][5] = 0.98;

    data[4][0] =  4. ;  data[4][1] = 0.1 ;
    data[4][2] =  4. ;  data[4][3] = 0.95;
    data[4][4] =  3.5;  data[4][5] = 0.98;

    data[5][0] =  4.2;  data[5][1] = 0.12;
    data[5][2] =  4.1;  data[5][3] = 0.94;
    data[5][4] =  3.6;  data[5][5] = 0.99;

    data[6][0] =  0.03; data[6][1] = 0.21;
    data[6][2] =  0.03; data[6][3] = 0.92;
    data[6][4] = -0.31; data[6][5] = 0.94;

    size_t rows[num_data] = { 3, 3, 3, 3, 3, 3, 3 };
    int labels[num_data] = { 0, 0, 1, 1, 1, 1, 0};

    DivParamsC div_params = {
        1, // k
        DEFAULT_FLANN_PARAMETERS,
        0, // num_threads
        1,
        print_progress
    };

    const double cvals[10] = { // 2^-9, 2^-6, ..., 2^18
        1./512., 1./64., 1./8., 1, 1<<3, 1<<6, 1<<9, 1<<12, 1<<15, 1<<18
    };

    double acc = crossvalidate_bags_float(
            data, num_data, rows, dim, labels, 
            "renyi:.9", "gaussian", 
            &div_params, 2, 0, 1, 1,
            cvals, 10, &default_svm_params,
            2);
    printf("%g\n", acc);
}
