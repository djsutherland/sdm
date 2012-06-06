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
    int i, j;
    double ** d = (double **) malloc(num_data * sizeof(double *));
    for (i = 0; i < num_data; i++)
        d[i] = malloc(dim * 3 * sizeof(double));

    d[0][0] =  0.;   d[0][1] = 0.1 ;
    d[0][2] =  0.01; d[0][3] = 0.89;
    d[0][2] = -0.2;  d[0][5] = 0.95;

    d[1][0] =  0.05; d[1][1] = 0.2 ;
    d[1][2] =  0.02; d[1][3] = 0.90;
    d[1][4] = -0.3;  d[1][5] = 0.96;

    d[2][0] =  4. ;  d[2][1] = 0.2 ;
    d[2][2] =  5. ;  d[2][3] = 0.96;
    d[2][4] =  4.6;  d[2][5] = 0.99;

    d[3][0] =  5. ;  d[3][1] = 0.1 ;
    d[3][2] =  4. ;  d[3][3] = 0.95;
    d[3][4] =  4.5;  d[3][5] = 0.98;

    d[4][0] =  4. ;  d[4][1] = 0.1 ;
    d[4][2] =  4. ;  d[4][3] = 0.95;
    d[4][4] =  3.5;  d[4][5] = 0.98;

    d[5][0] =  4.2;  d[5][1] = 0.12;
    d[5][2] =  4.1;  d[5][3] = 0.94;
    d[5][4] =  3.6;  d[5][5] = 0.99;

    d[6][0] =  0.03; d[6][1] = 0.21;
    d[6][2] =  0.03; d[6][3] = 0.92;
    d[6][4] = -0.31; d[6][5] = 0.94;

    const double ** data = (const double **) d;

    size_t rows[num_data] = { 3, 3, 3, 3, 3, 3, 3 };
    int labels[num_data] = { 0, 0, 1, 1, 2, 1, 0};

    struct FLANNParameters flann_params = DEFAULT_FLANN_PARAMETERS;
    flann_params.algorithm = FLANN_INDEX_KDTREE_SINGLE;

    DivParamsC div_params = {
        1, // k
        flann_params,
        1, // num_threads
        3, // how often to print progress
        NULL
    };

    const double cvals[10] = { // 2^-9, 2^-6, ..., 2^18
        1./512., 1./64., 1./8., 1, 1<<3, 1<<6, 1<<9, 1<<12, 1<<15, 1<<18
    };

    double acc = crossvalidate_bags_double(
            data, num_data, rows, dim, labels, 
            "renyi:.9", "gaussian", 
            &div_params, 2, 0, 1, 1,
            cvals, 10, &default_svm_params,
            2);
    printf("CV acc: %g\n", acc);

    printf("\n\nTraining SDM\n");
    SDMObjDouble *sdm = train_sdm_double(
            data, num_data - 2, dim, rows, labels,
            "renyi:.9", "gaussian",
            &div_params, cvals, 10, &default_svm_params,
            2, NULL);
    printf("Name: %s\n", SDMObjDouble_getName(sdm));

    printf("\n\nSingle predictions:\n");
    for (i = 0; i < num_data; i++) {
        printf("%d: %d\n", i, sdm_predict_double(sdm, data[i], rows[i]));
    }

    printf("\n\nMass predictions, with decision values:\n");
    int *pred_labels = (int *) malloc(num_data * sizeof(int));
    double **dec_vals;
    size_t num_vals;
    sdm_predict_many_vals_double(sdm, data, num_data, rows,
            pred_labels, &dec_vals, &num_vals);
    for (i = 0; i < num_data; i++) {
        printf("%d: %d   Vals: ", i, pred_labels[i]);
        for (j = 0; j < num_vals; j++)
            printf("%g ", dec_vals[i][j]);
        printf("\n");
    }

    SDMObjDouble_freeModel(sdm);
}
