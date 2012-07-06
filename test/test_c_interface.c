#include <sdm/sdm_c.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

#define NUM_BAGS 7
#define PTS_PER_BAG 3
#define DIM 2

void print_progress(size_t n) {
    printf("NN searches left: %zu\n", n);
}

int main() {
    int i, j;

    sdm_set_log_level(logWARNING);

    double ** d = (double **) malloc(NUM_BAGS * sizeof(double *));
    for (i = 0; i < NUM_BAGS; i++)
        d[i] = malloc(DIM * PTS_PER_BAG * sizeof(double));

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

    size_t rows[NUM_BAGS] = { 3, 3, 3, 3, 3, 3, 3 };
    int labels[NUM_BAGS] = { 0, 0, 1, 1, 2, 1, 0};
    double means[NUM_BAGS] = { 0.29, 0.305, 2.625, 2.589, 2.255, 2.325, 0.303};

    struct FLANNParameters flann_params = DEFAULT_FLANN_PARAMETERS;
    flann_params.algorithm = FLANN_INDEX_KDTREE_SINGLE;

    DivParamsC div_params = {
        1, // k
        flann_params,
        1, // num_threads
        3, // how often to print progress
        NULL
    };

    double acc = sdm_crossvalidate_classify_double(
            data, NUM_BAGS, rows, DIM, labels,
            "renyi:.9", "gaussian",
            &div_params, 2, 0, 1, 1,
            default_c_vals, num_default_c_vals, &default_svm_params, 2);
    printf("CV acc: %g\n", acc);

    printf("\n\nTraining SDM\n");
    SDM_ClassifyD *sdm = SDM_ClassifyD_train(
            data, NUM_BAGS - 2, DIM, rows, labels,
            "renyi:.9", "gaussian",
            &div_params, default_c_vals, num_default_c_vals,
            &default_svm_params,
            2, NULL);
    printf("Name: %s\n", SDM_ClassifyD_getName(sdm));

    printf("\n\nSingle predictions:\n");
    for (i = 0; i < NUM_BAGS; i++) {
        printf("%d: %d\n", i, SDM_ClassifyD_predict(sdm, data[i], rows[i]));
    }

    printf("\n\nMass predictions, with decision values:\n");
    int *pred_labels = (int *) malloc(NUM_BAGS * sizeof(int));
    double **dec_vals;
    size_t num_vals;
    SDM_ClassifyD_predict_many_vals(sdm, data, NUM_BAGS, rows,
            pred_labels, &dec_vals, &num_vals);
    for (i = 0; i < NUM_BAGS; i++) {
        printf("%d: %d   Vals: ", i, pred_labels[i]);
        for (j = 0; j < num_vals; j++)
            printf("%g ", dec_vals[i][j]);
        printf("\n");
    }

    SDM_ClassifyD_freeModel(sdm);

    double rmse = sdm_crossvalidate_regress_double(
            data, NUM_BAGS, rows, DIM, means,
            "renyi:.9", "gaussian",
            &div_params, 2, 0, 1, 1,
            default_c_vals, num_default_c_vals, &default_svm_params, 2);
    printf("\n\nCV mean prediction RMSE: %g\n", rmse);
}
