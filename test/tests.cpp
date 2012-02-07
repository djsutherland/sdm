#include "sdm/basics.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <stdexcept>

#include <np-divs/div-funcs/div_l2.hpp>
#include <np-divs/div_params.hpp>

#include "sdm/kernel_projection.hpp"
#include "sdm/kernels.hpp"
#include "sdm/sdm.hpp"

#include <svm.h>

namespace {

using namespace sdm;
using namespace std;

TEST(UtilitiesTest, PSDProjection) {
    double mat[25] = {
        0.8404, -0.6003, -2.1384,  0.1240,  2.9080,
       -0.8880,  0.4900, -0.8396,  1.4367,  0.8252,
        0.1001,  0.7394,  1.3546, -1.9609,  1.3790,
       -0.5445,  1.7119, -1.0722, -0.1977, -1.0582,
        0.3035, -0.1941,  0.9610, -1.2078, -0.4686
    };
    double psd[25] = {
        1.5021, -0.4976, -0.5390, -0.2882,  0.7051,
       -0.4976,  1.0476, -0.1796,  0.7752, -0.2417,
       -0.5390, -0.1796,  1.9070, -1.0633,  0.6634,
       -0.2882,  0.7752, -1.0633,  1.0847, -0.6603,
        0.7051, -0.2417,  0.6634, -0.6603,  0.8627
    };
    project_to_symmetric_psd(mat, 5);

    for (size_t i = 0; i < 25; i++)
        EXPECT_NEAR(mat[i], psd[i], 1e-4) << "i = " << i;
}

TEST(UtilitiesTest, CovarianceProjection) {
    double mat[25] = {
        0.8404, -0.6003, -2.1384,  0.1240,  2.9080,
       -0.8880,  0.4900, -0.8396,  1.4367,  0.8252,
        0.1001,  0.7394,  1.3546, -1.9609,  1.3790,
       -0.5445,  1.7119, -1.0722, -0.1977, -1.0582,
        0.3035, -0.1941,  0.9610, -1.2078, -0.4686
    };
    double kernel[25] = {
        1.0000, -0.4095, -0.2387, -0.2668,  0.6339,
       -0.4095,  1.0000, -0.2026,  0.7296, -0.2502,
       -0.2387, -0.2026,  1.0000, -0.7841,  0.5518,
       -0.2668,  0.7296, -0.7841,  1.0000, -0.6903,
        0.6339, -0.2502,  0.5518, -0.6903,  1.0000
    };
    project_to_covariance(mat, 5);

    for (size_t i = 0; i < 25; i++)
        EXPECT_NEAR(mat[i], kernel[i], 1e-4) << "i = " << i;
}

TEST(SDMTest, BasicTrainingTesting) {
    typedef flann::Matrix<double> Matrix;

    // set up a stupid dataset: 1D N(0,1) vs N(.5, .5) with size-ten samples
    double train11[10] = { 0.5377, 1.8339,-2.2588, 0.8622, 0.3188,-1.3077,-0.4336, 0.3426, 3.5784, 2.7694};
    double train12[10] = { 0.8884,-1.1471,-1.0689,-0.8095,-2.9443, 1.4384, 0.3252,-0.7549, 1.3703,-1.7115};
    double train13[10] = {-0.1022,-0.2414, 0.3192, 0.3129,-0.8649,-0.0301,-0.1649, 0.6277, 1.0933, 1.1093};
    double train14[10] = {-0.8637, 0.0774,-1.2141,-1.1135,-0.0068, 1.5326,-0.7697, 0.3714,-0.2256, 1.1174};
    double train15[10] = {-1.0891, 0.0326, 0.5525, 1.1006, 1.5442, 0.0859,-1.4916,-0.7423,-1.0616, 2.3505};

    double train21[10] = { 1.2097, 0.6458, 0.5989, 1.2938, 0.0978, 0.8483, 0.9175, 0.3781, 0.6078,-0.0829};
    double train22[10] = {-0.0740, 0.5524, 0.8611, 1.7927, 0.1666, 0.5937, 0.4588,-0.4665, 0.2805,-0.3973};
    double train23[10] = { 0.9202, 0.0560, 0.5500, 0.2277, 0.6518, 0.1998, 0.7450, 0.8697, 1.3559, 0.4029};
    double train24[10] = {-0.5692, 0.0802, 1.1773,-0.0361, 0.9805, 0.5620, 1.2183,-0.4804, 0.4012,-0.1039};
    double train25[10] = { 1.9540, 0.9126, 1.1895,-0.0291, 0.2657, 0.3638, 1.0492, 0.3611, 0.8508,-0.5259};

    Matrix train[10] = {
        Matrix(train11, 10, 1),
        Matrix(train12, 10, 1),
        Matrix(train13, 10, 1),
        Matrix(train14, 10, 1),
        Matrix(train15, 10, 1),
        Matrix(train21, 10, 1),
        Matrix(train22, 10, 1),
        Matrix(train23, 10, 1),
        Matrix(train24, 10, 1),
        Matrix(train25, 10, 1)
    };

    std::vector<int> labels(10);
    for (size_t i = 0; i < 5; i++) {
        labels[i] = 0;
        labels[i+5] = 1;
    }

    // set up parameters
    NPDivs::DivL2 div_func;
    GaussianKernel kernel(div_func, .0067);

    NPDivs::DivParams div_params;
    div_params.num_threads = 1;

    // TODO: write a convenince function to set up svm_params, or something
    svm_parameter svm_params;
    svm_params.svm_type = C_SVC;
    svm_params.kernel_type = PRECOMPUTED;
    svm_params.gamma = 0;
    svm_params.degree = 0;
    svm_params.cache_size = 10;
    svm_params.C = .002;
    svm_params.eps = 1e-3;
    svm_params.nr_weight = 0;
    svm_params.shrinking = 0;
    svm_params.probability = 0;

    // train a model
    const SDM<double> &model =
        train_sdm(train, 10, labels, kernel, div_params, svm_params);

    // an N(0, 1) bag
    double test1[10] = {-1.3499, 3.0349, 0.7254,-0.0631, 0.7147,-0.2050,-0.1241, 1.4897, 1.4090, 1.4172};

    // test an N(.5, .5) bag with 5 samples
    double test2[5] = { 0.8357,-0.1037, 0.8586, 1.3151, 0.7444 };

    Matrix test[2] = {
        Matrix(test1, 10, 1),
        Matrix(test2, 5, 1)
    };
    vector< vector<double> > vals(2);

    vector<int> pred_lab = model.predict(test, 2, vals);

    EXPECT_EQ(0, pred_lab[0]);
    cout << "Vals[0]: " << vals[0][0] << " " << vals[0][1] << endl;

    EXPECT_EQ(1, pred_lab[1]);
    cout << "Vals[1]: " << vals[1][0] << " " << vals[1][1] << endl;

}

} // end namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
