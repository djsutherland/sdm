#include "sdm/basics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <np-divs/div-funcs/div_l2.hpp>
#include <np-divs/div-funcs/div_linear.hpp>
#include <np-divs/div-funcs/div_renyi.hpp>
#include <np-divs/div_params.hpp>

#include "sdm/kernel_projection.hpp"
#include "sdm/kernels/gaussian.hpp"
#include "sdm/kernels/linear.hpp"
#include "sdm/kernels/polynomial.hpp"
#include "sdm/sdm.hpp"

#include <svm.h>

namespace {

// TODO: test using precomputed divergences

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

TEST(UtilitiesTest, CVSplitter) {
    double m[8*8] = {
          1,  2,  3,  4,  5,  6,  7,  8,
         11, 12, 13, 14, 15, 16, 17, 18,
         21, 22, 23, 24, 25, 26, 27, 28,
         31, 32, 33, 34, 35, 36, 37, 38,
         41, 42, 43, 44, 45, 46, 47, 48,
         51, 52, 53, 54, 55, 56, 57, 58,
         61, 62, 63, 64, 65, 66, 67, 68,
         71, 72, 73, 74, 75, 76, 77, 78
    };

    double train[6*6];
    double test[2*6];


    detail::copy_from_full_to_split(m, train, test, 0, 6, 2);
    double exp_train_1[6*6] = {
        23, 24, 25, 26, 27, 28,
        33, 34, 35, 36, 37, 38,
        43, 44, 45, 46, 47, 48,
        53, 54, 55, 56, 57, 58,
        63, 64, 65, 66, 67, 68,
        73, 74, 75, 76, 77, 78
    };
    double exp_test_1[2*6] = {
         3,  4,  5,  6,  7,  8,
        13, 14, 15, 16, 17, 18
    };
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(exp_train_1[i*6 + j], train[i*6 + j]);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(exp_test_1[i*6 + j], test[i*6 + j]);


    detail::copy_from_full_to_split(m, train, test, 2, 6, 2);
    double exp_train_2[6*6] = {
         1,  2,  5,  6,  7,  8,
        11, 12, 15, 16, 17, 18,
        41, 42, 45, 46, 47, 48,
        51, 52, 55, 56, 57, 58,
        61, 62, 65, 66, 67, 68,
        71, 72, 75, 76, 77, 78
    };
    double exp_test_2[2*6] = {
        21, 22, 25, 26, 27, 28,
        31, 32, 35, 36, 37, 38
    };
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(exp_train_2[i*6 + j], train[i*6 + j]);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(exp_test_2[i*6 + j], test[i*6 + j]);


    detail::copy_from_full_to_split(m, train, test, 6, 6, 2);
    double exp_train_3[6*6] = {
         1,  2,  3,  4,  5,  6,
        11, 12, 13, 14, 15, 16,
        21, 22, 23, 24, 25, 26,
        31, 32, 33, 34, 35, 36,
        41, 42, 43, 44, 45, 46,
        51, 52, 53, 54, 55, 56,
    };
    double exp_test_3[2*6] = {
        61, 62, 63, 64, 65, 66,
        71, 72, 73, 74, 75, 76
    };
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(exp_train_3[i*6 + j], train[i*6 + j]);
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 6; j++)
            EXPECT_EQ(exp_test_3[i*6 + j], test[i*6 + j]);

}

#define NUM_TRAIN 10
#define TRAIN_SIZE 10
#define NUM_TEST 4
class EasySmallSDMTest : public ::testing::Test {
    protected:

    typedef flann::Matrix<double> Matrix;

    size_t num_train;
    double train_raw[NUM_TRAIN][TRAIN_SIZE];
    Matrix train[NUM_TRAIN];
    std::vector<int> labels;
    std::vector<double> means;
    std::vector<double> stds;

    size_t num_test;
    double **test_raw;
    Matrix test[NUM_TEST];
    std::vector<int> test_labels;
    std::vector<double> test_means;
    std::vector<double> test_stds;

    npdivs::DivParams div_params;
    svm_parameter svm_params;

    EasySmallSDMTest() :
        num_train(NUM_TRAIN),
        num_test(NUM_TEST),
        div_params(3, flann::KDTreeSingleIndexParams(), flann::SearchParams(-1),
                0, false),
        svm_params(default_svm_params)
    {
        double train_bleh[NUM_TRAIN][TRAIN_SIZE] = {
            // 10 samples from N(0, 1)
            { 0.5377, 1.8339,-2.2588, 0.8622, 0.3188,-1.3077,-0.4336, 0.3426, 3.5784, 2.7694},
            { 0.8884,-1.1471,-1.0689,-0.8095,-2.9443, 1.4384, 0.3252,-0.7549, 1.3703,-1.7115},
            {-0.1022,-0.2414, 0.3192, 0.3129,-0.8649,-0.0301,-0.1649, 0.6277, 1.0933, 1.1093},
            {-0.8637, 0.0774,-1.2141,-1.1135,-0.0068, 1.5326,-0.7697, 0.3714,-0.2256, 1.1174},
            {-1.0891, 0.0326, 0.5525, 1.1006, 1.5442, 0.0859,-1.4916,-0.7423,-1.0616, 2.3505},

            // 10 samples from N(2, .5)
            { 2.3719, 1.7295, 2.1884, 1.6470, 2.4262, 1.0994, 1.4593, 1.6393, 1.9404, 1.2268},
            { 2.0136, 2.3392, 2.2173, 2.3234, 1.3115, 2.6987, 0.6268, 2.1646, 1.9145, 1.9381},
            { 1.7944, 1.5712, 1.8728, 0.9059, 2.2390, 1.6922, 2.1441, 1.7649, 2.9298, 2.8285},
            { 1.7341, 1.7434, 1.5150, 2.6200, 1.7308, 2.0120, 2.1880, 1.5013, 1.5420, 1.4232},
            { 1.6267, 1.6023, 3.1278, 2.2319, 2.2343, 2.1287, 2.4394, 2.4871, 2.3871, 2.6310}
        };

        for (size_t i = 0; i < NUM_TRAIN; i++) {
            std::copy(train_bleh[i], &train_bleh[i][TRAIN_SIZE], train_raw[i]);
            train[i] = Matrix(train_raw[i], TRAIN_SIZE, 1);
        }

        labels.resize(NUM_TRAIN);
        means.resize(NUM_TRAIN);
        stds.resize(NUM_TRAIN);
        for (size_t i = 0; i < 5; i++) {
            labels[i]   = 0; means[i]   = 0; stds[i]   = 1;
            labels[i+5] = 1; means[i+5] = 2; stds[i+5] = .5;
        }

        test_labels.resize(NUM_TEST);
        test_means.resize(NUM_TEST);
        test_stds.resize(NUM_TEST);

        test_raw = new double*[NUM_TEST];
        size_t i = 0;

        // 10 from N(0, 1)
        double test11[10] = {-1.3499, 3.0349, 0.7254,-0.0631, 0.7147,-0.2050,-0.1241, 1.4897, 1.4090, 1.4172};
        test_raw[i] = new double[10]; std::copy(test11, test11+10, test_raw[i]);
        test[i] = Matrix(test_raw[i], 10, 1);
        test_labels[i] = 0;
        test_means[i] = 0;
        test_stds[i] = 1;
        i++;

        // 5 from N(0, 1)
        double test12[5] = {-0.5377, 1.8339,-2.2588, 0.8622, 0.3188};
        test_raw[i] = new double[5]; std::copy(test12, test12+5, test_raw[i]);
        test[i] = Matrix(test_raw[i], 5, 1);
        test_labels[i] = 0;
        test_means[i] = 0;
        test_stds[i] = 1;
        i++;

        // 10 from N(2, .5)
        double test21[10] = { 1.6125, 2.4121, 2.7705, 2.0103, 1.0000, 2.5473, 1.7158, 2.6855, 2.0631, 1.8725};
        test_raw[i] = new double[10]; std::copy(test21, test21+10, test_raw[i]);
        test[i] = Matrix(test_raw[i], 5, 1);
        test_labels[i] = 1;
        test_means[i] = 2;
        test_stds[i] = .5;
        i++;

        // 5 from N(2, .5)
        double test22[5] = { 1.9902, 3.1628, 2.3821, 1.8842, 0.7164};
        test_raw[i] = new double[5]; std::copy(test22, test22+5, test_raw[i]);
        test[i] = Matrix(test_raw[i], 5, 1);
        test_labels[i] = 1;
        test_means[i] = 2;
        test_stds[i] = .5;
        i++;
    }

    ~EasySmallSDMTest() {
        for (size_t i = 0; i < NUM_TEST; i++)
            delete[] test_raw[i];
        delete[] test_raw;
    }

    vector< vector<double> > testTrainTest(
            const npdivs::DivFunc &div_func,
            const KernelGroup &kernel_group,
            const vector<double> &cs = default_c_vals,
            size_t tuning_folds = NUM_TRAIN)
    {
        // train up the model
        SDM<double, int> *model = train_sdm(train, num_train, labels, div_func,
                kernel_group, div_params, cs, svm_params, tuning_folds);

        // predict on test data
        vector< vector<double> > vals;
        vector<int> pred_lab = model->predict(test, num_test, vals);

        // check that labels are as expected
        for (size_t i = 0; i < num_test; i++) {
            EXPECT_EQ(test_labels[i], pred_lab[i]) << "mislabeled #" << i;
        }

        // clean up the model
        model->destroyModelAndProb();
        delete model;

        return vals;
    }

    void testTransduct(
            const npdivs::DivFunc &div_func,
            const KernelGroup &kernel_group,
            const vector<double> &cs = default_c_vals,
            size_t tuning_folds = NUM_TRAIN)
    {
        // get transductive predictions
        vector<int> pred_labs = transduct_sdm(
            train, num_train, labels, test, num_test,
            div_func, kernel_group, div_params, cs, svm_params, tuning_folds);

        // check that labels are as expected
        for (size_t i = 0; i < num_test; i++) {
            EXPECT_EQ(test_labels[i], pred_labs[i]) << "mislabeled #" << i;
        }
    }


    double testTrainTestRegression(
            const vector<double> &train_labs,
            const vector<double> &test_labs,
            double tol,
            const npdivs::DivFunc &div_func,
            const KernelGroup &kernel_group,
            const vector<double> &cs = default_c_vals,
            size_t tuning_folds = NUM_TRAIN)
    {
        // train up the model
        SDM<double, double> *model = train_sdm(train, num_train, train_labs,
                div_func, kernel_group, div_params, cs, svm_params,
                tuning_folds);

        // predict on test data
        vector<double> pred_lab = model->predict(test, num_test);

        // check that labels are as expected
        double rmse = 0;
        for (size_t i = 0; i < num_test; i++) {
            double diff = test_labels[i] - pred_lab[i];
            rmse += diff * diff;
            EXPECT_LT(std::abs(diff), tol) << "bad pred for #" << i;
        }

        // clean up the model
        model->destroyModelAndProb();
        delete model;

        return std::sqrt(rmse / NUM_TEST);
    }


    double testCV(
            size_t folds,
            const npdivs::DivFunc &div_func,
            const KernelGroup &kernel_group,
            size_t cv_threads = 0,
            const vector<double> &cs = default_c_vals,
            size_t tuning_folds = 3,
            bool project_all = true,
            bool shuffle = true)
    {
        return crossvalidate(train, num_train, labels, div_func, kernel_group,
            div_params, folds, cv_threads, project_all, shuffle, cs, svm_params,
            tuning_folds);
    }

    double testCVRegression(
            const vector<double> &labs,
            size_t folds,
            const npdivs::DivFunc &div_func,
            const KernelGroup &kernel_group,
            size_t cv_threads = 0,
            const vector<double> &cs = default_c_vals,
            size_t tuning_folds = 3,
            bool project_all = true,
            bool shuffle = true)
    {
        return crossvalidate(train, num_train, labs, div_func, kernel_group,
            div_params, folds, cv_threads, project_all, shuffle, cs, svm_params,
            tuning_folds);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Test basic classification

TEST_F(EasySmallSDMTest, BasicTrainingTesting) {
    npdivs::DivL2 div_func;

    std::vector<double> sigs(1, .00671082);
    GaussianKernelGroup kernel_group(sigs, false);

    std::vector<double> cs(1, 1./512.);

    svm_params.probability = 0;
    div_params.k = 2;

    const vector< vector<double> > &vals =
        testTrainTest(div_func, kernel_group, cs);
}

TEST_F(EasySmallSDMTest, RenyiCVTrainingTesting) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    const vector< vector<double> > &vals =
        testTrainTest(div_func, kernel_group);
}

TEST_F(EasySmallSDMTest, PolyCVTrainingTesting) {
    npdivs::DivLinear div_func;
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    const vector< vector<double> > &vals =
        testTrainTest(div_func, kernel_group);
}

TEST_F(EasySmallSDMTest, RenyiCVTransduction) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    testTransduct(div_func, kernel_group);
}

TEST_F(EasySmallSDMTest, PolyCVTransduction) {
    npdivs::DivLinear div_func;
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    testTransduct(div_func, kernel_group);
}

////////////////////////////////////////////////////////////////////////////////
// Test basic regression

TEST_F(EasySmallSDMTest, DISABLED_RenyiCVTrainingTestingMeans) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testTrainTestRegression(means, test_means, 1.5,
            div_func, kernel_group);
    cout << "RMSE: " << rmse << "\n";
}

TEST_F(EasySmallSDMTest, DISABLED_PolyCVTrainingTestingMeans) {
    npdivs::DivRenyi div_func(.99);
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testTrainTestRegression(means, test_means, 1.5,
            div_func, kernel_group);
    cout << "RMSE: " << rmse << "\n";
}

// TODO: these tests shouldn't take so long and do so poorly.
//       what's up with them?

TEST_F(EasySmallSDMTest, RenyiCVTrainingTestingStds) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testTrainTestRegression(stds, test_stds, 1,
            div_func, kernel_group);
    cout << "RMSE: " << rmse << "\n";
}

TEST_F(EasySmallSDMTest, DISABLED_PolyCVTrainingTestingStds) {
    npdivs::DivRenyi div_func(.99);
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testTrainTestRegression(stds, test_stds, 1,
            div_func, kernel_group);
    cout << "RMSE: " << rmse << "\n";
}

////////////////////////////////////////////////////////////////////////////////
// Test classification CV

TEST_F(EasySmallSDMTest, CVPolySingleThreaded) {
    npdivs::DivLinear div_func;
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;
    div_params.num_threads = 1;

    double acc = testCV(5, div_func, kernel_group, 1);
    cout << "Accuracy: " << acc << endl;
    EXPECT_GE(acc, .7);
}

TEST_F(EasySmallSDMTest, CVRenyiSingleThreaded) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;
    div_params.num_threads = 1;

    double acc = testCV(5, div_func, kernel_group, 1);
    cout << "Accuracy: " << acc << endl;
    EXPECT_GE(acc, .7);
}

TEST_F(EasySmallSDMTest, CVPolyTwoThreaded) {
    npdivs::DivLinear div_func;
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double acc = testCV(5, div_func, kernel_group, 2);
    cout << "Accuracy: " << acc << endl;
    EXPECT_GE(acc, .7);
}

TEST_F(EasySmallSDMTest, CVRenyiTwoThreaded) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double acc = testCV(5, div_func, kernel_group, 2);
    cout << "Accuracy: " << acc << endl;
    EXPECT_GE(acc, .7);
}

TEST_F(EasySmallSDMTest, CVPolyDefaultThreaded) {
    npdivs::DivLinear div_func;
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double acc = testCV(5, div_func, kernel_group, 0);
    cout << "Accuracy: " << acc << endl;
    EXPECT_GE(acc, .7);
}

TEST_F(EasySmallSDMTest, CVRenyiDefaultThreaded) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double acc = testCV(5, div_func, kernel_group, 0);
    cout << "Accuracy: " << acc << endl;
    EXPECT_GE(acc, .7);
}

////////////////////////////////////////////////////////////////////////////////
// Test regression CV

TEST_F(EasySmallSDMTest, CVPolyMeans) {
    npdivs::DivLinear div_func;
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testCVRegression(means, 5, div_func, kernel_group, 0);
    cout << "RMSE: " << rmse << endl;
    EXPECT_LT(rmse, .8);
}
TEST_F(EasySmallSDMTest, CVRenyiMeans) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testCVRegression(means, 5, div_func, kernel_group, 0);
    cout << "RMSE: " << rmse << endl;
    EXPECT_LT(rmse, .8);
}


TEST_F(EasySmallSDMTest, CVPolyStds) {
    npdivs::DivLinear div_func;
    PolynomialKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testCVRegression(stds, 5, div_func, kernel_group, 0);
    cout << "RMSE: " << rmse << endl;
    EXPECT_LT(rmse, .2);
}
TEST_F(EasySmallSDMTest, CVRenyiStds) {
    npdivs::DivRenyi div_func(.99);
    GaussianKernelGroup kernel_group;

    svm_params.probability = 0;
    div_params.k = 2;

    double rmse = testCVRegression(stds, 5, div_func, kernel_group, 0);
    cout << "RMSE: " << rmse << endl;
    EXPECT_LT(rmse, .2);
}


} // end namespace


int main(int argc, char **argv) {
    FILELog::ReportingLevel() = logWARNING;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
