#include "sdm/basics.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <stdexcept>

#include "sdm/kernel_projection.hpp"

namespace {

using namespace SDM;
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
    project_to_psd(mat, 5);

    for (size_t i = 0; i < 25; i++)
        EXPECT_NEAR(mat[i], psd[i], 1e-4) << "i = " << i;
}

TEST(UtilitiesTest, KernelProjection) {
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
    project_to_kernel(mat, 5);

    for (size_t i = 0; i < 25; i++)
        EXPECT_NEAR(mat[i], kernel[i], 1e-4) << "i = " << i;
}

} // end namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
