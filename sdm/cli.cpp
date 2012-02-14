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
#include "sdm/sdm.hpp"
#include "sdm/kernels/linear.hpp"
#include "sdm/kernels/polynomial.hpp"
#include "sdm/kernels/gaussian.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/utility.hpp>

#include <flann/flann.hpp>

#include <np-divs/matrix_io.hpp>
#include <np-divs/div-funcs/from_str.hpp>

// TODO: more general CLI (multiclass, etc)
//       probably switch to a "train_bags" that includes labels

using std::cerr;
using std::cin;
using std::cout;
using std::ifstream;
using std::endl;
using std::string;

namespace po = boost::program_options;

// TODO: leaks memory everywhere, but whatever...
struct ProgOpts : boost::noncopyable {
    string pos_bags_file;
    string neg_bags_file;

    string test_bags_file;
    string test_labels_file;

    npdivs::DivFunc * div_func;
    sdm::KernelGroup * kernel_group;

    size_t k;
    size_t num_threads;
    size_t tuning_folds;
    bool prob;

    flann::IndexParams * index_params;
    flann::SearchParams * search_params;

    void parse_div_func(const string spec) {
        div_func = npdivs::div_func_from_str(spec);
    }

    void parse_kernel(const string name) {
        // TODO: support specifying cs / sigmas / degrees / etc
        if (name == "gaussian") {
            kernel_group = new sdm::GaussianKernelGroup;
        } else if (name == "linear") {
            kernel_group = new sdm::LinearKernelGroup;
        } else if (name == "polynomial") {
            kernel_group = new sdm::PolynomialKernelGroup;
        } else {
            throw std::domain_error((
                        boost::format("unknown kernel type %s") % name).str());
        }
    }

    void parse_index(const string name) {
        // TODO: more index types, support arguments
        if (name == "linear" || name == "brute") {
            index_params = new flann::LinearIndexParams;

        } else if (name == "kdtree" || name == "kd") {
            index_params = new flann::KDTreeSingleIndexParams;

        } else {
            throw std::domain_error((
                        boost::format("unknown index type %s") % name).str());
        }
    }
};

bool parse_args(int argc, char ** argv, ProgOpts& opts);

int main(int argc, char ** argv) {
    typedef flann::Matrix<double> Matrix;

    try {
        ProgOpts opts;
        if (!parse_args(argc, argv, opts))
            return 1;

        // load positives  // TODO: gracefully handle nonexisting files
        size_t num_pos;
        Matrix* pos_bags;
        if (opts.pos_bags_file == "-") {
            pos_bags = npdivs::matrices_from_csv(std::cin, num_pos);
        } else {
            ifstream ifs(opts.pos_bags_file.c_str(), ifstream::in);
            pos_bags = npdivs::matrices_from_csv(ifs, num_pos);
        }

        // load negatives
        size_t num_neg;
        Matrix* neg_bags;
        if (opts.neg_bags_file == "-") {
            neg_bags = npdivs::matrices_from_csv(cin, num_neg);
        } else {
            ifstream ifs(opts.neg_bags_file.c_str(), ifstream::in);
            neg_bags = npdivs::matrices_from_csv(ifs, num_neg);
        }

        // combine training bags - TODO: this leaks memory
        Matrix* train_bags = new Matrix[num_pos + num_neg];
        for (size_t i = 0; i < num_pos; i++)
            train_bags[i] = pos_bags[i];
        for (size_t i = 0; i < num_neg; i++)
            train_bags[num_pos + i] = neg_bags[i];

        // load test
        size_t num_test;
        Matrix* test_bags;
        if (opts.test_bags_file == "-") {
            test_bags = npdivs::matrices_from_csv(cin, num_test);
        } else {
            ifstream ifs(opts.test_bags_file.c_str(), ifstream::in);
            test_bags = npdivs::matrices_from_csv(ifs, num_test);
        }

        // load test labels, maybe
        std::vector<int> test_labels;
        if (opts.test_labels_file == "-") {
            cout << "Input test labels, separated by whitespace: ";
            test_labels.resize(num_test);
            for (size_t i = 0; i < num_test; i++)
                cin >> test_labels[i];
        } else if (opts.test_labels_file != "") {
            ifstream ifs(opts.test_labels_file.c_str(), ifstream::in);

            test_labels.resize(num_test);
            for (size_t i = 0; i < num_test; i++)
                ifs >> test_labels[i];
        }

        // set up labels
        std::vector<int> labels(num_pos + num_neg, 0); // all 0s
        std::fill_n(labels.begin(), num_pos, 1); // 1s then 0s

        // div params
        npdivs::DivParams div_params(opts.k,
                *opts.index_params, *opts.search_params,
                opts.num_threads);

        // svm params
        svm_parameter svm_params(sdm::default_svm_params);
        svm_params.probability = (int) opts.prob;

        // train the model
        sdm::SDM<double>* model = train_sdm(
                train_bags, num_pos + num_neg, labels,
                *opts.div_func, *opts.kernel_group, div_params,
                sdm::default_c_vals, svm_params, opts.tuning_folds);

        // predict on test data
        const std::vector<int> &preds = model->predict(test_bags, num_test);

        // output predictions // TODO: optionally into file?
        cout << "Predicted labels: ";
        for (size_t i = 0; i < num_test; i++) {
            cout << preds[i] << " ";
        }
        cout << endl;

        if (test_labels.size() > 0) {
            size_t num_correct = 0;
            for (size_t i = 0; i < num_test; i++)
                if (preds[i] == test_labels[i])
                    num_correct++;
            cout << "Accuracy: " << num_correct * 100. / num_test << "%\n";
        }

        // clean up
        model->destroyModelAndProb();
        delete model;
        delete[] train_bags;

    } catch (std::exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

bool parse_args(int argc, char ** argv, ProgOpts& opts) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce this help message.")
        ("pos-bags,p",
            po::value<string>(&opts.pos_bags_file)->default_value("-"),
            "CSV-style file containing matrices separated by blank lines; "
            "- means stdin.")
        ("neg-bags,n",
            po::value<string>(&opts.neg_bags_file)->default_value("-"),
            "CSV-style file containing matrices separated by blank lines; "
            "- means stdin; if both pos and neg are read from stdin, pos "
            "is read first and the two groups must be separated by exactly "
            "two blank lines.")
        ("test-bags,t",
            po::value<string>(&opts.test_bags_file)->default_value("-"),
            "CSV-style file containing matrices separated by blank lines; "
            "- means stdin; again must be separated from previous groups by "
            "exactly two blank lines.")
        ("test-labels,l",
            po::value<string>(&opts.test_labels_file)->default_value(""),
            "A file containing labels for the test distributions, in the "
            "same order, one per line. Used to print out test accuracy; "
            "- means stdin, separate by exactly two blank lines.")
        ("div-func,d",
            po::value<string>()->default_value("l2")->notifier(
                boost::bind(&ProgOpts::parse_div_func, boost::ref(opts), _1)),
            "Divergence function to use. Format is name[:arg1[:arg2...]]. "
            "Options include alpha, bc, hellinger, l2, linear, renyi. "
            "alpha and renyi take an optional second argument, so that "
            "e.g. renyi:.8 is the Renyi-.8 divergence. All options take a "
            "last argument that specifies how large intermediate values are "
            "normalized; the default .99 means to cap certain values at the "
            "99th percentile of their distribution, and 1 means not to do "
            " this.")
        ("kernel,k",
            po::value<string>()->default_value("gaussian")->notifier(
                boost::bind(&ProgOpts::parse_kernel, boost::ref(opts), _1)),
            "Kernel type to use. Options are gaussian, linear, polynomial. "
            "Note that cross-validation is done to select gaussian kernel "
            "width or polynomial degree, in combination with SVM C; this is "
            "not yet configurable through this interface.")
        ("tuning-folds,f",
            po::value<size_t>(&opts.tuning_folds)->default_value(3),
            "The number of folds to use for the parameter tuning "
            "cross-validation.")
        ("probability,P",
            po::value<bool>(&opts.prob)->zero_tokens(),
            "Use probability estimates in the trained SVMs.")
        ("num-threads,T",
            po::value<size_t>(&opts.num_threads)->default_value(0),
            "Number of threads to use for calculations. 0 means one per core "
            "if compiled with recent-enough boost, or one thread otherwise.")
        ("neighbors,K",
            po::value<size_t>(&opts.k)->default_value(3),
            "The k for k-nearest-neighbor calculations.")
        ("index,i",
            po::value<string>()->default_value("kdtree")->notifier(
                boost::bind(&ProgOpts::parse_index, boost::ref(opts), _1)),
            "The nearest-neighbor index to use. Options: linear, kdtree. "
            "Note that this can have a large effect on calculation time: "
            "use kdtree for low-dimensional data and linear for relatively "
            "sparse high-dimensional data (about about 10).")
    ;

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        
        if (vm.count("help")) {
            cout << desc << endl;
            std::exit(0);
        }

        po::notify(vm);
    } catch (std::exception &e) {
        cerr << "Error: " << e.what() << endl;
        return false;
    }

    opts.search_params = new flann::SearchParams(64);

    return true;
}
