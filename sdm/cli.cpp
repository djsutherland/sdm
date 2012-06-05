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
#include "sdm/kernels/from_str.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/throw_exception.hpp>
#include <boost/utility.hpp>

#include <flann/flann.hpp>

#include <np-divs/matrix_arrays.hpp>
#include <np-divs/matrix_io.hpp>
#include <np-divs/div-funcs/from_str.hpp>
#include <np-divs/div_params.hpp>

// TODO: support CV
// TODO: warn about dumb parameter combos, like linear kernel with distance df

using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;

namespace po = boost::program_options;

// TODO: leaks memory everywhere, but whatever...
struct ProgOpts : boost::noncopyable {
    string train_bags_file;
    string test_bags_file;
    string cv_divs_file;

    npdivs::DivFunc * div_func;
    sdm::KernelGroup * kernel_group;

    size_t k;
    size_t num_threads;
    size_t cv_folds;
    size_t cv_threads;
    size_t tuning_folds;
    bool prob;
    bool proj_indiv;
    bool shuffle;

    flann::IndexParams index_params;
    flann::SearchParams search_params;

    void parse_div_func(const string &spec) {
        div_func = npdivs::div_func_from_str(spec);
    }

    void parse_kernel(const string &name) {
        kernel_group = sdm::kernel_group_from_str(name);
    }

    void parse_index(const string &name) {
        index_params = npdivs::index_params_from_str(name);
    }
};

bool parse_args(int argc, char ** argv, ProgOpts& opts);

int main(int argc, char ** argv) {
    typedef flann::Matrix<double> Matrix;

    try {
        ProgOpts opts;
        if (!parse_args(argc, argv, opts))
            return 1;

        // TODO: configurable logging
        FILELog::ReportingLevel() = logDEBUG2;
        // Output2FILE::Stream() = stderr;

        // TODO: gracefully handle nonexisting files
        // TODO: more robust input checking

        // load training bags
        size_t num_train;
        Matrix* train_bags;
        vector<string> train_labels;

        if (opts.train_bags_file == "-") {
            cout << "Enter training distributions in CSV-like "
                "format: an initial string label for each distribution, one "
                "line with comma-separated floating-point values for each "
                "point, one blank line between distributions, and an extra "
                "blank line when done.\n";
            train_bags = npdivs::labeled_matrices_from_csv(
                    std::cin, num_train, train_labels);
        } else {
            FILE_LOG(logINFO) << "Loading training bags...";
            ifstream ifs(opts.train_bags_file.c_str(), ifstream::in);
            train_bags = npdivs::labeled_matrices_from_csv(
                    ifs, num_train, train_labels);
            FILE_LOG(logINFO) << "done.";
        }

        // load divergences, maybe
        bool do_cv = !opts.cv_divs_file.empty() || opts.test_bags_file.empty();

        double* cv_divs = NULL;
        if (!opts.cv_divs_file.empty()) {
            flann::Matrix<double> cv_divs_m;
            if (opts.cv_divs_file == "-") {
                cout << "Enter the matrix of divergences among training bags "
                    "in CSV-like format.";
                cv_divs_m = npdivs::matrix_from_csv(cin);
            } else {
                FILE_LOG(logINFO) << "Loading divergences...";
                ifstream ifs(opts.cv_divs_file.c_str(), ifstream::in);
                cv_divs_m = npdivs::matrix_from_csv(ifs);
                FILE_LOG(logINFO) << "done.";
            }

            if (cv_divs_m.rows != num_train || cv_divs_m.cols != num_train) {
                FILE_LOG(logERROR) << boost::format(
                        "Input divergences are %dx%d, expected %dx%d. "
                        "Ignoring them.")
                    % cv_divs_m.rows % cv_divs_m.cols
                    % num_train % num_train;
                do_cv = opts.test_bags_file.empty();
            } else {
                cv_divs = cv_divs_m.ptr();
            }
        }

        // load test bags, maybe
        size_t num_test = 0;
        Matrix* test_bags;
        vector<string> test_labels;

        if (!do_cv) {
            if (opts.test_bags_file == "-") {
                cout << "Enter testing distributions in CSV-like "
                    "format: an initial string label for each distribution, "
                    "one line with comma-separated floating-point values for "
                    "each point, one blank line between distributions, and an "
                    "extra blank line when done.\n";
                test_bags = npdivs::labeled_matrices_from_csv(
                        cin, num_test, test_labels);
            } else {
                FILE_LOG(logINFO) << "Loading test bags...";
                ifstream ifs(opts.test_bags_file.c_str(), ifstream::in);
                test_bags = npdivs::labeled_matrices_from_csv(
                        ifs, num_test, test_labels);
                FILE_LOG(logINFO) << "done.";
            }
        }

        // convert labels into integers
        // make maps to convert between int / string labels
        int this_label = -1;
        std::map<string,int> labels_to_int;
        std::map<int,string> labels_to_string;

        labels_to_int[""] = -1;
        labels_to_int["?"] = -1;
        labels_to_string[-1] = "?";

        std::vector<int> train_labels_ints;
        train_labels_ints.reserve(num_train);
        for (size_t i = 0; i < num_train; i++) {
            const string &lbl = train_labels[i];

            if (labels_to_int.count(lbl) == 0) {
                labels_to_int[lbl] = ++this_label;
                labels_to_string[this_label] = lbl;
            }
            train_labels_ints.push_back(labels_to_int[lbl]);
        }

        std::vector<int> test_labels_ints;
        test_labels_ints.reserve(num_test);
        for (size_t i = 0; i < num_test; i++) {
            const string &lbl = test_labels[i];

            if (labels_to_int.count(lbl) == 0) {
                labels_to_int[lbl] = ++this_label;
                labels_to_string[this_label] = lbl;
            }
            test_labels_ints.push_back(labels_to_int[lbl]);
        }

        // div params
        npdivs::DivParams div_params(opts.k,
                opts.index_params, opts.search_params,
                opts.num_threads);

        // svm params
        svm_parameter svm_params(sdm::default_svm_params);
        svm_params.probability = (int) opts.prob;

        if (!do_cv) {
            // train the model
            sdm::SDM<double>* model = train_sdm(
                    train_bags, num_train, train_labels_ints,
                    *opts.div_func, *opts.kernel_group, div_params,
                    sdm::default_c_vals, svm_params, opts.tuning_folds);

            cout << "Trained model: " << model->name() << endl;

            // predict on test data
            const std::vector<int> &preds = model->predict(test_bags, num_test);

            // output predictions // TODO: optionally into file?
            // tally up accuracy at the same time
            size_t num_correct = 0;
            size_t total = num_test;

            cout << "Predicted labels:\n";
            for (size_t i = 0; i < num_test; i++) {
                cout << i << ":\t" << labels_to_string[preds[i]];
                if (test_labels_ints[i] == -1) {
                    total--;
                } else {
                    if (test_labels_ints[i] == preds[i]) {
                        cout << "\t -- correct";
                        num_correct++;
                    } else {
                        cout << "\t -- expected " << test_labels[i];
                    }
                }
                cout << endl;
            }
            cout << endl;

            if (total > 0) {
                cout << "Accuracy on " << total << " labeled test points: "
                    << num_correct * 100. / total << "%\n";
            }

            // cleanup
            model->destroyModelAndProb();
            delete model;
        } else {
            double acc = crossvalidate(train_bags, num_train,
                    train_labels_ints, *opts.div_func, *opts.kernel_group,
                    div_params, opts.cv_folds, opts.cv_threads,
                    !opts.proj_indiv, opts.shuffle,
                    sdm::default_c_vals, svm_params,
                    opts.tuning_folds, cv_divs);
            cout << opts.cv_folds << "-fold cross-validation accuracy on "
                << num_train << " bags: " << 100. * acc << "%" << endl;
        }

        // cleanup
        npdivs::free_matrix_array(train_bags, num_train);
        if (!do_cv)
            npdivs::free_matrix_array(test_bags, num_test);

    } catch (std::exception &e) {
         cerr << "Error: " << e.what() << endl;
         return 1;
    }
}

bool parse_args(int argc, char ** argv, ProgOpts& opts) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce this help message.")
        ("train-bags,b",
            po::value<string>(&opts.train_bags_file)->default_value("-"),
            "CSV-like file containing matrices separated by blank lines, "
            "with a string label on its own line before each matrix; "
            "- means stdin.")
        ("test-bags,t",
            po::value<string>(&opts.test_bags_file)->default_value(""),
            "CSV-style file containing matrices separated by blank lines, "
            "with a string label on its own line before each matrix."
            "- means stdin. "
            "\nUse a blank line or ? as the label if it's not known. If any "
            "test distributions are labeled, will report accuracy on them.")
        ("cv-divs,D",
            po::value<string>(&opts.cv_divs_file)->default_value(""),
            "CSV file containing precomputed divergences among all the bags "
            "to use for cross-validation. Still reads --train-bags for labels "
            "only; --test-bags is ignored.")
        // TODO: support giving labels only
        // TODO: test divs support
        ("cv-folds,c",
            po::value<size_t>(&opts.cv_folds)->default_value(10),
            "Do c-fold cross-validation on the data passed in --train-bags. "
            "Ignored if --test-bags is passed.")
        ("cv-threads",
            po::value<size_t>(&opts.cv_threads)->default_value(0),
            "The max number of CV folds to do in parallel; defaults to the "
            "value of --num-threads. Will not use more than "
            "max(cv_threads, num_threads) threads at any point.")
        ("div-func,d",
            po::value<string>()->default_value("l2")->notifier(
                boost::bind(&ProgOpts::parse_div_func, boost::ref(opts), _1)),
            "Divergence function to use. Format is name[:arg1[:arg2...]]. "
            "Options include alpha, bc, hellinger, l2, linear, renyi. "
            "\nalpha and renyi take an optional second argument, so that "
            "e.g. renyi:.8 is the Renyi-.8 divergence."
            "\nAll options take a last argument that specifies how large "
            " intermediate values are  normalized; the default .99 means to "
            "cap certain values at the 99th percentile of their distribution, "
            "and 1 means not to do this.")
        ("kernel,k",
            po::value<string>()->default_value("gaussian")->notifier(
                boost::bind(&ProgOpts::parse_kernel, boost::ref(opts), _1)),
            "Kernel type to use. Options are gaussian, linear, polynomial. "
            "\nUse '-k linear' or '-k polynomial' only with '-d linear'; any "
            "of the other -d arguments should be used with '-k gaussian' "
            "in order to get a meaningful kernel. "
            "\nNote that cross-validation is done to select gaussian kernel "
            "width or polynomial degree, in combination with SVM C; this is "
            "not yet configurable through this interface.")
        ("tuning-folds,f",
            po::value<size_t>(&opts.tuning_folds)->default_value(3),
            "The number of folds to use for the parameter tuning "
            "cross-validation.")
        ("probability,P",
            po::value<bool>(&opts.prob)->default_value(0)->zero_tokens(),
            "Use probability estimates in the trained SVMs.")
        ("project-individually,J",
            po::value<bool>(&opts.proj_indiv)->default_value(0)->zero_tokens(),
            "When cross-validating, do the PSD projection on each fold's "
            "training data only, rather than on the entire kernel matrix.")
        ("shuffle",
            po::value<bool>(&opts.shuffle)->default_value(1),
            "Whether to shuffle the order of folds in cross-validation.")
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
            "\nNote that this can have a large effect on calculation time: "
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

    opts.search_params = flann::SearchParams(64);

    return true;
}
