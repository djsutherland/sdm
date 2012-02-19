% Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%
%     * Neither the name of Carnegie Mellon University nor the
%       names of the contributors may be used to endorse or promote products
%       derived from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.


% This is a MATLAB wrapper class for the C++ SDM class.
%
% It maintains a pointer to the C++ object, and passes that into MEX functions
% that then call whatever C++ methods are appropriate.
%
% Loosely based on code by Oliver Woodford:
%   http://www.mathworks.com/matlabcentral/newsreader/view_thread/278243


% TODO: segfault when run with   echo some matlab commands | matlab -nojvm
% TODO: link mex file as statically as possible
% TODO: add warnings about possibly having to use LD_PRELOAD

classdef SDM < handle
    properties (Hidden = true, SetAccess = private)
        cpp_handle;
    end

    properties (SetAccess = immutable)
        name;
        num_classes;
        num_train;
        dim;
        C;
        kernel;
        div_func;
        labels;
    end

    methods
        function this = SDM(cpp_handle, labels, num_train, dim)
            % Make an SDM for a specific C++ instance.
            this.cpp_handle = cpp_handle;
            this.labels = labels;
            this.num_classes = numel(unique(labels));
            this.num_train = num_train;
            this.dim = dim;

            [this.name, this.kernel, this.div_func, this.C] = ...
                sdm_mex('info', cpp_handle);
        end

        function delete(this)
            % Destroy the related C++ object and its attributes.
            sdm_mex('delete', this.cpp_handle);
        end


        function [bags] = get_train_bags(this)
            % Get a copy of the SDM's training distributions.
            bags = sdm_mex('train_bags', this.cpp_handle);
        end


        function [labels vals] = predict(this, test_dists)
            % Run on new test data and predict labels.
            %
            % Arguments:
            %   test_dists: distributions of the same dimensionality as the
            %       training bags. Either a cell array or a single matrix.
            %
            % Returns:
            %   labels: a vector of predicted class labels for the distributions
            %   vals: a matrix of decision values / probabilities for each test
            %         point being of each class

            if ~iscell(test_dists); test_dists = {test_dists}; end

            if nargout == 2
                [labels vals] = sdm_mex('predict', this.cpp_handle, test_dists);
            else
                labels = sdm_mex('predict', this.cpp_handle, test_dists);
            end
        end
    end

    methods (Static)
        function [model] = train(train_bags, labels, options, divs)
            % Trains a support distribution machine on test data.
            %
            % Arguments:
            %   train_bags: a cell array of samples from distributions
            %       Must have the same number of columns, but may have different
            %       numbers of rows.
            %
            %   labels: a vector of integer labels for each training bag
            %
            %   options: a struct array whose elements might include:
            %
            %       div_func: the divergence function to use.
            %           examples: "renyi:.8", "l2", "bc", "linear".
            %           default: "l2"
            %
            %       kernel: the type of kernel to use.
            %           choices: "linear", "polynomial", "gaussian"
            %           default: "gaussian"
            %           "linear" and "polynomial" only make sense with
            %           "linear" div_func; others should use "gaussian"
            %
            %       k: the k of k-nearest-neighbors. default 3
            %
            %       tuning_folds: the number of folds to use for the tuning CV
            %           default 3
            %
            %       probability: whether to use SVMs with probability estimates
            %
            %       num_threads: the number of threads to use for calculations.
            %           0 (the default) means one per core
            %
            %       index: the nearest-neighbor index type.
            %           options: "linear", "kdtree"
            %           default: "kdtree"
            %           For high-dimensional data (over 10ish), use linear
            %
            %    divs: precomputed divergences among the bags. optional.

            % TODO: parameter checking
            dim = size(train_bags{1}, 2);

            if nargin < 3; options = struct(); end
            if nargin < 4; divs = []; end

            model_handle = sdm_mex('train', train_bags, labels, options, divs);
            model = SDM(model_handle, labels, numel(train_bags), dim);
        end

        function [acc] = crossvalidate(bags, labels, options, divs)
            % Trains a support distribution machine on test data.
            %
            % Arguments:
            %   bags: a cell array of samples from distributions
            %       Must have the same number of columns, but may have different
            %       numbers of rows.
            %
            %   labels: a vector of integer labels for each bag
            %
            %   options: a struct array whose elements might include:
            %
            %       folds: the number of CV folds (default 10).
            %           0 means to do leave-one-out CV.
            %
            %       cv_threads: the maximum number of CV folds to run in
            %           parallel; defaults to the value of num_threads. Will
            %           not use more than max(cv_threads, num_threads) at
            %           any point.
            %
            %       project_all: whether to project the entire estimated
            %           kernel matrix to be PSD; if false, only projects the
            %           training data for a given fold and leaves the rest
            %           unprojected. Default true.
            %
            %       div_func: the divergence function to use.
            %           examples: "renyi:.8", "l2", "bc", "linear".
            %           default: "l2"
            %
            %       kernel: the type of kernel to use.
            %           choices: "linear", "polynomial", "gaussian"
            %           default: "gaussian"
            %           "linear" and "polynomial" only make sense with
            %           "linear" div_func; others should use "gaussian"
            %
            %       k: the k of k-nearest-neighbors. default 3
            %
            %       tuning_folds: the number of folds to use for the tuning CV
            %           default 3
            %
            %       probability: whether to use SVMs with probability estimates
            %
            %       num_threads: the number of threads to use for calculations.
            %           0 (the default) means one per core
            %
            %       index: the nearest-neighbor index type.
            %           options: "linear", "kdtree"
            %           default: "kdtree"
            %           For high-dimensional data (over 10ish), use linear
            %
            %    divs: precomputed divergences among the bags. optional.

            if nargin < 3; options = struct(); end
            if nargin < 4; divs = []; end
            acc = sdm_mex('crossvalidate', bags, labels, options, divs);
        end

        function [Ds] = npdivs(x_bags, y_bags, opts)
            % Estimates divergences between distributions.
            %
            %   x_bags: a cell array of data matrices (each n_i x D)
            %
            %   y_bags: a cell array of data matrices (each n_i x D), or [],
            %         meaning the same thing as passing x_bags again (but is
            %         computed more efficiently)
            %
            %   options: a struct array with the following possible members:
            %
            %         div_funcs: a cell array of string specifications for
            %               divergence functions, such as 'l2', 'renyi:.99',
            %               'alpha:.2'.
            %
            %               Default is {'l2'}.
            %
            %               Possible functions include l2, alpha, bc,
            %               hellinger, renyi, linear.
            %
            %               Some support an argument specifying a divergence
            %               parameter: renyi:.99 means the Renyi-.99
            %               divergence.
            %
            %         k: the k for k-nearest-neighbor. Default 3.
            %
            %         index: the nearest-neighbor index to use. Options ar.
            %              linear, kdtree. Default is kdtree. Use linear for
            %              high-dimensional, relatively sparse data.
            %
            %         num_threads: the number of threads to use in calculation.
            %              0 (the default) means one per core.

            if nargin < 3; options = struct(); end
            if nargin < 2; y_bags = []; end

            Ds = sdm_mex('npdivs', x_bags, y_bags, opts);
        end
    end
end
