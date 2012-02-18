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


classdef SDM < handle
    properties (Hidden = true, SetAccess = private)
        cpp_handle;
    end

    methods
        % constructor...add a way to copy?
        % TODO: should probably be private
        function this = SDM(cpp_handle)
            this.cpp_handle = cpp_handle;
        end

        % TODO: destructor
        function delete(this)
            sdm_mex('delete', this.cpp_handle);
        end

        % get a string describing the model
        function [name] = name(this)
            name = sdm_mex('name', this.cpp_handle);
        end

        % TODO: predict on new data
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
        % TODO: train up a model
        function [model] = train(train_bags, labels, options)
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

            % TODO: parameter checking

            if nargin < 3; options = struct(); end
            model_handle = sdm_mex('train', train_bags, labels, options);
            model = SDM(model_handle);
        end

        % TODO: do cross-validation on a dataset
        function [acc] = crossvalidate()
        end
    end
end
