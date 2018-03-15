% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to evaluate the performance of the trained model on LFW dataset.
% We perform 10-fold cross validation, using cosine similarity as metric.
% More details about the testing protocol can be found at http://vis-www.cs.umass.edu/lfw/#views.
% 
% Usage:
% cd $SPHEREFACE_ROOT/test
% run code/evaluation.m
% --------------------------------------------------------

function evaluation_snd()

	clear;clc;close all;
	cd('../')

	%% caffe setttings
	matCaffe = fullfile(pwd, '../tools/caffe-sphereface/matlab');
	addpath(genpath(matCaffe));
	gpu = 1;
	if gpu
	   gpu_id = 0;
	   caffe.set_mode_gpu();
	   caffe.set_device(gpu_id);
	else
	   caffe.set_mode_cpu();
	end
	caffe.reset_all();

	model   = '../train/code/sphereface_deploy.prototxt';
	weights = '../train/result/sphereface_model_iter_28000.caffemodel';
	net     = caffe.Net(model, weights, 'test');
	net.save('result/sphereface_model.caffemodel');

        %% fold names
        foldfid=fopen('/home/arthur/sphereface/test/code/fold.txt');
	while ~feof(foldfid)
                foldname = fgetl(foldfid)
		id_file = fullfile('/home/wuqianliang/snd_id_photo-112X96',foldname,[foldname,'.jpg']);
		featureFold = extractDeepFeature(id_file, net);
		featureFoldN = bsxfun(@rdivide, featureFold, sqrt(sum(featureFold.^2)));   % ||feature|| =1
% 		fprintf('%s',id_file);
		fid=fopen('/home/arthur/sphereface/test/code/snd_pic.txt');
		while ~feof(fid)
		    line = fgetl(fid);
		    fname = fullfile('/home/wuqianliang/snd_id_photo-112X96', line);
%		    fprintf('extracting deep features from the %s ...\n', fname);
		    feature = extractDeepFeature(fname, net);
		    featureN = bsxfun(@rdivide, feature, sqrt(sum(feature.^2)));   % ||feature|| =1
		    scores    = sum(featureN .* featureFoldN);  % featureLs 
                    fprintf('%s,%s,%f\n',foldname,line,scores);
		end
	end


end
function feature = extractDeepFeature(file, net)
    img     = single(imread(file));
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    res_    = net.forward({flip(img, 1)});
    feature = double([res{1}; res_{1}]);
end
