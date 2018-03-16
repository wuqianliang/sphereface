% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to detect the faces in training & testing datasets (CASIA & LFW).
% Face and facial landmark detection are performed by MTCNN 
% (paper: http://kpzhang93.github.io/papers/spl.pdf, 
%  code: https://github.com/kpzhang93/MTCNN_face_detection_alignment).
%
% Note:
% If you want to use this script for other dataset, please make sure
% (a) the dataset is structured as `dataset/idnetity/image`, e.g. `casia/id/001.jpg`
% (b) the folder name and image format (bmp, png, etc.) are correctly specified.
% 
% Usage:
% cd $SPHEREFACE_ROOT/preprocess
% run code/face_detect_demo.m
% --------------------------------------------------------

function face_detect_demo_snd()

	clear;clc;close all;
	cd('../');

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%face detect%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	testFile = ('/home/wuqianliang/snd_id_photo/211322199001171510/211322199001171510.jpg');
	%% mtcnn settings
	minSize   = 20;
	factor    = 0.85;
	threshold = [0.6 0.7 0.9];

	%% add toolbox paths
	matCaffe       = fullfile(pwd, '../tools/caffe-sphereface/matlab');
	pdollarToolbox = fullfile(pwd, '../tools/toolbox');
	MTCNN          = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1');
	addpath(genpath(matCaffe));
	addpath(genpath(pdollarToolbox));
	addpath(genpath(MTCNN));

	%% MTCNN caffe  model settings
	gpu = 1;
	if gpu
	   gpu_id = 0;
	   caffe.set_mode_gpu();
	   caffe.set_device(gpu_id);
	else
	   caffe.set_mode_cpu();
	end
	caffe.reset_all();
	modelPath = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model');
	PNet = caffe.Net(fullfile(modelPath, 'det1.prototxt'), ...
			 fullfile(modelPath, 'det1.caffemodel'), 'test');
	RNet = caffe.Net(fullfile(modelPath, 'det2.prototxt'), ...
			 fullfile(modelPath, 'det2.caffemodel'), 'test');
	ONet = caffe.Net(fullfile(modelPath, 'det3.prototxt'), ...
			 fullfile(modelPath, 'det3.caffemodel'), 'test');

        %% sphereface caffe model setttings
        model   = '../train/code/sphereface_deploy.prototxt';
        weights = '../train/result/sphereface_model_iter_28000.caffemodel';
        net     = caffe.Net(model, weights, 'test');
        net.save('result/sphereface_model.caffemodel');		


	%%%%%%%%%%%%%%%%%%%%%%%%%%image process%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	img = imread(testFile);
	if size(img, 3)==1
	img = repmat(img, [1,1,3]);
	end
	% detection
	[bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, threshold, false, factor);

	if size(bboxes, 1)>1
		% pick the face closed to the center
		center   = size(img) / 2;
		distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
				      mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
		[~, Ix]  = min(distance);
		facial5point = reshape(landmarks(:, Ix), [5, 2]);

	elseif size(bboxes, 1)==1
		facial5point = reshape(landmarks, [5, 2]);
	else
		facial5point = [];
	end
        for i = 1:length(facial5point)
		fprintf('%d point: %f\n',i,facial5point(i));
	end
         
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%face align%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	imgSize     = [112, 96];
	coord5point = [30.2946, 51.6963;
		       65.5318, 51.5014;
		       48.0252, 71.7366;
		       33.5493, 92.3655;
		       62.7299, 92.2041];	
	if isempty(facial5point)
		fprintf('Image facial5point null!!\n');
	end
	facial5point = double(facial5point);

	transf   = cp2tform(facial5point, coord5point, 'similarity');
	cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
					'YData', [1 imgSize(1)], 'Size', imgSize);



	feature = extractDeepFeature(cropImg, net);
        featureN = bsxfun(@rdivide, feature, sqrt(sum(feature.^2)));   % ||feature|| =1	
	fprintf('%s %d',class(featureN),length(featureN));
	


end

function feature = extractDeepFeature(img, net)
    %img     = single(imread(file));
    img     = (img - 127.5)/128;
    img     = permute(img, [2,1,3]);
    img     = img(:,:,[3,2,1]);
    res     = net.forward({img});
    res_    = net.forward({flip(img, 1)});
    feature = double([res{1}; res_{1}]);
end
