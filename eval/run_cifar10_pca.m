clear;

bit = 512;
examples_to_show = 10;
% load CIFAR10 training set
data_folder = '/media/iis/Data/Data/cifar10/cifar-10-batches-mat/';
X_train = [];
Y_train = [];
for i = 1: 5
    datadict = load([data_folder 'data_batch_' num2str(i) '.mat']);
    X_train = [X_train; datadict.data];
    Y_train = [Y_train; datadict.labels];
end

% load CIFAR10 test set
datadict = load([data_folder 'test_batch.mat']);
X_test = double(datadict.data);
Y_test = datadict.labels;

% center the data
X_train = double(X_train);
sample_mean = mean(X_train, 1);
X_train = (X_train - repmat(sample_mean, size(X_train,1),1));

fprintf('performing PCA ...\n');
[pc, l] = eigs(double(cov(X_train)),bit);
%[coeff,score,latent] = pca(double(X_train));

% reconstruct images
% center data
XX = X_test - repmat(sample_mean, size(X_test, 1),1);

% projection
XX = XX * pc;

% reconstruction
XX = XX *pc' + repmat(sample_mean, size(XX,1),1);

% show reconstructed images
for i = 1: examples_to_show
    im = reshape(XX(i,:), [32 32 3]);
    im = permute(im, [2 1 3]);
    figure; imshow(uint8(im))
end
