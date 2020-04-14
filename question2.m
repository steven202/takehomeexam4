close all, clear all,
N=1000; n = 2; K=10;
% mu(:,1) = [-1;0]; mu(:,2) = [1;0]; 
% Sigma(:,:,1) = [2 0;0 1]; Sigma(:,:,2) = [1 0;0 4];
% p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
% % Generate samples
% label = rand(1,N) >= p(1); l = 2*(label-0.5);
% Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% x = zeros(n,N); % reserve space
% % Draw samples from each class pdf
% for lbl = 0:1
%     x(:,label==lbl) = randGaussian(Nc(lbl+1),mu(:,lbl+1),Sigma(:,:,lbl+1));
% end
numberOfClasses=2;
n_train=1000;
n_test=10000;
[x_train,y_train] = generateMultiringDataset(numberOfClasses,n_train);
[x_test,y_test] = generateMultiringDataset(numberOfClasses,n_test);

x=x_train;
label=y_train-1;
l = 2*(label-0.5);
%%
% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                %indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
                indTrain = [indPartitionLimits(1,1):indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):indPartitionLimits(K,2)];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
%%
figure(2), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
%%
x=x_train;
label=y_train-1;
l = 2*(label-0.5);
%%
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','rbf','KernelScale',sigmaBest);
%%
x=x_test;
label=y_test-1;
l = 2*(label-0.5);
%%
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(2), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Testing Data (RED: Incorrectly Classified)'),
pTestingError = length(indINCORRECT)/N, % Empirical estimate of training error probability
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(2), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,

