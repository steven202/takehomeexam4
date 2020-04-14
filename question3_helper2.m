function logLikelihood = question3_helper2(file,num_cluster)
    filenames{1,1} = '3096_color.jpg';
    filenames{1,2} = '42049_color.jpg';
    %filename = '';
    if file==1
        filename=filenames{1,1};
    else
        filename=filenames{1,2};
    end
    Kvalues = [num_cluster]; % desired numbers of clusters
    imageCounter=1;
    imdata = imread(filename);   

    figure(num_cluster), subplot(1,2,1),imshow(imdata);
    
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
    for d = 1:D
        imdatad = imdata(:,:,d); % pick one color at a time
        features = [features;imdatad(:)'];
    end
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
    options = statset('MaxIter',300);
    reg = 1e-2;

    d = size(x,1); % feature dimensionality
    % Train model parameters
    gm = fitgmdist(x', num_cluster, 'Options', options, 'RegularizationValue', reg);
    logLikelihood = sum(log(evalGMM(x, gm.ComponentProportion, gm.mu', gm.Sigma)));
    fprintf("logLikelihood %i with %i clusters\n",logLikelihood, num_cluster);
    %[params,H,error]= mleMLPwAWGN(xTrain, lTrain,xValidate,lValidate,0,nPerceptrons);  
    %%%%%%%%%%%% draw image %%%%%%%%%%%%
    labels = cluster(gm,x');
    labelImage = reshape(labels,R,C);
    figure(num_cluster), subplot(1,2,2), imshow(uint8(labelImage*255/num_cluster));
    %title(strcat({'Clustering with K = '},num2str(num_cluster),' ,with logLikelihood = ',num2str(logLikelihood))); 
    title(strcat({'Clustering with K = '},num2str(num_cluster)));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%     for k = 1:10
%         
%         %figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1+k), imshow(uint8(labelImage*255/Kvalues(k)));
%         %title(strcat({'Clustering with K = '},num2str(K)));
%     end
%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end