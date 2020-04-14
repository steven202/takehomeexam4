function avg_log = question3_helper1(file,num_cluster)
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
    %%%%%%%%%%%% draw image %%%%%%%%%%%%
    model = fitgmdist(x', num_cluster, 'Options', options, 'RegularizationValue', reg);
    labels = cluster(model,x');
    labelImage = reshape(labels,R,C);
    figure(num_cluster), subplot(1,2,2), imshow(uint8(labelImage*255/num_cluster));
    title(strcat({'Clustering with K = '},num2str(num_cluster))); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    d = size(x,1); % feature dimensionality
    K =10;
    MSEvalidate = zeros(K,1);
    dummy = ceil(linspace(0,N,K+1));

    for k = 1:K
        indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
    end

    

    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        %lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            %indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            indTrain = [indPartitionLimits(1,1):indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):indPartitionLimits(K,2)];
        end
        xTrain = x(:,indTrain); 
        %lTrain = l(indTrain);
        % Train model parameters
        gm = fitgmdist(xTrain', num_cluster, 'Options', options, 'RegularizationValue', reg);
        logLikelihood = sum(log(evalGMM(xValidate, gm.ComponentProportion, gm.mu', gm.Sigma)));
        MSEvalidate(k,1) = logLikelihood;
        fprintf("finish [%i/%i], logLikelihood %i\n",k,K,logLikelihood);
        %[params,H,error]= mleMLPwAWGN(xTrain, lTrain,xValidate,lValidate,0,nPerceptrons);        
    end
    avg_log = sum(MSEvalidate,1)/K;
    fprintf("average logLikelihood %i with %i clusters\n",avg_log, num_cluster);
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