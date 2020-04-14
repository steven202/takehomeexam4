data = exam4q1_generateData(1000);
x = data(1,:);
y = data(2,:);
l = y;
x_test = exam4q1_generateData(10000);

K=10;
N=1000;
dummy = ceil(linspace(0,N,K+1));

for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
P=15;
% Allocate space
MSEtrain = zeros(K,P); MSEvalidate = zeros(K,P); 
AverageMSEtrain = zeros(1,N); AverageMSEvalidate = zeros(1,N);

for nPerceptrons=1:P
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
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        % Train model parameters
        [params,H,error]= mleMLPwAWGN(xTrain, lTrain,xValidate,lValidate,0,nPerceptrons);
        MSEvalidate(k,nPerceptrons) = error;
        fprintf("finish [%i/%i], [%i/%i]\n",k,K,nPerceptrons,P);
    end
end
%%
MSEtrain = MSEvalidate(1:K,1:P);
%%
comparison = sum(MSEtrain,1)/P;
final_perceptrons= find(comparison==min(comparison));
disp(final_perceptrons);
%%    
X = x_test(1,:);
Y = x_test(2,:);
nPerceptrons=final_perceptrons;
[params,H,error]= mleMLPwAWGN(x, y,X,Y,0,nPerceptrons);
disp(error);


%objFncValue = objectiveFunction(X,Y,sizeParams,vecParamsInit,0)