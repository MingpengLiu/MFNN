clear;
rng(1);

%% load data for wall deformation
DataSummary = load("DataSummary.mat");
DataSummary = DataSummary.Datasummary;
net_low = DataSummary{5,1};
x_rule = DataSummary{4,1}; 
y_rule = DataSummary{4,2}; 

Data_ground_F = load('Ground_F.mat');
Data_ground_F = Data_ground_F.Save_Ground_field;
Data_ground_F = Data_ground_F(:,1);
%% prediction through low-fidelity neural network
X = [];
Y = [];
for i = 1:length(Data_ground_F)
    fieldmeasure = Data_ground_F{i};
    XData = fieldmeasure(:, 1:end-1)';
    YData = fieldmeasure(:, end)';
    xdata = mapminmax('apply', XData, x_rule);
    ysim = sim(net_low,  xdata);
    YSim = mapminmax('reverse', ysim, y_rule);

    X = [X; XData', YSim'];
    Y = [Y; YData'];
end
%% Training set and testing set
X = X';
Y = Y';

[x, x_highrule] = mapminmax(X, 0, 1);
[y, y_highrule] = mapminmax(Y, 0, 1);

k=5;
cv = cvpartition(size(x,2),'KFold',k);
%%
numepoch = 1000;
trainloss=zeros(k,numepoch+1);testloss=zeros(k,numepoch+1);
for i=1:k
    rng(i)
    net = newff(x, y, 15,{'tansig', 'purelin'});
    net.trainFcn = 'trainlm';
    net.trainParam.mc = 0.025;
    net.trainParam.epochs = 3000;
    net.trainParam.goal   = 1e-9;
    net.trainParam.lr = 0.0015;
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    net.trainparam.max_fail   = 3; 

    [net, info] = train(net, x, y);
    testIdx = info.testInd; valIdx = info.valInd;
    YTest = Y(testIdx);YVal = Y(valIdx);
    ysim = sim(net,x(:,testIdx));
    YSim = mapminmax('reverse',ysim,y_highrule);
    ysim2 = sim(net,x(:,valIdx));
    YSim2 = mapminmax('reverse',ysim2,y_highrule);

    R = 1 - norm(YTest - YSim)^2 / norm(YTest - mean(YTest))^2;
    MAE = mean(abs(YTest-YSim));
    MAPE = mape(YTest,YSim,'all');
    RMSE = rmse(YTest,YSim,'all');    
    indices(i,1:4)=[R,MAE,MAPE,RMSE];
end
