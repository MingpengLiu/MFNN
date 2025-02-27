clear;
rng(1);
%% load data for wall deformation
DataSummary = load("DataSummary.mat");
DataSummary = DataSummary.Datasummary;
net_low = DataSummary{5,1};
x_rule = DataSummary{4,1}; 
y_rule = DataSummary{4,2}; 

Data_wall_F = load('Wall_F.mat');
Data_wall_F = Data_wall_F.Save_Wall_field;
Data_wall_F = Data_wall_F(:,1);
%% prediction through low-fidelity neural network
X = [];
Y = [];
for i = 1:length(Data_wall_F)
    fieldmeasure = Data_wall_F{i};
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
%% train the network
for i=1:k
    trainIdx = training(cv,i);testIdx = test(cv,i);
    xTrain = x(:,trainIdx); yTrain = y(:,trainIdx); 

    net = feedforwardnet(30,'trainlm');
    net.trainParam.mc = 0.001;
    net.trainParam.epochs = 100;
    net.trainParam.goal   = 1e-7;
    net.trainParam.lr = 0.001;
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;

    [net, info] = train(net, xTrain, yTrain);

    xTest = x(:,testIdx);YTest = Y(:,testIdx);YTrain = Y(:,trainIdx);

    ysim = sim(net,xTest);
    YSim = mapminmax('reverse', ysim, y_highrule);
    
    ysim1 = sim(net,xTrain);
    YSim1 = mapminmax('reverse', ysim1, y_highrule);

    R = 1 - norm(YTest - YSim)^2 / norm(YTest - mean(YTest))^2;
    MAE = mean(abs(YTest-YSim));
    MAPE = mape(YTest,YSim,'all');
    RMSE = rmse(YTest,YSim,'all');

    indices(i,1:4)=[R,MAE,MAPE,RMSE];
    net_cell{i,1}={net};net_cell{i,2}={info};
end