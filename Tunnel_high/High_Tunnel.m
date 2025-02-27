clear;
rng(1);

%% load data 
DataSummary = load("DataSummary.mat");
DataSummary = DataSummary.Datasummary;
net_low = DataSummary{5,1};
x_rule = DataSummary{4,1}; 
y_rule = DataSummary{4,2}; 

Data_tunnel_F = load('Tunnel_F.mat');
Data_tunnel_F = Data_tunnel_F.Save_Tunnel_field;
Data_tunnel_F = Data_tunnel_F(:,1);
%% prediction through low-fidelity neural network
XFOrigin = [37.35000542, -12.25579776;
            40.45, -15.35;
            43.54999458, -12.25579776];
X = [];
Y = [];
for i = 1:length(Data_tunnel_F)
    fieldmeasure = Data_tunnel_F{i};
    [B, I] = sort(fieldmeasure(:,8));
    fieldmeasure = fieldmeasure(I,:);
    XData = [fieldmeasure(:, 1:end-2)'; XFOrigin'];
    YData = (fieldmeasure(:, end-1:end) - XFOrigin)';
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
%% train the network
k=5;
cv = cvpartition(size(x,2),'KFold',k);
for i=1:k
    trainIdx = training(cv,i);testIdx = test(cv,i);
    xTrain = x(:,trainIdx); yTrain = y(:,trainIdx); 
    net = feedforwardnet(15,'trainlm');
    net.trainParam.mc = 0.015;
    net.trainParam.epochs = 100;
    net.trainParam.goal   = 1e-9;
    net.trainParam.lr = 0.0015;
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;

    [net, info] = train(net, xTrain, yTrain);

    xTest = x(:,testIdx);YTest = Y(:,testIdx);
    ysim = sim(net,xTest);
    YSim = mapminmax('reverse', ysim, y_highrule);

    R = 1 - norm(YTest - YSim)^2 / norm(YTest - mean(YTest))^2;
    MAE = mean(abs(YTest-YSim),"all");
    MAPE = mape(YTest,YSim,'all');
    RMSE = rmse(YTest,YSim,'all');

    indices(i,1:4)=[R,MAE,MAPE,RMSE];
    net_cell{i,1}={net};net_cell{i,2}={info};
end