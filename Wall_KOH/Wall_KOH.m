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
error = [];
YSimall=[];
for i = 1:length(Data_wall_F)
    fieldmeasure = Data_wall_F{i};
    XData = fieldmeasure(:, 1:end-1)';
    YData = fieldmeasure(:, end)';
    xdata = mapminmax('apply', XData, x_rule);
    ysim = sim(net_low,  xdata);
    YSim = mapminmax('reverse', ysim, y_rule);

    X = [X; XData'];
    Y = [Y; YData'];
    YSimall = [YSimall;YSim'];
    error=[error;(YData-YSim)'];
end
%% GPR
gprMdl = fitrgp(X,error,'KernelFunction','squaredexponential','Sigma',0.1);
[error_pre,std_pre]=predict(gprMdl,X);
Y_calibration = YSimall + error_pre;

R_cal = 1 - norm(Y - Y_calibration)^2 / norm(Y - mean(Y))^2;
%% high-fidelity neural network
X = X';
Y = Y';

[x, x_highrule] = mapminmax(X, 0, 1);
[y, y_highrule] = mapminmax(Y, 0, 1);

% train the network
net = newff(x, y, 30,{'tansig', 'purelin'});
net.trainFcn = 'trainlm';
net.trainParam.mc = 0.001;
net.trainParam.epochs = 100;
net.trainParam.goal   = 1e-7;
net.trainParam.lr = 0.001;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

[net, info] = train(net, x, y);
% Prediction
idx_train = info.trainInd;idx_val = info.valInd;idx_test = info.testInd;

xtrain = x(:, idx_train);xval = x(:, idx_val);xtest = x(:, idx_test);
YTrain = Y(:, idx_train);YVal = Y(:, idx_val);YTest = Y(:, idx_test);

ysim1 = sim(net, xtrain);YSim1 = mapminmax('reverse', ysim1, y_highrule);
ysim2 = sim(net, xval);YSim2 = mapminmax('reverse', ysim2, y_highrule);
ysim3 = sim(net, xtest);YSim3 = mapminmax('reverse', ysim3, y_highrule);

R2 = 1 - norm(YVal - YSim2)^2 / norm(YVal - mean(YVal))^2;

