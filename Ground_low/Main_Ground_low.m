clear;
rng(1);
%% load data for wall deformation
Data_ground_N = load('Ground_N.mat');
Data_ground_N = Data_ground_N.Save_Ground;  %% M*9 matrix; input: tanfai, Eoed, B, t, Dh, Dv, H, depth of wall; output: deflection of wall
Data_ground_F = load('Ground_F.mat');
Data_ground_F = Data_ground_F.Save_Ground_field;
Data_ground_F = Data_ground_F(:,2);
Data_ground   = [Data_ground_N; Data_ground_F];

M = length(Data_ground);
Data_ground   = Data_ground(randperm(M(end)), :);  %% random combine 

%% Training set and testing set
data = [];
for i = 1:M
    datacell = Data_ground{i};
    Train_points = round(linspace(1, length(datacell), 40));
    traindata = datacell(Train_points, :);
    data = [data; traindata];
end
data = data';

X = data(1:end-1,:);
Y = data(end,:);
[x, x_rule] = mapminmax(X, 0, 1);
[y, y_rule] = mapminmax(Y, 0, 1);

%% train the network
net = newff(x, y, [40,30],{'tansig', 'tansig', 'purelin'});
net.trainFcn = 'trainlm';
net.trainParam.mc = 0.001;
net.trainParam.epochs = 100;
net.trainParam.goal   = 1e-9;
net.trainParam.lr = 0.001;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

[net, info] = train(net, x, y);

%% Prediction
idx_train = info.trainInd;idx_val = info.valInd;idx_test = info.testInd;

xtrain = x(:, idx_train);xval = x(:, idx_val);xtest = x(:, idx_test);
YTrain = Y(:, idx_train);YVal = Y(:, idx_val);YTest = Y(:, idx_test);

ysim1 = sim(net, xtrain);YSim1 = mapminmax('reverse', ysim1, y_rule);
ysim2 = sim(net, xval);YSim2 = mapminmax('reverse', ysim2, y_rule);
ysim3 = sim(net, xtest);YSim3 = mapminmax('reverse', ysim3, y_rule);

MSE1 = mse(YTrain,YSim1);MSE2 = mse(YVal,YSim2);MSE3 = mse(YTest,YSim3);

R1 = 1 - norm(YTrain - YSim1)^2 / norm(YTrain - mean(YTrain))^2;
R2 = 1 - norm(YVal - YSim2)^2 / norm(YVal - mean(YVal))^2;
R3 = 1 - norm(YTest - YSim3)^2 / norm(YTest - mean(YTest))^2;

