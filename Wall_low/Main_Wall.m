clear;
rng(1);
%% load data for wall deformation
Data_wall_N = load('Wall_N.mat');
Data_wall_N = Data_wall_N.Save_Wall;  %% M*9 matrix; input: tanfai, Eoed, B, t, Dh, Dv, H, depth of wall; output: deflection of wall
Data_wall_F = load('Wall_F.mat');
Data_wall_F = Data_wall_F.Save_Wall_field;
Data_wall_F = Data_wall_F(:,2);
Data_wall   = [Data_wall_N; Data_wall_F];

M = length(Data_wall);
Data_wall   = Data_wall(randperm(M(end)), :);  %% random combine
%% Training set and testing set
data = [];
for i = 1:M
    datacell = Data_wall{i};
    Train_points = round(linspace(1, length(datacell), 40));
    train_data = datacell(Train_points, :);
    data = [data; train_data];
end
data = data';
He = unique(data(7,:));

X = data(1:end-1,:);
Y = data(end,:);

testHe = He([3,6]);  % divide tetsing set according to He
testidx=[];
for i=testHe
    [~,col] = find(data(7,:)==i);
    testidx = [testidx,col];
end
trainidx = setdiff(1:length(data),testidx);
traindata = data(:,trainidx);

[x, x_rule] = mapminmax(X, 0, 1);
[y, y_rule] = mapminmax(Y, 0, 1);

X_train = X(:,trainidx);Y_train = Y(trainidx);
x_train = x(:,trainidx);y_train = y(trainidx);
X_test  = X(:,testidx); Y_test  = Y(testidx);
x_test  = x(:,testidx); y_test  = y(testidx);
%% train the network
net = feedforwardnet([40,30],'trainlm');
net.trainParam.mc = 0.001;
net.trainParam.epochs = 50;
net.trainParam.goal   = 1e-7;
net.trainParam.lr = 0.002;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0.;
net.divideParam.testRatio = 0.;

[net, info] = train(net, x_train, y_train);
%% Prediction
ysim1 = sim(net, x_train);YSim1 = mapminmax('reverse', ysim1, y_rule);
ysim2 = sim(net, x_test);YSim2 = mapminmax('reverse', ysim2, y_rule);

R1 = 1 - norm(Y_train - YSim1)^2 / norm(Y_train - mean(Y_train))^2;
R2 = 1 - norm(Y_test - YSim2)^2 / norm(Y_test - mean(Y_test))^2;
