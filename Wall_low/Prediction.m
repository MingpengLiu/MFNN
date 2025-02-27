% prediction of the numereucal results of field measurement

FieldCell = cell(length(Data_wall_F), 3);
for i = 1:length(Data_wall_F)
    datafield = (Data_wall_F{i})';
    B = datafield(3,1);
    dataX = datafield(1:end-1,:);
    dataY = datafield(end,:);
    datax = mapminmax('apply', dataX, x_rule);
    datay = mapminmax('apply', dataY, y_rule);
    ysim = sim(net, datax);
    YSim = mapminmax('reverse', ysim, y_rule);
    FieldCell(i, 1) = {[datafield(end-1:end, :)', YSim']};
    R = 1 - norm(dataY - YSim)^2 / norm(dataY - mean(dataY))^2;
    FieldCell(i, 2) = {R};
    FieldCell(i, 3) = {B};
end

save("FieldCell.mat", "FieldCell");
