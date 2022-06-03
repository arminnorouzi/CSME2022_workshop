clear all
close all
clc

%% Loading data


load('ML_Data.mat')

% creating enigne cycle vector
eng_cycle = linspace(1,size(Data(2:end,1), 1),size(Data(2:end,1), 1))';

% Start of injection [CAD]
SOI = Data(2:end,2); 

% Injection fuel amount [mg]
mf = Data(2:end,3);

% Output Tourque- Load [N.m]
Tout = Data(2:end,4);

% goal is modeling Load for given SOI and mf


%% Standardize Data 
% normalized = data-min(data) / max(data)-min(data)

[mf_n, min_mf, max_mf] = dataTrainStandardized(mf);
[SOI_n, min_SOI, max_SOI] = dataTrainStandardized(SOI);


% first 100000 engine cycle as training
mf_n_tr = mf_n(1:90000);
SOI_n_tr = SOI_n(1:90000);
Tout_n_tr = Tout(1:90000);


% from 90000 engine cycle to end as test
mf_ts_n = mf_n(90000:end);
SOI_ts_n = SOI_n(90000:end);
Tout_ts_n = Tout(90000:end);


%% ploting dataset


figure(1)
set(gcf, 'Position', [100, 100, 1600, 700]);
set(gcf,'color','w');

subplot(311)
plot(eng_cycle(1:90000)', Tout(1:90000)', 'b','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end)', Tout(90000:end)', 'b','LineWidth',1.5)
legend('train data','test data','Interpreter','latex')
ylabel('Load [N.m]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(312)
plot(eng_cycle(1:90000)', mf(1:90000)', 'k','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end)', mf(90000:end)', 'b','LineWidth',1.5)
ylabel('$m_{f,main}$ [mg]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(313)
plot(eng_cycle(1:90000)', SOI(1:90000)', 'k','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end)', SOI(90000:end)', 'b','LineWidth',1.5)
ylabel('SOI [CAD]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

%% Creating input-out for training

utrain = [mf_n_tr'; SOI_n_tr'];
ytrain = Tout_n_tr';


uts = [mf_ts_n'; SOI_ts_n'];
yts = Tout_ts_n';


%% import network from Tensorflow

% modelfile = 'DNNmodel.h5';
LoadModel = importKerasNetwork('DNNmodel_MPC.h5');

analyzeNetwork(LoadModel)
%% Predcition on whole data set
% predicition output
YPredt_load = predict(LoadModel,[SOI_n'; mf_n']');

%% Plotting results


figure(2)
set(gcf, 'Position', [100, 100, 1600, 700]);
set(gcf,'color','w');

subplot(311)
plot(eng_cycle(1:90000), Tout(1:90000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end), Tout(90000:end)', 'g','LineWidth',1.5)
plot(eng_cycle, YPredt_load, 'k--','LineWidth',1.5)
legend('train data' ,'test data', 'prediciton','Interpreter','latex')
ylabel('Load [N.m]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(312)
plot(eng_cycle(1:90000)', mf(1:90000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end)', mf(90000:end)', 'g','LineWidth',1.5)
ylabel('$m_{f,main}$ [mg]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(313)
plot(eng_cycle(1:90000)', SOI(1:90000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end)', SOI(90000:end)', 'g','LineWidth',1.5)
ylabel('SOI [CAD]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

%% Define function- we can use it later for MPC
% We have 128 hidden states and 128 cell states

% Fully connected layers
WFc1 = LoadModel.Layers(2, 1).Weights;
bFc1 = LoadModel.Layers(2, 1).Bias;

WFc2 = LoadModel.Layers(4, 1).Weights;
bFc2 = LoadModel.Layers(4, 1).Bias;

WFc3 = LoadModel.Layers(6, 1).Weights;
bFc3 = LoadModel.Layers(6, 1).Bias;

%% Assigining parameters to a structure

Par.WFc1 = double(WFc1);
Par.bFc1 = double(bFc1);

Par.WFc2 = double(WFc2);
Par.bFc2 = double(bFc2);

Par.WFc3 = double(WFc3);
Par.bFc3 = double(bFc3);


%% Evaluating function


uts = [SOI_n'; mf_n'];
xt1 = 0; % no effect as it is input-output model
for i = 1:120000

xt = dyn_dnn(xt1, uts(:,i),Par);

ydnn_hat(i) = xt(1);

xt1 = xt;
end

%% plotting

figure(3)
set(gcf, 'Position', [100, 100, 1600, 700]);
set(gcf,'color','w');

subplot(311)
plot(eng_cycle(1:90000), Tout(1:90000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end), Tout(90000:end)', 'g','LineWidth',1.5)
plot(eng_cycle, ydnn_hat, 'k--','LineWidth',1.5)
legend('train data' ,'test data', 'prediciton','Interpreter','latex')
ylabel('Load [N.m]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(312)
plot(eng_cycle(1:90000)', mf(1:90000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end)', mf(90000:end)', 'g','LineWidth',1.5)
ylabel('$m_{f,main}$ [mg]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

subplot(313)
plot(eng_cycle(1:90000)', SOI(1:90000)', 'c','LineWidth',1.5)
hold on
plot(eng_cycle(90000:end)', SOI(90000:end)', 'g','LineWidth',1.5)
ylabel('SOI [CAD]','Interpreter','latex')
xlabel('Engine Cycle','Interpreter','latex')
set(gca,'FontSize',12)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;



