%% Script Start
clc
close all
clear

load('./replayBuffer.mat', "replayBuffer2D")

rng(1);

%% Create DQN Agent
% Define the neural network architecture

% nodeNum1 = 256;
% nodeNum2 = 128;
% nodeNum3 = 64;
% 
% net = [ % Three Layer
%     featureInputLayer(1,'Normalization','none', 'Name','state')
%     fullyConnectedLayer(nodeNum1, 'Name', 'fc1')
%     batchNormalizationLayer('Name', 'bn1')
%     reluLayer('Name', 'relu1')
%     fullyConnectedLayer(nodeNum2, 'Name', 'fc2')
%     batchNormalizationLayer('Name', 'bn2')
%     reluLayer('Name', 'relu2')
%     fullyConnectedLayer(nodeNum3, 'Name','fc3')
%     batchNormalizationLayer('Name', 'bn3')
%     reluLayer('Name', 'relu3')
%     fullyConnectedLayer(5, 'Name', 'fc_out')
%     ];

nodeNum1 = 256;
nodeNum2 = 128;
nodeNum3 = 96;
nodeNum4 = 64;

net = [ % Four Layer
    featureInputLayer(1,'Normalization','none', 'Name','state')
    fullyConnectedLayer(nodeNum1, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(nodeNum2, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(nodeNum3, 'Name', 'fc3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(nodeNum4, 'Name', 'fc4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(5, 'Name', 'fc_out')
];

% nodeNum1 = 256;
% nodeNum2 = 128;
% nodeNum3 = 64;
% nodeNum4 = 32;
% nodeNum5 = 16;
% 
% net = [ % Five Layer
%     featureInputLayer(1,'Normalization','none', 'Name','state')
%     fullyConnectedLayer(nodeNum1, 'Name', 'fc1')
%     batchNormalizationLayer('Name', 'bn1')
%     reluLayer('Name', 'relu1')
%     fullyConnectedLayer(nodeNum2, 'Name', 'fc2')
%     batchNormalizationLayer('Name', 'bn2')
%     reluLayer('Name', 'relu2')
%     fullyConnectedLayer(nodeNum3, 'Name', 'fc3')
%     batchNormalizationLayer('Name', 'bn3')
%     reluLayer('Name', 'relu3')
%     fullyConnectedLayer(nodeNum4, 'Name', 'fc4')
%     batchNormalizationLayer('Name', 'bn4')
%     reluLayer('Name', 'relu4')
%     fullyConnectedLayer(nodeNum5, 'Name', 'fc5')
%     batchNormalizationLayer('Name', 'bn5')
%     reluLayer('Name', 'relu5')
%     fullyConnectedLayer(5, 'Name', 'fc_out')
% ];

net = dlnetwork(net);

% Hyperparameters
numEpisodes = 100;
maxStepsPerEpisode = 200;
gamma = 0.99; % Discout Factor
epsilon = 1.0; % Exploration Initial Value
epsilonDecay = 0.997;
minEpsilon = 0.01;
initialLearningRate =1e-3;

% Experience Replay Memory Initialize
bufferSize = 5000;
batchSize = 256;
replayBuffer = zeros(bufferSize, 5);  % [state, action, reward, next_state, done]
bufferCounter = 1;

% Initialize optimizer variables
gradThreshold = 1.0;  % 그래디언트 클리핑 임계값
beta1 = 0.9;
beta2 = 0.999;
epsilonOpt = 1e-8;  % 최적화에 사용하는 epsilon
movingAvg = [];
movingAvgSq = [];

targetNet = updateTargetNet(net);

% 학습 진행 상황 모니터링 설정
monitor = trainingProgressMonitor( ...
    Metrics="EpisodeReward", ...
    Info="Episode", ...
    XLabel="EpisodeNumber");

%% DQN 학습
updateIteration = 1;
learningRate = initialLearningRate;
dataIdxRange = 1:length(replayBuffer2D.state);
actArray = 1:5;
% targetPdr = 0.85;
targetPdr = 0.90;
successReward = 10;
failureReward = 0;

% Pre-allocate batch memory
states_batch = zeros(1, batchSize);
actions_batch = zeros(batchSize, 1);
rewards_batch = zeros(batchSize, 1);
nextStates_batch = zeros(1, batchSize);
dones_batch = zeros(batchSize, 1);

for episode = 1:numEpisodes
    totalReward = 0;
    
    for step_i = 1:maxStepsPerEpisode
        dataIdx = randsample(dataIdxRange, 1);

        state = unique(replayBuffer2D.state(:, dataIdx));
        nextState = state;

        % epsilon-greedy 정책에 따라 행동 선택
        if rand < epsilon
            actionIdx = randi([1, length(actArray)]);
        else
            dlState = dlarray(state, 'CB'); % 'CB' 차원 태그 사용
            qValues = predict(net, dlState);
            [~, actionIdx] = max(extractdata(qValues));
        end
        
        % 행동을 올바른 값으로 매핑
        action = actArray(actionIdx);
        
        % % Reward Decision
        pdrArray = replayBuffer2D.reward(:, dataIdx);
        rewardArray = pdrArray - targetPdr;       
        % 
        % % Reward #5 Ranked Reward:
        [~, rank] = sort(abs(rewardArray(:,1)));
        actionRank = find(rank == action);
        reward = successReward * (1 - (actionRank-1)/length(actArray)); 


        % alpha = 0.5; % Decay factor
        % reward = successReward * exp(-alpha * (actionRank - 1));

        % beta = 2; % Steepness
        % k = ceil(length(actArray)/2); % Midpoint
        % reward = successReward / (1 + exp(beta * (actionRank - k)));


        % proximityFactor = 1 / (1 + abs(pdrArray(action, 1) - targetPdr));
        % reward = successReward * (1 - (actionRank - 1) / length(actArray)) * proximityFactor;
        
        
        %%


        totalReward = totalReward + reward;
        
        isDone = 1;

        % 경험 저장
        experience = [state', actionIdx, reward, nextState', isDone];
        if bufferCounter <= bufferSize
            replayBuffer(bufferCounter, :) = experience;
        else
            idx = mod(bufferCounter - 1, bufferSize) + 1;
            replayBuffer(idx, :) = experience;
        end
        bufferCounter = bufferCounter + 1;
        
        state = nextState;
        
        % 경험 재생 메모리에서 미니배치 샘플링
        if bufferCounter > batchSize %256
            batchIndices = randi(min(bufferCounter-1, bufferSize), [batchSize, 1]);
            batch = replayBuffer(batchIndices, :);
            
            % 배치에서 데이터 추출
            states_batch(:) = batch(:, 1)';
            actions_batch(:) = batch(:, 2)';
            rewards_batch(:) = batch(:, 3)';
            nextStates_batch(:) = batch(:, 4)';
            dones_batch(:) = batch(:, 5);

            % 타깃 Q 값 계산
            dlNextStates = dlarray(nextStates_batch, 'CB'); % 'CB' 차원 태그 사용
            nextQValues = predict(targetNet, dlNextStates);
            maxNextQValues = max(extractdata(nextQValues));
            targetQValues = rewards_batch + gamma * maxNextQValues .* (1 - dones_batch);
            
            % Q 값 예측 및 손실 계산
            dlStates = dlarray(states_batch, 'CB'); % 'CB' 차원 태그 사용
            qValues = predict(net, dlStates);
            indices = sub2ind(size(qValues), actions_batch, (1:batchSize)');              
            
            % 경사 하강법으로 네트워크 업데이트
            [gradients, net] = dlfeval(@modelGradients, net, dlStates, targetQValues, indices);
            gradients = dlupdate(@(g) min(max(g, -gradThreshold), gradThreshold), gradients);  % 그래디언트 클리핑
            [net, movingAvg, movingAvgSq] = adamupdate(net, gradients, movingAvg, movingAvgSq, updateIteration, learningRate , beta1, beta2, epsilonOpt);
            updateIteration = updateIteration + 1;
            % epsilon 감소
            epsilon = max(minEpsilon, epsilon * epsilonDecay);
        end
    end
     % 학습 진행 모니터링 업데이트
        recordMetrics(monitor,episode, ...
            EpisodeReward=totalReward);
    
        updateInfo(monitor,Episode=episode);
        monitor.Progress = 100*episode/numEpisodes;
    
    
    % 타깃 네트워크 업데이트
    if mod(episode, 10) == 0  % 타깃 네트워크 업데이트 주기 증가
        targetNet = updateTargetNet(net);
    end
    
    % 학습 진행 상황 출력
    fprintf('Episode %d, Total Reward: %.2f, Epsilon: %.2f\n', episode, totalReward, epsilon);
    save(['optimalNetworkTargetPdr',num2str(targetPdr*100),'.mat'], 'net')
end
% save('net.mat','net');
% save('env.mat','env');
% save('actInfo.mat','actInfo');


%% 모형 경사 계산 함수
function [gradients, dlnet] = modelGradients(dlnet, dlStates, targetQValues, indices)
    qValues = predict(dlnet, dlStates);
    predictedQ = dlarray(qValues(indices), 'CB');
    targetQ = dlarray(targetQValues, 'CB');       
    loss = mean(sum((targetQ - predictedQ).^2));
    gradients = dlgradient(loss, dlnet.Learnables);
end

% 네트워크 복사 함수
function targetNet = updateTargetNet(dlnet)
    targetNet = dlnet;
end
