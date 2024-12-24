%% Script Start
clc
close all
clear

load('./replayBuffer.mat', "replayBuffer2D")
rng(1);

%% Create DQN Agent
nodeNum1 = 128;
nodeNum2 = 64;

net = [
 featureInputLayer(1, 'Normalization', 'none', 'Name', 'state')
 fullyConnectedLayer(nodeNum1, 'Name', 'fc1')  
 batchNormalizationLayer('Name', 'bn1')
 reluLayer('Name', 'relu1')
 fullyConnectedLayer(nodeNum2, 'Name', 'fc2')   
 reluLayer('Name', 'relu2')
 fullyConnectedLayer(5, 'Name', 'fc_out')
 ];

net = dlnetwork(net);

% Hyperparameters
numEpisodes = 500;
maxStepsPerEpisode = 200;
gamma = 0.99;
epsilon = 1;
epsilonDecay = 0.995;
minEpsilon = 0.01;
initialLearningRate = 1e-3;

% Experience Replay
bufferSize = 10000;
batchSize = 128;
replayBuffer = zeros(bufferSize, 5);
bufferCounter = 1;

% Initialize optimizer variables
gradThreshold = 1.0;
beta1 = 0.9;
beta2 = 0.999;
epsilonOpt = 1e-8;
movingAvg = [];
movingAvgSq = [];

targetNet = updateTargetNet(net);

monitor = trainingProgressMonitor( ...
  Metrics="EpisodeReward", ...
  Info="Episode", ...
  XLabel="EpisodeNumber");

%% Training
updateIteration = 1;
learningRate = initialLearningRate;
dataIdxRange = 1:length(replayBuffer2D.state);
actArray = 1:5;
targetPdr = 0.85;
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
      dataIdx = dataIdxRange(randi(length(dataIdxRange)));
      state = unique(replayBuffer2D.state(:, dataIdx));
      nextState = state;

      if rand < epsilon
          actionIdx = randi([1, length(actArray)]);
      else
          dlState = dlarray(state, 'CB');
          qValues = predict(net, dlState);
          [~, actionIdx] = max(extractdata(qValues));
      end
      
      action = actArray(actionIdx);
      pdrArray = replayBuffer2D.reward(:, dataIdx);
      rewardArray = pdrArray - targetPdr;
      
      % Enhanced reward design
      % if length(rewardArray(rewardArray >= 0)) == length(actArray)
      %     if action == 1
      %         reward = successReward;
      %     else
      %         reward = successReward/2;
      %     end
      % else
      %     pdrDiff = abs(pdrArray(action) - targetPdr);
      %     reward = -pdrDiff * 10;
      % end
      reward = failureReward;

      if abs(rewardArray(action,1)) == min(abs(rewardArray(:,1)))
          reward = successReward;
      end

      totalReward = totalReward + reward;
      isDone = 1;

      experience = [state', actionIdx, reward, nextState', isDone];
      if bufferCounter <= bufferSize
          replayBuffer(bufferCounter, :) = experience;
      else
          idx = mod(bufferCounter - 1, bufferSize) + 1;
          replayBuffer(idx, :) = experience;
      end
      bufferCounter = bufferCounter + 1;
      
      % DDQN Training
      if bufferCounter > batchSize
          batchIndices = randi(min(bufferCounter-1, bufferSize), [batchSize, 1]);
          batch = replayBuffer(batchIndices, :);
          
          states_batch(:) = batch(:, 1)';
          actions_batch(:) = batch(:, 2);
          rewards_batch(:) = batch(:, 3);
          nextStates_batch(:) = batch(:, 4)';
          dones_batch(:) = batch(:, 5);

          dlNextStates = dlarray(nextStates_batch, 'CB');
          nextQValues = predict(net, dlNextStates);
          [~, maxActions] = max(extractdata(nextQValues));
          targetNextQValues = predict(targetNet, dlNextStates);
          
          maxNextQValues = zeros(size(maxActions));
          for i = 1:length(maxActions)
              maxNextQValues(i) = targetNextQValues(maxActions(i), i);
          end
          
          targetQValues = rewards_batch + gamma * maxNextQValues' .* (1 - dones_batch);
          
          dlStates = dlarray(states_batch, 'CB');
          qValues = predict(net, dlStates);
          indices = sub2ind(size(qValues), actions_batch, (1:batchSize)');
          
          [gradients, net] = dlfeval(@modelGradients, net, dlStates, targetQValues, indices);
          gradients = dlupdate(@(g) min(max(g, -1), 1), gradients);
          [net, movingAvg, movingAvgSq] = adamupdate(net, gradients, movingAvg, movingAvgSq, updateIteration, learningRate, beta1, beta2, epsilonOpt);
          updateIteration = updateIteration + 1;
          
          epsilon = max(minEpsilon, epsilon * epsilonDecay);
      end
  end

  recordMetrics(monitor, episode, EpisodeReward=totalReward);
  updateInfo(monitor, Episode=episode);
  monitor.Progress = 100*episode/numEpisodes;
  
  if mod(episode, 5) == 0
      targetNet = updateTargetNet(net);
  end
  
  fprintf('Episode %d, Total Reward: %.2f, Epsilon: %.2f\n', episode, totalReward, epsilon);
  save(['optimalNetworkTargetPdr',num2str(targetPdr*100),'.mat'], 'net')
end

%% Helper Functions
function [gradients, dlnet] = modelGradients(dlnet, dlStates, targetQValues, indices)
   qValues = predict(dlnet, dlStates);
   predictedQ = dlarray(qValues(indices), 'CB');
   targetQ = dlarray(targetQValues, 'CB');
   loss = mean(sum((targetQ - predictedQ).^2));
   gradients = dlgradient(loss, dlnet.Learnables);
end

function targetNet = updateTargetNet(dlnet)
  targetNet = dlnet;
end