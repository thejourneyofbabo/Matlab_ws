%% Script Start
clc
% close all
clear
%% state mapping action
% load('optimalNetworkTargetPdr85.mat', 'net')
% targetPdr = 0.85;
load('optimalNetworkTargetPdr90.mat', 'net')
targetPdr = 0.9;

load('./replayBuffer.mat', "replayBuffer2D")

stateMax = round(max(replayBuffer2D.state, [], 'all'));
stateMin = round(min(replayBuffer2D.state, [], 'all'));
%% rho PDR mapping
rhoArray = 5:5:400;

act1PdrArray = zeros(1, length(rhoArray));
act2PdrArray = zeros(1, length(rhoArray));
act3PdrArray = zeros(1, length(rhoArray));
act4PdrArray = zeros(1, length(rhoArray));
act5PdrArray = zeros(1, length(rhoArray));

for rhoIdx = 1:length(rhoArray)
    rho = rhoArray(rhoIdx);
    % action 1 pdr Array
    windowMatrix = (replayBuffer2D.rho(1, :) == rho);
    pdr = replayBuffer2D.reward(1, windowMatrix);
    averPdr = mean(pdr,'all');
    if isnan(averPdr)
        stop = 0;
    end
    act1PdrArray(rhoIdx) = averPdr;
    
    % action 2 pdr Array
    windowMatrix = (replayBuffer2D.rho(2, :) == rho);
    pdr = replayBuffer2D.reward(2, windowMatrix);
    averPdr = mean(pdr,'all');
    if isnan(averPdr)
        stop = 0;
    end
    act2PdrArray(rhoIdx) = averPdr;
    
    % action 3 pdr Array
    windowMatrix = (replayBuffer2D.rho(3, :) == rho);
    pdr = replayBuffer2D.reward(3, windowMatrix);
    averPdr = mean(pdr);
    if isnan(averPdr)
        stop = 0;
    end
    act3PdrArray(rhoIdx) = averPdr;
    
    % action 4 pdr Array
    windowMatrix = (replayBuffer2D.rho(4, :) == rho);
    pdr = replayBuffer2D.reward(4, windowMatrix);
    averPdr = mean(pdr);
    if isnan(averPdr)
        stop = 0;
    end
    act4PdrArray(rhoIdx) = averPdr;

    % action 5 pdr Array
    windowMatrix = (replayBuffer2D.rho(5, :) == rho);
    pdr = replayBuffer2D.reward(5, windowMatrix);
    averPdr = mean(pdr);
    if isnan(averPdr)
        stop = 0;
    end
    act5PdrArray(rhoIdx) = averPdr;
    
end

actPdrArray = [act1PdrArray; act2PdrArray; act3PdrArray; act4PdrArray; act5PdrArray];

f = figure;
axPlot = subplot(1, 1, 1);
hold on;
grid on;
grid minor;
plot([0, 400], [targetPdr, targetPdr], 'Color', 'black', 'LineWidth', 5);
plot(rhoArray, act1PdrArray, 'LineStyle','-', 'Marker','o', 'MarkerFaceColor', '#ffffff', 'LineWidth', 1.5, 'Color', 'red');
plot(rhoArray, act2PdrArray, 'LineStyle','-', 'Marker','o', 'MarkerFaceColor', '#ffffff', 'LineWidth', 1.5, 'Color', 'blue');
plot(rhoArray, act3PdrArray, 'LineStyle','-', 'Marker','o', 'MarkerFaceColor', '#ffffff', 'LineWidth', 1.5, 'Color', 'green');
plot(rhoArray, act4PdrArray, 'LineStyle','-', 'Marker','o', 'MarkerFaceColor', '#ffffff', 'LineWidth', 1.5, 'Color', 'cyan');
plot(rhoArray, act5PdrArray, 'LineStyle','-', 'Marker','o', 'MarkerFaceColor', '#ffffff', 'LineWidth', 1.5, 'Color', 'magenta');

plotExecution = plot(axPlot, [0], [0], 'LineStyle',':', 'Marker','^', 'MarkerFaceColor', '#000000', 'LineWidth', 3, 'Color', 'black');

hold off;

%% rho mapping action
rhoArray = 5:5:400;
stateArray = linspace(stateMin, stateMax, length(rhoArray));
actionSelectPdrArray = zeros(1, length(stateArray));

for rhoIdx = 1:length(rhoArray)
    rho = rhoArray(rhoIdx);
    state = stateArray(rhoIdx);

    state = dlarray(state, 'CB');
    qValues = predict(net, state);
    [~, actionIdx] = max(extractdata(qValues));

    actionArray(rhoIdx) = actionIdx;
    actionSelectPdrArray(rhoIdx) = actPdrArray(actionIdx, rhoIdx);
end

plotExecution.XData = rhoArray;
plotExecution.YData = actionSelectPdrArray;