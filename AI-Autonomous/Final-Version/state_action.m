%% State에 따른 Action 선택 분석
clear all;
close all;

% Load trained networks for both target PDRs
load('optimalNetworkTargetPdr85.mat');  % For 85% target
net85 = net;
load('optimalNetworkTargetPdr90.mat');  % For 90% target
net90 = net;

% Generate state space
states = 0:80;
actions85 = zeros(size(states));
actions90 = zeros(size(states));

% Get actions for each state
for i = 1:length(states)
   state = states(i);
   
   % For 85% target PDR
   dlState = dlarray(state, 'CB');
   qValues = predict(net85, dlState);
   [~, actionIdx] = max(extractdata(qValues));
   actions85(i) = actionIdx * 100;  % Convert to ms
   
   % For 90% target PDR
   qValues = predict(net90, dlState);
   [~, actionIdx] = max(extractdata(qValues));
   actions90(i) = actionIdx * 100;  % Convert to ms
end

% Plot results
figure;
plot(states, actions85, 'b-', 'LineWidth', 2, 'DisplayName', 'Target PDR 85%');
hold on;
plot(states, actions90, 'r--', 'LineWidth', 2, 'DisplayName', 'Target PDR 90%');
xlabel('Number of nodes within 100m');
ylabel('Selected message rate (ms)');
title('Message Rate Selection vs Node Density');
grid on;
legend('show');

% Add annotations for changing points
[~, change_point85] = find(diff(actions85) ~= 0, 1, 'first');
[~, change_point90] = find(diff(actions90) ~= 0, 1, 'first');

fprintf('Change point for 85%% target PDR: Around %d nodes\n', states(change_point85));
fprintf('Change point for 90%% target PDR: Around %d nodes\n', states(change_point90));

%% 설명: Target PDR이 높을 때 더 낮은 State에서 Action이 변화하는 이유
% 1. 높은 Target PDR(90%)은 더 엄격한 통신 품질을 요구
% 2. 노드 수가 증가하면 채널 혼잡도가 증가
% 3. 따라서 90% PDR 달성을 위해서는 더 일찍(낮은 노드 밀도에서) 
%    전송 주기를 조절해야 함
% 4. 85% Target의 경우 상대적으로 여유가 있어 더 높은 노드 밀도까지
%    낮은 전송 주기 유지 가능