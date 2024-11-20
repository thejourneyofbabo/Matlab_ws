clc
clear
close all
%%

% System parameters
ms = 240;    % Sprung mass (kg)
mu = 36;     % Unsprung mass (kg)
ks = 16000;  % Spring stiffness (N/m)
kt = 160000; % Tire stiffness (N/m)

% Different damping coefficients
bs_cases = [3920, 980, 196];  % Cases A, B, C (N·s/m)
line_styles = {'-', '--', '-.'};  % Different line styles for each case

% Frequency range (matching the plot)
w = logspace(-1, 3, 1000);  % 0.1 to 1000 rad/s

% Create figure with log-log scale
figure('Position', [100 100 800 500])
hold on

% Calculate and plot for each damping case
for i = 1:length(bs_cases)
    bs = bs_cases(i);
    
    % Calculate transfer function
    for j = 1:length(w)
        s = 1i*w(j);
        
        % Numerator
        num = kt*s*(bs*s + ks);
        
        % Denominator
        d = ms*mu*s^4 + (mu+ms)*bs*s^3 + ((ms+mu)*ks + ms*kt)*s^2 + bs*kt*s + ks*kt;
        
        % Transfer function (absolute value for magnitude)
        H(j) = abs(num/d);
    end
    
    % Plot on log-log scale
    loglog(w, H, line_styles{i}, 'LineWidth', 2)
end

% Configure plot
grid on
xlabel('Frequency (rad/s)')
ylabel('Vertical Acceleration (m/s²/(m/s))')
title('Frequency Response with Different Damping Coefficients')

% Set specific axis ticks
xticks([0.1 1 10 100 1000])
yticks([0.01 0.1 1 10 100])
xticklabels({'0.1', '1', '10', '100', '1000'})
yticklabels({'0.01', '0.1', '1', '10', '100'})

% Set axis limits
xlim([0.1 1000])
ylim([0.01 100])

% Add legend
legend('Cs = 3920 N·s/m (A)', 'Cs = 980 N·s/m (B)', 'Cs = 196 N·s/m (C)', ...
    'Location', 'southeast')

% Add system parameters text box
txt = sprintf('ms = %d kg\nmu = %d kg\nks = %d N/m\nkt = %d N/m', ...
    ms, mu, ks, kt);
annotation('textbox', [0.7 0.7 0.2 0.2], 'String', txt, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white')

% Add grid
grid on
ax = gca;
ax.GridAlpha = 0.3;

% Make the grid lines match the main ticks only
ax.XMinorGrid = 'off';
ax.YMinorGrid = 'off';
ax.XScale = 'log';
ax.YScale = 'log';

% Add text annotations for key observations
text(0.2, 0.02, 'Smooth ride at low frequencies', 'FontSize', 8)
text(50, 0.5, 'High damping: harsh ride', 'FontSize', 8)
text(2, 50, 'Low damping: poor control', 'FontSize', 8)