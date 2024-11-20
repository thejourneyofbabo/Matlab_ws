clc
clear
close all
%%

% Define normalized frequency range (log scale)
w_wn = logspace(-1, 2, 1000); % 0.1 to 100

% Define different damping ratios
zeta = [0.2 0.4 0.6 1 10 inf]; % ζ values
line_styles = {'-', '-', '-', '-', '-', '-'}; % Line styles for each curve
colors = {'r', 'g', 'b', 'm', 'c', 'k'}; % Colors for each curve

% Create figure
figure('Position', [100 100 800 500])

% Plot for each damping ratio
for i = 1:length(zeta)
    if isinf(zeta(i))
        % For infinite damping
        mag_db = 20*log10(w_wn.^2); % Acceleration response for rigid body
    else
        % Calculate acceleration magnitude response
        % Multiply by (ω)² for acceleration
        num = w_wn.^2 .* sqrt((2*zeta(i)*w_wn).^2 + 1);
        den = sqrt((1-(w_wn).^2).^2 + (2*zeta(i)*w_wn).^2);
        mag_db = 20*log10(num./den);
    end
    
    % Plot with different colors and add legend entry
    semilogx(w_wn, mag_db, [colors{i} line_styles{i}], 'LineWidth', 2)
    hold on
end

% Add stationary point
% For acceleration, stationary point occurs at ω/ωₙ = √2
stationary_w_wn = sqrt(2);
% Calculate magnitude at stationary point
% At ω/ωₙ = √2, the magnitude is 2 (or 6 dB)
stationary_mag = 6; % 6 dB
plot(stationary_w_wn, stationary_mag, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 10)
text(stationary_w_wn*1.2, stationary_mag+2, 'Stationary point', ...
    'FontSize', 10, 'FontWeight', 'bold')

% Configure plot
grid on
xlabel('Normalized frequency \omega/\omega_n')
ylabel('|X¨/U| [dB]')
title('Acceleration Response for Different Damping Ratios')

% Add legend
legend('\zeta = 0.2', '\zeta = 0.4', '\zeta = 0.6', '\zeta = 1', ...
    '\zeta = 10', '\zeta = \infty', 'Stationary point', ...
    'Location', 'northwest')

% Set axis limits
xlim([0.1 100])
ylim([-20 60])

% Add minor grid
grid minor
ax = gca;
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.15;

% Format grid for easier reading
ax.XScale = 'log';
ax.YGrid = 'on';
ax.XGrid = 'on';

% Add frequency ticks
xticks([0.1 0.2 0.5 1 2 5 10 20 50 100])
yticks(-20:10:60)

% Add annotation to explain stationary point
dim = [.15 .3 .3 .2];
str = {'Stationary Point:', ...
       'ω/ωₙ = √2', ...
       'Magnitude = 6 dB', ...
       'All damping ratios produce', ...
       'same acceleration response'};
annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');