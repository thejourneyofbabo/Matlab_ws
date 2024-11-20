clc
clear
close all
%%

% Define normalized frequency range (log scale)
w_wn = logspace(-1, 2, 1000); % 0.1 to 100

% Define optimal damping ratio
zeta_optimal = 1/(2*sqrt(2)); % ≈ 0.354

% Calculate acceleration magnitude response
num = w_wn.^2 .* sqrt((2*zeta_optimal*w_wn).^2 + 1);
den = sqrt((1-(w_wn).^2).^2 + (2*zeta_optimal*w_wn).^2);
mag_db = 20*log10(num./den);

% Create figure
figure('Position', [100 100 800 500])

% Plot the response curve
semilogx(w_wn, mag_db, 'b-', 'LineWidth', 2)
hold on

% Mark the stationary point at ω/ωₙ = √2
stationary_w_wn = sqrt(2);
stationary_mag = 6; % 6 dB
plot(stationary_w_wn, stationary_mag, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10)

% Configure plot
grid on
xlabel('Normalized frequency \omega/\omega_n')
ylabel('|X¨/U| [dB]')
title('Acceleration Response for Optimal Damping Ratio \zeta = 1/(2\sqrt{2})')

% Add legend
legend(['Optimal \zeta = ' num2str(zeta_optimal, '%.3f')], 'Stationary point', ...
    'Location', 'northwest')

% Set axis limits
xlim([0.1 100])
ylim([-20 60])

% Add minor grid
grid minor
ax = gca;
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.15;

% Add frequency ticks
xticks([0.1 0.2 0.5 1 2 5 10 20 50 100])
yticks(-20:10:60)

% Add annotation for the stationary point
annotation('textbox', [.15 .5 .3 .2], 'String', ...
    {'Stationary Point:', ...
     '\omega/\omega_n = \sqrt{2}', ...
     '|X¨/U| = 6 dB', ...
     'd/d\omega|X¨/U| = 0'}, ...
    'FitBoxToText', 'on', ...
    'BackgroundColor', 'white', ...
    'EdgeColor', 'black');

% Add vertical line at stationary point
plot([stationary_w_wn stationary_w_wn], [-20 60], 'k:', 'LineWidth', 1)

% Calculate and plot derivative near stationary point to show zero slope
w_near = linspace(sqrt(2)*0.9, sqrt(2)*1.1, 100);
num_near = w_near.^2 .* sqrt((2*zeta_optimal*w_near).^2 + 1);
den_near = sqrt((1-w_near.^2).^2 + (2*zeta_optimal*w_near).^2);
mag_db_near = 20*log10(num_near./den_near);