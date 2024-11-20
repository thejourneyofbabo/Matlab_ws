clc
clear
close all
%%

% Define frequency range (log scale)
f = logspace(-1, 2, 1000); % 0.1 Hz to 100 Hz
omega = 2*pi*f;

% System parameters
m = 400;  % kg
k_nominal = 20000;  % N/m (20 kN/m)
c_nominal = 2000;   % N.s/m (2 kN.s/m)

% Create variations of K
k_high1 = k_nominal * 1.5;    % First level increase
k_high2 = k_nominal * 2;      % Second level increase (larger)
k_low1 = k_nominal * 0.75;    % First level decrease
k_low2 = k_nominal * 0.5;     % Second level decrease (smaller)

% Function to calculate acceleration frequency response
function mag_db = calculate_acceleration_response(m, k, c, omega)
    % For acceleration, multiply transfer function by ω²
    num = omega.^2 .* sqrt(k^2 + (c*omega).^2);
    den = sqrt((k - m*omega.^2).^2 + (c*omega).^2);
    mag = num./den;
    mag_db = 20*log10(mag);
end

% Calculate responses
mag_db_nominal = calculate_acceleration_response(m, k_nominal, c_nominal, omega);
mag_db_high1 = calculate_acceleration_response(m, k_high1, c_nominal, omega);
mag_db_high2 = calculate_acceleration_response(m, k_high2, c_nominal, omega);
mag_db_low1 = calculate_acceleration_response(m, k_low1, c_nominal, omega);
mag_db_low2 = calculate_acceleration_response(m, k_low2, c_nominal, omega);

% Create plot
figure('Position', [100 100 800 500])

% Plot all curves
semilogx(f, mag_db_nominal, 'k-', 'LineWidth', 2, 'DisplayName', 'Nominal K')
hold on
semilogx(f, mag_db_high1, 'b--', 'LineWidth', 2, 'DisplayName', 'K increase (1.5K)')
semilogx(f, mag_db_high2, 'b:', 'LineWidth', 2, 'DisplayName', 'K increase (2K)')
semilogx(f, mag_db_low1, 'r--', 'LineWidth', 2, 'DisplayName', 'K decrease (0.75K)')
semilogx(f, mag_db_low2, 'r:', 'LineWidth', 2, 'DisplayName', 'K decrease (0.5K)')

% Configure plot
grid on
xlabel('Frequency [Hz]')
ylabel('|X¨/U| [dB]')
title('Effects of Spring Constant Variation on Acceleration Response')
legend('Location', 'northwest')

% Set axis limits
xlim([0.1 100])
ylim([-10 70])

% Add minor grid
grid minor
ax = gca;
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.15;

% Add frequency ticks
xticks([0.1 0.2 0.5 1 2 5 10 20 50 100])
yticks(-10:10:70)

% Format grid for easier reading
ax.XScale = 'log';
ax.YGrid = 'on';
ax.XGrid = 'on';