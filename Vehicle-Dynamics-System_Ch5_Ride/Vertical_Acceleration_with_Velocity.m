clc
clear
close all
%%

% Create velocity vector (x-axis)
velocity_kmh = linspace(0, 140, 1000); % 0 to 140 km/h
velocity_ms = velocity_kmh / 3.6;  % Convert km/h to m/s

% Parameters from the Quarter-Car Model
m = 400;    % Sprung mass [kg]
k = 20000;  % Spring stiffness [N/m]
c = 2000;   % Damping coefficient [Ns/m]

% Calculate acceleration transfer function
wavelength = 10;  % [m] road wavelength
omega = 2 * pi * velocity_ms / wavelength;  % Angular frequency [rad/s]

% Initialize acceleration array
acc_magnitude = zeros(size(velocity_ms));

% Calculate acceleration transfer function
for i = 1:length(omega)
    w = omega(i);
    if w > 0  % Avoid division by zero at w = 0
        % Transfer function for acceleration
        numerator = k^2 + (c*w)^2;
        denominator = (k - m*w^2)^2 + (c*w)^2;
        H = w^2 * sqrt(numerator/denominator);
        
        % Convert to g (acceleration due to gravity)
        acc_magnitude(i) = H * 0.02 / 9.81;  % 0.02m is bump height
    end
end

% Create positive and negative acceleration around 1g
acc_with_suspension_pos = 1 + acc_magnitude;  % Above 1g
acc_with_suspension_neg = 1 - acc_magnitude;  % Below 1g

% Without suspension - simplified model
acc_without_magnitude = (omega.^2 * 0.02) / 9.81;  % Direct acceleration
acc_without_suspension_pos = 1 + acc_without_magnitude;  % Above 1g
acc_without_suspension_neg = 1 - acc_without_magnitude;  % Below 1g

% Create the plot
figure('Position', [100 100 800 500])

% Plot without suspension
plot(velocity_kmh, acc_without_suspension_pos, 'b--', 'LineWidth', 1.5)
hold on
plot(velocity_kmh, acc_without_suspension_neg, 'b--', 'LineWidth', 1.5)

% Plot with suspension
plot(velocity_kmh, acc_with_suspension_pos, 'r-', 'LineWidth', 1.5)
plot(velocity_kmh, acc_with_suspension_neg, 'r-', 'LineWidth', 1.5)

% Add 1g reference line
plot(velocity_kmh, ones(size(velocity_kmh)), 'k:', 'LineWidth', 1)

% Add labels and formatting
xlabel('Velocity [km/h]')
ylabel('Vertical Acceleration [g]')
%title('Vertical Acceleration vs. Velocity')
title('Vertical Acceleration vs. Velocity', 'Position', [10, 2.1, 0]) 
legend('Without suspension (upper)', 'Without suspension (lower)', ...
       'With suspension (upper)', 'With suspension (lower)', ...
       '1g reference', 'Location', 'best')
grid on
grid minor

% Set axis limits
xlim([0 140])
ylim([0 2.0])  % Adjust based on your needs

% Format axes
ax = gca;
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.15;
xticks(0:20:140)
yticks(0:0.2:2.0)

% Add secondary x-axis for frequency
ax2 = axes('Position', get(ax, 'Position'), ...
    'XAxisLocation', 'top', ...
    'YAxisLocation', 'right', ...
    'Color', 'none', ...
    'XColor', 'k', ...
    'YColor', 'none');

% Calculate and set frequency axis
freq = velocity_ms / wavelength;  % Frequency in Hz
ax2.XLim = [0 max(freq)];
ax2.XTick = 0:0.5:ceil(max(freq));
xlabel(ax2, 'Frequency [Hz]')