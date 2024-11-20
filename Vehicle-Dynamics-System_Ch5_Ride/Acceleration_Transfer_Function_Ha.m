clc
clear
close all
%%

% System parameters
ms = 400;    % Sprung mass (kg)
mu = 40;     % Unsprung mass (kg)
ks = 20000;  % Spring stiffness (N/m)
kt = 200000; % Tire stiffness (N/m)
bs = 1000;   % Damping coefficient (N·s/m)

% Frequency range (log scale)
f = logspace(-3, 3, 1000);  % 0.001 Hz to 1000 Hz
w = 2*pi*f;                 % Convert to rad/s

% Calculate transfer function Ha(s) = z̈s/żr
% Ha(s) = kt*s(bs*s + ks)/d(s)
for i = 1:length(w)
    s = 1i*w(i);
    
    % Numerator
    num = kt*s*(bs*s + ks);
    
    % Denominator
    d = ms*mu*s^4 + (mu+ms)*bs*s^3 + ((ms+mu)*ks + ms*kt)*s^2 + bs*kt*s + ks*kt;
    
    % Transfer function
    H(i) = num/d;
end

% Calculate magnitude in dB
mag_db = 20*log10(abs(H));

% Create figure
figure('Position', [100 100 800 500])
semilogx(f, mag_db, 'LineWidth', 2)
grid on

% Configure plot
xlabel('Frequency (Hz)')
ylabel('Acceleration (dB)')
title('Frequency Response of Acceleration Transfer Function Ha(s)')
ylim([-70 30])
xlim([1e-3 1e3])

% Add minor grid
grid minor
ax = gca;
ax.GridAlpha = 0.3;
ax.MinorGridAlpha = 0.15;

% Add asymptotic slopes
hold on
% Low frequency asymptote (+20 dB/decade)
f_low = logspace(-3, 0, 100);
mag_low = 20*log10(2*pi*f_low) + mag_db(1);
semilogx(f_low, mag_low, 'r--', 'LineWidth', 1)

% High frequency asymptote (-40 dB/decade)
f_high = logspace(1, 3, 100);
mag_high = -40*log10(f_high/f_high(1)) + mag_db(find(f >= f_high(1), 1));
semilogx(f_high, mag_high, 'r--', 'LineWidth', 1)

% Add text annotations
text(1e-2, 0, '+20 dB/decade', 'Color', 'r', 'FontSize', 10)
text(1e2, -40, '-40 dB/decade', 'Color', 'r', 'FontSize', 10)

% Add resonance frequencies
text(1.25, 10, 'Primary ride freq: 1.25 Hz', 'FontSize', 10)
text(10.9, -10, 'Secondary ride freq: 10.9 Hz', 'FontSize', 10)

legend('Frequency Response', 'Asymptotic Slopes', 'Location', 'southwest')