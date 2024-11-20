clc
clear
close all
%%

% System parameters (given values)
ms = 317.5;   % Sprung mass (kg)
mu = 45.4;    % Unsprung mass (kg)
ks = 22000;   % Spring stiffness (N/m)
kt = 192000;  % Tire stiffness (N/m)
bs = 1500;    % Damping coefficient (N·s/m)

% Natural frequencies (given)
f1 = 1.25;    % Primary ride frequency (Hz)
f2 = 10.9;    % Secondary ride frequency (Hz)

% Frequency range (log scale)
f = logspace(-3, 3, 1000);  % 0.001 Hz to 1000 Hz
w = 2*pi*f;                 % Convert to rad/s

% Calculate transfer function Ht(s) = (zu-zr)/żr
for i = 1:length(w)
    s = 1i*w(i);
    
    % Numerator
    num = -(mu*ms*s^3 + (mu+ms)*bs*s^2 + (mu+ms)*ks*s);
    
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
ylabel('Tire Deflection (dB)')
title('Frequency Response of Tire Deflection Transfer Function Ht(s)')
ylim([-100 -20])  % Adjusted based on your image
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

% High frequency asymptote (-20 dB/decade)
f_high = logspace(1, 3, 100);
mag_high = -20*log10(f_high/f_high(1)) + mag_db(find(f >= f_high(1), 1));
semilogx(f_high, mag_high, 'r--', 'LineWidth', 1)

% Add vertical lines at resonance frequencies
semilogx([f1 f1], ylim, 'g--', 'LineWidth', 1)
semilogx([f2 f2], ylim, 'g--', 'LineWidth', 1)

% Add text annotations
text(1e-2, -60, '+20 dB/decade', 'Color', 'r', 'FontSize', 10)
text(1e2, -80, '-20 dB/decade', 'Color', 'r', 'FontSize', 10)
text(f1, -40, 'Primary Ride: 1.25 Hz', 'FontSize', 10)
text(f2, -50, 'Secondary Ride: 10.9 Hz', 'FontSize', 10)

legend('Frequency Response', 'Asymptotic Slopes', 'Resonance Frequencies', ...
    'Location', 'southwest')

% Add system parameters text box
dim = [.15 .6 .3 .3];
str = {['ms = ' num2str(ms) ' kg'], ...
       ['mu = ' num2str(mu) ' kg'], ...
       ['ks = ' num2str(ks) ' N/m'], ...
       ['kt = ' num2str(kt) ' N/m'], ...
       ['bs = ' num2str(bs) ' Ns/m']};
annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');