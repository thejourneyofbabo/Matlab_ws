clc
clear
close all
%%

% System parameters
ms = 240;    % Sprung mass (kg)
mu = 36;     % Unsprung mass (kg)
bs = 980;    % Fixed damping coefficient (N·s/m)
ks = 16000;  % Spring stiffness (N/m)

% Different tire stiffness cases (±20%)
kt_cases = [128000, 160000, 192000];  % [0.8Kt, Kt, 1.2Kt] (N/m)
line_styles = {'-', '--', '-.'};  
labels = {'0.8Kt (Low pressure)', 'Kt (Normal)', '1.2Kt (High pressure)'};

% Frequency range
w = logspace(-1, 3, 1000);  % 0.1 to 1000 rad/s

% Create subplots for all three transfer functions
figure('Position', [100 100 1200 400])

% Function to set common axis properties
function setAxisProperties(ax)
    ax.XMinorGrid = 'off';
    ax.YMinorGrid = 'off';
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.GridAlpha = 0.3;
    grid(ax, 'on');
    ax.XTick = [0.1 1 10 100 1000];
    ax.XTickLabel = {'0.1', '1', '10', '100', '1000'};
end

% 1. Acceleration Transfer Function (Ha)
subplot(1,3,1)
hold on

for i = 1:length(kt_cases)
    kt = kt_cases(i);
    
    for j = 1:length(w)
        s = 1i*w(j);
        num = kt*s*(bs*s + ks);
        d = ms*mu*s^4 + (mu+ms)*bs*s^3 + ((ms+mu)*ks + ms*kt)*s^2 + bs*kt*s + ks*kt;
        H(j) = abs(num/d);
    end
    
    loglog(w, H, line_styles{i}, 'LineWidth', 2)
end

xlabel('Frequency (rad/s)')
ylabel('Vertical Acceleration (m/s²/(m/s))')
title('Acceleration Response')
legend(labels, 'Location', 'southeast')
yticks([0.01 0.1 1 10 100])
yticklabels({'0.01', '0.1', '1', '10', '100'})
xlim([0.1 1000])
ylim([0.01 100])
setAxisProperties(gca)

% 2. Suspension Deflection Transfer Function (Hs)
subplot(1,3,2)
hold on

for i = 1:length(kt_cases)
    kt = kt_cases(i);
    
    for j = 1:length(w)
        s = 1i*w(j);
        num = -kt*ms*s;
        d = ms*mu*s^4 + (mu+ms)*bs*s^3 + ((ms+mu)*ks + ms*kt)*s^2 + bs*kt*s + ks*kt;
        H(j) = abs(num/d);
    end
    
    loglog(w, H, line_styles{i}, 'LineWidth', 2)
end

xlabel('Frequency (rad/s)')
ylabel('Suspension Deflection (m/(m/s))')
title('Suspension Deflection Response')
legend(labels, 'Location', 'southeast')
yticks([0.001 0.01 0.1 1 10])
yticklabels({'0.001', '0.01', '0.1', '1', '10'})
xlim([0.1 1000])
ylim([0.001 10])
setAxisProperties(gca)

% 3. Tire Deflection Transfer Function (Ht)
subplot(1,3,3)
hold on

for i = 1:length(kt_cases)
    kt = kt_cases(i);
    
    for j = 1:length(w)
        s = 1i*w(j);
        num = -(mu*ms*s^3 + (mu+ms)*bs*s^2 + (mu+ms)*ks*s);
        d = ms*mu*s^4 + (mu+ms)*bs*s^3 + ((ms+mu)*ks + ms*kt)*s^2 + bs*kt*s + ks*kt;
        H(j) = abs(num/d);
    end
    
    loglog(w, H, line_styles{i}, 'LineWidth', 2)
end

xlabel('Frequency (rad/s)')
ylabel('Tire Deflection (m/(m/s))')
title('Tire Deflection Response')
legend(labels, 'Location', 'southeast')
yticks([0.0001 0.001 0.01 0.1 1])
yticklabels({'0.0001', '0.001', '0.01', '0.1', '1'})
xlim([0.1 1000])
ylim([0.0001 1])
setAxisProperties(gca)

% Add title for the whole figure
sgtitle('Effect of Tire Pressure Variation (Fixed Ks = 16 kN/m, Cs = 980 N·s/m)', 'FontSize', 14)

% Add system parameters text box
txt = sprintf('ms = %d kg\nmu = %d kg\nKs = %d N/m\nCs = %d N·s/m', ...
    ms, mu, ks, bs);
annotation('textbox', [0.02 0.7 0.1 0.2], 'String', txt, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white')