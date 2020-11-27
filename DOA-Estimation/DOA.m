clear all; close all; clc;

c = 343.0;
d = 0.2 / sqrt(2);

% Fsnew = 15000;
% y11 = bandpass(y1, [200 Fsnew], Fs);
% y21 = bandpass(y2, [200 Fsnew], Fs);
% y31 = bandpass(y3, [200 Fsnew], Fs);
% y41 = bandpass(y4, [200 Fsnew], Fs);
% y1 = resample(y1, 2*Fsnew, Fs);
% y2 = resample(y2, 2*Fsnew, Fs);
% y3 = resample(y3, 2*Fsnew, Fs);
% y4 = resample(y4, 2*Fsnew, Fs);

% N = 2;
% mic = phased.OmnidirectionalMicrophoneElement;
% array = phased.URA([N, N],[d, d],'Element',mic);
% 
% estimator = phased.GCCEstimator('SensorArray',array,...
%     'PropagationSpeed', c, 'SampleRate', Fs);
% ang = estimator([y1, y2, y3, y4]);
% 
% uv = azel2uv(ang);
% 
% [theta, ~] = cart2pol(uv(2), uv(1));
% theta = mod(rad2deg(theta) - 45, 360)

out = zeros(1, 14);

for ii = 6

    [s, Fs] = read('train/', ii);

    gcc = reshape(gccphat(s, Fs), 4, 4);
    tau1 = (gcc(1, 2) + gcc(4, 3)) / 2; % horizontal
    tau2 = (gcc(1, 4) + gcc(2, 3)) / 2; % vertical
    theta1_ori = rad2deg(acos(-tau1 * c / d));
    theta2_ori = rad2deg(acos(-tau2 * c / d));
    theta1 = real(theta1_ori);
    theta2 = real(theta2_ori);

    % flag = 1 when theta1 and theta2 are same direction
    if theta2 < 90
        flag = abs(theta1 - theta2 - 90) < abs(theta2 + theta1 - 90);
    else
        flag = abs(theta2 - theta1 - 90) < abs(theta2 + theta1 - 270);
    end

    if abs(theta2 - 90) > 40 || ~isreal(theta2_ori)
        theta = (2 * (theta2 < 90) - 1) * theta1 - 45;
    elseif abs(theta1 - 90) > 40 || ~isreal(theta1_ori)
        theta = 45 - (2 * xor(flag, theta2 < 90) - 1) * theta2;
    else
        theta = (((2 * flag - 1) * theta2 + theta1) / 2) * (2 * (theta2 < 90) - 1);
    end

    theta = mod(theta, 360);

    out(ii) = theta;

end
