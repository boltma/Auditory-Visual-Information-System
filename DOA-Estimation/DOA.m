clear all; close all; clc;

c = 343.0;
d = 0.2 / sqrt(2);

% Fsnew = 15000;
% y1 = bandpass(y1, [200 20000], Fs);
% y2 = bandpass(y2, [200 20000], Fs);
% y3 = bandpass(y3, [200 20000], Fs);
% y4 = bandpass(y4, [200 20000], Fs);
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
out2 = zeros(1, 14);
out3 = zeros(1, 14);

for ii = 3
    disp(ii);

    [s, Fs] = read('train/', ii);
    
    mics = [0, d / sqrt(2), 0;...
        d/sqrt(2), 0, 0;...
        0, -d/sqrt(2), 0;...
        -d/sqrt(2), 0, 0;];
    
    %try
        %[finalpos,finalsrp,finalfe] = srplems(s, mics, Fs, [-5 -5 -5], [5 5 5]);
        [finalpos, minim] = SRP_PHAT_SRC(mics, Fs, s, 2000000, -5, 5);
        [theta, ~] = cart2pol(finalpos(2), finalpos(1));
        theta = rad2deg(theta);
        %theta = finalpos;
        out2(ii) = minim;
%         if minim < 0.05
%             gcc = reshape(gccphat(s, Fs), 4, 4);
% %         theta1 = rad2deg(acos(-tau1 * c / d));
% %         theta2 = rad2deg(acos(-tau2 * c / d));
%             tau1 = gcc(2, 4);
%             tau2 = gcc(1, 3);
%             x = -tau1 * c / sqrt(2) / d;
%             y = -tau2 * c / sqrt(2) / d;
%             theta = rad2deg(cart2pol(y, x));
%         end
       
    %catch
%         disp([num2str(ii), "failed"]);
        gcc = reshape(gccphat(s, Fs), 4, 4);
%         tau1 = (gcc(1, 2) + gcc(4, 3)) / 2; % horizontal
%         tau2 = (gcc(1, 4) + gcc(2, 3)) / 2; % vertical
%         theta1 = rad2deg(acos(-tau1 * c / d));
%         theta2 = rad2deg(acos(-tau2 * c / d));
        flag1 = 0;
        flag2 = 0;
        flag3 = 0;
        flag4 = 0;
        if abs(gcc(2, 3) - gcc(1, 4)) > 1 / Fs
            if abs(gcc(2, 3)) < abs(gcc(1, 4))
                gcc(1, 4) = gcc(2, 3);
                gcc(4, 1) = gcc(3, 2);
                flag1 = 1;
            else
                gcc(2, 3) = gcc(1, 4);
                gcc(3, 2) = gcc(4, 1);
                flag2 = 1;
            end
        end
        if abs(gcc(2, 1) - gcc(3, 4)) > 1 / Fs
            if abs(gcc(2, 1)) < abs(gcc(3, 4))
                gcc(3, 4) = gcc(2, 1);
                gcc(4, 3) = gcc(1, 2);
                flag3 = 1;
            else
                gcc(2, 1) = gcc(3, 4);
                gcc(1, 2) = gcc(4, 3);
                flag4 = 1;
            end
        end
        %tau2 = gcc(1, 3);
        if ~flag1 && ~flag2 && ~flag3 && ~flag4
            tau2 = (gcc(1, 4) + gcc(4, 3) + gcc(1, 2) + gcc(2, 3) + 2 * gcc(1, 3)) / 4;
        elseif ~flag1 && ~flag3
            tau2 = gcc(1, 4) + gcc(4, 3);
        elseif ~flag2 && ~flag4
            tau2 = gcc(1, 2) + gcc(2, 3);
        else
            tau2 = gcc(1, 3);
        end
        %tau2 = (gcc(1, 4) + gcc(4, 3) + 2 * gcc(1, 3)) / 3;
        %tau1 = gcc(2, 4);
        if ~flag1 && ~flag2 && ~flag3 && ~flag4
            tau1 = (gcc(3, 4) - gcc(3, 2) + gcc(1, 4) - gcc(1, 2) + 2 * gcc(2, 4)) / 4;
        elseif ~flag2 && ~flag3
            tau1 = gcc(3, 4) - gcc(3, 2);
        elseif ~flag1 && ~flag4
            tau1 = gcc(1, 4) - gcc(1, 2);
        else
            tau1 = gcc(2, 4);
        end
        %tau1 = (gcc(1, 4) - gcc(1, 2) + 2 * gcc(2, 4)) / 3;
        
        %tau1 = (gcc(1, 4) - gcc(1, 2) + gcc(3, 4) - gcc(3, 2)) / 2; % gcc(2, 4);
        %tau2 = (gcc(1, 2) + gcc(2, 3) + gcc(1, 4) + gcc(4, 3)) / 2; % gcc(1, 3);
%         tau1 = gcc(2, 4);
%         tau2 = gcc(1, 3);
        x = -tau1 * c / sqrt(2) / d;
        y = -tau2 * c / sqrt(2) / d;
        out3(ii) = mod(rad2deg(cart2pol(y, x)), 360);
%     %end
    
    
%         tau1 = (gcc(1, 2) + gcc(4, 3)) / 2; % horizontal
%         tau2 = (gcc(1, 4) + gcc(2, 3)) / 2; % vertical
%         theta1 = rad2deg(acos(-tau1 * c / d));
%         theta2 = rad2deg(acos(-tau2 * c / d));
% 
%         if false && (~isreal(theta1) || ~isreal(theta2))
%             tau1 = gcc(2, 4);
%             tau2 = gcc(1, 3);
%             x = -tau1 * c / sqrt(2) / d;
%             y = -tau2 * c / sqrt(2) / d;
%             theta = rad2deg(cart2pol(y, x));
%         else
% 
%         % flag = 1 when theta1 and theta2 are same direction
%         if theta2 < 90
%             flag = abs(theta1 - theta2 - 90) < abs(theta2 + theta1 - 90);
%         else
%             flag = abs(theta2 - theta1 - 90) < abs(theta2 + theta1 - 270);
%         end
% 
%         if abs(theta2 - 90) > 40
%             theta = (2 * (theta2 < 90) - 1) * theta1 - 45;
%         elseif abs(theta1 - 90) > 40
%             theta = 45 - (2 * xor(flag, theta2 < 90) - 1) * theta2;
%         else
%             theta = (((2 * flag - 1) * theta2 + theta1) / 2) * (2 * (theta2 < 90) - 1);
%         end
%     end
%     
    theta = mod(theta, 360);
% 
    out(ii) = theta;

end
