clear all; close all; clc;

c = 343.0;
d = 0.2 / sqrt(2);

out = zeros(1, 140);
out2 = zeros(1, 140);
out3 = zeros(1, 140);
out4 = zeros(1, 140);
d2 = zeros(1, 140);
d3 = zeros(1, 140);

for ii = 1:140
    disp(ii);

    [s, Fs] = read('test/', ii);
    
    mics = [0, d / sqrt(2);...
        d/sqrt(2), 0;...
        0, -d/sqrt(2);...
        -d/sqrt(2), 0;];
    
     [finalpos, minim] = srpphat(mics, c, Fs, s, 2000000, -5, 5);
     [theta, ~] = cart2pol(finalpos(2), finalpos(1));
     theta = rad2deg(theta);
    % out2(ii) = minim;
    out(ii) = mod(theta, 360);
    gcc = reshape(gccphat(s, Fs), 4, 4);
    tau1 = gcc(2, 4);
    tau2 = gcc(1, 3);
    
    x2 = -tau1 * c / sqrt(2) / d;
    y2 = -tau2 * c / sqrt(2) / d;
    out2(ii) = mod(rad2deg(cart2pol(y2, x2)), 360);
    d2(ii) = x2*x2 + y2*y2;
    flag1 = 0;
    flag2 = 0;
    flag3 = 0;
    flag4 = 0;
    if abs(gcc(2, 3) - gcc(1, 4)) > 1 / Fs || abs(gcc(2, 3)) > d / c + 1 / Fs || abs(gcc(1, 4)) > d / c + 1 / Fs
        if abs(gcc(2, 3)) < abs(gcc(1, 4)) || abs(gcc(1, 4)) > d / c + 1 / Fs
            gcc(1, 4) = gcc(2, 3);
            gcc(4, 1) = gcc(3, 2);
            flag1 = 1;
        end
        if abs(gcc(2, 3)) > abs(gcc(1, 4)) || abs(gcc(2, 3)) > d / c + 1 / Fs
            gcc(2, 3) = gcc(1, 4);
            gcc(3, 2) = gcc(4, 1);
            flag2 = 1;
        end
    end
    if abs(gcc(2, 1) - gcc(3, 4)) > 1 / Fs || abs(gcc(1, 2)) > d / c + 1 / Fs || abs(gcc(3, 4)) > d / c + 1 / Fs
        if abs(gcc(2, 1)) < abs(gcc(3, 4)) || abs(gcc(3, 4)) > d / c + 1 / Fs
            gcc(3, 4) = gcc(2, 1);
            gcc(4, 3) = gcc(1, 2);
            flag3 = 1;
        end
        if abs(gcc(2, 1)) > abs(gcc(3, 4)) || abs(gcc(1, 2)) > d / c + 1 / Fs
            gcc(2, 1) = gcc(3, 4);
            gcc(1, 2) = gcc(4, 3);
            flag4 = 1;
        end
    end
    if ~flag1 && ~flag2 && ~flag3 && ~flag4
        tau2 = (gcc(1, 4) + gcc(4, 3) + gcc(1, 2) + gcc(2, 3) + 2 * gcc(1, 3)) / 4;
    elseif ~flag1 && ~flag3
        tau2 = gcc(1, 4) + gcc(4, 3);
    elseif ~flag2 && ~flag4
        tau2 = gcc(1, 2) + gcc(2, 3);
    else
        tau2 = gcc(1, 3);
    end

    if ~flag1 && ~flag2 && ~flag3 && ~flag4
        tau1 = (gcc(3, 4) - gcc(3, 2) + gcc(1, 4) - gcc(1, 2) + 2 * gcc(2, 4)) / 4;
    elseif ~flag2 && ~flag3
        tau1 = gcc(3, 4) - gcc(3, 2);
    elseif ~flag1 && ~flag4
        tau1 = gcc(1, 4) - gcc(1, 2);
    else
        tau1 = gcc(2, 4);
    end

    x3 = -tau1 * c / sqrt(2) / d;
    y3 = -tau2 * c / sqrt(2) / d;
    d3(ii) = x3*x3 + y3*y3;
    out3(ii) = mod(rad2deg(cart2pol(y3, x3)), 360);
    
    %theta = out3(ii);
    %theta = mod(theta, 360);
    %out(ii) = theta;
    
    
    if abs(d2(ii) - 1) > abs(d3(ii) - 1)
        out4(ii) = out3(ii);
    else
        out4(ii) = out2(ii);
    end
    

end
