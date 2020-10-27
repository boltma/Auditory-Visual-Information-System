function s = angleConversion(ori)
%ANGLECONVERSION 此处显示有关此函数的摘要
%   此处显示详细说明
s = zeros(3, size(ori, 1));
s(1, :) = cos(ori(:, 3) / 180 * pi).*sin(ori(:, 2) / 180 * pi);
s(2, :) = sin(ori(:, 3) / 180 * pi);
s(3, :) = cos(ori(:, 3) / 180 * pi).*cos(ori(:, 2) / 180 * pi);
end

