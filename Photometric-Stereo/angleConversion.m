function s = angleConversion(ori)
%ANGLECONVERSION angle conversion for train vectors
%   s = angleConversion(ori)
s = zeros(3, size(ori, 1));
s(1, :) = cos(ori(:, 3) / 180 * pi).*sin(ori(:, 2) / 180 * pi);
s(2, :) = sin(ori(:, 3) / 180 * pi);
s(3, :) = cos(ori(:, 3) / 180 * pi).*cos(ori(:, 2) / 180 * pi);
end

