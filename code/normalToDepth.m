function z = normalToDepth(zx, zy)
%NORMALTODEPTH 此处显示有关此函数的摘要
%   此处显示详细说明

[wx, wy] = meshgrid(linspace(-pi/2, pi/2, size(zx, 1)), linspace(pi/2, -pi/2, size(zx, 2)));

wx = ifftshift(wx);
wy = ifftshift(wy);

fx = fft2(zx);
fy = fft2(zy);

fd = (1j * wx .* fx + 1j * wy .* fy) ./ (wx.^2 + wy.^2 + eps);

z = real(ifft2(fd))/2;
end

