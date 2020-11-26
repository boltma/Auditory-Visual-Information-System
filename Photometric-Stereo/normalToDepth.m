function z = normalToDepth(zx, zy)
%NORMALTODEPTH Convert normal vectors into depth by Frankot-Chellappa
%algorithm
%   z = normalToDepth(zx, zy)

[wx, wy] = meshgrid(linspace(-pi/2, pi/2, size(zx, 1)), linspace(pi/2, -pi/2, size(zx, 2)));

wx = ifftshift(wx);
wy = ifftshift(wy);

fx = fft2(zx);
fy = fft2(zy);

fd = (1j * wx .* fx - 1j * wy .* fy) ./ (wx.^2 + wy.^2 + eps);

z = real(ifft2(fd))/2;
end

