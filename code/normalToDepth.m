function z = normalToDepth(zx, zy)
%NORMALTODEPTH 此处显示有关此函数的摘要
%   此处显示详细说明

[wx, wy] = meshgrid(linspace(-pi/2, pi/2, size(zx, 1)), linspace(pi/2, -pi/2, size(zx, 2)));

wx = ifftshift(wx);
wy = ifftshift(wy);

fx = fft2(zx);
fy = fft2(zy);

fd = (1j * wx .* fx - 1j * wy .* fy) ./ (wx.^2 + wy.^2 + eps);

z = real(ifft2(fd))/2;
end

% function z = normalToDepth(zx, zy)
% 
% % using the Discrete Cosine Transform to represent Zx and the Discrete Sine Transform DST to represent Zy
% Px = dct(dst(zx')');
% Py = dst(dct(zy')');
% 
% [wx, wy] = meshgrid(linspace(pi / size(zx, 1), pi, size(zx, 1)), linspace(pi / size(zx, 2), pi, size(zx, 2)));
% C = (-wx .* Px - wy .* Py) ./ (wx.^2 + wy.^2 + eps);
% z = idct2(C);
% 
% % % here the integration starts
% % 
% % for y = 1:Y 
% %     AyWy = pi * y / Y; % the differentiation operator is approximated as pi * (u,v)/(M,N)
% %     for x = 1:X
% %         AxWx = pi * x / X;
% %         
% %         c(y,x) =(-AxWx*tQx(y,x) - AyWy*tQy(y,x))/(((1+Delta)*(AxWx^2+AyWy^2))+(Miu*(AxWx^2+AyWy^2)^2)); 
% %         
% %     end
% % end
% % 
% % Z = idct2(c);  % Height map delivered by the Inverse Discrete Cosine Transform
% % 
% % Z = Z + max(max(abs(Z))); % normalizing

% end

