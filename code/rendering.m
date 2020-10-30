function [z,imgs] = rendering(pth)
%RENDERING �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%   z�ĳ߶���x��y��ͬ����С��ͬ�ڲ���ͼ���С��λ�������ͼ�����ص�һһ��Ӧ
%   imgsΪ��Ⱦ�������С��ͬ�ڲ���ͼ���С��λ�������ͼ�����ص�һһ���?
z=zeros(168,168);
imgs=zeros(168,168,10);
imgs=uint8(imgs);
m = zeros(168*168, 7);
for j = 1:7
    TrainPath = [pth, '/train/', num2str(j), '.bmp'];
    TempImg = double(imread(TrainPath, 'bmp'));
    m(:, j) = reshape(TempImg, [], 1);
end
Source = load([pth, '/train.txt']);
Test = load([pth, '/test.txt']);
s = angleConversion(Source);
t = angleConversion(Test);

b = s'\m';

% len = numel(b);
% TSfunc = @(x) norm(TSmodel(reshape(x(1:len), 3, []), s, x(len+1), x(len+2), reshape(x(end-8:end), 3, 3)) - m);
% x0 = [reshape(b, 1, []), 0, 0, reshape(eye(3), 1, 9)];
% options = optimset('Display','iter','MaxIter', 1,'Algorithm','quasi-newton');
% x = fminunc(TSfunc, x0, options);

k = zeros(1, 168*168);
z = zeros(3, 168*168);
for ind = 1:168*168
    k(ind) = norm(b(:, ind));
    z(:, ind) = b(:, ind) / k(ind);
    z(:, ind) = -z(:, ind) / z(3, ind);
end

depth = normalToDepth(reshape(z(1,:), 168, 168), reshape(z(2,:), 168, 168));

[bx, by, bz] = surfnorm(depth);
b = [reshape(bx, 1, []); reshape(by, 1, []); reshape(bz, 1, [])] .* k;

% b = reshape(x(1:len), 3, []);
% as = x(len+1);
% nu = x(len+2);
% g = reshape(x(end-8:end), 3, 3);
for ii = 1:10
    imgs(:,:,ii) = uint8(reshape(b' * t(:, ii), 168, 168));
    % imgs(:,:,ii) = uint8(reshape(TSmodel(b, t(:, ii), as, nu, g), 168, 168));
end

end

