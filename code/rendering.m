function [z,imgs] = rendering(pth)
% rendering function
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

th_l = 5;
th_h = 255;
m(m < th_l | m > th_h) = NaN;

b = s'\m';

% len = numel(b);
% TSfunc = @(x) norm(TSmodel(reshape(x(1:len), 3, []), s, x(len+1), x(len+2), reshape(x(end-8:end), 3, 3)) - m);
% x0 = [reshape(b, 1, []), 0, 0, reshape(eye(3), 1, 9)];
% options = optimset('Display','iter','MaxIter', 1,'Algorithm','quasi-newton');
% x = fminunc(TSfunc, x0, options);

k = zeros(168, 168);
z = zeros(3, 168*168);
for ind = 1:168*168
    k(ind) = norm(b(:, ind));
    z(:, ind) = b(:, ind) / k(ind);
    z(:, ind) = -z(:, ind) / z(3, ind);
end

zx = reshape(z(1,:), 168, 168);
zy = reshape(z(2,:), 168, 168);
zx = Inpaint(zx);
zy = Inpaint(zy);
k = reshape(Inpaint(k), 1, []);

depth = normalToDepth(zx, zy);
z = depth;

[zx_new, zy_new] = partial(depth);
b_new = [reshape(zx_new, 1, []); reshape(zy_new, 1, []); -ones(1, 168*168)];
b_new = -b_new ./ vecnorm(b_new) .* k;

% b = reshape(x(1:len), 3, []);
% as = x(len+1);
% nu = x(len+2);
% g = reshape(x(end-8:end), 3, 3);
for ii = 1:10
    imgs(:,:,ii) = uint8(reshape(b_new' * t(:, ii), 168, 168));
    % imgs(:,:,ii) = uint8(reshape(TSmodel(b, t(:, ii), as, nu, g), 168, 168));
end

end

