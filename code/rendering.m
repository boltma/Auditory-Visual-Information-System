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
m_ori = m; % save original m before removing saturated or shadows
m(m < th_l | m > th_h) = NaN;

as = 0;
nu = 0;
g = reshape(eye(3), 1, 9);

Im = zeros(size(m));

for ii = 1:10
    b = s'\(m - Im)';

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

    z = normalToDepth(zx, zy);

    [zx_new, zy_new] = partial(z);
    b_new = [-reshape(zx_new, 1, []); -reshape(zy_new, 1, []); -ones(1, 168*168)];
    b_new = -b_new ./ vecnorm(b_new) .* k;
    
    TSfunc = @(x) norm((TSmodel(b_new, s, x(1), x(2), reshape(x(3:end), 3, 3)) - m_ori) .* ~isnan(m));
    x0 = [0, 0, reshape(eye(3), 1, 9)];
    options = optimset('Display', 'iter', 'Algorithm', 'quasi-newton');
    x = fminunc(TSfunc, x0, options);

    as = x(1);
    nu = x(2);
    g = reshape(x(3:end), 3, 3);
    
    for jj = 1:7
        [~, Im(:, jj)] = TSmodel(b_new, s(:, jj), as, nu, g);
    end
end

for ii = 1:10
    %imgs(:,:,ii) = uint8(reshape(b_new' * t(:, ii), 168, 168));
    [I, ~] = TSmodel(b_new, t(:, ii), as, nu, g);
    imgs(:,:,ii) = uint8(reshape(I, 168, 168));
end

end

