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

k = zeros(1, 168*168);
z = zeros(3, 168*168);
for ind = 1:168*168
    k(ind) = norm(b(:, ind));
    z(:, ind) = b(:, ind)/k(ind);
end

for ii = 1:10
    % imgs(:,:,ii) = uint8(reshape(b' * t(:, ii), 168, 168));
    imgs(:,:,ii) = uint8(reshape(TSmodel(b, t(:, ii), 0, 0, eye(3)), 168, 168));
end

end

