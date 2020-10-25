function [z,imgs] = rendering(pth)
%RENDERING �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%   z�ĳ߶���x��y��ͬ����С��ͬ�ڲ���ͼ���С��λ�������ͼ�����ص�һһ��Ӧ
%   imgsΪ��Ⱦ�������С��ͬ�ڲ���ͼ���С��λ�������ͼ�����ص�һһ��Ӧ
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
s = zeros(3, 7);
s(1, :) = cos(Source(:, 3)).*sin(Source(:, 2));
s(2, :) = sin(Source(:, 3));
s(3, :) = cos(Source(:, 3)).*cos(Source(:, 2));
% b = s'\m';

% k = zeros(1, 168*168);
% z = zeros(3, 168*168);
% for ind = 1:168*168
%     k(ind) = norm(b(:, ind));
%     z(:, ind) = b(:, ind)/k(ind);
% end
end

