function I = TSmodel(b, s, as, nu, G)
%TSMODEL �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    v = [0; 0; 1];
    Gb = G.' \ b;
    Gs = G * s;
    Gsnorm = Gs / norm(Gs);
    I = b.' * s + (as * norm(Gs) * exp(-nu^2 * (acos(Gb.' * (Gsnorm + v) / norm(Gb) / norm(Gsnorm + v))).^2)) ./ (Gb' * v / norm(Gb));
end

