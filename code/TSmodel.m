function [I, Im] = TSmodel(b, s, as, nu, G)
%TSMODEL Torrance-Sparrow Model, Im for mirror reflection intensity, I for
%total intensity
%   [I, Im] = TSmodel(b, s, as, nu, G)
    v = [0; 0; 1];
    Gb = G.' \ b;
    Gs = G * s;
    Gsnorm = Gs / norm(Gs);
    Id = b.' * s;
    Im = (as .* norm(Gs) * exp(-nu.^2 .* (acos(Gb.' * (Gsnorm + v) / norm(Gb) / norm(Gsnorm + v))).^2)) ./ (Gb' * v / norm(Gb));
    I = Id + Im;
end

