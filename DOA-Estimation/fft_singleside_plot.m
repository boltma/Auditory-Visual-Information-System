function fft_singleside_plot(x, Fs)
%FFT_SINGLESIDE_PLOT Plots single-side fft
%   fft_singleside_plot(x, Fs)
%   refer to document of fft
    L = length(x);
    f = (0:L/2) * Fs / L;           % frequency sample, half of length(x)
    y = abs(fft(x) / L);            % get amplitude
    y = y(1:L/2+1);                 % get single side
    y(2:end-1) = 2 * y(2:end-1);    % double the side
    plot(f, y);
end

