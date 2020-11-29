%read all audio files

function [s, Fs] = read(name, num)
    [y1, ~] = audioread([name num2str(num) '_mic1.wav']);
    [y2, ~] = audioread([name num2str(num) '_mic2.wav']);
    [y3, ~] = audioread([name num2str(num) '_mic3.wav']);
    [y4, Fs] = audioread([name num2str(num) '_mic4.wav']);
    
%     n = 2;
%     beta = 20;
    
    %y11 = resample(y1, (1:length(y1))/Fs, 2 * Fs, 'spline');
%     y11 = resample(y1, 5 * Fs, Fs, n, beta);
%     y21 = resample(y2, 5 * Fs, Fs, n, beta);
%     y31 = resample(y3, 5 * Fs, Fs, n, beta);
%     y41 = resample(y4, 5 * Fs, Fs, n, beta);
%     y11 = resample(y1, (1:length(y1))/Fs, 2 * Fs, 'linear');
%     y21 = resample(y2, (1:length(y1))/Fs, 2 * Fs, 'linear');
%     y31 = resample(y3, (1:length(y1))/Fs, 2 * Fs, 'linear');
%     y41 = resample(y4, (1:length(y1))/Fs, 2 * Fs, 'linear');
%     y11 = highpass(y1, 100, Fs);
%     y21 = highpass(y2, 100, Fs);
%     y31 = highpass(y3, 100, Fs);
%     y41 = highpass(y4, 100, Fs);
    
    s = [y1 y2 y3 y4];
%     Fs = 5 * Fs;
end