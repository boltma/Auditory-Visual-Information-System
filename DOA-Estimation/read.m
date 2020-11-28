%read all audio files

function [s, Fs] = read(name, num)
    [y1, ~] = audioread([name num2str(num) '_mic1.wav']);
    [y2, ~] = audioread([name num2str(num) '_mic2.wav']);
    [y3, ~] = audioread([name num2str(num) '_mic3.wav']);
    [y4, Fs] = audioread([name num2str(num) '_mic4.wav']);
    
    sound(y1);
    
    s = [y1 y2 y3 y4];
end