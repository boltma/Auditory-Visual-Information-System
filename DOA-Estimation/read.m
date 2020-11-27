%read all audio files



function s = read(num)
    [y1,Fs] = audioread(['train/' num2str(num) '_mic4.wav']);
    [y2,Fs] = audioread(['train/' num2str(num) '_mic3.wav']);
    [y3,Fs] = audioread(['train/' num2str(num) '_mic1.wav']);
    [y4,Fs] = audioread(['train/' num2str(num) '_mic2.wav']);
    s = [y1 y2 y3 y4];
end