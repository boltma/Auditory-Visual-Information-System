%input is the numbering of data, output is an n*4 matrix
function s = read(num)
    num = num+'0';
    pth = ['train/' num];
    [y1,Fs] = audioread([pth '_mic1.wav']);
    [y2,Fs] = audioread([pth  '_mic2.wav']);
    [y3,Fs] = audioread([pth  '_mic3.wav']);
    [y4,Fs] = audioread([pth  '_mic4.wav']);

    s = [y1 y2 y3 y4];
end