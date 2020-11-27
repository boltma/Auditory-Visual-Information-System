%read all audio files
function s = read()
    audios = zeros(70000, 56);
    for i=1:14
        len = size(read_group(i));
        audios(1:len(1), i:i+3) = read_group(i);
    end
end
function s = read_group(num)
    [y1,Fs] = audioread(['train/' num2str(num) '_mic1.wav']);
    [y2,Fs] = audioread(['train/' num2str(num) '_mic2.wav']);
    [y3,Fs] = audioread(['train/' num2str(num) '_mic3.wav']);
    [y4,Fs] = audioread(['train/' num2str(num) '_mic4.wav']);
    s = [y1 y2 y3 y4];
end