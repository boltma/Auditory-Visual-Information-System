c = 343.0;

[y1, Fs] = audioread('train/10_mic4.wav');
[y2, Fs] = audioread('train/10_mic3.wav');
[y3, Fs] = audioread('train/10_mic1.wav');
[y4, Fs] = audioread('train/10_mic2.wav');

N = 2;
d = L / sqrt(2);
mic = phased.OmnidirectionalMicrophoneElement;
array = phased.URA([N, N],[d, d],'Element',mic);

estimator = phased.GCCEstimator('SensorArray',array,...
    'PropagationSpeed',c,'SampleRate',Fs);
ang = estimator([y1, y2, y3, y4]);

uv = azel2uv(ang);

[theta, ~] = cart2pol(uv(2), uv(1));
theta = mod(rad2deg(theta) - 45, 360);
