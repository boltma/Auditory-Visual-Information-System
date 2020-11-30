function [source, minim]=SRP_PHAT_SRC(mics, fs, s, n, lsb, usb)
       % mics - microphones location         
       % fs - sampling rate
       % s - signal
       % n - number of points
       % 

 
       
        [nr_mic,b]=size(mics);

        corelation=zeros( (nr_mic*(nr_mic-1)/2), length(s)+length(s)-1);
        idx=1;
        for(i=1:nr_mic-1)
            for(j=i+1:nr_mic)
               
                [c cor cor2]=gccphat(s(:,i),s(:,j));
                corelation(idx, :)=abs(cor);
                idx=idx+1;
            end
        end

% ll = linspace(lsb, usb, n);
% [x, y] = meshgrid(ll, ll);
% points=[reshape(x, 1, []); reshape(y, 1, []); zeros(1, n*n)];

% n = n * n;
% points(3, :) = 0;

points=lsb + (usb - lsb).*rand(3,n);
points(3,:) = 0;

distances=zeros((nr_mic*(nr_mic-1)/2), n);

idx=1;
for(i=1:nr_mic)
    microphone=mics(i, :)';
    microphone_rep=repmat(microphone, 1, 1);
    distances(idx, :)=sqrt( (points(1, :)-microphone_rep(1,:)).^2  +(points(2, :)-microphone_rep(2,:)).^2 + (points(3, :)-microphone_rep(3,:)).^2 );
    idx=idx+1;
    
end


idx=1;
distanceDifference=zeros((nr_mic*(nr_mic-1)/2), n);

for(i=1:nr_mic-1)
    for(j=i+1:nr_mic)
        
        distanceDifference(idx, :)=distances(i,:)-distances(j,:);
        idx=idx+1;
    end
end

% theta = linspace(0, 360, n);
% 
% idx = 1;
% 
% dis = [theta + 45; theta; 45 - theta; 45 - theta; 90 - theta; theta + 225];
% dis = -cos(dis * pi / 180);
% 
% distanceDifference=zeros(6, n);
% for i = 1:3
%     for j = i+1:4
%         if idx == 2 || idx == 5
%             distanceDifference(idx, :)=dis(idx, :) * 0.2;
%         else
%             distanceDifference(idx, :)=dis(idx, :) * 0.2 / sqrt(2);
%         end
%         idx = idx + 1;
%     end
% end

sampleDifference=distanceDifference*fs/343;

sampleDifferenceIdx=fix(sampleDifference)+length(s);



[m indexMax]=max(corelation');

indexMax=indexMax';
[no1, no2]=size(sampleDifferenceIdx);
indexMaxim2=repmat(indexMax, 1,no2 );




c=indexMaxim2-sampleDifferenceIdx;
% cc=c-length(s)
cc=abs(c);
ccc=sum(cc);
[minim, index]=min(ccc); 
%source=theta(index);
source = points(:, index);
minim = max(m);
end