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



points=lsb + (usb - lsb).*rand(3,n);
points(3, :) = 0;
%%

distances=zeros((nr_mic*(nr_mic-1)/2), n);

idx=1;
for(i=1:nr_mic)
    microphone=mics(i, :)';
    microphone_rep=repmat(microphone, 1, length(points));
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
% 

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
source=points(:,index);

end