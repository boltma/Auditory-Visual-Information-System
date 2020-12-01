function [source, minim]=SRP_PHAT_SRC(mics, fs, s, n, lsb, usb)
    c = 343;
    [nr_mic,~]=size(mics);

    corelation=zeros( (nr_mic*(nr_mic-1)/2), 2*length(s)-1);
    idx=1;
    for i=1:nr_mic-1
        for j=i+1:nr_mic
            [~, cor, ~]=gccphat(s(:,i),s(:,j));
            corelation(idx, :)=abs(cor);
            idx=idx+1;
        end
    end
    points=lsb + (usb - lsb).*rand(3,n);
    points(3,:) = 0;

    distance=zeros((nr_mic*(nr_mic-1)/2), n);
    idx=1;
    for i=1:nr_mic
        microphone=mics(i, :)';
        microphone_rep=repmat(microphone, 1, 1);
        distance(idx, :)=sqrt( (points(1, :)-microphone_rep(1,:)).^2  +(points(2, :)-microphone_rep(2,:)).^2 + (points(3, :)-microphone_rep(3,:)).^2 );
        idx=idx+1;
    end
    
    distancediff=zeros((nr_mic*(nr_mic-1)/2), n);
    idx=1;
    for i=1:nr_mic-1
        for j=i+1:nr_mic
            distancediff(idx, :)=distance(i,:)-distance(j,:);
            idx=idx+1;
        end
    end

    samplediff=distancediff*fs/c;
    samplediffIdx=fix(samplediff)+length(s);

    [m, indexMax]=max(corelation');
    
    indexMax=indexMax';
    [~, no]=size(samplediffIdx);
    indexMaxim2=repmat(indexMax, 1,no );

    c1=indexMaxim2-samplediffIdx;
    c2=abs(c1);
    c3=sum(c2);
    [~, index]=min(c3); 
    source = points(:, index);
    minim = max(m);
end