function [source, m]=srpphat(mics, c, fs, s, n, lsb, usb)
    correlation=zeros(6, 2*length(s)-1);
    idx=1;
    for ii=1:3
        for jj=ii+1:4
            [~, cor, ~]=gccphat(s(:,ii), s(:,jj));
            correlation(idx, :)=abs(cor);
            idx=idx+1;
        end
    end
    points = lsb + (usb - lsb) .* rand(2, n);

    distance=zeros(6, n);
    idx=1;
    for ii=1:4
        m=mics(ii, :)';
        m_rep=repmat(m, 1, 1);
        distance(idx, :)=sqrt((points(1, :)-m_rep(1,:)).^2 + (points(2, :)-m_rep(2,:)).^2);
        idx=idx+1;
    end
    
    distancediff=zeros(6, n);
    idx=1;
    for ii=1:3
        for jj=ii+1:4
            distancediff(idx, :)=distance(ii,:)-distance(jj,:);
            idx=idx+1;
        end
    end

    samplediff=distancediff*fs/c;
    samplediffIdx=fix(samplediff)+length(s);

    [m, indexMax]=max(correlation');
    
    indexMax=indexMax';
    [~, no]=size(samplediffIdx);
    indexMaxim2=repmat(indexMax, 1, no);

    c1=indexMaxim2-samplediffIdx;
    c2=abs(c1);
    c3=sum(c2);
    % [~, index]=min(c3);
    [~, index] = sort(c3);
    source = mean(points(:, index(1:2)), 2);
    m = max(m);
end