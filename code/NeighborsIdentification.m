function Neighbors = NeighborsIdentification(n, m, nanList)
Offset = [1 0; -1 0; 0 1; 0 -1];
nanCnt = size(nanList, 1);

NeighborList = zeros(nanCnt*4, 2);
for i = 1:4
    NeighborList((i-1)*nanCnt+1:i*nanCnt, :) = nanList(:, 2:3)+repmat(Offset(i, :), nanCnt, 1);
end
OutOfBound = (NeighborList(:, 1)<1) | (NeighborList(:, 1)>n) | (NeighborList(:, 2)<1) | (NeighborList(:, 2)>m);
NeighborList(OutOfBound, :) = []; % delete nodes out of boundary

Neighbors = [sub2ind([n, m], NeighborList(:, 1), NeighborList(:, 2)), NeighborList]; % 3 column array
Neighbors = unique(Neighbors, 'rows'); % delete replicates
Neighbors = setdiff(Neighbors, nanList, 'rows'); % delete same nodes in nanList
end
