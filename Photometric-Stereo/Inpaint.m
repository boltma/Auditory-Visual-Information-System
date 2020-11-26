function InpaintA = Inpaint(A)
[n, m] = size(A);
nm = n*m;
A = A(:);
InpaintA = A;
AIsNan = isnan(A(:));

nanList = find(AIsNan); % nodes need interpolate
[nanRow, nanColunm] = ind2sub([n, m], nanList);
nanList = [nanList, nanRow, nanColunm]; % (unrolled ind, row ind, column ind)
Neighbors = NeighborsIdentification(n, m, nanList); % neighbors
X = [nanList; Neighbors];

L1 = find((X(:,2) > 1) & (X(:,2) < n));
i = repmat(X(L1, 1), 1, 3);
j = repmat(X(L1, 1), 1, 3)+repmat([-1 0 1], length(L1), 1);
k = repmat([1 -2 1], length(L1), 1);
ScndPartial1 = sparse(i, j, k, nm, nm);

L2 = find((X(:,3) > 1) & (X(:,3) < m));
i = repmat(X(L2, 1), 1, 3);
j = repmat(X(L2, 1), 1, 3)+repmat([-n 0 n], length(L2), 1);
k = repmat([1 -2 1], length(L2), 1);
ScndPartial2 = sparse(i, j, k, nm, nm);
SecondPartials = ScndPartial1+ScndPartial2;

KnownList = find(~AIsNan); % nodes already known
Elimination = -SecondPartials(:, KnownList)*A(KnownList); % elimination
AIsNan = find(any(SecondPartials(:, nanList(:,1)), 2));
InpaintA(nanList(:, 1)) = SecondPartials(AIsNan, nanList(:, 1))\Elimination(AIsNan);
InpaintA = reshape(InpaintA, n, m);
end