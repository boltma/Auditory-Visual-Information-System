%depth = [1 3 5;7 11 13;17 19 23];
%[dx, dy] = partial1(depth)

function [dx, dy] = partial(depth)
    [m, n] = size(depth);
    diffx = diff(depth, 1, 2);
    diffy = diff(depth, 1);
    dx = zeros(size(depth));
    dy = zeros(size(depth));
    dx(:, 1) = diffx(:, 1);
    dy(1, :) = diffy(1, :);
    for i = 2:n-1 %168*168
        dx(:, i) = (diffx(:,i-1)+diffx(:,i)) / 2;
        dy(i, :) = (diffy(i-1,:)+diffy(i,:)) / 2;
    end
    dx(:,n) = diffx(:,n-1);
    dy(m,:) = diffy(m-1,:);
end