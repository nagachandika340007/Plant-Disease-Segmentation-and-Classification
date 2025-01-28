function [Bestfit, Convergence, Bestsol, Time] = EOO(X, objfun, LB, UB, maxIter)
tic;
lb = LB(1, :);
ub = UB(1, :);
[N, dim] = size(X);
Fitness = feval(objfun, X);
Bestsol = zeros(1, dim);
Bestfit = inf;

for t = 1:maxIter
    L = randi([3, 5], 1);
    T = (((L - 3) / (5 - 3)) * 10) - 5;
    C = (((L - 3) / (5 - 3)) * 2) + 0.6;
    E = ((t - 1) / (maxIter - 1)) - 0.5;
    for i=1:N
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    end
    
    for i = 1:N
        Y = T + E + L * rand * (Bestsol - X(i, :));
        X(i, :) = Y * C;
    end
    
    for i=1:N
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
    end
    
    Fitness = feval(objfun, X);
    [minf, iminf] = min(Fitness);
    if minf < Bestfit
        Bestfit =  minf;
        Bestsol = X(iminf,:);
    end
    
    Convergence(t) = Bestfit;
end
Time = toc;
end