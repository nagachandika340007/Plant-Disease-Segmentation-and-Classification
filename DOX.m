function [bestfit,Convergence_curve,Leader_pos,time]=DOX(Positions,objfun,Lb,Ub,Max_iter)
ub = Ub(1,:);
lb = Lb(1,:);
[N,D] = size(Positions);
p=0.8;                       % probabibility switch
power_exponent=0.1;             % Power exponent
sensory_modality=0.01;          % sensor modality

Leader_pos=zeros(1,D);
Leader_score=inf; %change this to -inf for maximization problems



t=0;% Loop counter
tic;
% Main loop
while t<Max_iter
    t
    
    for i=1:size(Positions,1)
        % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;  
  
        Fitness(i) = feval(objfun,Positions(i,:));
        Frg_bf(i) = (sensory_modality*Fitness(i)^power_exponent);
        
        % Update the leader
        if Fitness(i)<Leader_score % Change this to > for maximization problem
            Leader_score=Fitness(i); % Update alpha
            Leader_pos=Positions(i,:);
        end
    end
    
    % need to find the best solution
     
    for i=1:size(Positions,1)
        r1=rand();                          % r1 is a random number in [0,1]
        if r1<p
            dis = r1^2.*(Leader_pos-Positions(i,:));        %Eq. (2) in paper
            Positions(i,:)=Positions(i,:)+dis*Frg_bf(i);
        else
            dis = r1^2.*(Leader_pos-Positions(i,:));
            J=randi(N);
            K=randi(N);
            Positions(i,:)=Positions(i,:)+dis*Frg_bf(i); 
            dis = r1^2.*(Positions(J,:)-Positions(K,:));               %Eq. (3) in paper
        end
    end
    
   
    t=t+1;
    Convergence_curve(t)=Leader_score;
   
    % update the value of power exponent
    power_exponent = rand;
end
time = toc;
bestfit = Convergence_curve(end);
end