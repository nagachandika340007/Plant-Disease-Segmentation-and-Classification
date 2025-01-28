function[EVAL1] = evaluation1(sp,act)
for i = 1:length(sp)
    p = sp{i};
    a = act{i};
    tp =0; tn =0; fp =0; fn =0;
    for j = 1:length(p)
        if a(j) == 1 && p(j) == 1
            tp = tp+1;
        elseif a(j) == 0 && p(j) == 0
            tn = tn+1;
        elseif a(j) == 0 && p(j) == 1
            fp = fp+1;
        elseif a(j) == 1 && p(j) == 0
            fn = fn+1;
        end
    end
    Tp(i) = tp;
    Fp(i) = fp;
    Tn(i) = tn;
    Fn(i) = fn;
end
tp = sum(Tp);
fp = sum(Fp);
tn = sum(Tn);
fn = sum(Fn);
Dice = (2 * tp) / ((2 * tp) + fp + fn);
Jaccard = tp / (tp + fp + fn);
accuracy = (tp+tn)/(tp+tn+fp+fn);
EVAL1 = [Dice Jaccard accuracy];
end