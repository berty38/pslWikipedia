function p = svm2prob(s)

s = s - sum(s)/length(s);
e = exp(s);
p = e / sum(e);
