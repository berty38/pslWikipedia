
clear

% load file1.mat
% actions = 1:5;

load file2.mat
actions = [1,2,3,5,6,7];

maxFrames = 1000;
maxBoxes = 100;

ctp = 0; cfp = 0;
accuracies = zeros(length(anno),1);

for s=1:length(anno)
	tp = 0; fp = 0;
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		for b=1:length(anno{s}{f})
			% bounding box ID
			bbid = frid*maxBoxes + b;
			% filter certain labels
			if any(actions == anno{s}{f}(b).act)
				% convert SVM scores to probabilities
				hogprob = svm2prob(feat{s}{f}(b).hogscore);
				acdprob = svm2prob(feat{s}{f}(b).acscore);
				[~,pred] = max(acdprob);
				if pred == anno{s}{f}(b).act
					tp = tp + 1;
				else
					fp = fp + 1;
				end
			end
		end
	end
	ctp = ctp + tp;
	cfp = cfp + fp;
	accuracies(s) = tp / (tp+fp);
end
cacc = ctp / (ctp+cfp)
aacc = mean(accuracies)

