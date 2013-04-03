
%% 5-action Dataset

clear
load file1.mat

actions = 1:5;

fid_pfx = 'd1_';
fid_inframe = fopen([fid_pfx 'inframe.txt'], 'w');
fid_coords = fopen([fid_pfx 'coords.txt'], 'w');
fid_action = fopen([fid_pfx 'action.txt'], 'w');
fid_hogaction = fopen([fid_pfx 'hogaction.txt'], 'w');
fid_acdaction = fopen([fid_pfx 'acdaction.txt'], 'w');
fid_sameobj = fopen([fid_pfx 'sameobj.txt'], 'w');

for s=1:length(anno)
	for f=1:length(anno{s})
		for b=1:length(anno{s}{f})
			% bounding box ID
			id = s*1000 + f*10 + b;
			% filter certain labels
			if any(actions == anno{s}{f}(b).act)
				% action ground-truth
				fprintf(fid_action, '%d\t%d\n', id, anno{s}{f}(b).act);
				% identity maintenance ground-truth
				if f > 1
					for b_=1:length(anno{s}{f-1})
						id_ = s*1000 + (f-1)*10 + b_;
						if anno{s}{f-1}(b_).id == anno{s}{f}(b).id
							fprintf(fid_sameobj, '%d\t%d\t1\n', id_, id);
						end
					end
				end
				% in-frame predicate
				fprintf(fid_inframe, '%d\t%d\t%d\n', id,s,f);
				% coordinates predicate
				fprintf(fid_coords, '%d\t%d\t%d\t%d\t%d\n', id, ...
					anno{s}{f}(b).x, anno{s}{f}(b).y, ...
					anno{s}{f}(b).w, anno{s}{f}(b).h);
				% convert SVM scores to probabilities
				hogprob = svm2prob(feat{s}{f}(b).hogscore);
				acdprob = svm2prob(feat{s}{f}(b).acscore);
				for a=1:length(feat{s}{f}(b).hogscore)
					fprintf(fid_hogaction, '%d\t%d\t%f\n', id, actions(a), hogprob(a));
					fprintf(fid_acdaction, '%d\t%d\t%f\n', id, actions(a), acdprob(a));
				end
			end
		end
	end
end

fclose('all');

%% 6-action Dataset

clear
load file2.mat

actions = [1 2 3 5 6 7];

fid_pfx = 'd2_';
fid_inframe = fopen([fid_pfx 'inframe.txt'], 'w');
fid_coords = fopen([fid_pfx 'coords.txt'], 'w');
fid_action = fopen([fid_pfx 'action.txt'], 'w');
fid_hogaction = fopen([fid_pfx 'hogaction.txt'], 'w');
fid_acdaction = fopen([fid_pfx 'acdaction.txt'], 'w');
fid_sameobj = fopen([fid_pfx 'sameobj.txt'], 'w');

for s=1:length(anno)
	for f=1:length(anno{s})
		for b=1:length(anno{s}{f})
			% bounding box ID
			id = s*1000 + f*10 + b;
			% filter certain labels
			if any(actions == anno{s}{f}(b).act)
				% action ground-truth
				fprintf(fid_action, '%d\t%d\n', id, anno{s}{f}(b).act);
				% identity maintenance ground-truth
				if f > 1
					for b_=1:length(anno{s}{f-1})
						id_ = s*1000 + (f-1)*10 + b_;
						if anno{s}{f-1}(b_).id == anno{s}{f}(b).id
							fprintf(fid_sameobj, '%d\t%d\t1\n', id_, id);
						end
					end
				end
				% in-frame predicate
				fprintf(fid_inframe, '%d\t%d\t%d\n', id,s,f);
				% coordinates predicate
				fprintf(fid_coords, '%d\t%d\t%d\t%d\t%d\n', id, ...
					anno{s}{f}(b).x, anno{s}{f}(b).y, ...
					anno{s}{f}(b).w, anno{s}{f}(b).h);
				% convert SVM scores to probabilities
				hogprob = svm2prob(feat{s}{f}(b).hogscore);
				acdprob = svm2prob(feat{s}{f}(b).acscore);
				for a=1:length(feat{s}{f}(b).hogscore)
					fprintf(fid_hogaction, '%d\t%d\t%f\n', id, actions(a), hogprob(a));
					fprintf(fid_acdaction, '%d\t%d\t%f\n', id, actions(a), acdprob(a));
				end
			end
		end
	end
end

fclose('all');

