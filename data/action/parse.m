
%% 5-action Dataset

clear

maxFrames = 1000;
maxBoxes = 100;

load file1.mat

actions = 1:5;

fid_pfx = 'd1_';
fid_inframe = fopen([fid_pfx 'inframe.txt'], 'w');
fid_sameframe = fopen([fid_pfx 'insameframe.txt'], 'w');
fid_coords = fopen([fid_pfx 'coords.txt'], 'w');
fid_action = fopen([fid_pfx 'action.txt'], 'w');
fid_hogaction = fopen([fid_pfx 'hogaction.txt'], 'w');
fid_acdaction = fopen([fid_pfx 'acdaction.txt'], 'w');
fid_sameobj = fopen([fid_pfx 'sameobj.txt'], 'w');
fid_seqframes = fopen([fid_pfx 'inseqframes.txt'], 'w');
% fid_nhogscores = fopen([fid_pfx 'nhogscores.txt'], 'w');
fid_acdframelabel = fopen([fid_pfx 'acdframelabel.txt'], 'w');
fid_hogframelabel = fopen([fid_pfx 'hogframelabel.txt'], 'w');

for s=1:length(anno)
	if length(anno{s}) > maxFrames
		error('Too many frames in seq %d: %d frames', s, length(anno{s}))
	end
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		hoglabscores = zeros(length(actions),1);
		acdlabscores = zeros(length(actions),1);
		if length(anno{s}{f}) > maxBoxes
			error('Too many boxes in seq %d, frame %d: %d boxes', s, f, length(anno{s}{f}))
		end
		for b=1:length(anno{s}{f})
			% bounding box ID
			bbid = frid*maxBoxes + b;
			% filter certain labels
			if any(actions == anno{s}{f}(b).act)
				inframe = [inframe bbid];
				% action ground-truth
				fprintf(fid_action, '%d\t%d\n', bbid, anno{s}{f}(b).act);
				% identity maintenance ground-truth
				if f > 1
					for b_=1:length(anno{s}{f-1})
						bbid_ = (frid-1)*maxBoxes + b_;
						fprintf(fid_seqframes, '%d\t%d\n', bbid_, bbid);
						if anno{s}{f-1}(b_).id == anno{s}{f}(b).id
							fprintf(fid_sameobj, '%d\t%d\n', bbid_, bbid);
						end
					end
				end
				% in-frame predicate
				fprintf(fid_inframe, '%d\t%d\t%d\n', bbid,s,frid);
				% coordinates predicate
				fprintf(fid_coords, '%d\t%d\t%d\t%d\t%d\n', bbid, ...
					anno{s}{f}(b).x, anno{s}{f}(b).y, ...
					anno{s}{f}(b).w, anno{s}{f}(b).h);
				% convert SVM scores to probabilities
				hogprob = svm2prob(feat{s}{f}(b).hogscore);
				acdprob = svm2prob(feat{s}{f}(b).acscore);
				for a=1:length(feat{s}{f}(b).hogscore)
					fprintf(fid_hogaction, '%d\t%d\t%f\n', bbid, actions(a), hogprob(a));
					fprintf(fid_acdaction, '%d\t%d\t%f\n', bbid, actions(a), acdprob(a));
				end
				% count predicted frame labels
				hoglabscores = hoglabscores + feat{s}{f}(b).hogscore';
				acdlabscores = acdlabscores + feat{s}{f}(b).acscore';
				% write nhogscores
% 				for l=1:length(feat{s}{f}(b).nhogscore)
% 					fprintf(fid_nhogscores, '%d\t%d\t%f\n', bbid, l, feat{s}{f}(b).nhogscore(l));
% 				end
			end
		end
		% after determining who's in the current frame, write same-frame
		for i=1:length(inframe)
			bbid1 = inframe(i);
			for j=1:length(inframe)
				bbid2 = inframe(j);
				% in-same-frame predicate
				fprintf(fid_sameframe, '%d\t%d\n', bbid1, bbid2);
			end
		end
		% convert frame label scores to probs
		hoglabprobs = svm2prob(hoglabscores);
		for a=1:length(hoglabscores)
			fprintf(fid_hogframelabel, '%d\t%d\t%f\n', frid, actions(a), hoglabprobs(a));
		end
		% convert frame label scores to probs
		acdlabprobs = svm2prob(acdlabscores);
		for a=1:length(acdlabscores)
			fprintf(fid_acdframelabel, '%d\t%d\t%f\n', frid, actions(a), acdlabprobs(a));
		end
	end
end

fclose('all');

%% 6-action Dataset

clear

maxFrames = 1000;
maxBoxes = 100;

load file2.mat

actions = [1 2 3 5 6 7];

fid_pfx = 'd2_';
fid_inframe = fopen([fid_pfx 'inframe.txt'], 'w');
fid_sameframe = fopen([fid_pfx 'insameframe.txt'], 'w');
fid_coords = fopen([fid_pfx 'coords.txt'], 'w');
fid_action = fopen([fid_pfx 'action.txt'], 'w');
fid_hogaction = fopen([fid_pfx 'hogaction.txt'], 'w');
fid_acdaction = fopen([fid_pfx 'acdaction.txt'], 'w');
fid_sameobj = fopen([fid_pfx 'sameobj.txt'], 'w');
fid_seqframes = fopen([fid_pfx 'inseqframes.txt'], 'w');
% fid_nhogscores = fopen([fid_pfx 'nhogscores.txt'], 'w');
fid_acdframelabel = fopen([fid_pfx 'acdframelabel.txt'], 'w');
fid_hogframelabel = fopen([fid_pfx 'hogframelabel.txt'], 'w');

for s=1:length(anno)
	if length(anno{s}) > maxFrames
		error('Too many frames in seq %d: %d frames', s, length(anno{s}))
	end
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		hoglabscores = zeros(length(actions),1);
		acdlabscores = zeros(length(actions),1);
		if length(anno{s}{f}) > maxBoxes
			error('Too many boxes in seq %d, frame %d: %d boxes', s, f, length(anno{s}{f}))
		end
		for b=1:length(anno{s}{f})
			% bounding box ID
			bbid = frid*maxBoxes + b;
			% filter certain labels
			if any(actions == anno{s}{f}(b).act)
				inframe = [inframe bbid];
				% action ground-truth
				fprintf(fid_action, '%d\t%d\n', bbid, anno{s}{f}(b).act);
				% identity maintenance ground-truth
				if f > 1
					for b_=1:length(anno{s}{f-1})
						bbid_ = (frid-1)*maxBoxes + b_;
						fprintf(fid_seqframes, '%d\t%d\n', bbid_, bbid);
						if anno{s}{f-1}(b_).id == anno{s}{f}(b).id
							fprintf(fid_sameobj, '%d\t%d\n', bbid_, bbid);
						end
					end
				end
				% in-frame predicate
				fprintf(fid_inframe, '%d\t%d\t%d\n', bbid,s,frid);
				% coordinates predicate
				fprintf(fid_coords, '%d\t%d\t%d\t%d\t%d\n', bbid, ...
					anno{s}{f}(b).x, anno{s}{f}(b).y, ...
					anno{s}{f}(b).w, anno{s}{f}(b).h);
				% convert SVM scores to probabilities
				hogprob = svm2prob(feat{s}{f}(b).hogscore);
				acdprob = svm2prob(feat{s}{f}(b).acscore);
				for a=1:length(feat{s}{f}(b).hogscore)
					fprintf(fid_hogaction, '%d\t%d\t%f\n', bbid, actions(a), hogprob(a));
					fprintf(fid_acdaction, '%d\t%d\t%f\n', bbid, actions(a), acdprob(a));
				end
				% count predicted frame labels
				hoglabscores = hoglabscores + feat{s}{f}(b).hogscore';
				acdlabscores = acdlabscores + feat{s}{f}(b).acscore';
				% write nhogscores
% 				for l=1:length(feat{s}{f}(b).nhogscore)
% 					fprintf(fid_nhogscores, '%d\t%d\t%f\n', bbid, l, feat{s}{f}(b).nhogscore(l));
% 				end
			end
		end
		% after determining who's in the current frame, write same-frame
		for i=1:length(inframe)
			bbid1 = inframe(i);
			for j=1:length(inframe)
				bbid2 = inframe(j);
				% in-same-frame predicate
				fprintf(fid_sameframe, '%d\t%d\n', bbid1, bbid2);
			end
		end
		% convert frame label scores to probs
		hoglabprobs = svm2prob(hoglabscores);
		for a=1:length(hoglabscores)
			fprintf(fid_hogframelabel, '%d\t%d\t%f\n', frid, actions(a), hoglabprobs(a));
		end
		% convert frame label scores to probs
		acdlabprobs = svm2prob(acdlabscores);
		for a=1:length(acdlabscores)
			fprintf(fid_acdframelabel, '%d\t%d\t%f\n', frid, actions(a), acdlabprobs(a));
		end
	end
end

fclose('all');

