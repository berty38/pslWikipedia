
clear

maxFrames = 1000;
maxBoxes = 100;

%% 5-action Dataset

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
fid_seqframes = fopen([fid_pfx 'seqframes.txt'], 'w');

for s=1:length(anno)
	if length(anno{s}) > maxFrames
		error('Too many frames in seq %d: %d frames', s, length(anno{s}))
	end
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		if length(anno{s}{f}) > maxBoxes
			error('Too many boxes in seq %d, frame %d: %d boxes', s, f, length(anno{s}{f}))
		end
		% sequential frames predicate (because PSL is stupid)
% 		if f > 1
% 			fprintf(fid_seqframes, '%d\t%d\n', frid-1, frid);
% 		end
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
	end
end

fclose('all');

%% 6-action Dataset

clear feat; clear anno; clear seqlab;
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
fid_seqframes = fopen([fid_pfx 'seqframes.txt'], 'w');

for s=1:length(anno)
	if length(anno{s}) > maxFrames
		error('Too many frames in seq %d: %d frames', s, length(anno{s}))
	end
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		if length(anno{s}{f}) > maxBoxes
			error('Too many boxes in seq %d, frame %d: %d boxes', s, f, length(anno{s}{f}))
		end
		% sequential frames predicate (because PSL is stupid)
% 		if f > 1
% 			fprintf(fid_seqframes, '%d\t%d\n', frid-1, frid);
% 		end
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
	end
end

fclose('all');

