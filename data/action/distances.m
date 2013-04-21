maxFrames = 1000;
maxBoxes = 100;
dx = cell(5,1);
dy = cell(5,1);
for a=1:5
	dx{a} = [];
	dy{a} = [];
end
actions = 1:5;
for s=1:length(anno)
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		for b=1:length(anno{s}{f})
			% bounding box ID
			bbid = frid*maxBoxes + b;
			% filter certain labels
			a = anno{s}{f}(b).act;
			if any(actions == a)
				% identity maintenance ground-truth
				if f > 1
					for b_=1:length(anno{s}{f-1})
						bbid_ = (frid-1)*maxBoxes + b_;
						if anno{s}{f-1}(b_).id == anno{s}{f}(b).id
							dx{a} = [dx{a} ; anno{s}{f-1}(b_).x - anno{s}{f}(b).x];
							dy{a} = [dy{a} ; anno{s}{f-1}(b_).y - anno{s}{f}(b).y];
						end
					end
				end
			end
		end
	end
end


for a=1:5
	figure(a)
% 	hist(exp(-max(0,abs(dx{a})-6)))
	hist(exp(-max(0,abs(dx{a})+abs(dy{a})-10) / 10))
end
