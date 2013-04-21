dx = [];
dy = [];
actions = 1:5;
for s=1:length(anno)
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		for b=1:length(anno{s}{f})
			% bounding box ID
			bbid = frid*maxBoxes + b;
			% filter certain labels
			if any(actions == anno{s}{f}(b).act)
				% identity maintenance ground-truth
				if f > 1
					for b_=1:length(anno{s}{f-1})
						bbid_ = (frid-1)*maxBoxes + b_;
						if anno{s}{f-1}(b_).id == anno{s}{f}(b).id ...
						&& (anno{s}{f-1}(b_).act==2 || anno{s}{f-1}(b_).act==3)
							dx = [dx ; anno{s}{f-1}(b_).x - anno{s}{f}(b).x];
							dy = [dy ; anno{s}{f-1}(b_).y - anno{s}{f}(b).y];
						end
					end
				end
			end
		end
	end
end
