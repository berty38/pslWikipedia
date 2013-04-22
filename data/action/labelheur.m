actions = 1:5;
maxFrames = 1000;
maxBoxes = 100;

wtp = 0; wfp = 0; wfn = 0;
qtp = 0; qfp = 0; qfn = 0;
ttp = 0; tfp = 0; tfn = 0;
for s=1:length(anno)
	for f=1:length(anno{s})
		frid = s*maxFrames + f;
		inframe = [];
		queueing = [];
		waiting = [];
		talking = [];
		for b=1:length(anno{s}{f})
			% bounding box ID
			bbid = frid*maxBoxes + b;
			% filter certain labels
			a = anno{s}{f}(b).act;
			if any(actions == a)
				inframe = [inframe ; bbid];
				% waiting
				if a == 2
					waiting = [waiting ; bbid];
				end
				% queueing
				if a == 3
					queueing = [queueing ; bbid];
				end
				% talking
				if a == 5
					talking = [talking ; bbid];
				end
			end
		end
		if length(waiting) >= (0.55*length(inframe))
			wtp = wtp + length(waiting);
			wfp = wfp + (length(inframe) - length(waiting));
		else
			wfn = wfn + length(waiting);
		end
		if length(queueing) >= (0.65*length(inframe))
			qtp = qtp + length(queueing);
			qfp = qfp + (length(inframe) - length(queueing));
		else
			qfn = qfn + length(queueing);
		end
		if length(talking) >= (0.65*length(inframe))
			ttp = ttp + length(talking);
			tfp = tfp + (length(inframe) - length(talking));
		else
			tfn = tfn + length(talking);
		end
	end
end
wpre = wtp / (wtp+wfp)
wrec = wtp / (wtp+wfn)
qpre = qtp / (qtp+qfp)
qrec = qtp / (qtp+qfn)
tpre = ttp / (ttp+tfp)
trec = ttp / (ttp+tfn)


