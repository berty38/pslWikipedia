SPNfiles = {
    'spn/olive-left.dat',...
    'spn/olive-btm.dat',...
    'spn/Faces_easy-left.dat',...
    'spn/Faces_easy-btm.dat'};

taskName = {'images/ol', 'images/ob', 'images/cl', 'images/cb' };



oDir = '~/Dropbox/Research/psl/wikipedia/olivettiRunQuad225';
cDir = '~/Dropbox/Research/psl/wikipedia/caltechRunQuad';
clDir = '~/Dropbox/Research/psl/wikipedia/caltechRunLinear';
olDir = '~/Dropbox/Research/psl/wikipedia/olivettiRunLinear';

files = {...
    sprintf('%s/olivetti-left-same-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-left-rand-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-left-same-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/olivetti-left-rand-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/olivetti-bottom-same-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-bottom-rand-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-bottom-same-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/olivetti-bottom-rand-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/caltech-left-same-quad-mle-200-5.0.txt', cDir), ...
    sprintf('%s/caltech-left-rand-quad-mle-200-5.0.txt', cDir), ...
    sprintf('%s/caltech-left-same-linear-mle-200-5.0.txt', clDir), ...
    sprintf('%s/caltech-left-rand-linear-mle-200-5.0.txt', clDir), ...
    sprintf('%s/caltech-bottom-same-quad-mle-200-5.0.txt', cDir),...
    sprintf('%s/caltech-bottom-rand-quad-mle-200-5.0.txt', cDir),...
    sprintf('%s/caltech-bottom-same-linear-mle-200-5.0.txt', clDir),...
    sprintf('%s/caltech-bottom-rand-linear-mle-200-5.0.txt', clDir)};

for task = 1:length(SPNfiles)
    
    file = SPNfiles{task};
    
    pad = 10;
    
    h = 64;
    w = 64;
    
    X = load(file);
    
    for i = 1:50
        start = (i-1) * (h+pad);
        truth{i} = X(start+1:start+h, 1:w);
        spn{i} = X(start+1:start+h, w+pad+1:w+pad+w);
    end

    for i = 1:4
        IJ = dlmread(files{(task-1)*4+i}, ',', 1, 0);
        X = full(sparse(IJ(:,1)+1, IJ(:,2)+1, 255*IJ(:,3)));
        
        for j = 1:50
            psl{i}{j} = reshape(X(:,j), w, h);
        end
    end
    
    bigImage = [];
    
    for i = 1:50
        row = [truth{i} spn{i} psl{1}{i} psl{2}{i} psl{3}{i} psl{4}{i}];
        bigImage = [bigImage; row];
    end
    imagesc(row); colormap gray; axis image;
    imwrite(uint8(round(bigImage)), [taskName{task} '.png'], 'png');
end
