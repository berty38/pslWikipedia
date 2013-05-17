clear;


SPNfiles = {
    'spn/olive-left.dat',...
    'spn/olive-btm.dat',...
    'spn/Faces_easy-left.dat',...
    'spn/Faces_easy-btm.dat'};



oDir = 'olivettiRunQuad225';
cDir = 'caltechRunQuad';
clDir = 'caltechRunLinear';
olDir = 'olivettiRunLinear';

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
h = 64;
w = 64;

vpad = zeros(h,1);

for task = 1:length(SPNfiles)
    
    file = SPNfiles{task};
    
    pad = 10;
    
    
    X = load(file);
    
    for i = 1:50
        start = (i-1) * (h+pad);
        truth{task}{i} = X(start+1:start+h, 1:w);
        spn{task}{i} = X(start+1:start+h, w+pad+1:w+pad+w);
    end
    
    for i = 1:4
        IJ = dlmread(files{(task-1)*4+i}, ',', 1, 0);
        X = full(sparse(IJ(:,1)+1, IJ(:,2)+1, 255*IJ(:,3)));
        
        for j = 1:50
            psl{task}{i}{j} = reshape(X(:,j), w, h);
        end
    end
end


%%

finalInds = {[], []};

for task = 1:2
    finalImage{task} = [];
    
    for k = 1:5
        bigImage = [];
        for j = 1:10
            i = (k-1) * 10 + j;
            t = (task-1)*2+1;
            row = [truth{t}{i} vpad psl{t}{2}{i} vpad spn{t}{i} vpad psl{t+1}{2}{i} vpad spn{t+1}{i}];
            bigImage = [bigImage;  row; zeros(1, size(row,2))];
        end
        imagesc(bigImage); colormap gray; axis image;
        [x,y] = ginput;
            
        inds = (k-1)*10 + ceil(y/h);
        
        for j = 1:length(inds) 
            i = inds(j);
            row = [truth{t}{i} vpad psl{t}{2}{i} vpad spn{t}{i} vpad psl{t+1}{2}{i} vpad spn{t+1}{i}];
            finalInds{task}(end+1) = i;
            finalImage{task} = [finalImage{task}; row; zeros(1, size(row,2))];
        end
        
        imagesc(finalImage{task});
        pause;
    end
end

%%


imwrite(uint8(round(finalImage{1})), 'images/olivettiFigNew.png', 'png');
imwrite(uint8(round(finalImage{2})), 'images/caltechFigNew.png', 'png');


