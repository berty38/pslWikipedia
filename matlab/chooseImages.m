clear;


SPNfiles = {
    'spn/olive-left.dat',...
    'spn/olive-btm.dat',...
    'spn/Faces_easy-left.dat',...
    'spn/Faces_easy-btm.dat'};



oDir = 'olivettiRunQuad';
cDir = 'caltechRunQuad';

files = {...
    sprintf('%s/olivetti-left-rand-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-bottom-rand-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/caltech-left-rand-quad-mle-200-5.0.txt', cDir), ...
    sprintf('%s/caltech-bottom-rand-quad-mle-200-5.0.txt', cDir)};
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
    
    IJ = dlmread(files{task}, ',', 1, 0);
    X = full(sparse(IJ(:,1)+1, IJ(:,2)+1, 255*IJ(:,3)));
    
    for j = 1:50
        psl{task}{j} = reshape(X(:,j), w, h);
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
            row = [truth{t}{i} vpad psl{t}{i} vpad spn{t}{i} vpad psl{t+1}{i} vpad spn{t+1}{i}];
            bigImage = [bigImage;  row; zeros(1, size(row,2))];
        end
        imagesc(bigImage); colormap gray; axis image;
        [x,y] = ginput;
        
        inds = (k-1)*10 + ceil(y/h);
        
        for j = 1:length(inds)
            i = inds(j);
            row = [truth{t}{i} vpad psl{t}{i} vpad spn{t}{i} vpad psl{t+1}{i} vpad spn{t+1}{i}];
            finalInds{task}(end+1) = i;
            finalImage{task} = [finalImage{task}; row; zeros(1, size(row,2))];
        end
        
        imagesc(finalImage{task});
        pause;
    end
end

%%


imwrite(uint8(round(finalImage{1})), 'images/olivettiFigNew200.png', 'png');
imwrite(uint8(round(finalImage{2})), 'images/caltechFigNew200.png', 'png');


