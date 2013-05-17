oDir = '~/Dropbox/Research/psl/wikipedia/olivettiRunQuad225';
cDir = '~/Dropbox/Research/psl/wikipedia/caltechRunQuad';
clDir = '~/Dropbox/Research/psl/wikipedia/caltechRunLinear';
olDir = '~/Dropbox/Research/psl/wikipedia/olivettiRunLinear';

files = {...
    sprintf('%s/olivetti-left-same-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-left-rand-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-bottom-same-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/olivetti-bottom-rand-quad-mle-200-5.0.txt', oDir), ...
    sprintf('%s/caltech-left-same-quad-mle-200-5.0.txt', cDir), ...
    sprintf('%s/caltech-left-rand-quad-mle-200-5.0.txt', cDir), ...
    sprintf('%s/caltech-bottom-same-quad-mle-200-5.0.txt', cDir),...
    sprintf('%s/caltech-bottom-rand-quad-mle-200-5.0.txt', cDir),...
    sprintf('%s/olivetti-left-same-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/olivetti-left-rand-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/olivetti-bottom-same-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/olivetti-bottom-rand-linear-mle-200-5.0.txt', olDir), ...
    sprintf('%s/caltech-left-same-linear-mle-200-5.0.txt', clDir), ...
    sprintf('%s/caltech-left-rand-linear-mle-200-5.0.txt', clDir), ...
    sprintf('%s/caltech-bottom-same-linear-mle-200-5.0.txt', clDir),...
    sprintf('%s/caltech-bottom-rand-linear-mle-200-5.0.txt', clDir)};



for file = 1:length(files)
    if ~exist(files{file}, 'file')
        continue;
    end
    
    data = dlmread(files{file}, ',', 1, 0);
    
    %
    n = max(data(:,2))+1;
    m = max(data(:,1))+1;
    
    X = zeros(n, m);
    
    X = X + sparse(data(:,2) + 1, data(:,1) + 1, data(:,3), n, m);
    
    % X = load('spn/olive-btm.dat');
    
    figure(file);
    clf;
    
    predCount = size(X,1);
    
    rows = ceil(predCount / 10);
    cols = ceil(predCount / rows);
    
    width = round(sqrt(m));
    
    bigImage = zeros(rows*width, cols*width);
    
    k = 1;
    for i = 1:rows
        for j = 1:cols
            bigImage((i-1)*width + 1 : i * width, (j-1) * width + 1 : j * width) = reshape(X(k,:), width, width);
            k = k + 1;
            if (k > predCount)
                break;
            end
        end
        if (k > predCount)
            break;
        end
    end
    %%
    imagesc(bigImage);
    colormap gray; axis image;
    title(files{file});
    
    [pathstr, name, ext] = fileparts(files{file});
    print('-dpng', sprintf('images/%s.png', name));
    
end

