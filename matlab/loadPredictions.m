oDir = '~/Dropbox/Research/pslWikipedia/output/vision/latent/';

files = {...
    sprintf('%s/olivetti-small-left.txt', oDir)};



for file = 1:length(files)
    if ~exist(files{file}, 'file')
        continue;
    end
    
    data = dlmread(files{file}, ',', 1, 0);
    
    %
    n = max(data(:,1))+1;
    m = max(data(:,2))+1;
    
    X = zeros(n, m);
    
    X = X + sparse(data(:,1) + 1, data(:,2) + 1, data(:,3), n, m);
        
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
            bigImage((i-1)*width + 1 : i * width, (j-1) * width + 1 + width/2: j * width) = 0;
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

