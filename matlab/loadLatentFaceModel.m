clear;
filename = '../output/vision/latent/olivetti-left-same-model.txt';


width = 64;
height = 64;

patternPos =  '\{(.+?)\} \( PICTURE\(pictureVar\) & PICTURETYPE\(pictureVar, (.+?)\) \) >> PIXELBRIGHTNESS\((.+?),';
patternNeg =  '\{(.+?)\} \( PICTURE\(pictureVar\) & PICTURETYPE\(pictureVar, (.+?)\) \) >> ~\( PIXELBRIGHTNESS\((.+?),';

fid = fopen(filename);
while ~feof(fid)
    line = fgetl(fid);
    
    pos = true;
    
    tokens = regexp(line, patternPos, 'tokens');
    if isempty(tokens)
        tokens = regexp(line, patternNeg, 'tokens');
        pos = false;
    end
    
    if isempty(tokens) || length(tokens{1}) ~= 3
        fprintf('Could not parse line\n%s\n', line);
        continue;
    end
    tokens = tokens{1};
    
    weight = str2double(tokens{1});
    group = str2double(tokens{2});
    pixel = str2double(tokens{3});
    
    if pos
        posWeight{group+1}(pixel+1) = weight;
    else
        negWeight{group+1}(pixel+1) = weight;
    end
    
end
fclose(fid);

bigImage = [];

for i = 1:length(posWeight)
    posWeight{i} = reshape(posWeight{i}, width, height);
    negWeight{i} = reshape(negWeight{i}, width, height);
    bigImage = [bigImage; posWeight{i}; negWeight{i}];
end
%%
imagesc(bigImage);
axis image
