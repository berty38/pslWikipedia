clear;
filename = '../output/vision/latent/olivetti-left-same-model.txt';


width = 64;
height = 64;

patternPos =  '\{(.+?)\} \( PICTURE\(pictureVar\) & PICTURETYPE\(pictureVar, (.+?)\) \) >> PIXELBRIGHTNESS\((.+?),';
% patternNeg =  '\{(.+?)\} \( PICTURE\(pictureVar\) & PICTURETYPE\(pictureVar, (.+?)\) \) >> ~\( PIXELBRIGHTNESS\((.+?),';
patternPrior =  '\{(.+?)\} PICTURE\(pictureVar\) >> ~\( PIXELBRIGHTNESS\((.+?),';

fid = fopen(filename);
while ~feof(fid)
    line = fgetl(fid);
    
    pos = true;
    
    tokens = regexp(line, patternPos, 'tokens');
    if ~isempty(tokens)
        tokens = tokens{1};
        weight = str2double(tokens{1});
        group = str2double(tokens{2});
        pixel = str2double(tokens{3});
        posWeight{group+1}(pixel+1) = weight;
    else
        tokens = regexp(line, patternPrior, 'tokens');
        if ~isempty(tokens)
            tokens = tokens{1};
            weight = str2double(tokens{1});
            pixel = str2double(tokens{2});
            priorWeight(pixel+1) = weight;
        end
    end
    
end
fclose(fid);
%%
figure(2);
subplot(121);
priorWeight = reshape(priorWeight, width, height);
bigImage = priorWeight(:, 1:width/2);

for i = 1:length(posWeight)
    posWeight{i} = reshape(posWeight{i}, width, height);
    bigImage = [bigImage; posWeight{i}(:, 1:width/2)];
end

imagesc(bigImage);
colorbar
axis image

%%
subplot(122);
imagesc(posWeight{1} - posWeight{2})
axis image;
colorbar