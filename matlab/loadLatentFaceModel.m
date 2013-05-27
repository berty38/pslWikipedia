clear;
filename = '../output/vision/latent/olivetti-left-same-model.txt';


width = 64;
height = 64;

patternPos = '{%f} ( PICTURE(pictureVar) & PICTURETYPE(pictureVar, %f) ) >> HASMEAN(pictureVar, %f, %f) {squared}';
patternNeg = '{%f} ( PICTURE(pictureVar) & PICTURETYPE(pictureVar, %f) ) >> ~( HASMEAN(pictureVar, %f, %f) ) {squared}';
patternPriorPos = '{%f} PICTURE(pictureVar) >> HASMEAN(pictureVar, %f, %f) {squared}';
patternPriorNeg = '{%f} PICTURE(pictureVar) >> ~( HASMEAN(pictureVar, %f, %f) ) {squared}';

fid = fopen(filename);
while ~feof(fid)
    line = fgetl(fid);
    A = sscanf(line, patternPos, 4);
    if length(A) == 4
        posWeight{A(4)+1}{A(2)+1}(A(3)+1) = A(1);
        continue;
    end
    A = sscanf(line, patternNeg, 4);
    if length(A) == 4
        negWeight{A(4)+1}{A(2)+1}(A(3)+1) = A(1);
        continue;
    end
    
    A = sscanf(line, patternPriorPos, 3);
    if length(A) == 3
        posPrior{A(3)+1}(A(2)+1) = A(1);
        continue;
    end
    
    A = sscanf(line, patternPriorNeg, 3);
    if length(A) == 3
        negPrior{A(3)+1}(A(2)+1) = A(1);
        continue;
    end
end

fclose(fid);


%%


bigImage = [];

for i = 1:length(posPrior)
    smallImage = [reshape(posPrior{i}, width, height), ...
        reshape(negPrior{i}, width, height)];
    
    for j = 1:length(posWeight{i})
        smallImage = [smallImage, ...
            reshape(posWeight{i}{j}, width, height),...
            reshape(negWeight{i}{j}, width, height)];
    end
    bigImage = [bigImage; smallImage];
end

%%
figure(2);
imagesc(bigImage);
ylabel('gaussian mean id');
xlabel(sprintf('prior (pos-neg), types %d through %d (pos-neg)', 0, length(posWeight{1})));
colormap gray;

