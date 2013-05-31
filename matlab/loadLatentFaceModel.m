clear;
filename = '../output/vision/latent/caltech-small-left-model-0.txt';


width = 32;
height = 32;

patternPos = '{%f} ( PICTURE(pictureVar) & PICTURETYPE(pictureVar, %f) ) >> PIXELBRIGHTNESS(pictureVar, %f) {squared}';
patternNeg = '{%f} ( PICTURE(pictureVar) & PICTURETYPE(pictureVar, %f) ) >> ~( PIXELBRIGHTNESS(pictureVar, %f) ) {squared}';
patternPriorPos = '{%f} PICTURE(pictureVar) >> PIXELBRIGHTNESS(pictureVar, %f) {squared}';
patternPriorNeg = '{%f} PICTURE(pictureVar) >> ~( PIXELBRIGHTNESS(pictureVar, %f) ) {squared}';

fid = fopen(filename);
while ~feof(fid)
    line = fgetl(fid);
    A = sscanf(line, patternPos, 3);
    if length(A) == 3
        posWeight{A(2)+1}(A(3)+1) = A(1);
        continue;
    end
    A = sscanf(line, patternNeg, 3);
    if length(A) == 3
        negWeight{A(2)+1}(A(3)+1) = A(1);
        continue;
    end
    
    A = sscanf(line, patternPriorPos, 2);
    if length(A) == 2
        posPrior(A(2)+1) = A(1);
        continue;
    end
    
    A = sscanf(line, patternPriorNeg, 2);
    if length(A) == 2
        negPrior(A(2)+1) = A(1);
        continue;
    end
end

fclose(fid);


%%


for i = 1:length(posWeight)
    posWeight{i} = reshape(posWeight{i}, height, width/2);
    negWeight{i} = reshape(negWeight{i}, height, width/2);
end

posPrior = reshape(posPrior, height, width/2);
negPrior = reshape(negPrior, height, width/2);

%
figure(3);
subplot(131);
imagesc([posPrior; negPrior]);
xlabel('Prior');
ylabel('Negative    Positive');
axis image;
colorbar;

subplot(1,3,[2, 3]);
img = [];
for i = 1:length(posWeight)
    img = [img, [posWeight{i}; negWeight{i}]];
end
imagesc(img);
xlabel('Type');
ylabel('Negative    Positive');
colormap gray;
axis image;
colorbar;
