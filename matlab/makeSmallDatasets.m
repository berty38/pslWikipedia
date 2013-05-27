datasets = {'olivetti', 'caltech'};

smallSize = 32;

for set = 1:length(datasets)
    
    images = load(sprintf('../data/vision/%s01.txt', datasets{set}));

    imagesSmall = zeros(size(images,1), smallSize, smallSize);
    
    for i = 1:size(images,1)
        tmp = reshape(images(i,:), 64, 64);
        tmpSmall = imresize(tmp, [smallSize smallSize]);
        imagesSmall(i,:) = tmpSmall(:)';
    end
   
    dlmwrite(sprintf('../data/vision/%s-small01.txt', datasets{set}), imagesSmall, '\t');
end
