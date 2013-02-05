% load labels

iv = load('newCategoryBelonging.txt');
Y = zeros(max(iv(:,1)),1);

Y(iv(:,1)+1) = iv(:,2);
N = length(Y);

% load words

fid = fopen('document.txt', 'r');

I = [];
J = [];
V = [];

I2 = zeros(10,1);
J2 = zeros(10,1);
V2 = zeros(10,1);

next = 1;

while ~feof(fid)
    line = fgetl(fid);
    if line==-1
        break;
    end
    
    [token, remain] = strtok(line);
    i = str2double(token) + 1;
    
    vals = sscanf(remain, '%d:%f');
    
    j = vals(1:2:end-1) + 1;
    v = vals(2:2:end-1);
    
    while length(I2) < next + length(j)
        I2 = [I2; zeros(size(I2))];
        J2 = [J2; zeros(size(I2))];
        V2 = [V2; zeros(size(I2))];
    end
    
    I2(next:next+length(j)-1) = i;
    J2(next:next+length(j)-1) = j;
    V2(next:next+length(j)-1) = v;
    
    next = next + length(j);
    
%     I = [I; repmat(i, length(j),1)];
%     J = [J; j(:)];
%     V = [V; v(:)];
    
    if rand < .001
        plot(J2(1:next),I2(1:next),'.');
        drawnow;
    end
end

fclose(fid);

I = I2(1:next-1);
J = J2(1:next-1);
V = V2(1:next-1);

W = sparse(I,J,V,N,max(J));

%% get ordering
[~,inds] = sort(Y);

%% compute kernel

K = W*W';
K = K ./ bsxfun(@times, sqrt(diag(K)), sqrt(diag(K))');
figure(1);
imagesc(K(inds, inds));
%% binary feature kernel
B = double(W>0);

Kb = B*B';
Kb = Kb ./ bsxfun(@times, sqrt(diag(Kb)), sqrt(diag(Kb))');
figure(2);
imagesc(Kb(inds, inds));

%%


IJ = load('uniqueLinks.txt')-20000;

sameClass = 0;

for k = 1:length(IJ)
    if Y(IJ(k,1)+1) == Y(IJ(k,2)+1)
        sameClass = sameClass + 1;
    end
end

Au = sparse(IJ(:,1)+1, IJ(:,2)+1, ones(length(IJ), 1));
figure(3);
imagesc(Au(inds, inds));
%%
IJ = load('perfectLinks.txt')-20000;

sameClass = 0;

for k = 1:length(IJ)
    if Y(IJ(k,1)+1) == Y(IJ(k,2)+1)
        sameClass = sameClass + 1;
    end
end

Au = sparse(IJ(:,1)+1, IJ(:,2)+1, ones(length(IJ), 1));
figure(4);
imagesc(Au(inds, inds));


% %%
% IJ = load('WithinWithinLinks.txt');
% 
% sameClass = 0;
% 
% for k = 1:length(IJ)
%     if Y(IJ(k,1)+1) == Y(IJ(k,2)+1)
%         sameClass = sameClass + 1;
%     end
% end
% Aa = sparse(IJ(:,1)+1, IJ(:,2)+1, ones(length(IJ), 1));

%% 
figure(4);
imagesc(bsxfun(@eq, Y(inds), Y(inds)'));
