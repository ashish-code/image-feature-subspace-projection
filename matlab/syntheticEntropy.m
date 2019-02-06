% Generate synthetic data and measure Renyi entropy to motivate its use as
% measure of information content in high dimensional data distribution

% initialize matlab
cdir = pwd;
cd ~;startup;cd (cdir);

rootDir = '/vol/vssp/diplecs/ash/Data/';
swiss = generate_data('swiss', 500);
swissfilename = strcat(rootDir,'Thesis/dictionarylearning/swiss.3');
dlmwrite(swissfilename,swiss,',');

intersect = generate_data('intersect',500);
intersectfilename = strcat(rootDir,'Thesis/dictionarylearning/intersect.3');
dlmwrite(intersectfilename,intersect,',');

realdata = load('/vol/vssp/diplecs/ash/Data/Thesis/dictionarylearning/car.3');
realdata = realdata(1:500,1:3);
realdatafilename = strcat(rootDir,'Thesis/dictionarylearning/realdata.3');
dlmwrite(realdatafilename,realdata,',');

swisspdist = pdist(swiss);
intersectpdist = pdist(intersect);
realdatapdist = pdist(realdata);

swisshist = hist(swisspdist,100);
intersecthist = hist(intersectpdist,100);
realdatahist = hist(realdatapdist,100);

swisshist = swisshist/sum(swisshist);
intersecthist = intersecthist/sum(intersecthist);
realdatahist = realdatahist/sum(realdatahist);

swisshistfilename = strcat(rootDir,'Thesis/dictionarylearning/swisshist.3');
intersecthistfilename = strcat(rootDir,'Thesis/dictionarylearning/intersecthist.3');
realdatahistfilename = strcat(rootDir,'Thesis/dictionarylearning/realdatahist.3');

stem(swisshist,'k.');
hold on
stem(intersecthist,'ro');
stem(realdatahist,'bd');

Hswiss = renyi_entro(swisspdist',0.2);
Hintersect = renyi_entro(intersectpdist',0.2);
Hrealdata = renyi_entro(realdatapdist',0.2);

figure(2),scatter3(swiss(:,1),swiss(:,2),swiss(:,3),'ks');
figure(3),scatter3(intersect(:,1),intersect(:,2),intersect(:,3),'b*');
figure(4),scatter3(realdata(:,1),realdata(:,2),realdata(:,3),'rd');

figure(5)
plot(swisshist,'--ks','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',4);
hold on
plot(intersecthist,'-.b*','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',4);
plot(realdatahist,'-.rd','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',4);
