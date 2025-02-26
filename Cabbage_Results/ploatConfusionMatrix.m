% This code is for calculating top1 accuracy, precision/recall and plot confusion matrix of the trained network

networkName = input('Type Network name : ', 's');
net = trainedNetwork_1;

imageFolder = fullfile('~/mnt/nfs_clientshare/','Final-Dataset');
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

imageSize = net.Layers(1).InputSize;
augmentedTestSet = augmentedImageDatastore(imageSize, imds, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'classoutput';

labels = imds.Labels;
predictedLabels = predict(net, augmentedTestSet)';

[~, pli] = max(predictedLabels);
predictedLabels = predictedLabels';



plc = categorical();
pred_Backmoth = zeros(2, length(pli))';
pred_Clubroot = zeros(2, length(pli))';
pred_Leafminer = zeros(2, length(pli))';
pred_Mildew = zeros(2, length(pli))';
pred_healthy = zeros(2, length(pli))';

resp_Backmoth = zeros(1, length(pli))';
resp_Clubroot = zeros(1, length(pli))';
resp_Leafminer = zeros(1, length(pli))';
resp_Mildew = zeros(1, length(pli))';
resp_healthy = zeros(1, length(pli))';

for i = 1 : length(pli)
    if pli(i) == 1
        plc(i) = 'Backmoth';
        resp_Backmoth(i, 1) = 1;

    elseif pli(i) == 2
        plc(i)= 'Clubroot';
        resp_Clubroot(i, 1) = 1;

    elseif pli(i) == 3
        plc(i)= 'Leafminer';
        resp_Leafminer(i, 1) = 1;

    elseif pli(i) == 4
        plc(i)= 'Mildew';
        resp_Mildew(i, 1) = 1;

    elseif pli(i) == 5
        plc(i)= 'healthy';
        resp_healthy(i, 1) = 1;
    
    else
        disp('Error occured during labeling')
    end

    pred_Backmoth(i, 1) = predictedLabels(i, 1);
    pred_Backmoth(i, 2) = sum(predictedLabels(i, 2:5));
    pred_Clubroot(i, 1) = predictedLabels(i, 2);
    pred_Clubroot(i, 2) = sum([predictedLabels(i, 1), sum(predictedLabels(i, 3:5))]);
    pred_Leafminer(i, 1) = predictedLabels(i, 3);
    pred_Leafminer(i, 2) = sum([sum(predictedLabels(i, 1:2)), sum(predictedLabels(i, 4:5))]);
    pred_Mildew(i, 1) = predictedLabels(i, 4);
    pred_Mildew(i, 2) = sum([sum(predictedLabels(i, 1:3)), predictedLabels(i, 5)]);
    pred_healthy(i, 1) = predictedLabels(i, 5);
    pred_healthy(i, 2) = sum(predictedLabels(i, 1:4));
end

plc = plc';


mdl_Backmoth = fitglm(pred_Backmoth,resp_Backmoth,'Distribution','binomial','Link','logit');
mdl_Clubroot = fitglm(pred_Clubroot,resp_Clubroot,'Distribution','binomial','Link','logit');
mdl_Leafminer = fitglm(pred_Leafminer,resp_Leafminer,'Distribution','binomial','Link','logit');
mdl_Mildew = fitglm(pred_Mildew,resp_Mildew,'Distribution','binomial','Link','logit');
mdl_healthy = fitglm(pred_healthy,resp_healthy,'Distribution','binomial','Link','logit');


scores_Backmoth = mdl_Backmoth.Fitted.Probability;
scores_Clubroot = mdl_Clubroot.Fitted.Probability;
scores_Leafminer = mdl_Leafminer.Fitted.Probability;
scores_Mildew = mdl_Mildew.Fitted.Probability;
scores_healthy = mdl_healthy.Fitted.Probability;


[X_Backmoth, Y_Backmoth, T_Backmoth, AUC_Backmoth] = perfcurve(labels, scores_Backmoth, 'Backmoth');
[X_Clubroot, Y_Clubroot, T_Clubroot, AUC_Clubroot] = perfcurve(labels, scores_Clubroot, 'Clubroot');
[X_Leafminer, Y_Leafminer, T_Leafminer, AUC_Leafminer] = perfcurve(labels, scores_Leafminer, 'Leafminer');
[X_Mildew, Y_Mildew, T_Mildew, AUC_Mildew] = perfcurve(labels, scores_Mildew, 'Mildew');
[X_healthy, Y_healthy, T_healthy, AUC_healthy] = perfcurve(labels, scores_healthy, 'healthy');








top1Accuracy = mean(plc == labels);
confMat = confusionmat(plc, labels);

tp = 0;
fp = 0;
fn = 0;

for i=1 : length(confMat(:, 1))
    for j=1 : length(confMat(1,:))
        if i == j
            tp = tp + confMat(i, j);
        elseif i > j
            fp = fp + confMat(i, j);
        elseif i < j
            fn = fn + confMat(i, j);
        else
            disp('Error occured during calculating precision/recall');
        end
    end
end


precision = tp/(tp + fp);
recall = tp/(tp + fn);

fprintf("Top 1 accuracy : %f\n", top1Accuracy);
fprintf('Precision : %f\n', precision);
fprintf('Recall : %f\n\n', recall);



figure(1)
plotconfusion(labels, plc, networkName);
grid

X_AUC1 = [0; 0; 1;];
Y_AUC1 = [0; 1; 1;];

X_AUC_05 = [0; 0.5; 1;];
Y_AUC_05 = [0; 0.5; 1;];

figure(2)
plot(X_AUC1, Y_AUC1, '-.')
hold on
plot(X_Backmoth, Y_Backmoth)
plot(X_Clubroot, Y_Clubroot)
plot(X_Leafminer, Y_Leafminer)
plot(X_Mildew, Y_Mildew)
plot(X_healthy, Y_healthy)
plot(X_AUC_05, Y_AUC_05, ':')


l_Backmoth = strcat('Backmoth     AUC=', string(AUC_Backmoth));
l_Clubroot = strcat('Clubroot        AUC=', string(AUC_Clubroot));
l_Leafminer = strcat('Leafminer      AUC=', string(AUC_Leafminer));
l_Mildew = strcat('Mildew          AUC=', string(AUC_Mildew));
l_healthy = strcat('Healthy          AUC=', string(AUC_healthy));


legend('AUC = 1', l_Backmoth, l_Clubroot, l_Leafminer, l_Mildew, l_healthy, 'AUC = 0.5', 'Location', 'Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Chinese-Cabbage')
hold off

grid
