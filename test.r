library(igraph)
library(mlr)
library(caret)
library(knitr)
library(dplyr)

set.seed(1)

trainOrig <- read.csv("lucene2.2train.csv", header = TRUE, sep = ",")
testOrig <- read.csv("lucene2.4test.csv", header = TRUE, sep = ",")

print(head(trainOrig, n=2))

print(mlr::summarizeColumns(trainOrig))

print(summary(trainOrig))

# par(mfrow=c(1,4)) 
# for(i in 5:8) { 
#     boxplot(trainOrig[,i], main=names(trainOrig[i]))
# }


# iv <- trainOrig[,5:8]
# dv <- trainOrig[,29]
# caret::featurePlot(
#     x=iv, 
#     y=as.factor(dv), 
#     plot="box", 
#     auto.key = list(columns=2),
#     scales=list(x=list(relation="free"), y=list(relation="free"))
# )


train <- trainOrig %>% mutate(dataset = "train")
test <- testOrig %>% mutate(dataset = "test")
combined <- dplyr::bind_rows(train,test)
# print(summarizeColumns(combined) %>% kable(digits = 2))

# delete columns
combined <- combined %>% select(-c(X, Project, Version, Class))
# print(summarizeColumns(combined) %>% kable(digits = 2))

imp <- impute(combined,
    # replace NaNs with mean values
    classes = list( factor=imputeMode(), integer=imputeMean(), numeric=imputeMean())
)
combined <- imp$data
# print(summarizeColumns(combined) %>% kable(digits = 2))

# normalize values / features
combined<-normalizeFeatures(combined, target="isBuggy")
# print(summarizeColumns(combined) %>% kable(digits = 2))

# divide set for training 
train <- combined %>% filter(dataset=="train") %>% select(-dataset) 
test <- combined %>% filter(dataset=="test") %>% select(-dataset)
# print(summarizeColumns(train) %>% kable(digits = 2))

#create tasks
trainTask <- makeClassifTask(data=train, target="isBuggy", positive="TRUE")
testTask <- makeClassifTask(data=test, target="isBuggy", positive="TRUE")
print(trainTask)

# feature importance in dataset
featureImportance<-mlr::generateFilterValuesData(testTask, method = "information.gain")
# print(mlr::plotFilterValues(featureImportance))

# training models
print("Logistic Regression")
logisticRegression.learner <- mlr:::makeLearner("classif.logreg", predict.type = "response")
logisticRegression.model <- mlr::train(logisticRegression.learner, task = trainTask)
pred = predict(logisticRegression.model, task = testTask)
perfMeasures<-mlr::performance(pred, measures=list(mcc, mmce, acc, f1, kappa))

print(perfMeasures)
print(mlr::calculateConfusionMatrix(pred))

print("Quadratic Disciminant Analysis")
quadraticDiscirminantAnalysis.learner <- makeLearner("classif.qda", predict.type = "response")
quadraticDiscirminantAnalysis.model <- mlr::train(quadraticDiscirminantAnalysis.learner, task = trainTask)
pred = predict(quadraticDiscirminantAnalysis.model, task = testTask)
perfMeasures<-mlr::performance(pred, measures=list(mcc, mmce, acc, f1, kappa))

print(perfMeasures)
print(mlr::calculateConfusionMatrix(pred))

print("Decision Tree")
decisionTree.learner <- makeLearner("classif.rpart", predict.type = "response")
decisionTree.model <- mlr::train(decisionTree.learner, task = trainTask)
pred = predict(decisionTree.model, task = testTask)
perfMeasures<-mlr::performance(pred, measures=list(mcc, mmce, acc, f1, kappa))

print(perfMeasures)
print(mlr::calculateConfusionMatrix(pred))

print(getParamSet("classif.rpart"))

# Tuned Decistion Tree

print("Tuned Decision Tree")
decisionTree.learner <- makeLearner("classif.rpart", predict.type = "response")

setCV <- makeResampleDesc("CV", iters = 3L)
paramSet <- makeParamSet(
    makeIntegerParam("minsplit", lower=10, upper=50),
    makeIntegerParam("minbucket", lower=5, upper=50),
    makeNumericParam("cp", lower=0.001, upper=0.3)
)

tuneParams <- tuneParams(learner = decisionTree.learner, resampling = setCV,
    task = trainTask, par.set = paramSet, control = makeTuneControlGrid(),
    measures = mcc)

print(tuneParams$x)

decisionTreeTuned.learner <- setHyperPars(decisionTree.learner, par.vals = tuneParams$x)
decisionTreeTuned.model <- mlr::train(decisionTreeTuned.learner, task = trainTask)
pred = predict(decisionTreeTuned.model, task = testTask)
perfMeasures<-mlr::performance(pred, measures=list(mcc, mmce, acc, f1, kappa))

print(perfMeasures)
print(mlr::calculateConfusionMatrix(pred))