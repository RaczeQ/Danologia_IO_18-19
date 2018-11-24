library(caret)
library(mlr)
# library(scales) 
library(igraph)
library(e1071)
library(knitr)
library(dplyr)

set.seed(1)

# read csv files
trainOrig <- read.csv("lucene2.2train.csv", header = TRUE, sep = ",")
testOrig <- read.csv("lucene2.4test.csv", header = TRUE, sep = ",")

train <- trainOrig %>% mutate(dataset = "train")
test <- testOrig %>% mutate(dataset = "test")
combined <- dplyr::bind_rows(train, test)

# delete columns
combined <- combined %>% select(-c(X, Project, Version, Class))

# replace NaNs with mean values
imp <- mlr::impute(combined,
    classes = list(
        factor = imputeMode(),
        integer = imputeMean(),
        numeric = imputeMean()
    )
)
combined <- imp$data

# normalize values / features
combined <- normalizeFeatures(combined, target = "isBuggy")

# divide set for training
train <- combined %>% filter(dataset == "train") %>% select(-dataset)
test <- combined %>% filter(dataset == "test") %>% select(-dataset)

# create tasks
trainTask <- makeClassifTask(data = train, target = "isBuggy", positive = "TRUE")
testTask <- makeClassifTask(data = test, target = "isBuggy", positive = "TRUE")

trainTask <- createDummyFeatures(obj = trainTask)
testTask <- createDummyFeatures(obj = testTask)

# set parallel backend
library(parallelMap)
library(parallel)
parallelStartSocket(cpus = detectCores())
# parallel start

# Random Forest Tuning
print("Random Forest Tuning")
rf.lrn <- makeLearner("classif.randomForest")
rf.lrn$par.vals <- list(ntree = 100L,
                        importance=TRUE,
                        cutoff=c(0.75,0.25)
)

# set parameter space
params <- makeParamSet(
        makeIntegerParam("mtry",lower = 2,upper = 10)
        ,makeIntegerParam("nodesize",lower = 10,upper = 50)
        # ,makeIntegerParam("ntree",lower = 50,upper = 500)
)

# set validation strategy
rdesc <- makeResampleDesc("CV",iters=5L)

# set optimization technique
ctrl <- makeTuneControlRandom(maxit = detectCores() * 10)

library(parallelMap)
library(parallel)
parallelStartSocket(cpus = detectCores())

tune <- tuneParams(learner = rf.lrn
                   ,task = trainTask
                   ,resampling = rdesc
                   ,measures = list(mcc)
                   ,par.set = params
                   ,control = ctrl
                   ,show.info = T)

print(tune$x)

final.learner <- setHyperPars(rf.lrn, par.vals = tune$x)
final.model <- mlr::train(final.learner, task = trainTask)
pred = predict(final.model, task = testTask)
perfMeasures<-mlr::performance(pred, measures=list(mcc, mmce, acc, f1, kappa))

print(perfMeasures)
print(mlr::calculateConfusionMatrix(pred))

#stop parallelization
parallelStop()