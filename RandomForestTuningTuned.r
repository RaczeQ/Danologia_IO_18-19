library(caret)
library(mlr)
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
        makeIntegerParam("mtry", lower = 13, upper = 22),
        makeIntegerParam("nodesize", lower = 58, upper = 72)
)
# Few iterations of tuning:

# mtry = 6      nodesize = 11                           mcc test: 0.5451814
# mtry = 27     nodesize = 87                           mcc test: 0.5540268
# mtry = 14     nodesize = 50                           mcc test: 0.5624794
# mtry = 18     nodesize = 71   mcc train: 0.5754450    mcc test: 0.5712314 <- BEST TEST
# mtry = 15     nodesize = 75   mcc train: 0.6223846    mcc test: 0.5528764
# mtry = 21     nodesize = 72   mcc train: 0.6152079    mcc test: 0.5500041
# mtry = 15     nodesize = 70   mcc train: 0.6181142    mcc test: 0.5664719

# set validation strategy
rdesc <- makeResampleDesc("CV",iters=5L)

# set optimization technique
# ctrl <- makeTuneControlRandom(maxit = detectCores() * 10)
resolution <- ceiling(sqrt(detectCores() * 10))
ctrl <- makeTuneControlGrid(resolution = resolution)

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

####################
# BEST TUNED MODEL #
####################

# rf.lrn <- makeLearner("classif.randomForest")
# rf.lrn$par.vals <- list(
#         ntree = 100L,
#         importance = TRUE,
#         cutoff = c(0.75,0.25),
#         mtry = 15,
#         nodesize = 70
# )

# final.model <- mlr::train(rf.lrn, task = trainTask)
# pred = predict(final.model, task = testTask)
# perfMeasures<-mlr::performance(pred, measures=list(mcc, mmce, acc, f1, kappa))

# print(perfMeasures)
# print(mlr::calculateConfusionMatrix(pred))

#       mcc      mmce       acc        f1     kappa
# 0.5664719 0.2238806 0.7761194 0.7435897 0.5510198
#         predicted
# true     FALSE TRUE -err.-
#   FALSE    242   91     91
#   TRUE      29  174     29
#   -err.-    29   91    120
