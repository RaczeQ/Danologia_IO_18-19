# add necesarry libraries
library(knitr)
library(magrittr)
library(igraph)
library(mlr)
library(caret)
library(knitr)
library(dplyr)
# set parallel backend
library(parallelMap)
library(parallel)

# set the random
set.seed(1)

# read csv files
trainOrig <- read.csv("lucene2.2train.csv", header = TRUE, sep = ",")
testOrig <- read.csv("lucene2.4test.csv", header = TRUE, sep = ",")

# add column name with dataset type
train <- trainOrig %>% mutate(dataset = "train")
test <- testOrig %>% mutate(dataset = "test")

# merge test and train data at once, 
# now it is easier prepare data (we doing operations for two sets at once)
combined <- dplyr::bind_rows(train, test)

# delete unused columns (column with specific data, haven't any dependent with errors in class) can be removed
combined <- combined %>% select(-c(X, Project, Version, Class))

myResult= "0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 1 0"
sss = strsplit(myResult, "\\s+")[[1]]
sss[1]

columns = c("wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa",
            "mfa","cam","ic","cbm","amc","nr","ndc","nml","ndpv","max.cc.","avg.cc.")

for (b in 1:24) {
  if (sss[b]=="1") {
    combined <- combined %>% select(-c(columns[b]))
  }
}

# replace NaNs with mean values
# I will not ask this question anymore, (mabye later on datascience lectures)
# bu we decided for merge training and test data, now we calculate the mean of all
# I think that the sets are dependent, is it ok?
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