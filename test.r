library(igraph)
library(mlr)
library(caret)

set.seed(1)

trainOrig <- read.csv("lucene2.2train.csv", header = TRUE, sep = ",")
testOrig <- read.csv("lucene2.4test.csv", header = TRUE, sep = ",")

print(head(trainOrig, n=2))

print(mlr::summarizeColumns(trainOrig))

print(summary(trainOrig))

par(mfrow=c(1,4)) 
for(i in 5:8) { 
    boxplot(trainOrig[,i], main=names(trainOrig[i]))
}