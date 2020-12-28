#Load Principle libraries
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(gbm)

rfilename <- "pml-training.csv"


# Checking if folder exists
if (!file.exists(rfilename)) { 
  fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileURL, rfilename)
  rfilename <- "pml-testing.csv"
  fileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileURL, rfilename)
}

trainingraw <- read.csv(file="pml-training.csv", header=TRUE)
testingraw <- read.csv(file="pml-testing.csv", header=TRUE)

dim(trainingraw)



trainingraw$classe <- as.factor(trainingraw$classe)



trainData<- trainingraw[, colSums(is.na(trainingraw)) == 0]
testingData <- testingraw[, colSums(is.na(testingraw)) == 0]
trainData <- trainData[, -c(1:7)]
testingData <- testingData[, -c(1:7)]



dim(testingData)

set.seed(202012) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
testData <- trainData[-inTrain, ]
trainData <- trainData[inTrain, ]

dim(trainData)

NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]

dim(trainData)

# Boosted Model
set.seed(202012)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=trainData, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel
print(modFitGBM)

predictGBM <- predict(modFitGBM, newdata=testData)
cmGBM <- confusionMatrix(predictGBM, testData$classe)
cmGBM

plot(cmGBM$table, col = cmGBM$byClass, 
     main = paste("GBM - Accuracy =", round(cmGBM$overall['Accuracy'], 4)))


finalpredict <- predict(modFitGBM, newdata=testingData)
finalpredict


# Random Forests
set.seed(202012)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=trainData, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel

predictRandForest <- predict(modFitRandForest, newdata=testData)
confMatRandForest <- confusionMatrix(predictRandForest, testData$classe)
confMatRandForest

plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =", round(confMatRandForest$overall['Accuracy'], 4)))


finalpredict <- predict(modFitRandForest, newdata=testingData)
finalpredict


# Decision Trees
set.seed(202012)
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
fancyRpartPlot(decisionTreeMod1, sub = "Rattle 2020-12-27 23:41:00 WernerGarcia")

predictTreeMod1 <- predict(decisionTreeMod1, testData, type = "class")
cmtree <- confusionMatrix(predictTreeMod1, testData$classe)
cmtree

plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Decision Tree - Accuracy =", round(cmtree$overall['Accuracy'], 4)))

finalpredict <- predict(decisionTreeMod1, newdata=testingData, type = "class")
finalpredict
