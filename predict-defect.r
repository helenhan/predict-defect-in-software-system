#  *------------------------------------------------------------------*
#  | prepare train data
#  *------------------------------------------------------------------* 
 
traindata <- read.csv('/project/data/data.csv')

head(traindata)
 
#  *------------------------------------------------------------------*
#  | train the network
#  *------------------------------------------------------------------* 
 
library(neuralnet)
 
nn <- neuralnet(
 recession~spread+spindex+cbindex,
 data=traindata, hidden=2, err.fct="sse",learningrate = 0.01,algorithm="backprop",
 linear.output=FALSE)
 
#  *------------------------------------------------------------------*
#  | output training results
#  *------------------------------------------------------------------*  
 
# basic
nn
 
# reults options
names(nn)
 
 
# result matrix
nn$result.matrix
 
# The given data is saved in nn$covariate and
# nn$response as well as in nn$data for the whole data
# set inclusive non-used variables. The output of the
# neural network, i.e. the fitted values o(x), is provided
# by nn$net.result:
 
out <- cbind(nn$covariate,nn$net.result[[1]])
 
#dimnames(out) <- list(NULL, c("date", "recession","spread"))
 
head(out)
 
# generalized weights
 
# The generalized weight expresses the effect of each
# ovariate xi and thus has an analogous interpretation
# as the ith regression parameter in regression models.
# However, the generalized weight depends on all
# other covariates. Its distribution indicates whether
# the effect of the covariate is linear since a small variance
# suggests a linear effect
 
# The columns refer to the four covariates age (j =
# 1), parity (j = 2), induced (j = 3), and spontaneous (j=4)
 
head(nn$generalized.weights[[1]])
 
# visualization
 
plot(nn)
gwplot(nn)

par(mfrow=c(2,2))
gwplot(nn,selected.covariate="spread",min=-2.5, max=5)
gwplot(nn,selected.covariate="spindex",min=-2.5, max=5)
gwplot(nn,selected.covariate="cbindex",min=-2.5, max=5)

prediction(nn)
summary(nn)

#  *------------------------------------------------------------------*
#  | test data
#  *------------------------------------------------------------------* 


testdata <- read.csv('/project/data/tdata.csv')

y <- compute(nn,testdata[,3:5])$net.result

out<-cbind(testdata[,1:2],y)

print(out)

plot(y ~ testdata[,2])

(nmse2 <- mean((y-testdata[,2])^2)/mean((mean(testdata[,2])- testdata[,2])^2)) # 0.06823054333

#  *------------------------------------------------------------------*
#  | test Canada data
#  *------------------------------------------------------------------* 

cadata <- read.csv('/project/data/catest.csv')
y2 <- compute(nn,cadata[,3:5])
out<-cbind(cadata[,1:2],y2$net.result)
print(out)
plot(y2$net.result ~ cadata[,2:2])



#  *------------------------------------------------------------------*
#  | feature selection
#  *------------------------------------------------------------------*
install.packages("caret", repos="http://R-Forge.R-project.org")
library(caret)

feature <- read.csv('/project/data/feature.csv')
mdrrDescr <- feature[,3:12]
mdrrClass <- feature[,2]
Process <- preProcess(mdrrDescr)
str(mdrrDescr)
newdata3 <- predict(Process, mdrrDescr)

# install.packages("randomForest")
profile <- rfe(newdata3,mdrrClass,sizes = c(1,2,3,4,5,6,7,8,9,10),rfeControl = rfeControl(functions=rfFuncs,method='cv'))
plot(profile,type=c('o','g'))
print(profile)

#  *------------------------------------------------------------------*
#  | data partition
#  *------------------------------------------------------------------*
data <- read.csv('/project/data/data.csv')
inTrain <- createDataPartition(y = data$recession,p = .75,list = FALSE)
training <- data[ inTrain,]
testing <- data[-inTrain,]
nrow(training)
nrow(testing)

#  *------------------------------------------------------------------*
#  | train with nnet
#  *------------------------------------------------------------------*
ctrl <- trainControl(method = "repeatedcv",repeats = 3,summaryFunction = twoClassSummary,classProbs = TRUE)
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))
#myGrid <- data.frame(layer1 = 3, layer2=0,layer3=0)
#nn <- train(recession~spread+spindex+cbindex, data = training, method = "neuralnet",tuneGrid=myGrid,metric="ROC" trControl = ctrl,preProc = "range")
training$recession<-as.factor(training$recession)

set.seed(1)
nn <- train(recession~spread+spindex+cbindex, data = training, method = "nnet",tuneGrid=my.grid,metric="ROC" ,trControl = ctrl,preProc = "range")
nnresult <- predict(nn, newdata = testing)
nn
plot(nn)


#threshold <- 0.5
#pred <- factor( ifelse(nnresult > threshold, "1", "0") )
#pred <- relevel(pred, "1")
confusionMatrix(nnresult, testing$recession)


#  *------------------------------------------------------------------*
#  | train with svm
#  *------------------------------------------------------------------*
set.seed(2)
svmTune <- train(recession~spread+spindex+cbindex, data = training,method = "svmRadial",tuneLength = 9,preProc = c("center", "scale"),metric = "ROC",trControl = ctrl)
svmTune
plot(svmTune)
svmresult <- predict(svmTune, newdata = testing)
confusionMatrix(svmresult, testing$recession)

#  *------------------------------------------------------------------*
#  | train with knn
#  *------------------------------------------------------------------*
set.seed(3)
knnTune <- train(recession~spread+spindex+cbindex, data = training,method = "knn",tuneLength = 12,preProc = c("center", "scale"),metric = "ROC",trControl = ctrl)
knnTune
plot(knnTune)
knnresult <- predict(knnTune, newdata = testing)
plot(knnresult)
confusionMatrix(knnresult, testing$recession)

#  *------------------------------------------------------------------*
#  | resample models
#  *------------------------------------------------------------------*
cvValues <- resamples(list(NNET = nn, SVM = svmTune, KNN = knnTune))
summary(cvValues)
splom(cvValues, metric = "ROC")
xyplot(cvValues, metric = "ROC")
parallelplot(cvValues, metric = "ROC")
dotplot(cvValues, metric = "ROC")

#  *------------------------------------------------------------------*
#  | compare with models
#  *------------------------------------------------------------------*
rocDiffs <- diff(cvValues, metric = "ROC")
summary(rocDiffs)
dotplot(rocDiffs, metric = "ROC")


plot(nnresult)
ggplot(testing) + geom_point(aes(date,nnresult,color = "predictive")) +geom_point(aes(date,recession,color = "actual"))
# nn <- train(recession~spread+spindex+cbindex, data = data, method = "neuralnet", algorithm = 'backprop', learningrate = 0.01, hidden = layer-layer3, trControl = ctrl, linout = FALSE)

