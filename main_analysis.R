### Predictive modelling on boston housing data
### Elio Amicarelli

### Load packages
#
library(MASS)
library(car)
library(caret)
library(MuMIn)
library(randomForest)
library(xgboost)

### Data and summaries
#
data<-data.frame(Boston)
data$chas<-as.factor(data$chas)
summary(data)
hist(data$medv,main="Response",xlab="medv", breaks=20,col="black",probability=T)
lines(density(data$medv),col="violet")

### Train/test split
#
set.seed(1)
j.train<-sample(1:nrow(data), nrow(data)*0.8)
data.train<-data[j.train,]
data.test<-data[setdiff(1:nrow(data),j.train),]

### GLM
#
glm.naive<-glm(medv ~ ., data=data.train,family=gaussian(link = "identity"))
vif(glm.naive) # Variance inflation factors
glm.naive<-glm(medv ~ .-tax, data=data.train,family=gaussian(link = "identity")) # updated glm
vif(glm.naive) #  way better, this is going to be more stable during model selection
#
## model selection
options(na.action="na.fail")
drAICc <-dredge(glm.naive, rank="AICc")
head(drAICc)
model.glm<-glm(medv ~ .-tax -age -indus,data=data.train,family=gaussian(link="identity"))
#
## check and fix functional forms
par(mfrow=c(2,2))
termplot(model.glm, partial.resid=TRUE, pch=19,col.res = rgb(0,0,0,0.1),
         col.term="blue", smooth=panel.smooth, col.smth = "red",lty.smth=1,
         se=T,col.se = "blue")
#
data.train$rm2<-(1+1)/(max(data.train$rm)-min(data.train$rm))*
  (data.train$rm-max(data.train$rm))+1
data.train$rm2<-data.train$rm2^2
data.train$lstat2<-log(data.train$lstat)
data.train$crim2<-log(data.train$crim)
#
## Final GLM
workingmodel.glm<-glm(medv ~ zn + chas + nox + dis + rad + ptratio + black + lstat2 + 
                        rm2 + crim2,data=data.train,family=gaussian(link = "identity"))
AIC(workingmodel.glm, model.glm )
finalmodel.glm<-workingmodel.glm
#
## Model checking
#
par(mfrow=c(1,2))
plot(data.train$medv,finalmodel.glm$fitted.values, main="GLM - Train Set", 
     ylab="Predicted",xlab="Observed", pch=19, col=rgb(0,0,0,0.1))
abline(0,1, col="violet")

plot(data.train$medv,finalmodel.glm$residuals, main="GLM - Train Set", ylab="Residuals",
     xlab="Response", pch=19, col=rgb(0,0,0,0.1))
abline(h=0, col="violet")
lines(lowess(data.train$medv,finalmodel.glm$residuals), col="blue")
#
par(mfrow=c(2,2))
termplot(finalmodel.glm, partial.resid=TRUE, pch=19,col.res = rgb(0,0,0,0.1),col.term="blue", 
         smooth=panel.smooth, col.smth = "red",lty.smth=1,se=T,col.se = "blue")
#
par(mfrow=c(1,2))
hist(finalmodel.glm$residuals, main="GLM -Train Set",xlab="Residuals", 
     breaks=20,col="black",probability=T)
lines(density(finalmodel.glm$residuals),col="violet")
qqnorm(finalmodel.glm$residuals, main="GLM - Train Set", pch=19, col=rgb(0,0,0,0.1))
qqline(finalmodel.glm$residuals,col="violet")
#
## GLM predictive performances
data.test$rm2<-(1+1)/(max(data.test$rm)-min(data.test$rm))*
  (data.test$rm-max(data.test$rm))+1
data.test$rm2<-data.test$rm2^2
data.test$lstat2<-log(data.test$lstat)
data.test$crim2<-log(data.test$crim)
glm.preds<-predict(finalmodel.glm, data.test)
plot(data.test$medv, main="GLM - Test Set", ylab="medv",type="l", col= "skyblue")
lines(glm.preds,col="violet")
legend(0, 50, c("Observed","Predicted"), lty=c(1,1), col=c("skyblue","violet"))
GLM.MSE<-mean((data.test$medv-glm.preds)^2)
cat("GLM out of sample MSE:",GLM.MSE)

### RF
#
# Clean and recode data
data.train<-data.train[,!names(data.train)%in%c("lstat2","rm2","crim2")]
data.test<-data.test[,!names(data.test)%in%c("lstat2","rm2","crim2")]
data.train$chas<-as.numeric(data.train$chas)
data.train$rad<-as.numeric(data.train$rad)
data.test$chas<-as.numeric(data.test$chas)
data.test$rad<-as.numeric(data.test$rad)
#
# settings for grid search and validation
rf.ctrl <-trainControl(method="repeatedcv",
                       number=10,
                       repeats=10,
                       verboseIter=TRUE,
                       savePredictions = TRUE)
rf.myGrid<-expand.grid(mtry=seq(1,13,by=1))
#
# train
set.seed(1)
rf.modelspace = train(medv ~ ., data = data.train, 
                      method = "rf",        
                      trControl = rf.ctrl, 
                      tuneGrid = rf.myGrid,
                      ntree=500,
                      metric="RMSE")
finalmodel.rf<-rf.modelspace$finalModel
#
plot(rf.modelspace$results$mtry,rf.modelspace$results$RMSE, main="RF - Train Set",ylab="RMSE (Repeated Cross-Validation)", xlab="mtry", col="violet")
lines(rf.modelspace$results$mtry,rf.modelspace$results$RMSE, col="violet")
abline(v=rf.modelspace$results$mtry[rf.modelspace$results$RMSE==min(rf.modelspace$results$RMSE)], lty=2, col="darkgrey")
#
# model checking
par(mfrow=c(1,2))
plot(data.train$medv,finalmodel.rf$predicted, main="RF - Train Set", 
     ylab="Predicted",xlab="Observed", pch=19, col=rgb(0,0,0,0.1))
abline(0,1, col="violet")
plot(data.train$medv,data.train$medv-finalmodel.rf$predicted, main=" RF - Train Set", 
     ylab="Residuals",xlab="Response", pch=19, col=rgb(0,0,0,0.1))
abline(h=0, col="violet")
lines(lowess(data.train$medv,data.train$medv-finalmodel.rf$predicted),col="blue")
#
par(mfrow=c(1,2))

hist(data.train$medv-finalmodel.rf$predicted,main="RF -Train Set",xlab="Residuals",
     breaks=20,col="black",probability=T)
lines(density(data.train$medv-finalmodel.rf$predicted),col="violet")

qqnorm(data.train$medv-finalmodel.rf$predicted, main="RF - Train Set",
       pch=19, col=rgb(0,0,0,0.1))
qqline(data.train$medv-finalmodel.rf$predicted,col="violet")
#
# RF predictive performances
rf.preds<-predict(finalmodel.rf, data.test)
plot(data.test$medv, main="GLM, RF - Test Set", ylab="medv",type="l", col= "skyblue")
lines(glm.preds,col="violet")
lines(rf.preds,col="green")
legend(0, 50, c("Observed","Predicted GLM", "Predicted RF"), lty=c(1,1,1), 
       col=c("skyblue","violet", "green"))
RF.MSE<-mean((data.test$medv-rf.preds)^2)
cat("GLM out of sample MSE: ",GLM.MSE,"\n",
    "RF out of sample MSE: ",RF.MSE, sep="")

### GBM
#
# Train
gbm.ctrl<- trainControl(
  method = "repeatedcv",
  number= 10,
  repeats = 10,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",
  allowParallel = TRUE
)
gbm.myGrid<-expand.grid(
  nrounds = seq(1000, 10000, 100),
  eta = c(0.01, 0.001),
  max_depth = seq(1:6),
  gamma = 1,
  colsample_bytree=1,
  min_child_weight = 1
)
set.seed(1)
gbm.modelspace = train(medv ~ ., data = data.train, 
                       method = "xgbTree",        
                       trControl = gbm.ctrl, 
                       tuneGrid = gbm.myGrid,
                       metric="RMSE")
plot(gbm.modelspace)
#
# predictive performances
gbm.preds<-predict(gbm.modelspace, data.test)
plot(data.test$medv, main="GLM, RF, GBM - Test Set",ylab="medv",type="l", col= "skyblue")
lines(glm.preds,col="violet")
lines(rf.preds,col="green")
lines(gbm.preds,col="orange")
legend(0, 50, c("Observed","Predicted GLM", "Predicted RF", "Predicted GBM"), 
       lty=c(1,1,1), col=c("skyblue","violet", "green", "orange"))
GBM.MSE<-mean((data.test$medv-gbm.preds)^2)
cat("GLM out of sample MSE: ",GLM.MSE,"\n",
    "RF out of sample MSE: ",RF.MSE,"\n",
    "GBM out of sample MSE: ",GBM.MSE, sep = "")