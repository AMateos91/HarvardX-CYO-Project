## Credit Card Fraud Detection Project 2##

# HarvardX Data Science Professional Certificate Program #

# Module 9 - Capstone: Choose Your Own Project #

# GitHub repo: https://github.com/AMateos91

library(dplyr)
library(tidyr)
library(tidyverse)
library(data.table)
library(stringr)
library(caTools)
library(rpart)
library(rpart.plot)
library(pROC)
library(gbm)
library(neuralnet) 
library(ranger)
library(car)
library(caret)
library(data.table)
dataset <- read.csv("C:/Users/Usuario/Desktop/creditcard.csv")

# Data exploration / cleaning #

dim(dataset)
head(dataset, 5)
names(dataset)
var(dataset$Amount)
summary(dataset$Amount)
table(dataset$Class)

##Number of missing values in each variable:
colSums(is.na(dataset))

##Replacing NA values by 0:
dataset[is.na(dataset)] <- 0

head(dataset, 10)

##Means
mean(dataset$Amount, na.rm = TRUE)
mean(dataset$Class, na.rm = TRUE)

df = dataset
summary(df)
head(df, 5)
boxplot(df)
  
##Winsorize: When an outlier is negatively impacting a model results, it is possible to replace this with a less extreme maximum value. 
#In Winsorizing, values located out of a predetermined percentile range of the data are identified and set to this percentile.
#Rather, winsorizing a vector means a predefined quantum of the smallest and/or the largest values is replaced instead by less extreme values. 
#Thus, the substitution values are the most extreme retained values in reference to those ones above 95th percentile. 
install.packages("robustHD", repos="https://cran.rstudio.com")
require(robustHD)
sum(df$Amount > quantile(df$Amount, .95))
df <- df %>% mutate(wins_total_amount = winsorize(Amount))
head(df, 5)

##########################################
  
#Data visualization 
  
# visualizing the distribution of transactions across time
df %>%
ggplot(aes(x = Time, fill = factor(Class))) + 
geom_histogram(bins = 100) + 
labs(x = "Time elapsed since first transaction (seconds)", 
     y = "Number of transactions", 
     title = "Distribution of the transactions during time") +
facet_grid(Class ~ ., scales = 'free_y') + theme()

# correlation of anonymous variables with amount and class
#install.packages("corrplot")
correlation <- cor(df[, -1], method = "pearson")
corrplot::corrplot(correlation, number.cex = 1, 
                   method = "color", type = "full", 
                   tl.cex=0.7, tl.col="black")

#####################################
  
# Data wrangling #
  
df$Amount <- scale(df$Amount)
df_1 <- df[,-c(1)]
head(df_1)

# Data modeling #

set.seed(123)
split <- sample.split(df_1$Class, SplitRatio=.70)
train <- df_1[split==TRUE, ]
test <- df_1[split==FALSE, ]
dim(train) 
dim(test)

# Decision Tree model #

dt_model <- rpart(Class~., df, method = 'class')
predicted <- predict(dt_model, df, type = 'class')
probability <- predict(dt_model, df, type = 'prob')
rpart.plot(dt_model)

# Logistic regression model #

lr_model <- glm(Class~., train, family=binomial()) 

summary(lr_model)
plot(lr_model)

predicted_2 <- predict(lr_model,test, probability = TRUE)

auc_curve <- roc(test$Class, predicted_2, plot= TRUE, col="green")

##################################################
#This is just other possible algorithm model to implement instead of the previous
#LR model. Nevertheless, it does not fit at the most the regression problem.

#Linear Discriminant Analysis (LDA) #
  
#lda<- train(Class~., data=train, method = "lda", metric = accuracy, trControl=control)
#summary(lda)
#plot(lda)

#prediction <- predict(lda, train, probability = TRUE)
#auc_curve = roc(test$Class, prediction, plot = TRUE, col = "green")

###################################################

# Artificial Neural Network #

nn_model <- neuralnet(Class~., train, linear.output=FALSE)
plot(nn_model)

predicted_3 <- compute(nn_model, test)
result_nn_model <- predicted_3$net.result
result_nn_model <- ifelse(result_nn_model>0.5, 1, 0)
  
#XGBoosting regression model #

set.seed(9560)
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
require(xgboost)

labels <- train$Class

xgb_fit <- xgboost(data = data.matrix(train[,-30]), 
                   label = labels,
                   eta = 0.1,
                   gamma = 0.1,
                   max_depth = 10, 
                   nrounds = 300, 
                   objective = "binary:logistic",
                   colsample_bytree = 0.6,
                   verbose = 0,
                   nthread = 7
)
# XGBoost predictions
xgb_pred <- predict(xgb_fit, data.matrix(test[,-30]))
curve <- roc(test$Class, xgb_pred, plot = TRUE)
curve
plot(curve)

xgb_pred <- predict(xgb_fit, data.matrix(train[,-30]))
curve <- roc(train$Class, xgb_pred, plot = TRUE)
curve
plot(curve)