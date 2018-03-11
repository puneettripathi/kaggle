##### Import Libraries #####
library(MASS)
library(car)
library(e1071)
library(caret)
library(ggplot2)
library(cowplot)
library(caTools)
library(plyr) #data wrangling
library(dplyr) #data wrangling

setwd("D:/pgdds/kaggle/titanic")
##### reading titanic dataset #####
titanic = read.csv("train.csv", stringsAsFactors = F)

# Creating derived metrics
titanic$family_size <- titanic$SibSp+titanic$Parch

titanic$fare_per_person <- titanic$Fare/(titanic$family_size + 1)

titanic$traveling_alone <- ifelse(titanic$family_size == 0, 'Yes', 'No')

# Treating Missing Values 
cols_with_na <- colnames(titanic)[colSums(is.na(titanic)) > 0]

titanic$Age[is.na(titanic$Age)] <- median(titanic$Age, na.rm = TRUE)

titanic$Embarked[is.na(titanic$Embarked)] <- 'S'

titanic$CabinAvl <- ifelse(titanic$Cabin == "", "No", "Yes")

titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)
VIP <- c("Capt","Col","Don","Dona","Dr","Jonkheer","Lady","Major",
         "Mlle", "Mme","Rev","Sir","the Countess")

titanic$Title[titanic$Title %in% VIP] <- "VIP"


titanic$is_Adult <- ifelse(titanic$Age <= 18, "No", "Yes")

# Creating ticketSize
titanic <- ddply(titanic,.(Ticket),transform,Ticketsize=length(Ticket))

str(titanic)

##### EDA ####
bar_theme1<- theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
                   legend.position="none")

plot_grid(ggplot(titanic, aes(x=Sex,fill=factor(Survived)))+ geom_bar(),
          ggplot(titanic, aes(x=Embarked,fill=factor(Survived)))+ geom_bar()+bar_theme1,
          ggplot(titanic, aes(x=factor(Parch),fill=factor(Survived)))+ geom_bar()+bar_theme1,
          ggplot(titanic, aes(x=factor(SibSp),fill=factor(Survived)))+ geom_bar()+bar_theme1,
          ggplot(titanic, aes(x=factor(Pclass),fill=factor(Survived)))+ geom_bar()+bar_theme1,
          align = "h")

plot_grid(ggplot(titanic, aes(x=factor(family_size),fill=factor(Survived)))+ geom_bar()+bar_theme1,
          ggplot(titanic, aes(x=factor(traveling_alone),fill=factor(Survived)))+ geom_bar()+bar_theme1,
          align = "h")
# People who are not traveling alone have a higher chance of survival

ggplot(titanic, aes(x=Title,fill=factor(Survived))) + 
  geom_bar() +
  facet_wrap(~Pclass) + labs(title="Survivor split by class and Title") +
  bar_theme1

ggplot(titanic, aes(x=factor(Ticketsize),fill=factor(Survived))) + 
  geom_bar() +
  bar_theme1

##### Data Preparation #####
titanic$Age <- scale(titanic$Age)
titanic$fare_per_person = scale(titanic$fare_per_person)

cor(titanic$Fare, titanic$fare_per_person)


# Removing columns which are not needed
titanic <- titanic[, !(colnames(titanic) %in% c('Name', 'Cabin', 'Fare', 'Ticket', 'SibSp', 'Parch'))]

# Convert categorical columns to factor
titanic$Pclass = factor(titanic$Pclass)

str(titanic)


factor_var <- c("Sex", "Title", "Embarked", "traveling_alone", "CabinAvl", "Pclass", "is_Adult")
dummy_input <- titanic[,factor_var]
num_input <- titanic[,! colnames(titanic) %in% factor_var]

#creating dummy vairable
dummies <- data.frame(sapply(titanic, 
                             function(x) data.frame(model.matrix(~x-1,data = titanic))[,-1]))

# column bind dummies and num_input
master <- cbind(num_input,dummies)

# Create train test
set.seed(7)

#indices = sample.split(master$Survived, SplitRatio = 0.8)

train = master

test = read.csv('test.csv')
test$family_size <- test$SibSp+test$Parch

test$fare_per_person <- test$Fare/(test$family_size + 1)

test$traveling_alone <- ifelse(test$family_size == 0, 'Yes', 'No')

# Treating Missing Values 
cols_with_na <- colnames(test)[colSums(is.na(test)) > 0]

test$Age[is.na(test$Age)] <- median(test$Age, na.rm = TRUE)

test$Embarked[is.na(test$Embarked)] <- 'S'

test$CabinAvl <- ifelse(test$Cabin == "", "No", "Yes")

test$Title <- gsub('(.*, )|(\\..*)', '', test$Name)
VIP <- c("Capt","Col","Don","Dona","Dr","Jonkheer","Lady","Major",
         "Mlle", "Mme","Rev","Sir","the Countess")

test$Title[test$Title %in% VIP] <- "VIP"


test$is_Adult <- ifelse(test$Age <= 18, "No", "Yes")

# Creating ticketSize
test <- ddply(test,.(Ticket),transform,Ticketsize=length(Ticket))

str(test)
test$Age <- scale(test$Age)
test$fare_per_person = scale(test$fare_per_person)

cor(test$Fare, test$fare_per_person)


# Removing columns which are not needed
test <- test[, !(colnames(test) %in% c('Name', 'Cabin', 'Fare', 'Ticket', 'SibSp', 'Parch'))]

# Convert categorical columns to factor
test$Pclass = factor(test$Pclass)

str(test)


factor_var <- c("Sex", "Title", "Embarked", "traveling_alone", "CabinAvl", "Pclass", "is_Adult")
dummy_input <- test[,factor_var]
num_input <- test[,! colnames(test) %in% factor_var]

#creating dummy vairable
dummies <- data.frame(sapply(dummy_input, 
                             function(x) data.frame(model.matrix(~x-1,data = dummy_input))[,-1]))

# column bind dummies and num_input
testf <- cbind(num_input,dummies)


# Logistic Regression: 

#Initial model
model_1 = glm(Survived ~ ., data = train, family = "binomial")
summary(model_1) 

# Stepwise Selection
model_2<- stepAIC(model_1, direction="both")


model_2
summary(model_2)

# Removing multicollinearity through VIF check
vif(model_2)

model_3 <- glm(formula = Survived ~ Age + family_size + fare_per_person + 
                 Pclass.x3 + Embarked.xS + CabinAvl + Title.xMiss + 
                 Title.xMr + Title.xMrs + Title.xVIP, family = "binomial", 
               data = train)
summary(model_3)

# Removing multicollinearity through VIF check
vif(model_3)

model_4 <- glm(formula = Survived ~ Age + family_size + fare_per_person + 
                 Pclass.x3 + Embarked.xS + CabinAvl +
                 Title.xMr + Title.xMrs + Title.xVIP, family = "binomial", 
               data = train)
summary(model_4)

# Removing multicollinearity through VIF check
vif(model_4)

model_5 <- glm(formula = Survived ~ Age + family_size + 
                 Pclass.x3 + Embarked.xS + CabinAvl +
                 Title.xMr + Title.xMrs + Title.xVIP, family = "binomial", 
               data = train)
summary(model_5)


# final model
final_model<- model_5

#######################################################################

### Model Evaluation

### Test Data ####

#predicted probabilities of Churn 1 for test data

test_pred = predict(final_model, type = "response", 
                    newdata = testf)


# Let's see the summary 

summary(test_pred)

test$prob <- test_pred
View(test)

test_pred_survived <- factor(ifelse(test_pred >= 0.50, "Yes", "No"))
test_actual_survived <- factor(ifelse(test$Survived==1,"Yes","No"))


table(test_actual_survived,test_pred_survived)


#######################################################################
perform_fn <- function(cutoff) 
{
  predicted_survived <- factor(ifelse(test_pred >= cutoff, "Yes", "No"))
  conf <- confusionMatrix(predicted_survived, test_actual_survived, positive = "Yes")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  out <- t(as.matrix(c(sens, spec, acc))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}

# Creating cutoff values from 0.003575 to 0.812100 for plotting and initiallizing a matrix of 100 X 3.

# Summary of test probability

summary(test_pred)

s = seq(.01,.80,length=100)

OUT = matrix(0,100,3)


for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
} 


plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))


cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.01)]
cutoff

test_cutoff_survived <- factor(ifelse(test_pred >=0.4, "1", "0"))

conf_final <- confusionMatrix(test_cutoff_survived, test_actual_survived, positive = "Yes")

acc <- conf_final$overall[1]

sens <- conf_final$byClass[1]

spec <- conf_final$byClass[2]

acc

sens

spec

test_cutoff_survived <- factor(ifelse(test_pred >=0.64, "1", "0"))
submission = data.frame(testf$PassengerId,test_cutoff_survived)
colnames(submission) <- c("PassengerId", "Survived")
write.csv(submission, "submission.csv", row.names = FALSE)

