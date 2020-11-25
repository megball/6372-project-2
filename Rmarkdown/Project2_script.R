#title: "r6 + MB edits"
#author: "Simerpreet & Megan & Rinku"
#date: "November 20, 2020"

###################
## Load Packages ##
###################

library(dplyr)
library(tidyverse)
library(ggplot2)
library(caret)
library(e1071)
library(class)
library(gridExtra)
library(summarytools)
library(gt)
library(corrplot)
library(janitor)
library(tidyselect)
library(GGally)
library(randomForest)
library(car)
library(ROCR)
library(MASS)
library(glmnet)

################
## Load Data ##
###############

#full <- read_delim(here::here("data", "bank-additional-full.csv"),';')
full <- read.csv(file.choose(), sep=';')
str(full)
head(full)
nrow(full) 
ncol(full)

# Clean up column names
full <- janitor::clean_names(full)
summary(full)
#print(dfSummary(full, graph.magnif = 0.75), method = 'browser')
str(full)
# Check for missing values
tibble(variable = names(colSums(is.na(full))),
       missing = colSums(is.na(full))) %>% 
  gt() %>% 
  tab_header(title = "Missing Values in Data") 


#Looking at the dfsummary, there doesnt seem to be missing data in terms of just not having values. However, there are some fields that have explicit unknown or non-existent classes that could be considered as 'missing'. For example, loan and housing have 990 "unknown" values. And 'default' has 8597 "unknown" values representing 20.9% 

#pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

# Remove missing values
#```{r}
#remove "unknowns" based on small sample sizes compared to full data set
df <- full %>%  filter(loan != "unknown")
nrow(df)
#down to 40,198 obs
df <- df %>%  filter(marital != "unknown")
nrow(df)
#down to 40,119 obs
df <- df %>%  filter(education != "unknown")
nrow(df)
#down to 38,437 obs

#Simer: remove unknowns from job
df <- df %>%  filter(job != "unknown")
nrow(df)
#down to 38,245 obs
#Remove column default from the analysis. REasons for removing the columns: 
#1) After cleaning up the data set of the other 'unknowns', this column has 7,757 unknown values as well. With only 3 values as 'yes',and 30,485 as 'no', it is difficult to impute values
#2) Practically, column default would need to be used before the campaign, the sales person needs to decide if the person wilth a default needs to be approached to or not, not after the campaign,
#so it appears that it is ok to let go of this column when predicting the outcome of the campaign. 

#Keeping default in for now
#df <- df %>%  dplyr::select(age:education,housing: y) #for some reason select(default is not working for me)
#Closing Simer's changes
str(df)
#recheck summary
summary(df)


#```
#Simer changed the number of decreased 'yes' from 400 to 480 and the final number as well
message ("Our yes group has decreased by about ~480 to 4,258.")
#Our yes group has decreased by about ~400 to 4,277.

#```{r}
#change some variables to factor
#Simer removed default from the list of columns
#cols <- c("job", "marital", "education", "default","housing","loan","contact","month","day_of_week","poutcome","y")
cols <- c("job", "marital", "education", "housing","loan","contact","month","day_of_week","poutcome","y")
df[cols] <- lapply(df[cols], factor) 
str(df)
#make sure "success" level is defined as "yes"
str(df$y)
#```

################################
## Exploratory Data Analysis ##
################################

#run first pass PCA to see if we have useful numeric predictors
df.numeric <- df[ , sapply(df, is.numeric)]

pc.result<-prcomp(df.numeric,scale.=TRUE)
pc.scores<-pc.result$x
pc.scores<-data.frame(pc.scores)
pc.scores$y<-df$y
pc.scores

#Scree plot
eigenvals<-(pc.result$sdev)^2
eigenvals
plot(1:8,eigenvals/sum(eigenvals),type="l",main="Scree Plot PC's",ylab="Prop. Var. Explained",ylim=c(0,1))
cumulative.prop<-cumsum(eigenvals/sum(eigenvals))
lines(1:8,cumulative.prop,lty=2)

#Use ggplot2 to plot the first few pc's
ggplot(data = pc.scores, aes(x = PC1, y = PC2)) +
  geom_point(aes(col=y), size=1)+
  ggtitle("PCA of Numeric Data")
#There is some separation, but it is not in a way we would hope for our response variable

ggplot(data = pc.scores, aes(x = PC2, y = PC3)) +
  geom_point(aes(col=y), size=1)+
  ggtitle("PCA of Numeric Data")

ggplot(data = pc.scores, aes(x = PC3, y = PC4)) +
  geom_point(aes(col=y), size=1)+
  ggtitle("PCA of Numeric Data")

#ggpairs(df,columns=1:18, aes(colour=y))

ggpairs(df,columns=3:7, aes(colour=y))

ggpairs(df, columns=14:18, aes(colour=y))

df_yes <- df %>%   filter(y=="yes")
#summary(df_yes)
# Nothing interesting found in the below code so commenting it out
# ggplot(bank_additional_full, aes(x=age, y=emp.var.rate)) +
#   geom_point(size=1, shape="circle") +
#   ggtitle("Employment Variation Rate vs Age") + 
#   facet_wrap(~ y)
ggplot(df, aes(x=age, y=duration, color = y)) +  geom_point(size=1, shape="circle") +   ggtitle("Duration vs Age")

message("Duration vs Age: The duration of last contact (in seconds) was longer for ages 25-50. And it was understandably longer for 'yes'' vs for 'no'. ")

#ggplot(df, aes(x=age, y=cons_price_idx, color = y)) +  geom_point(size=1, shape="circle") +   ggtitle("Consumer Price Index vs Age")


#Checking collinearlity using box plots
#Simer: Added box plot for Consumer Price Index vs Age
ggplot(df, aes(x=age, y=cons_price_idx, color = y)) +  geom_boxplot() +   ggtitle("Consumer Price Index vs Age") 

#Simer: Added box plot for Age vs. duration
ggplot(df, aes(x=duration , y=age, color = y)) +  geom_boxplot() +   ggtitle("Age vs. duration")


#Simer: Added box plot for cons.price.idx vs. cons.conf.idx
ggplot(df, aes(x=cons_price_idx , y=cons_conf_idx, color = y)) +  geom_boxplot() +   ggtitle("cons.price.idx vs. cons.conf.idx")


#Simer: Added box plot for cons.price.idx vs. emp.var.rate
ggplot(df, aes(x=cons_price_idx , y=emp_var_rate, color = y)) +  geom_boxplot() +   ggtitle("cons.price.idx vs. emp.var.rate")

#Simer: Analysing nr.employed
ggplot(df) + geom_histogram(mapping = aes(x=nr_employed, fill=y)) +ggtitle("Distribution of 'y' by nr.employed")

# ggplot(bank_additional_full, aes(x=age, y=education)) +
#   geom_point(size=1, shape="circle") + 
#   ggtitle("Education vs Age")  + 
#    facet_wrap(~ y)


#Analysing Age
ggplot(df) + geom_histogram(mapping = aes(x=age, fill=y)) +ggtitle("Distribution of 'y' by age")


#Creating new variables
#Age_Grp - split the data into age groups "17-31","32-37" ,"38-47", "47-55", ">55" (based in IQR)
df$Age_Grp <- cut(df$age, breaks = c(16,31,37,46,55,98), labels = c("17-31","32-37" ,"38-47", "47-55", ">55"))
#validate the cut command
#df %>% filter(!$Age_Grp  %in% c("17-31","32-37" ,"38-47", "47-55", ">55"))
#df %>% filter(df$age==55)
ggplot(df) + geom_bar(mapping = aes(x=Age_Grp, fill = y)) + ggtitle("Distribution of 'y' by Age_Grp") +  ylab("Cnt") + xlab("Age Group")


message("We will keep both age and age group in our model to see if one is selected over the other. We need to make sure to not use both in our model building.")


#Analyzing pdays
ggplot(df) + geom_histogram(mapping = aes(x=pdays, fill=y))


message("Analyzing 'pdays' ie., number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted")
message(" Most folks had no previous campaign but if they did, it looks like most who had a previous campaign decided to subscribe")

#zoom in for ones that were previously contacted
df %>%  filter(pdays < 999) %>%   ggplot() +  geom_histogram(mapping = aes(x=pdays, fill=y))

message(" Highest frequency appears to be less than 10 days since last contact. Let''s make this into a y/n variable instead due to the large gap between days contacted and the '999' variable.")


message(" prevly_Cntctd Yes/No. TO see the distribution or 'Y' on first time contact vs. a follow up")
df$prevly_Cntctd <- as.factor(case_when(df$pdays==999 ~ "No", !df$pdays==999 ~ "Yes"))
#Validate previously contacted variable
#df %>% filter(!df$pdays==999)
ggplot(df) + geom_bar(mapping = aes(x=prevly_Cntctd, fill = y)) + ggtitle("Number of 'y' by whether customers were prev.contacted or not") +
  ylab("Cnt") + xlab("Previously contacted?")


message(" Same observation here as above: Most folks had no previous campaign but if they did, it looks like most who had a previous campaign decided to subscribe / likely to say 'Yes'. Since we have now created a new variable dependent on pdays, proceed to remove pdays to avoid issues with multicollinearity. Additionally, poutcome is dependent on whether or not someone has previously been contacted, and we have 'nonexistent' at 86% of the data. Remove this variable as well since it doesn''t add much value and is dependent on pdays/previously contacted.")


#Simer: Let's keep poutcome, these are independent variables. Let's check in refression or by VIF''s to check collinearily betweent he two
#df <- df %>% select(-pdays)

#df <- df %>% select(-pdays, -poutcome)
summary(df)

#Analysing campaign
ggplot(df) +   geom_histogram(mapping = aes(x=campaign, fill=y)) +  ggtitle("Distribution of 'y' by campaign")

# Just visually, when we decided to stop contacting a person it didn''t affect our closing ratio which still dropped off precipitously 

# Ideally, the campaign would stop contacting people who are less likely to subscribe, and keep contacting people if they are more likely to subscribe.  Then we should see the ratio of Yes to No go up as the number of no contacts goes up.  Instead, it looks like the ratio stays the same and the number of Yes''s drops proportionately with the number of No''s. 

#Analyzing job
ggplot(df) +   geom_bar(mapping = aes(x=job, fill = y)) +   coord_flip() +     #Added coord flip here to make it more readable
  ggtitle("Number of 'y' by job") +  ylab("Count") +   xlab("Job")


#"y" - has a client subscribed a term deposit? : admin, technician and blue collar jobs are the top 3 subscribers by volume 


#df2 <- df %>%  group_by(job) %>%  count(y) %>%  mutate(job_conv = n/sum(n)) %>%  filter(y == "yes")
#ggplot(df2, aes(x=job, y=job_conv)) +  geom_point() +  coord_flip() 


#Above, I looked at the ratio of "yes" vs "no" and see that students and retired persons convert at much higher rates than those of other professions. And 'blue collar' has the lowest conversion rate

#So, if they were to want to improve the cost effectiveness of their campaigns they might want to target more 'students' and 'retirees'



#Analyzing marital
ggplot(data=df) +   geom_bar(mapping = aes(x=marital, fill = y)) +   ggtitle("Number of 'y' by marital") +  ylab("Cnt") +   xlab("marital")


# More 'married' people are represented in the campaign
#Visually looking, conversion rate seems to be higher for 'single' people


#Analyzing duration and creating duration group variable
summary(df$duration)
df$duration_group <-   cut(df$duration,       breaks = c(-Inf,100,60,300,600,Inf),       labels = c("0-30s", "30-60s", "1-5 min", "5-10min","10+ min"))
# Check for missing values
tibble(variable = names(colSums(is.na(df))),
       missing = colSums(is.na(df))) %>% 
  gt() %>% 
  tab_header(title = "Missing Values in Data")
df3 <- df %>%  group_by(duration_group) %>%  count(y) %>%  mutate(duration_group_conv = n/sum(n)) %>%  filter(y == "yes")
#ggplot(df3, aes(x=duration_group, y=duration_group_conv)) +  geom_point() +  facet_wrap(~ y)


#Looking above, clearly conversion rate goes up the longer the most recent call

# Checking for correlation

# Convert data to numeric
corrs <- data.frame(lapply(df, as.integer))
# Plot the graph
ggcorr(corrs,
       method = c("pairwise", "spearman"),
       nbreaks = 6,
       hjust = 0.8,
       label = TRUE,
       label_size = 3,
       color = "grey50")


#Based on the correlation plot above, we see high correlation between 'euribor3m' and 'emp_var_rate' 
#and to a lesser degree with 'nr_employed.' We also see 'nr_employed' and 'emp_var_rate' also highly 
#correlated, which makes sense since you would expect the number of employees to vary at the same time
#as the employment variation rate. We will use VIF and feature selection tools in our model building
#to determine which to remove.


#remove emp_var_rate
summary(df)
#df <- df %>% select(-emp_var_rate)
#Simer: COmmented out the above step. 


# Run random forest on down-sampled data set to check for variable importance 

#I am running RF on a subset of the data to do a gross check for important variables and to determine if the new variables duration group and age group are deemed more important than the continuous variables of just raw duration and raw age.


#move response variable to end of data set
df <- df %>% relocate(y, .after = last_col())
#randomly sample 10k obs
#sample10k <- sample_n(df, 10000)
#down sample to balance response
#set.seed(1)
#downsample <- downSample(x = sample10k[, -21],
#y = sample10k$y)
#table(downsample$Class)
#RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, verbose = FALSE)
#set.seed(123)
#subsets <- c(1:5, 10, 15, 20)
#RFresults <- rfe(downsample[,1:20], downsample[[21]], sizes=subsets, rfeControl=RFcontrol)
#RFresults
#varImp(RFresults)

#Based on this, remove 'duration_group' and 'age_group' as the continuous version of those variables had higher importance on the final model.
str(df)
#Simer
#RFcontrol <- rfeControl(functions=rfFuncs, method="cv", number=5, verbose = FALSE)
#set.seed(123)
#subsets <- c(1:5, 10, 15, 20)
#RFresults <- rfe(df[,1:20], df[[21]], sizes=subsets, rfeControl=RFcontrol)
#RFresults
#varImp(RFresults)


#df <- df %>% select(-duration_group, -Age_Grp)

#save dataset to this point
df_clean <- write.csv(df, "data/df_clean.csv", row.names = FALSE)

#open saved dataframe
#df <- read.csv(here::here("data", "df_clean.csv"), stringsAsFactors = TRUE)
#str(df)

######################
## Train/Test Split ##
######################

#Simer train/test split. Making sure the train and test set get enough 'yes' variables. 
summary(df)
#38245 obs. of 24 variables

set.seed(1234) 

df_yes <- df %>% filter(y=='yes')
df_No <- df %>% filter(y=='no')
num_rows_yes <- nrow(df_yes) #4,258
num_rows_no <- nrow(df_No) #33,987

train_idx_yes <- sample(1:num_rows_yes, 0.8 * num_rows_yes)
train_yes <- df_yes[train_idx_yes, ]
test_yes <- df_yes[-train_idx_yes, ]
nrow(train_yes) #3,406
nrow(test_yes)  #852

train_idx_no <- sample(1:num_rows_no, 0.8 * num_rows_no)
train_no <- df_No[train_idx_no, ]
test_no <- df_No[-train_idx_no, ]
nrow(train_no) #27,189
nrow(test_no)  #6798

train <- rbind(train_yes, train_no)
test <- rbind(test_yes, test_no)

nrow(train) #30,595
nrow(test) #7,650

nrow(train %>% filter(y=='yes')) #3,406
nrow(test %>% filter(y=='yes'))  #852

summary(train)
#30595 obs. of 24 variables


##############################
### Simple Logistic Model ###
##############################

# Run Initial Logistic Regression
#Simple regression model
simple.log<-glm(y~.,family="binomial",data=train)
summary(simple.log)
exp(cbind("Odds ratio" = coef(simple.log), confint.default(simple.log, level = 0.95)))
vif(simple.log)

#Remove variables with high vifs and run the model again
#pdays / prevly_Cntctd
#emp_var_rate/euribor3m/nr_employed
#age/ Age_Grp
#duration/duration_group
train_simple <- train %>% dplyr::select(-age, -pdays,-emp_var_rate, -duration_group )

#Check vifs again
simple.log<-glm(y~.,family="binomial",data=train_simple)
summary(simple.log)
exp(cbind("Odds ratio" = coef(simple.log), confint.default(simple.log, level = 0.95)))
vif(simple.log)

#VIFs are still high for euribor3m and nr_employed, but model shows euribor3m as insignificant. As euribor3m looks important practically, remove nr_employed and see if things change.
train_simple_2 <- train_simple %>% dplyr::select(-nr_employed )

simple.log<-glm(y~.,family="binomial",data=train_simple_2)
summary(simple.log)
exp(cbind("Odds ratio" = coef(simple.log), confint.default(simple.log, level = 0.95)))
vif(simple.log)
#poutcome and prevly_Cntctd have higher vifs but let's keep both of them.
#MB comment: I think we should take out poutcome at VIF of 24!

#Remove statistically insignificant variables and run the model again
train_simple_3 <- train_simple_2 %>% dplyr::select(-marital,-day_of_week, -loan, -housing,-previous)

#Check model again
simple.log<-glm(y~.,family="binomial",data=train_simple_3)
summary(simple.log)
exp(cbind("Odds ratio" = coef(simple.log), confint.default(simple.log, level = 0.95)))
vif(simple.log)

#simple model -1 
simple.log<-glm(y~job+education+default+contact+month+duration+campaign+poutcome+cons_price_idx+cons_conf_idx+euribor3m+Age_Grp+prevly_Cntctd,family="binomial",data=train)
#simple.log<-glm(y~.,family="binomial",data=train_simple_3)
summary(simple.log)
exp(cbind("Odds ratio" = coef(simple.log), confint.default(simple.log, level = 0.95)))
vif(simple.log)
#Prediction using simple model
fit.pred.simple<-predict(simple.log,newdata=test, type="response")

class.simple<-factor(ifelse(fit.pred.simple>0.5,"yes","no"),levels=c("no","yes"))
# use caret and compute a confusion matrix
confusionMatrix(class.simple,test$y)

#MB add - running without poutcome
#simple model -2
simple.log2<-glm(y~job+education+default+contact+month+duration+campaign+cons_price_idx+cons_conf_idx+euribor3m+Age_Grp+prevly_Cntctd,family="binomial",data=train)
#simple.log<-glm(y~.,family="binomial",data=train_simple_3)
summary(simple.log2)
exp(cbind("Odds ratio" = coef(simple.log2), confint.default(simple.log2, level = 0.95)))
vif(simple.log2)

summary(simple.log2)

#Prediction using simple model
fit.pred.simple2<-predict(simple.log2,newdata=test,type="response")
class.simple2<-factor(ifelse(fit.pred.simple2>0.5,"yes","no"),levels=c("no","yes"))
# use caret and compute a confusion matrix
confusionMatrix(class.simple2,test$y)

#No change in metrics, both around 92% and sens. at 97%, spec. at 42%. OK to keep in poutcome

##########
## STEP ##
##########

# Feature selection using step
full.log<-glm(y~.,family="binomial",data=train)
step.log<-full.log %>% stepAIC(trace=FALSE)
summary(step.log)
#exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
vif(step.log)

#Remove variables with high vifs and run the model again
train_step <- train %>% dplyr::select(-emp_var_rate)
#Check vifs again
full.log<-glm(y~.,family="binomial",data=train_step)
step.log<-full.log %>% stepAIC(trace=FALSE)
summary(step.log)
#exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
vif(step.log)  

#euribor and nr_employed are both statistically significant in the model but have high VIFs.Removing nr_employed
train_step_2 <- train_step %>% dplyr::select(-nr_employed)

full.log<-glm(y~.,family="binomial",data=train_step_2)
step.log<-full.log %>% stepAIC(trace=FALSE)
summary(step.log)
#exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
vif(step.log)  
#pdays and prevly_Cntctd are have high VIFs.Removing pdays
train_step_3 <- train_step_2 %>% dplyr::select(-pdays )
#Check vifs again
full.log<-glm(y~.,family="binomial",data=train_step_3)
step.log<-full.log %>% stepAIC(trace=FALSE)
summary(step.log)
#exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
vif(step.log)


full.log<-glm(y~education+default+contact+month+duration+campaign+poutcome+cons_price_idx+euribor3m+Age_Grp,family="binomial",data=train)
#full.log<-glm(y~.,family="binomial",data=train_step_3)
step.log<-full.log %>% stepAIC(trace=FALSE)
summary(step.log)
#exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
vif(step.log) 
#poutcome and prevly_Cntctd have high VIFs but these are not interchangable.
#Remove statistically insignificant variables and run the model again
#str(train_step_3)
#Run step model again
#full.log<-glm(y~education+default+contact+month+duration+campaign+poutcome+cons_price_idx+euribor3m+Age_Grp,family="binomial",data=train)
#full.log<-glm(y~.,family="binomial",data=train_step_3)

#step.log<-full.log %>% stepAIC(trace=FALSE)
#summary(step.log)
#exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
#vif(step.log) 

#education is border line and contact became insignificant. Remove contact from the model

#Run step model again
#final model using stepwise for feature selection
full.log<-glm(y~education+default+month+duration+campaign+poutcome+cons_price_idx+euribor3m+Age_Grp,family="binomial",data=train)
#full.log<-glm(y~.,family="binomial",data=train)
step.log<-full.log %>% stepAIC(trace=FALSE)
summary(step.log)

#education becomes statistically significant after removing contact. VIFs look good.
#exp(cbind("Odds ratio" = coef(step.log), confint.default(step.log, level = 0.95)))
vif(step.log)
#Predicting using step 
fit.pred.step<-predict(step.log,newdata=test,type="response")
test$y[1:15]
fit.pred.step[1:15]

class.step1<-factor(ifelse(fit.pred.step>0.5,"yes","no"),levels=c("no","yes"))
# use caret and compute a confusion matrix
confusionMatrix(class.step1,test$y)
  #Acc 91%, Sens. 97%, Spec. 42%

###########
## LASSO ##
###########

# Feature selection using lasso

dat.train.x <- model.matrix(y~.,train)
dat.train.y<-train[,24]
cvfit <- cv.glmnet(dat.train.x, dat.train.y, family = "binomial", type.measure = "class", nlambda = 1000)
plot(cvfit)
coef(cvfit, s = "lambda.min")
#CV misclassification error rate is little below .1
print("CV Error Rate:")
cvfit$cvm[which(cvfit$lambda==cvfit$lambda.min)]
#"CV Error Rate:"
#0.09053767

#Optimal penalty
print("Penalty Value:")
cvfit$lambda.min
#"Penalty Value:"
#0.001054887
finalmodel<-glmnet(dat.train.x, dat.train.y, family = "binomial",lambda=cvfit$lambda.min)
finalmodel$call
finalmodel

dat.test.x<-model.matrix(y~.,test)
fit.pred.lasso <- predict(finalmodel, newx = dat.test.x, type = "response")

test$y[1:15]
fit.pred.lasso[1:15]

#confusion matrix at 0.5 cutoff
class.lasso1<-factor(ifelse(fit.pred.lasso>0.5,"yes","no"),levels=c("no","yes"))
# use caret and compute a confusion matrix
confusionMatrix(class.lasso1,test$y)
#Acc 91%, Sens. 97%, Spec. 45%

#ROCR
results.lasso<-prediction(fit.pred.lasso, test$y,label.ordering=c("no","yes"))
roc.lasso = performance(results.lasso, measure = "tpr", x.measure = "fpr")
plot(roc.lasso,colorize = TRUE)
abline(a=0, b= 1)


results.step<-prediction(fit.pred.step, test$y,label.ordering=c("no","yes"))
roc.step = performance(results.step, measure = "tpr", x.measure = "fpr")


simple.log<-glm(y~.,family="binomial",data=train)
fit.pred.origin<-predict(simple.log,newdata=test,type="response")
results.origin<-prediction(fit.pred.origin,test$y,label.ordering=c("no","yes"))
roc.origin=performance(results.origin,measure = "tpr", x.measure = "fpr")

plot(roc.lasso)
plot(roc.step,col="orange", add = TRUE)
plot(roc.origin,col="blue",add=TRUE)
legend("bottomright",legend=c("Lasso","Stepwise","Simple"),col=c("black","orange","blue"),lty=1,lwd=1)
abline(a=0, b= 1)

#Playing with different cut offs
cutoff<-0.5
class.lasso<-factor(ifelse(fit.pred.lasso>cutoff,"yes","no"),levels=c("no","yes"))
class.step<-factor(ifelse(fit.pred.step>cutoff,"yes","no"),levels=c("no","yes"))
class.simple<-factor(ifelse(fit.pred.simple>cutoff,"yes","no"),levels=c("no","yes"))

#Confusion Matrix for Lasso
conf.lasso<-table(class.lasso,test$y)
print("Confusion matrix for LASSO")
conf.lasso

#Confusion Matrix for step
conf.step<-table(class.step,test$y)
print("Confusion matrix for Stepwise")
conf.step

#Confusion Matrix for simple
conf.simple<-table(class.simple,test$y)
print("Confusion matrix for Stepwise")
conf.simple

#Accuracy of LASSO and Stepwise
print("Overall accuracy for LASSO and Stepwise respectively")
sum(diag(conf.lasso))/sum(conf.lasso)
sum(diag(conf.step))/sum(conf.step)
print("Alternative calculations of accuracy")
Acc_LASSO_0.5 <- mean(class.lasso==test$y)
Acc_STEP_0.5 <-mean(class.step==test$y)
Acc_SIMPLE_0.5<-mean(class.simple==test$y)

#Confusion Matrix for cut off =05
lasso_0.5<-confusionMatrix(conf.lasso)
step_0.5<-confusionMatrix(conf.step)
simple_0.5<-confusionMatrix(conf.simple)

cutoff<-0.1
class.lasso<-factor(ifelse(fit.pred.lasso>cutoff,"yes","no"),levels=c("no","yes"))
class.step<-factor(ifelse(fit.pred.step>cutoff,"yes","no"),levels=c("no","yes"))
class.simple<-factor(ifelse(fit.pred.simple>cutoff,"yes","no"),levels=c("no","yes"))

#Confusion Matrix for Lasso
conf.lasso<-table(class.lasso,test$y)
print("Confusion matrix for LASSO")
conf.lasso

#Confusion Matrix for step
conf.step<-table(class.step,test$y)
print("Confusion matrix for Stepwise")
conf.step

#Confusion Matrix for simple
conf.simple<-table(class.simple,test$y)
print("Confusion matrix for Stepwise")
conf.simple

#Accuracy of LASSO and Stepwise
print("Overall accuracy for LASSO and Stepwise respectively")
sum(diag(conf.lasso))/sum(conf.lasso)
sum(diag(conf.step))/sum(conf.step)
print("Alternative calculations of accuracy")
Acc_LASSO_0.1 <- mean(class.lasso==test$y)
Acc_STEP_0.1 <-mean(class.step==test$y)
Acc_SIMPLE_0.1<-mean(class.simple==test$y)
#Confusion Matrix for cut off =0.1
lasso_0.1<-confusionMatrix(conf.lasso)
step_0.1<-confusionMatrix(conf.step)
simple_0.1<-confusionMatrix(conf.simple)

cutoff<-0.15
class.lasso<-factor(ifelse(fit.pred.lasso>cutoff,"yes","no"),levels=c("no","yes"))
class.step<-factor(ifelse(fit.pred.step>cutoff,"yes","no"),levels=c("no","yes"))
class.simple<-factor(ifelse(fit.pred.simple>cutoff,"yes","no"),levels=c("no","yes"))

#Confusion Matrix for Lasso
conf.lasso<-table(class.lasso,test$y)
print("Confusion matrix for LASSO")
conf.lasso

#Confusion Matrix for step
conf.step<-table(class.step,test$y)
print("Confusion matrix for Stepwise")
conf.step

#Confusion Matrix for simple
conf.simple<-table(class.simple,test$y)
print("Confusion matrix for Stepwise")
conf.simple

#Accuracy of LASSO and Stepwise
print("Overall accuracy for LASSO and Stepwise respectively")
sum(diag(conf.lasso))/sum(conf.lasso)
sum(diag(conf.step))/sum(conf.step)
print("Alternative calculations of accuracy")
Acc_LASSO_0.15 <- mean(class.lasso==test$y)
Acc_STEP_0.15 <-mean(class.step==test$y)
Acc_SIMPLE_0.15<-mean(class.simple==test$y)

#Confusion Matrix for cut off =0.15
lasso_0.15<-confusionMatrix(conf.lasso)
step_0.15<-confusionMatrix(conf.step)
simple_0.15<-confusionMatrix(conf.simple)

cutoff<-0.2
class.lasso<-factor(ifelse(fit.pred.lasso>cutoff,"yes","no"),levels=c("no","yes"))
class.step<-factor(ifelse(fit.pred.step>cutoff,"yes","no"),levels=c("no","yes"))
class.simple<-factor(ifelse(fit.pred.simple>cutoff,"yes","no"),levels=c("no","yes"))

#Confusion Matrix for Lasso
conf.lasso<-table(class.lasso,test$y)
print("Confusion matrix for LASSO")
conf.lasso

#Confusion Matrix for step
conf.step<-table(class.step,test$y)
print("Confusion matrix for Stepwise")
conf.step

#Confusion Matrix for simple
conf.simple<-table(class.simple,test$y)
print("Confusion matrix for Stepwise")
conf.simple

#Accuracy of LASSO and Stepwise
print("Overall accuracy for LASSO and Stepwise respectively")
sum(diag(conf.lasso))/sum(conf.lasso)
sum(diag(conf.step))/sum(conf.step)
print("Alternative calculations of accuracy")
Acc_LASSO_0.2 <- mean(class.lasso==test$y)
Acc_STEP_0.2 <-mean(class.step==test$y)
Acc_SIMPLE_0.2<-mean(class.simple==test$y)

#Confusion Matrix for cut off =0.2
lasso_0.2<-confusionMatrix(conf.lasso)
step_0.2<-confusionMatrix(conf.step)
simple_0.2<-confusionMatrix(conf.simple)


Sensitivity_simple<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"Simple_Sensitivty"=c(simple_0.1$byClass[1],simple_0.15$byClass[1],simple_0.2$byClass[1],simple_0.5$byClass[1] ) )
Sensitivity_step<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"Step_Sensitivity"=c(step_0.1$byClass[1],step_0.15$byClass[1],step_0.2$byClass[1],step_0.5$byClass[1] ) )
Sensitivity_lasso<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"LASSO_Sensitivity"=c(lasso_0.1$byClass[1],lasso_0.15$byClass[1],lasso_0.2$byClass[1],lasso_0.5$byClass[1] ) )

Specificity_simple<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"Simple_Specificity"=c(simple_0.1$byClass[2],simple_0.15$byClass[2],simple_0.2$byClass[2],simple_0.5$byClass[2] ) )
Specificity_step<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"Step_Specificity"=c(step_0.1$byClass[2],step_0.15$byClass[2],step_0.2$byClass[2],step_0.5$byClass[2] ) )
Specificity_lasso<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"LASSO_Specificity"=c(lasso_0.1$byClass[2],lasso_0.15$byClass[2],lasso_0.2$byClass[2],lasso_0.5$byClass[2] ) )

Accuracy_simple<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"Simple_Accuracy"=c(simple_0.1$overall[1],simple_0.15$overall[1],simple_0.2$overall[1],simple_0.5$overall[1] ) )
Accuracy_step<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"Step_Accuracy"=c(step_0.1$overall[1],step_0.15$overall[1],step_0.2$overall[1],step_0.5$overall[1] ) )
Accuracy_lasso<- data.frame("CutOff"= c("0.1", "0.15","0.2","0.5"),"LASSO_Accuracy"=c(lasso_0.1$overall[1],lasso_0.15$overall[1],lasso_0.2$overall[1],lasso_0.5$overall[1] ) )

Sensitivity <- cbind(Sensitivity_simple,Sensitivity_step$Step_Sensitivity,Sensitivity_lasso$LASSO_Sensitivity)
Specificity <- cbind(Specificity_simple, Specificity_step$Step_Specificity,Specificity_lasso$LASSO_Specificity)
Accuracy <- cbind(Accuracy_simple,Accuracy_step$Step_Accuracy, Accuracy_lasso$LASSO_Accuracy)
Sensitivity
Specificity
Accuracy

##############################
### Complex Logistic Model ###
##############################

# Run Initial Logistic Regression allowing for interaction
#start with only variables from best simple model
complex.log<-glm(y~ education * default * month * duration * campaign * poutcome * cons_price_idx * euribor3m * Age_Grp,family="binomial",data=train)
summary(complex.log)
exp(cbind("Odds ratio" = coef(complex.log), confint.default(complex.log, level = 0.95)))
vif(complex.log)


#################
## LDA & QDA ###
################

#Training Set
train.lda.x <- train[ , sapply(train, is.numeric)]

train.lda.y <- train$y

fit.lda <- lda(train.lda.y ~ ., data = train.lda.x)
pred.lda <- predict(fit.lda, newdata = train.lda.x)

preds <- pred.lda$posterior
preds <- as.data.frame(preds)

pred <- prediction(preds[,2],train.lda.y)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))
#AUC = 0.921

# Test Set
test.lda.x <- test[ , sapply(test, is.numeric)]

test.lda.y <- test$y

pred.lda1 <- predict(fit.lda, newdata = test.lda.x)

preds1 <- pred.lda1$posterior
preds1 <- as.data.frame(preds1)

pred1 <- prediction(preds1[,2],test.lda.y)
roc.perf = performance(pred1, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred1, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .40, y = .6,paste("AUC = ", round(auc.train[[1]],3), sep = ""))
#AUC = 0.919

#running cv on train set using LDA
nloops<-50   #number of CV loops
ntrains<-dim(train.lda.x)[1]  #No. of samples in training data set
cv.aucs<-c()
dat.train.yshuf<-train.lda.y[sample(1:length(train.lda.y))]

set.seed(123)
for (i in 1:nloops){
  index<-sample(1:ntrains,ntrains*.8)
  cvtrain.x<-train.lda.x[index,]
  cvtest.x<-train.lda.x[-index,]
  cvtrain.y<-dat.train.yshuf[index]
  cvtest.y<-dat.train.yshuf[-index]
  
  cvfit <- lda(cvtrain.y ~ ., data = cvtrain.x)
  fit.pred <- predict(cvfit, newdata = cvtest.x)
  preds.cv <- fit.pred$posterior
  preds.cv <- as.data.frame(preds.cv)
  pred.cv <- prediction(preds.cv[,2], cvtest.y)
  roc.perf = performance(pred.cv, measure = "tpr", x.measure = "fpr")
  auc.train <- performance(pred.cv, measure = "auc")
  auc.train <- auc.train@y.values
  
  cv.aucs[i]<-auc.train[[1]]
}

hist(cv.aucs)
summary(cv.aucs)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.4763  0.4916  0.4975  0.4974  0.5031  0.5145

###################
## Random Forest ##
###################

cv_5 <- trainControl(method="cv", 
                     number = 5)

set.seed(1234)
rf_grid <- expand.grid(
  mtry = 3:8,
  splitrule = c("gini","extratrees", "hellinger"),
  min.node.size = c(1)
)

fitRF <- train(y ~ ., 
               data = train, 
               method = "ranger", 
               trControl = cv_5,
               num.threads = 4,
               tuneGrid=rf_grid)  

fitRF
plot(fitRF)
confusionMatrix(fitRF)

predRF <- predict(fitRF, newdata = test)
confusionMatrix(predRF, test$y)

# I chose to tune 2 hyper parameters for Random Forest
# - mtry which represents the number of predictors considered when splitting a node in a tree
# - splitrule which determines the rule used for the actual splitting based on the above predictors
# I set min.node.size to 1 as appropriate for classification

# Optimizing for accuracy, an mtry of 6 predictors and the Hellinger split rule gave the best test accuracy: 0.9179, with Sensitivity 0.9665 and Specificity 0.5305.
# It's interesting that Hellinger won. Hellinger handles imbalanced data well; being insensitive to skew.*
# * CITATION: https://www3.nd.edu/~nchawla/papers/DMKD11.pdf
# * CITATION: https://medium.com/@evgeni.dubov/classifying-imbalanced-data-using-hellinger-distance-f6a4330d6f9a


######################
## Model Comparison ##
######################