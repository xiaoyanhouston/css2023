## project7v81. 
## xiaoyan Zhang
## purpose
#######################################################
## The project is to build the model to forcast the ###
## house price, using house data from redfin and     ##
## Dallas HPI, use the output from Doc2vec model to  ## 
## build a gradient boosting tree model              ##
#######################################################
library(ggplot2)
library(dplyr)
library(lubridate)
library(caret)
library(zoo)
library(randomForest)
library(rsample)
library(gbm)

################ input data  #############
#read house data
house_data<- read.csv("C:\\Users\\xiaoy\\Desktop\\UTDS23\\summer2023\\project\\7v81data\\house_data2.csv")

# read HPI data
hpi<- read.csv("C:\\Users\\xiaoy\\Desktop\\UTDS23\\summer2023\\project\\7v81data\\Dallas_HPI.csv")

embed_d = 24

#read training embedding vector with dimention 24
train_vector<- read.csv(paste("C:\\Users\\xiaoy\\Desktop\\UTDS23\\summer2023\\project\\emd_out\\vec_",
                              embed_d,"\\train_vector_",embed_d,".csv", sep=""))
train_vector$split_ind<- 1

# read validation embedding vector with dimention 24
valid_vector<- read.csv(paste("C:\\Users\\xiaoy\\Desktop\\UTDS23\\summer2023\\project\\emd_out\\vec_",
                              embed_d,"\\valid_vector_",embed_d,".csv", sep=""))
valid_vector$split_ind<- 0
#read oot embedding vector 
oot_vector<- read.csv(paste("C:\\Users\\xiaoy\\Desktop\\UTDS23\\summer2023\\project\\emd_out\\vec_",
                            embed_d,"\\oot_vector_",embed_d,".csv", sep=""))
oot_vector$split_ind<- 2
#combine train_vector_ind 1 and 0 and 2

train_val_oot_combined<- rbind(train_vector,valid_vector,oot_vector)

# transformation
# convert house data sold_month to date
#library(lubridate)

house_data$SOLD_MONTH<-mdy(house_data$SOLD_MONTH)
house_data$base_month<- floor_date(house_data$SOLD_MONTH, unit = "month") - months(1)


house_data<- house_data[house_data$SOLD_MONTH< as.Date("2023-05-01"),]

house_data1<-house_data[house_data$BEDS ==1, ]
#drop beds outlier
house_data<- house_data[-c(2209,4088,3905,3976), ]

# convert HPI date string  to date
hpi$DATE<-mdy(hpi$DATE)

#create lag hpi
#library(dplyr)
hpi<- hpi %>%
  mutate(hpi_lag1 = lag(HPI)) %>%
  mutate(hpi_lag3 = lag(HPI, n=3)) %>%
  mutate(hpi_lag6= lag(HPI, n=6))

#create the hpi percentage change
hpi$hpi_1m_pct <- (hpi$HPI-hpi$hpi_lag1)/hpi$hpi_lag1
hpi$hpi_3m_pct <- (hpi$HPI-hpi$hpi_lag3)/hpi$hpi_lag3
hpi$hpi_6m_pct <- (hpi$HPI-hpi$hpi_lag6)/hpi$hpi_lag6

#join house table and hpi table by date
house_df<- merge(house_data, hpi,by.x="base_month", by.y="DATE", all.x= TRUE, all.y=FALSE)

# do histogram of price, squarefeet, lotsize
hist(house_df$PRICE)
hist(house_df$SQUAREFEET)
hist(house_df$LOTSIZE)
hist(house_df$BEDS)

## create new variables of house data
# extract sold year from house data
house_df$sold_year <- year(as.Date(house_df$SOLD_MONTH))
print(house_df$sold_year)

#  house age 
house_df$houseage <- house_df$sold_year- house_df$YEARBUILT
house_df$houseage

#total beds and bath
house_df$total_beds_baths <- house_df$BEDS + house_df$BATHS

#baths and beds ratio
house_df$bath_bed_ratio <- ifelse(house_df$BEDS==0,NA, house_df$BATHS/house_df$BEDS)

#living squarefeet and beds ratio
house_df$sqrt_beds <- ifelse(house_df$BEDS==0, NA, house_df$SQUAREFEET/house_df$BEDS)

############################ data clean #################
# check price
quantile(na.omit((house_df$PRICE)),c(0,0.01,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
# check beds
quantile(na.omit((house_df$BEDS)),c(0,0.01,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
#check baths
quantile(na.omit((house_df$BATHS)),c(0,0.01,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
#check lotsize 
quantile(na.omit((house_df$LOTSIZE)),c(0,0.01,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))
#check living squarefeet
quantile(na.omit((house_df$SQUAREFEET)),c(0,0.01,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1))

# impute the lotsize outlier to median
med_lot<- median(house_df$LOTSIZE, na.rm = TRUE )
house_df$LOTSIZE_imputed <- ifelse(is.na(house_df$LOTSIZE)|house_df$LOTSIZE > 30000|house_df$LOTSIZE< 2000, med_lot,
                                   house_df$LOTSIZE)
house_df$LOTSIZE_imputed_ind<-ifelse(is.na(house_df$LOTSIZE)|house_df$LOTSIZE > 30000|house_df$LOTSIZE< 2000, 1,0)
  

#check  missing value 
sum(is.na(house_df$PRICE))
sum(is.na(house_df$LOTSIZE))
sum(is.na(house_df$BEDS))
sum(is.na(house_df$BATHS))
sum(is.na(house_df$SQUAREFEET))

#rows missing
#rows_missing <- setdiff(1:nrow(house_df["PRICE"]),which(complete.cases((house_df$PRICE))))

# drop rows of missing price
house_df <- house_df[complete.cases(house_df$PRICE), ]

## variable transformation 
#log transformation for house price, lotsize, squarefeet

hist(house_df$PRICE)
hist(log(house_df$PRICE))
house_df$log_price <- log(house_df$PRICE)

hist(house_df$SQUAREFEET)
hist(log(house_df$SQUAREFEET))
# log sqrt is skewed to the left, original is better

hist(house_df$LOTSIZE_imputed)
hist(log(house_df$LOTSIZE_imputed))
house_df$log_lotsize <- log(house_df$LOTSIZE_imputed)
hist(house_df$BEDS)
hist(house_df$BATHS)

############################# data analysis ####################
## cross table
# frequency of zip code
zip_freq<- table(house_df$ZIP)
zip_freq
#frequency of house age
houseage_freq <- table(house_df$houseage)
houseage_freq

# frequency of number of beds
beds_freq<- table(house_df$BEDS)
beds_freq
# frequency of number of baths
baths_freq<- table(house_df$BATHS)
baths_freq
#cross table by zip *number of beds
ZIP_BEDS<- xtabs(~ ZIP + BEDS, data= house_df)
ZIP_BEDS
#cross table by zip *number of baths
ZIP_BATHS<- xtabs(~ ZIP + BATHS, data = house_df)
ZIP_BATHS
#cross table by number of beds * number of baths
beds_baths<- xtabs(~ BEDS + BATHS, data = house_df )
beds_baths

#mean price by month
price_aver_month<-aggregate(PRICE~ SOLD_MONTH, data= house_df, mean)
price_aver_month
plot(price_aver_month$SOLD_MONTH,price_aver_month$PRICE, xlab="SOLD_MONTH", ylab ="PRICE",
     main="Mean Price by Month", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_aver_month$SOLD_MONTH, labels = price_aver_month$SOLD_MONTH) 

#mean price by number of beds
price_aver_beds<-aggregate(PRICE ~ BEDS , data= house_df, mean)
price_aver_beds
plot(price_aver_beds$BEDS,price_aver_beds$PRICE, xlab="BEDS", ylab ="PRICE",
     main="Mean Price by Beds", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_aver_beds$BEDS, labels = price_aver_beds$BEDS) 

#mean price by number of baths
price_aver_BATHS<-aggregate(PRICE ~ BATHS , data= house_df, mean)
price_aver_BATHS
plot(price_aver_BATHS$BATHS,price_aver_BATHS$PRICE, xlab="BATHS", ylab ="PRICE",
     main="Mean Price by Baths", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_aver_BATHS$BATHS, labels = price_aver_BATHS$BATHS)
#mean price by houseage
price_aver_houseage<-aggregate(PRICE ~ houseage , data= house_df, mean)
price_aver_houseage
plot(price_aver_houseage$houseage,price_aver_houseage$PRICE, xlab="houseage", ylab ="PRICE",
     main="Mean Price by houseage", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_aver_houseage$houseage, labels = price_aver_houseage$houseage)

# scatter plot of price by living square feet

p <- ggplot(house_df,aes(x=SQUAREFEET, y=PRICE ))+geom_point()+ geom_smooth(method="lm", se = FALSE)
print(p)

#median price by month
price_med_month <- aggregate(PRICE ~ SOLD_MONTH, data=house_df, median)
plot(price_med_month$SOLD_MONTH, price_med_month$PRICE, xlab="SOLD_MONTH", ylab ="PRICE",
     main="Median Price by Month", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_med_month$SOLD_MONTH, labels = price_med_month$SOLD_MONTH) 

#median price by number of beds
price_med_beds <- aggregate(PRICE ~ BEDS, data=house_df, median)
plot(price_med_beds$BEDS, price_med_beds$PRICE, xlab="BEDS", ylab ="PRICE",
     main="Median Price by Beds", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_med_beds$BEDS, labels = price_med_beds$BEDS) 

#median price by baths
price_med_baths <- aggregate(PRICE ~ BATHS, data=house_df, median)
plot(price_med_baths$BATHS, price_med_baths$PRICE, xlab="BATHS", ylab ="PRICE",
     main="Median Price by Baths", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_med_baths$BATHS, labels = price_med_baths$BATHS) 

#median price by houseage
price_med_houseage <- aggregate(PRICE ~ houseage, data=house_df, median)
plot(price_med_houseage$houseage, price_med_houseage$PRICE, xlab="houseage", ylab ="PRICE",
     main="Median Price by houseage", type= "l",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_med_houseage$houseage, labels = price_med_houseage$houseage) 

# create new variable age group 

#library(dplyr)
house_df <- house_df %>%
mutate(age_group = cut(houseage, breaks = c(-0.5,5,10,15,20,25, Inf), labels = c("0-5", "6-10", "11-15","16-20","21-25","25+"))) 
house_df
#mean price by age group
price_aver_age_group<-aggregate(PRICE ~ age_group , data= house_df, mean)
price_aver_age_group
plot(price_aver_age_group$age_group,price_aver_age_group$PRICE, xlab="age_group", ylab ="PRICE",
     main="Mean Price by age_group ", type= "p",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_aver_age_group$age_group, labels = price_aver_age_group$age_group)

#median price by age group
price_med_age_group<-aggregate(PRICE ~ age_group , data= house_df, median)
price_med_age_group
plot(price_med_age_group$age_group,price_med_age_group$PRICE, xlab="age_group", ylab ="PRICE",
     main="Median Price by age_group ", type= "p",xaxt = "n")  # xaxt = "n" removes default x-axis labels
axis(side = 1, at =  price_med_age_group$age_group, labels = price_med_age_group$age_group)


# create new variable month from sold month 
house_df$month_ind<- as.factor(month(house_df$SOLD_MONTH))

# join the house data and combined training and validation vectors
house_df_emb<- merge(house_df, train_val_oot_combined,by.x = "MLS", by.y = "id", all.x=TRUE)


emb_list<- c("vec_1")

for (i in 2:embed_d){
  var_vec = paste('vec_', i, sep = '')
  emb_list = c(emb_list, var_vec)
}

# keep list
#
raw_list <-c("BEDS","BATHS","SQUAREFEET","HPI","hpi_lag1","hpi_lag3","hpi_lag6","hpi_1m_pct", 
             "hpi_3m_pct","hpi_6m_pct","houseage","total_beds_baths","bath_bed_ratio","sqrt_beds","LOTSIZE_imputed_ind",
             "log_price","log_lotsize", "month_ind")
keep_list<-  c(raw_list, "split_ind", emb_list) 


house_df_emb<-house_df_emb[ ,keep_list]
#house_df_emb <- na.omit(house_df_emb)
#save to csv

write.csv(house_df_emb, file= paste("C:\\Users\\xiaoy\\Desktop\\UTDS23\\summer2023\\project\\emd_out\\vec_",
       embed_d,"\\house_df_emb_",embed_d,".csv", sep=""),row.names = FALSE)
################## split the data ##################################
#split the data to training data as 70% and testing data as 30%

train_data <- subset(house_df_emb[house_df_emb$split_ind == 1,], select = -split_ind)
test_data <-  subset(house_df_emb[house_df_emb$split_ind == 0,], select = -split_ind)
oot_data <- subset(house_df_emb[house_df_emb$split_ind == 2,], select = -split_ind)

library(rsample)


########################## START GBM MODEL ##########################

## build gradient boosting tree model 
library(gbm)
# drop  similar variables based on importance
train_data2<-subset(train_data, select = -c(hpi_lag3,hpi_1m_pct,
                                            hpi_6m_pct,hpi_lag1, HPI))

#test with vectors
set.seed(1)
n_trees <- 150
learning_rate <- 0.11
max_depth <- 5
tree_seq <- seq(1, n_trees, by = 5)

pred_train<-  matrix(0, nrow = nrow(train_data2), ncol = length(tree_seq))

pred_test <- matrix(0, nrow = nrow(test_data), ncol = length(tree_seq))
pred_oot <- matrix(0, nrow = nrow(oot_data), ncol = length(tree_seq))
#use for loop to make predict
col_ind = 0
for (i in tree_seq) {
  model <- gbm(log_price ~ ., data = train_data2, n.trees = i, 
               interaction.depth = max_depth, shrinkage = learning_rate,
               distribution = "gaussian")
  col_ind = col_ind +1
  pred_train[, col_ind] <- predict(model, newdata = train_data2, n.trees = i)
  pred_test[, col_ind] <- predict(model, newdata = test_data, n.trees = i)
  pred_oot[, col_ind] <- predict(model, newdata = oot_data, n.trees = i)
}

#combine the prediction
error_list_train <- apply((pred_train-train_data2$log_price)^2, 2, mean)

error_list_test <- apply((pred_test-test_data$log_price)^2, 2, mean)
error_list_oot <- apply((pred_oot-oot_data$log_price)^2, 2, mean)

plot(tree_seq, error_list_train, type="l", col= "blue", xlab="nTree", 
     ylab="Error",ylim=c(0,0.15) )

lines(tree_seq,error_list_test, type="l", col = "red")
lines(tree_seq,error_list_oot, type="l", col = "green")
legend("topright", legend = c("error_list_train", "error_list_test","error_list_oot"), 
      col = c("blue", "red","green"), lty = 1)

#legend("topright", legend = c("error_list_train", "error_list_test"), 
      # col = c("blue", "red"), lty = 1)


#best model
best_model <- gbm(log_price ~ ., data = train_data2, n.trees = 96, 
                  interaction.depth = max_depth, shrinkage = learning_rate,
                  distribution = "gaussian")
best_pred <- predict(best_model, newdata = test_data)
mse<-mean((test_data$log_price - best_pred)^2)
mse
# pred_oot
best_pred_oot <- predict(best_model, newdata = oot_data)
mse<-mean((oot_data$log_price - best_pred_oot)^2)
mse




# calculate var importance
var_importance<- summary(best_model)
# sort var importance in descending order
var_importance_sorted <- var_importance[order(var_importance[,2], decreasing = FALSE),]
par(mar = c(4, 7, 2, 4))
barplot(var_importance_sorted$rel.inf,
        names.arg = rownames(var_importance_sorted), 
        xlab = "Score",
        main = "variable importance",
        horiz = TRUE,las=1,
        cex.axis = 0.8,
        cex.names = 0.8)
































#best model
best_model <- gbm(log_price ~ ., data = train_data, n.trees = 60, 
                  interaction.depth = 10, shrinkage = 0.1,n.minobsinnode = 5,
                  distribution = "gaussian")
best_pred <- predict(best_model, newdata = test_data)
mse<-mean((test_data$log_price - best_pred)^2)


# calculate var importance
var_importance<- summary(model)
# sort var importance in descending order
var_importance_sorted <- var_importance[order(var_importance[,2], decreasing = FALSE),]
barplot(var_importance_sorted$rel.inf,
        names.arg = rownames(var_importance_sorted), 
        xlab = "Score",
        main = "variable importance",
        horiz = TRUE,las=1,)
par(mar = c(3, 12, 2, 4))


# # drop  similar variables based on importance
train_data2<-subset(train_data, select = -c(hpi_lag3,hpi_1m_pct,hpi_6m_pct,hpi_lag1, HPI))
# 
# ## best tune
gbmgrid<- expand.grid(interaction.depth = c(3,5,7,9), n.trees= c(10,50,100), 
                       shrinkage =c(0.01, 0.05, 0.1), n.minobsinnode= 5)
set.seed(123)
#
gbm_model<- train(log_price~ ., data=na.fill(train_data2,0), method ="gbm",
                   trControl = trainControl(method ="cv", number= 10),
                   tuneGrid= gbmgrid)
# 
# ## best tune
gbmgrid2<- expand.grid(interaction.depth = c(9,12), n.trees= c(100, 150), 
                       shrinkage =c(0.05, 0.1, 0.15), n.minobsinnode= 5)
set.seed(123)
# 
gbm_model2<- train(log_price~ ., data=na.fill(train_data2,0), method ="gbm",
                   trControl = trainControl(method ="cv", p = 0.8,
                                            allowParallel = TRUE, number= 3),
                   tuneGrid= gbmgrid2)
# 
# 
# #best model
best_model <- gbm(log_price ~ ., data = train_data, n.trees = 100, 
              interaction.depth = 9, shrinkage = 0.1,n.minobsinnode = 5,
              distribution = "gaussian",verbose = TRUE)
best_pred <- predict(best_model, newdata = test_data)
mse<-mean((test_data$log_price - best_pred)^2)
mse

##

