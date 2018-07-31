#install.packages("tensorflow")
library(tensorflow)
#install.packages("keras")
library(keras)
#install_keras()
## To activate this environment, use:
# > activate r-tensorflow
#
# To deactivate an active environment, use:
# > deactivate

#install.packages(c("faraway", "corrplot", "clusterGeneration", "devtools"))
library(neuralnet)
library(nnet)
library(NeuralNetTools)
library(MASS)
library(ISLR)
library(caTools) # sample.split
library(boot) # cv.glm
library(faraway) # compact lm summary "sumary" function
library(caret) # useful tools for machine learning
library(corrplot)
library(RSNNS)
library(clusterGeneration)
library(devtools)

library(plyr)

library(ggplot2)

library(dplyr)

setwd("D:/BODT Camp IYKRA/Capstone Project/")
newss <- read.csv("dataset/OnlineNewsPopularity.csv")
# shares <- read.csv("df_new.csv")

# dim(shares)
# str(shares)
# head(shares, 10)
# summary(shares)
# View(shares)
# summary(shares$shares)
#Mean = 3395
#Median = 1400
# ggplot(shares, aes(x = X, y = shares)) + geom_line() #masih salah, harusnya berupa bar atau histogram. X-nya harusnya bukan ini

dim(newss)
str(newss)
head(newss, 10)
summary(newss)
#View(newss)


#df
df <- newss
df <- df %>%
  mutate(ol = case_when(shares > 38275 ~ 1,
                        TRUE ~ 0
  ))
df$ol <- as.double(df$ol)

df <- df %>%
  mutate(new_shares = case_when(ol==1 ~ 38275,
                                TRUE~ as.double(shares)))

df <- df %>%
  filter(ol==0)


df <- df %>%
  filter(n_unique_tokens <= 1)

columns <- c("LDA_00","LDA_01","LDA_02","LDA_03","LDA_04","is_weekend","data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world","avg_negative_polarity","average_token_length","min_negative_polarity","max_negative_polarity","rate_positive_words","rate_negative_words","self_reference_min_shares","self_reference_max_shares","self_reference_avg_sharess","num_hrefs","num_imgs","global_subjectivity","abs_title_sentiment_polarity","num_videos","shares")
columns_new <- c("LDA_00","LDA_01","LDA_02","LDA_03","LDA_04","is_weekend","data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world","average_token_length","min_negative_polarity","max_negative_polarity","rate_positive_words","rate_negative_words","self_reference_min_shares","self_reference_max_shares","num_hrefs","num_imgs","global_subjectivity","abs_title_sentiment_polarity","num_videos","shares")


df <- df %>%
  select(columns_new)
write.csv(df, file = "df_newest.csv")
df <- df %>% 
  mutate(popularity = case_when(shares >= 1400 ~ 1, #sebelumnya pakai >= 1400, kemudian ganti jadi >1400
                                TRUE ~ 0
                                ))
summary(df)
df <- df[,-25] #perlu dihitung terus jumlah variabelnya

corrplot(cor(df), method = "number", type = "lower", tl.pos = "ld", number.cex = 0.6)
corrplot.mixed(cor(df), lower.col = "black", number.cex = 0.6, tl.col = "black", tl.cex = 0.4)
#Hilangkan yang bukan untuk prediction
newss.corr <- newss[,-c(1,2)]

corrplot(cor(newss.corr), tl.pos = "n")
corrplot(cor(newss.corr), method = "number", tl.pos = "n", number.cex = 0.5)
corrplot.mixed(cor(newss.corr), lower.col = "black", number.cex = 0.5, tl.col = "black", tl.cex = 0.4)

#filter setelah EDA, bisa pakai nama variabel juga
filter <- newss[,c(8, 10:12, 14:38, 40:45, 49:50, 54:56, 60, 61)]
#View(filter)

set.seed(123)

# Regression

filter.reg <- filter

#Data Partition, bukan cuma untuk NN
index <- caret::createDataPartition(filter.reg$shares, p = 0.7, list = FALSE)

train.reg <- filter.reg[index,]
test.reg <- filter.reg[-index,]

#Scaling data untuk NN
max <- apply(filter, 2, max)
min <- apply(filter, 2, min)
scaled <- as.data.frame(scale(filter, center = min, scale = max - min))

trainNN.reg <- scaled[index,]
testNN.reg <- scaled[-index,]

set.seed(100)
# Untuk melihat parameter apa saja yang perlu diperhatikan: ?neuralnet
# 3 dense hidden layers c(30, 30, 30)

n <- names(df)
f <- as.formula(paste("shares ~", paste(n[!n %in% "shares"], collapse = " + ")))

NNbp.reg <- neuralnet(f,
                        data = train,
                        hidden = 30,
                        #algorithm = 'backprop', #algorithm yang dipilih untuk sementara: simple back propagation dulu supaya gak lama
                        #learningrate = 0.0001, #'learningrate' must be a numeric value, if the backpropagation algorithm is used
                        linear.output = TRUE)

outputt <- capture.output(NNbp.reg)
cat("NN30new.reg", outputt, file = "NN30new.reg.txt", sep = " ", fill = TRUE, append = TRUE)

#Making formula:
n.reg <- names(filter.reg)
f.reg <- as.formula(paste("shares ~", paste(n.reg[!n.reg %in% "shares"], collapse = " + ")))
#f.reg nanti akan ngegantikan shares~ dst.

#Hidden layers = 40

NN40bp.reg <- neuralnet(f.reg,
                      data = trainNN.reg,
                      hidden = 40,
                      algorithm = 'backprop', #algorithm yang dipilih untuk sementara: simple back propagation dulu supaya gak lama
                      learningrate = 0.0001, #'learningrate' must be a numeric value, if the backpropagation algorithm is used
                      linear.output = TRUE)

NN40.reg <- neuralnet(f.reg,
                      data = trainNN.reg,
                      hidden = 40,
                      #algorithm = 'backprop', #algorithm yang dipilih untuk sementara: simple back propagation dulu supaya gak lama
                      #learningrate = 0.01, #'learningrate' must be a numeric value, if the backpropagation algorithm is used
                      linear.output = TRUE) #algorithm by default = Resilient Propagation +
#NN40.reg
summary(NN40.reg)
outputt <- capture.output(NN40.reg)
cat("NN40.reg", outputt, file = "NN40.reg.txt", sep = " ", fill = TRUE, append = TRUE)
head(NN40.reg$result.matrix)
#Hasil:
#error                    1.896173894891
#reached.threshold        0.009825922532
#steps                 7413.000000000000

OutputVsPred <- cbind(trainNN.reg$shares, NN40.reg$net.result[[1]])
OutputVsPred

# plotting NN
plot(NN40.reg)

#Prediction using Neural Network

predict_testNN40.reg = compute(NN40.reg, testNN.reg[,c(1:40)])
predict_testNN40.reg = (predict_testNN40.reg$net.result * (max(filter$shares) - min(filter$shares))) + min(filter$shares)

#Hitung RMSE
#bisa pakai test.reg (sebelum di-scale) atau testNN.reg*(max(filter$shares) - min(filter$shares)) + min(filter$shares)
RMSE.NN40.reg = (sum((test.reg$shares - predict_testNN40.reg)^2) - nrow(test.reg)) ^ 0.5
RMSE.NN40.reg


#Hidden layers = 60

NN60.reg <- neuralnet(f.reg,
                    data = trainNN.reg,
                    hidden = 60,
                    #algorithm = 'backprop', #algorithm yang dipilih untuk sementara: simple back propagation dulu supaya gak lama
                    #learningrate = 0.01, #'learningrate' must be a numeric value, if the backpropagation algorithm is used
                    linear.output = TRUE) #algorithm by default = Resilient Propagation +
#NN60.reg
summary(NN60.reg)
outputt <- capture.output(NN60.reg)
cat("NN60.reg", outputt, file = "NN60.reg.txt", sep = " ", fill = TRUE, append = TRUE)
head(NN60.reg$result.matrix)
#Hasil:
#error                    1.896173894891
#reached.threshold        0.009825922532
#steps                 7413.000000000000

OutputVsPred <- cbind(trainNN.reg$shares, NN60.reg$net.result[[1]])
OutputVsPred

# plotting NN
plot(NN60.reg)

#Prediction using Neural Network

predict_testNN60.reg = compute(NN60.reg, testNN.reg[,c(1:30)])
predict_testNN60.reg = (predict_testNN60.reg$net.result * (max(filter$shares) - min(filter$shares))) + min(filter$shares)

#Hitung RMSE
#bisa pakai test.reg (sebelum di-scale) atau testNN.reg*(max(filter$shares) - min(filter$shares)) + min(filter$shares)
RMSE.NN60.reg = (sum((test.reg$shares - predict_testNN60.reg)^2) - nrow(test.reg)) ^ 0.5
RMSE.NN60.reg


#Cross-Validation

cv.error <- NULL
k <- 10 #k-folds

pbar <- create_progress_bar('text')

pbar$init(k)

for (i in 1:k){
  index.cv <- caret::createDataPartition(filter.reg$shares, p = 0.7, list = FALSE)
  trainNN.reg.cv <- scaled[index,]
  testNN.reg.cv <- scaled[-index,]
  NN.reg.cv <- neuralnet(shares~num_hrefs + num_imgs + num_videos + average_token_length + data_channel_is_entertainment + data_channel_is_bus + data_channel_is_tech + data_channel_is_world + kw_max_min + kw_avg_min + kw_avg_max + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + self_reference_max_shares + self_reference_avg_sharess + weekday_is_tuesday + weekday_is_thursday + LDA_01 + LDA_02 + LDA_03 + LDA_04 + global_subjectivity + rate_positive_words + rate_negative_words + avg_negative_polarity + min_negative_polarity + max_negative_polarity + abs_title_sentiment_polarity,
                      data = trainNN.reg.cv,
                      hidden = c(30, 30),
                      linear.output = TRUE)
  predict_testNN.reg.cv = compute(NN.reg.cv, testNN.reg.cv[,c(1:30)])
  predict_testNN.reg.cv = (predict_testNN.reg.cv$net.result * (max(filter$shares) - min(filter$shares))) + min(filter$shares)
  #cek lagi test.reg-nya, mungkin perlu dibuat versi cv?
  cv.error[i] <- (sum((test.reg$shares - predict_testNN.reg.cv)^2) - nrow(test.reg)) ^ 0.5
  pbar$step()
}


#  Classification :: Binary Classification (Popular/Not Popular)
filter.cla <- filter

#Buat variabel baru sebagai batas berdasarkan jumlah shares
#Batas = mean? Batas = median?
filter.cla$popularity <- as.numeric(ifelse(filter.cla$shares >= 1400, 1, 0))
#Hapus variabel shares
filter.cla <- filter.cla[,-42]
str(filter.cla)
#Data Partition, bukan cuma untuk NN
index <- caret::createDataPartition(filter.cla$popularity, p = 0.7, list = FALSE)

train.cla <- filter.cla[index,]
test.cla <- filter.cla[-index,]

#Scaling data untuk NN
max <- apply(filter.cla[,1:41], 2, max) #masih pakai filter?
min <- apply(filter.cla[,1:41], 2, min)
scaled <- as.data.frame(scale(filter.cla[,1:41], center = min, scale = max - min))

trainNN.cla <- cbind(scaled[index,], filter.cla[index,42])
testNN.cla <- cbind(scaled[index,], filter.cla[index,42])

names(trainNN.cla) <- c(names(scaled), "popularity")
names(testNN.cla) <- c(names(scaled), "popularity")

set.seed(100)
# Untuk melihat parameter apa saja yang perlu diperhatikan: ?neuralnet
# 3 dense hidden layers c(30, 30, 30)

#Making formula:
n.cla <- names(filter.cla) #awalnya newss
f.cla <- as.formula(paste("popularity ~", paste(n.cla[!n.cla %in% "popularity"], collapse = " + ")))
#f.cla nanti akan ngegantikan popularity~ dst.

NN40.cla <- neuralnet(f.cla,
                    data = trainNN.cla,
                    hidden = 40,
                    algorithm = 'backprop', #algorithm yang dipilih untuk sementara: simple back propagation dulu supaya gak lama
                    learningrate = 0.01, #'learningrate' must be a numeric value, if the backpropagation algorithm is used
                    linear.output = FALSE) #algorithm by default = Resilient Propagation +


#KERAS
df <- as.matrix(df)
dimnames(df) <- NULL

summary(df)
df[,1:24] <- normalize(df[,1:24])
summary(df)

set.seed(100)
ind <- sample(2, nrow(df), replace = T, prob = c(0.7, 0.3))
training <- df[ind==1, 1:24]
test <- df[ind==2, 1:24]
trainingtarget <- df[ind==1, 25]
testtarget <- df[ind==2, 25]

trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)
testLabels

model <- keras_model_sequential()
model %>%
  layer_dense(units=30, activation = 'relu', input_shape = c(24)) %>%
  layer_dense(units=9, activation = 'relu') %>% 
  layer_dense(units=2, activation = 'softmax')
summary(model)

model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

history <- model %>% 
  fit(training,
      trainLabels,
      epoch = 150,
      batch_size = 10,
      validation_split = 0.2)
save(model, file = "goodmodel-1.RData")
save(history, file = "goodmodel1.RData")
plot(history)

model %>% 
  evaluate(test, testLabels)

#UNITS = 9
#11661/11661 [==============================] - 0s 29us/step
#$`loss`
#[1] 0.6857434

#$acc
#[1] 0.5531258

#UNITS = 30
#11661/11661 [==============================] - 0s 33us/step
#$`loss`
#[1] 0.679858

#$acc
#[1] 0.5352886

#UNITS = c(30, 9)
#11661/11661 [==============================] - 1s 54us/step
#$`loss`
#[1] 0.674248

#$acc
#[1] 0.5921448

#UNITS = c(100, 30)
#11661/11661 [==============================] - 0s 17us/step
#$`loss`
#[1] 0.6745392

#$acc
#[1] 0.5873424

#UNITS = c(100, 60, 32)
#11661/11661 [==============================] - 0s 18us/step
#$`loss`
#[1] 0.6724508

#$acc
#[1] 0.5834834

#UNITS = c(78, 30)
#11661/11661 [==============================] - 0s 16us/step
#$`loss`
#[1] 0.6764085

#$acc
#[1] 0.5887145

#UNITS = c(26, 9)
#11661/11661 [==============================] - 1s 94us/step
#$`loss`
#[1] 0.6767246

#$acc
#[1] 0.5806535

#Batas populer >1400, optimizer = sgd, loss = categorical_crossentropy
#UNITS = c(30,9)
#11661/11661 [==============================] - 0s 15us/step
#$`loss`
#[1] 0.6880543

#$acc
#[1] 0.5237115

#Batas populer >=1400, optimizer = adam, loss = categorical_crossentropy ((model = goodmodel-1.RData, history = goodmodel1.RData))
#UNITS = c(30,9)
#11661/11661 [==============================] - 0s 16us/step
#$`loss`
#[1] 0.6916751

#$acc
#[1] 0.6183003

prob <- model %>% 
  predict_proba(test)

pred <- model %>% 
  predict_classes(test)

table(Predicted = pred, Actual = testtarget)
#UNITS = 9
#          Actual
#Predicted    0    1
#        0  549  294
#        1 4917 5901
#accuracy = (549+5901)/(549+294+4917+5901) = 0.5531258
#UNITS = 30
#          Actual
#Predicted    0    1
#        0 1105 1058
#        1 4361 5137
#accuracy = (1105+5137)/(1105+1058+4361+5137) = 0.5352886
#UNITS = c(30, 9)
#          Actual
#Predicted    0    1
#        0 3079 2369
#        1 2387 3826
#accuracy = (3079+3826)/(3079+2369+2387+3826) = 0.5921448
#UNITS = c(100, 30)
#          Actual
#Predicted    0    1
#        0 3440 2786
#        1 2026 3409
#accuracy = (3440+3409)/(3440+2786+2026+3409) = 0.5873424
#UNITS = c(100, 60, 32)
#          Actual
#Predicted    0    1
#       0 2519 1910
#       1 2947 4285
#accuracy = (2519+4285)/(2519+1910+2947+4285) = 0.5834834
#UNITS = c(78, 30)
#          Actual
#Predicted    0    1
#        0 3499 2829
#        1 1967 3366
#accuracy = (3499+3366)/(3499+2829+1967+3366) = 0.5887145
#UNITS = c(26, 9)
#          Actual
#Predicted    0    1
#        0 3719 3143
#        1 1747 3052
#accuracy = (3719+3052)/(3719+3143+1747+3052) = 0.5806535

#Batas populer >1400, optimizer = sgd, loss = categorical_crossentropy
#UNITS = c(30,9)
#          Actual
#Predicted    0    1
#        0 1340  953
#        1 4601 4767
#Batas populer >1400, optimizer = sgd, loss = categorical_crossentropy
#UNITS = c(30,9)
#          Actual
#Predicted    0    1
#        0 2386 1371
#        1 3080 4824
cbind(prob, pred, testtarget)


#KERAS Regression
df <- as.matrix(df)
dimnames(df) <- NULL

summary(df)
df[,1:24] <- normalize(df[,1:24])
summary(df)

set.seed(100)
ind <- sample(2, nrow(df), replace = T, prob = c(0.7, 0.3))
training <- df[ind==1, 1:24]
test <- df[ind==2, 1:24]
trainingtarget <- df[ind==1, 25]
testtarget <- df[ind==2, 25]

model <- keras_model_sequential()
model %>%
  layer_dense(units=30, activation = 'relu', input_shape = c(24)) %>%
  layer_dense(units=9, activation = 'relu') %>% 
  layer_dense(units=1)
summary(model)

model %>% 
  compile(loss = 'mse',
          optimizer = 'adam',
          metrics = c("mae"))

history <- model %>% 
  fit(training,
      trainingtarget,
      epoch = 150,
      batch_size = 10,
      validation_split = 0.2)

plot(history)

model %>% 
  evaluate(test, testtarget)

#11661/11661 [==============================] - 0s 17us/step
#$`loss` = MSE
#[1] 15629234
#RMSE = 3953.383
sqrt(15629234)
#$mean_absolute_error
#[1] 2280.194

prob <- model %>% 
  predict_proba(test)

pred <- model %>% 
  predict(test)

cbind(prob, pred, testtarget)

plot(pred, testtarget)


#Cross-Validation
#CLASSIFICATION
#Cross-Validation

cv.error <- NULL
k <- 10 #k-folds

pbar <- create_progress_bar('text')

pbar$init(k)

df <- as.matrix(df)
dimnames(df) <- NULL

summary(df)
df[,1:24] <- normalize(df[,1:24])


for (i in 1:k){
  ind <- sample(2, nrow(df), replace = T, prob = c(0.7, 0.3))
  training <- df[ind==1, 1:24]
  test <- df[ind==2, 1:24]
  trainingtarget <- df[ind==1, 25]
  testtarget <- df[ind==2, 25]
  
  model <- keras_model_sequential()
  model %>%
    layer_dense(units=30, activation = 'relu', input_shape = c(24)) %>%
    layer_dense(units=9, activation = 'relu') %>% 
    layer_dense(units=1)
  summary(model)
  
  model %>% 
    compile(loss = 'mse',
            optimizer = 'adam',
            metrics = c("mae"))
  
  history <- model %>% 
    fit(training,
        trainingtarget,
        epoch = 150,
        batch_size = 10,
        validation_split = 0.2,
        verbose = 0)
  
  plot(history)
  
  cv.error[i] <- model %>% 
    evaluate(test, testtarget)
  pbar$step()
}