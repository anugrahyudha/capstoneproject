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

ggplot(data = newss, aes(x = shares)) + geom_density()
ggplot(data = newss, aes(x = shares)) + geom_histogram(binwidth = 100)
#Hasil: target sangat skewed

summary(newss$shares)
sd(newss$shares)
#Minimum = 1
#1st Quartile = 946
#Median = 1400
#3rd Quartile = 2800
#Maximum = 843300
#Mean = 3395
#Count = 39644
#Standar deviasi = 11626.95


#Visualisasi
df <- newss
df <- df %>% 
  mutate(day = factor(case_when(weekday_is_monday == 1 ~ 1,
                         weekday_is_tuesday == 1 ~ 2,
                         weekday_is_wednesday == 1 ~ 3,
                         weekday_is_thursday == 1 ~ 4,
                         weekday_is_friday == 1 ~ 5,
                         weekday_is_saturday == 1 ~ 6,
                         weekday_is_sunday == 1 ~ 7)),
         channel = factor(case_when(data_channel_is_bus == 1 ~ "business",
                             data_channel_is_entertainment == 1 ~ "entertainment",
                             data_channel_is_lifestyle == 1 ~ "lifestyle",
                             data_channel_is_socmed == 1 ~ "socmed",
                             data_channel_is_tech == 1 ~ "tech",
                             data_channel_is_world == 1 ~ "world",
                             TRUE ~ "viral"))
         )
summary(df)
str(df)
#Kalau outlier di targetnya dihapus, akan terlihat lebih jelas
ggplot(data = df, aes(x = day, y = shares)) + geom_boxplot() + coord_flip()
ggplot(data = df, aes(x = channel, y = shares)) + geom_boxplot() + coord_flip()


ggplot(data = df, aes(x = n_tokens_title, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = n_tokens_content, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = num_hrefs, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = num_self_hrefs, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = num_imgs, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = num_keywords, y = shares)) + geom_jitter() #bisa jadi geom_bar juga

ggplot(data = df, aes(y = self_reference_min_shares, x = shares)) + geom_jitter()
ggplot(data = df, aes(y = self_reference_max_shares, x = shares)) + geom_jitter()
ggplot(data = df, aes(y = self_reference_avg_sharess, x = shares)) + geom_jitter()
#Insight menarik:
#Artikel yang mendapatkan share paling viral, tidak mempunyai artikel referensi (sebelumnya) yang viral juga (cek kanan bawah)
#Sementara itu, artikel yang mencantumkan referensi (sebelumnya) yang viral, tidak mendapatkan banyak share (cek kiri atas)
#Seharusnya, ada yang mendapatkan kanan atas
#Sepertinya untuk pemodelan, variabel ini bisa digunakan dengan batasan nilai tertentu saja

ggplot(data = df, aes(x = kw_min_min, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = kw_avg_min, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = kw_min_avg, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = kw_min_max, y = shares)) + geom_jitter()
#Susah untuk ditarik kesimpulan... salah cara?

ggplot(data = df, aes(x = LDA_00, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = LDA_01, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = LDA_03, y = shares)) + geom_jitter()
ggplot(data = df, aes(x = LDA_04, y = shares)) + geom_jitter()
#Susah untuk ditarik kesimpulan... salah cara?


#df
df <- newss

#Hapus outlier karena nilai untuk variabel Unique Tokens tidak berada dalam rentang seharusnya judgement
#Menghapus 1 observasi
df <- df %>%
  filter(n_unique_tokens <= 1)

df <- df %>%
  mutate(ol = case_when(shares > 38275 ~ 1,
                        TRUE ~ 0
  ))
df$ol <- as.double(df$ol)

df <- df %>%
  filter(ol==0)
#Hapus outlier sejumlah 308 variabel karena diluar batas rata-rata + 3 standar deviasi #1

summary(df$shares)
nrow(df)
sd(df$shares)
# #1
#Minimum = 1
#1st Quartile = 942
#Median = 1400
#3rd Quartile = 2700
#Maximum = 38200
#Mean = 2755
#Count = 39335
#Standar deviasi = 3949.67

ggplot(data = df, aes(x = shares)) + geom_density()
#Hasil: target (masih) sangat skewed, tapi sudah lebih normal

#Nilai kedua dari rata-rata + 3 standar deviasi = 14604.01


columns <- c("LDA_00","LDA_01","LDA_02","LDA_03","LDA_04","is_weekend","data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world","avg_negative_polarity","average_token_length","min_negative_polarity","max_negative_polarity","rate_positive_words","rate_negative_words","self_reference_min_shares","self_reference_max_shares","self_reference_avg_sharess","num_hrefs","num_imgs","global_subjectivity","abs_title_sentiment_polarity","num_videos","shares")
columns_new <- c("LDA_00","LDA_01","LDA_02","LDA_03","LDA_04","is_weekend","data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world","average_token_length","min_negative_polarity","max_negative_polarity","rate_positive_words","rate_negative_words","self_reference_min_shares","self_reference_max_shares","num_hrefs","num_imgs","global_subjectivity","abs_title_sentiment_polarity","num_videos","shares")

df <- df %>%
  select(columns_new)
write.csv(df, file = "df_cleaner2.csv")
df <- df %>% 
  mutate(popularity = case_when(shares >= 1400 ~ 1, #klasifikasi awal
                                TRUE ~ 0
  ))

df <- df %>% 
  mutate(popularity = case_when(shares >= 2755 ~ 2, #sebelumnya pakai >= 1400 ~ 1, TRUE ~ 0; pernah juga pakai > 1400 ~ 1
                                shares < 942 ~ 0, #kalau ada 3 kelas, nanti sesuaikan jumlah units-nya di model juga ya
                                TRUE ~ 1
                                ))
summary(df)
df <- df[,-25] #perlu dihitung terus jumlah variabelnya

df <- df %>%
  filter(popularity!=1)
df <- df %>%
  mutate(popularity = case_when(popularity == 2 ~ 1,
                         TRUE ~ 0))
summary(df)


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
  layer_dense(units=2, activation = 'softmax') #kalau mau pakai 2 kelas popular atau tidak, ganti units jadi 2
summary(model)

model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = 'rmsprop',
          metrics = 'accuracy')

history <- model %>% 
  fit(training,
      trainLabels,
      epoch = 150,
      batch_size = 10,
      validation_split = 0.2)
save(model, file = "goodmodel-3.RData")
save(history, file = "goodmodel3.RData")
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

#Batas populer >=1400, optimizer = rmsprop, loss = categorical_crossentropy ((model = goodmodel-2.RData, history = goodmodel2.RData))
#UNITS = c(30,9)
#11661/11661 [==============================] - 0s 15us/step
#$`loss`
#[1] 0.6932605

#$acc
#[1] 0.6288483

#Batas populer >=2755, unpopular < 942, optimizer = rmsprop, loss = categorical_crossentropy ((model = goodmodel-3.RData, history = goodmodel3.RData))
#UNITS = c(30,9)
#5746/5746 [==============================] - 0s 14us/step
#$`loss`
#[1] 0.7113757

#$acc
#[1] 0.6816916

#Batas populer >=1400, optimizer = rmsprop, loss = categorical_crossentropy ((model = goodmodel-2.RData, history = goodmodel2.RData))
#UNITS = c(30,9)
#11661/11661 [==============================] - 0s 16us/step
#$`loss`
#[1] 0.6781046

#$acc
#[1] 0.6276477

prob <- model %>% 
  predict_proba(test)
str(prob)
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
#Batas populer >1400, optimizer = rmsprop, loss = categorical_crossentropy
#UNITS = c(30,9)
#          Actual
#Predicted    0    1
#        0 2194 1056
#        1 3272 5139
#Batas populer >=2755, unpopular < 942, optimizer = rmsprop, loss = categorical_crossentropy ((model = goodmodel-3.RData, history = goodmodel3.RData))
#UNITS = c(30,9)
#         Actual
#Predicted    0    1
#        0 1788  712
#        1 1117 2129
#Batas populer >1400, optimizer = rmsprop, loss = categorical_crossentropy
#UNITS = c(30,9)
#          Actual
#Predicted    0    1
#        0 2287 1163
#        1 3179 5032
cbind(prob, pred, testtarget)

# Perhitungan ROC, AUC
pred.val <- prediction(prob[,2],testtarget)
auc.perf = performance(pred.val, measure = "auc")
auc.perf@y.values

#Batas populer >=2755, unpopular < 942, optimizer = rmsprop, loss = categorical_crossentropy ((model = goodmodel-3.RData, history = goodmodel3.RData))
#UNITS = c(30,9)
#[[1]]
#[1] 0.7412682

#Batas populer >1400, optimizer = rmsprop, loss = categorical_crossentropy
#UNITS = c(30,9)
#[[1]]
#[1] 0.6667672
perf = performance(pred.val,measure="tpr",x.measure = "fpr")
plot(perf)

rocCurve.df <- roc(response = testtarget,
                    predictor = prob[,2],
                    levels = levels(factor(testtarget)))
auc(rocCurve.df)
#Batas populer >=2755, unpopular < 942, optimizer = rmsprop, loss = categorical_crossentropy ((model = goodmodel-3.RData, history = goodmodel3.RData))
#UNITS = c(30,9)
#Area under the curve: 0.7413

#Batas populer >1400, optimizer = rmsprop, loss = categorical_crossentropy
#UNITS = c(30,9)
#Area under the curve: 0.6668
plot(rocCurve.df,legacy.axes=TRUE)
title("ROC Curve Mashable Classification w/ 24 Var.",line=+3)


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
          optimizer = 'rmsprop',
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

#Optimizer = 'adam'
#11661/11661 [==============================] - 0s 17us/step
#$`loss` = MSE
#[1] 15629234
#RMSE = 3953.383
#=sqrt(15629234)
#$mean_absolute_error
#[1] 2280.194

#Optimizer = 'rmsprop'
#11661/11661 [==============================] - 0s 14us/step
#$`loss`
#[1] 15540156
#RMSE = 3942.1
#=sqrt(15540156)
#$mean_absolute_error
#[1] 2231.32

prob <- model %>% 
  predict_proba(test)
str(prob)
pred <- model %>% 
  predict(test)

cbind(prob, pred, testtarget)

plot(pred, testtarget)


#Cross-Validation
#CLASSIFICATION

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


#Logistic Regression
dft <- newss
dft <- dft[,-c(1,2)]

dft <- dft %>% 
  mutate(popularity = case_when(shares <= 946 ~ 1,
                                shares >= 3395 ~ 2,
                                TRUE ~ 0
  )) %>% 
  filter(popularity > 0) %>% 
  mutate(popularity = popularity - 1)
dft <- dft[,-59]
summary(dft)

set.seed(100)
index <- caret::createDataPartition(dft$popularity, p = 0.7, list = FALSE)

train_logit <- dft[index,]
test_logit <- dft[-index,]

model_logit = glm(formula = popularity ~ ., data = train_logit, family = "binomial")
model_logit

summary(model_logit)
# *
#n_tokens_title
#n_tokens_content
#data_channel_is_bus
#kw_min_max
#global_rate_positive_words

# **
#kw_avg_max
#LDA_01

# ***
#num_hrefs
#num_self_hrefs
#average_token_length
#num_keywords
#data_channel_is_entertainment
#data_channel_is_socmed
#data_channel_is_tech
#kw_min_min
#kw_max_min
#kw_avg_min
#kw_min_avg
#kw_max_avg
#kw_avg_avg
#weekday_is_monday
#weekday_is_tuesday
#weekday_is_wednesday
#weekday_is_thursday
#weekday_is_friday 
#LDA_00
#LDA_02
#LDA_03
#global_subjectivity
#title_sentiment_polarity
#abs_title_subjectivity

#AIC: 14070
#Number of Fisher Scoring iterations: 5

model_logit$fitted.values

#result of model
train_logit$prediction <- model_logit$fitted.values > 0.5

#prediction
fitted.results <- predict(model_logit,newdata=test_logit)

?findLinearCombos

fitted.results
#nilai prediksi dari implementasi model terhadap variabel-variabel pada data test_logit.
#Bukan data popularity sebenarnya (label, pada supervised learning)
fitted.results <- ifelse(fitted.results > 0.5,1,0)

table(fitted.results, test_logit$popularity)

#accuracy
misClasificError <- mean(fitted.results != test_logit$popularity)
accr <- 1 - misClasificError
accr
#Accuracy = 0.707886
#Kalau hapus outlier di variabel independen (1 row)= 0.7082562

#AUC
pr <- prediction(fitted.results, test_logit$popularity)
pr
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
#AUC = 0.6837943 // 0.6833857


#Logistic Regression #2
dft <- newss
dft <- dft %>%
  filter(n_unique_tokens <= 1)

dft <- dft %>%
  mutate(ol = case_when(shares > 38275 ~ 1,
                        TRUE ~ 0
  ))
dft$ol <- as.double(dft$ol)

dft <- dft %>%
  filter(ol==0)
dft <- dft[,-c(1, 2, 62)] #perlu dihitung terus jumlah variabelnya
###################
columns_new <- c("LDA_00","LDA_01","LDA_02","LDA_03","LDA_04","is_weekend","data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus","data_channel_is_socmed","data_channel_is_tech","data_channel_is_world","average_token_length","min_negative_polarity","max_negative_polarity","rate_positive_words","rate_negative_words","self_reference_min_shares","self_reference_max_shares","num_hrefs","num_imgs","global_subjectivity","abs_title_sentiment_polarity","num_videos","shares")

dft <- dft %>%
  select(columns_new)

dft <- dft %>% 
  mutate(popularity = case_when(shares >= 2755 ~ 2, #sebelumnya pakai >= 1400 ~ 1, TRUE ~ 0; pernah juga pakai > 1400 ~ 1
                                shares < 942 ~ 1, #kalau ada 3 kelas, nanti sesuaikan jumlah units-nya di model juga ya
                                TRUE ~ 0
  )) %>%
  filter(popularity > 0) %>% 
  mutate(popularity = popularity - 1)
summary(dft)
dft <- dft[,-25] #perlu dihitung terus jumlah variabelnya

set.seed(100)
index <- caret::createDataPartition(dft$popularity, p = 0.7, list = FALSE)

train_logit <- dft[index,]
test_logit <- dft[-index,]

model_logit = glm(formula = popularity ~ ., data = train_logit, family = "binomial")
model_logit

summary(model_logit)

model_logit$fitted.values

#result of model
train_logit$prediction <- model_logit$fitted.values > 0.5

#prediction
fitted.results <- predict(model_logit,newdata=test_logit)

?findLinearCombos

fitted.results
#nilai prediksi dari implementasi model terhadap variabel-variabel pada data test_logit.
#Bukan data popularity sebenarnya (label, pada supervised learning)
fitted.results <- ifelse(fitted.results > 0.5,1,0)

table(fitted.results, test_logit$popularity)

#accuracy
misClasificError <- mean(fitted.results != test_logit$popularity)
accr <- 1 - misClasificError
accr
#Accuracy = 0.6566832

#AUC
pr <- prediction(fitted.results, test_logit$popularity)
pr
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
#AUC = 0.6518797