library(tidyverse)
library(tuneR)
library(ggplot2)
library(caret)
library(tsfeatures)
library(e1071)
library(randomForest)
library(tree)
library(plotrix)
library(reprtree)
library(cvTools)
library(MLmetrics)
library(reshape2)
library(gridExtra)



########## Read Louis Short Files ##########
files <- list.files(path="Spiker_box_Louis/Short/")
short_wave <- lapply(files, function(x) readWave(paste("Spiker_box_Louis/Short/", x, sep='')))
names(short_wave) <- lapply(files, function(x) substr(x,1,6))



########## Read Zoe Short Files ##########
all_files_short <- list.files("zoe_spiker/Length3/")
wave_file_short <- list()
for (i in all_files_short){
  wave_file_short <- c(wave_file_short, list(readWave(file.path("zoe_spiker/Length3/",i))))}
wave_label_short <- lapply(strsplit(all_files_short, "_"), "[[", 1)
wave_label_short <- lapply(wave_label_short, function(x) strsplit(x, "")[[1]])



########## Create Data Frame Function ##########
create_df <- function(waveSeq) {
  timeSeq <- seq_len(length(waveSeq))/waveSeq@samp.rate
  df <- data.frame(time = timeSeq, 
                   Y = waveSeq@left, 
                   event_type = "none", 
                   event_time = NA, 
                   event_pos = NA)
  df$event_type <- as.character(df$event_type)
  return(df = df)
}

df <- create_df(short_wave$LRL_L3)



########## identify_event_time_zeroCrossing ##########
identify_event_time_zeroCrossing <-
  function(Y, time, 
           windowSize = 0.5, 
           thresholdEvents = 20, 
           downSampleRate = 50) {
    ind <- seq_len(which(time == time[length(time)] - windowSize) + 1)
    # until the first element of the last window
    ind <- seq(1, ind[length(ind)], by = downSampleRate)

    timeMiddle <- time[ind] + windowSize/2
    testStat <- rep(NA, length(ind))
    
    for (i in 1:length(ind)) {
      Y_subset <- Y[time >= time[ind[i]] & time < time[ind[i]] + windowSize]
      testStat[i] <- sum(Y_subset[1:(length(Y_subset) - 1)] * Y_subset[2:(length(Y_subset))] <= 0)
    }
    
    predictedEvent <- which(testStat < thresholdEvents)
    eventTimes <- timeMiddle[predictedEvent]
    
    deltaSameEvent <- windowSize
    gaps <- which(diff(eventTimes) > deltaSameEvent)
    
    event_time_interval <- c()
    event_time_interval <- min(eventTimes)
    for (i in 1:length(gaps)) {
      event_time_interval <- append(event_time_interval, c(eventTimes[gaps[i]], eventTimes[gaps[i] + 1]))
    }
    event_time_interval <- append(event_time_interval, max(eventTimes))
    event_time_interval <- matrix(event_time_interval, ncol = 2, byrow = TRUE)
    
    predictedEventTimes <- rep(FALSE, length(Y))
    for (i in 1:nrow(event_time_interval)) {
      predictedEventTimes[event_time_interval[i, 1] <= time & 
                            event_time_interval[i,2] >= time] <- TRUE
    }
    
    num_event <- length(gaps) + 1
    return(list(num_event = num_event, 
                predictedEventTimes = predictedEventTimes, 
                predictedInterval = event_time_interval))
  }

res = identify_event_time_zeroCrossing(df$Y, df$time)



########## identify_event_sd ##########
identify_event_sd <- function(Y,  xtime, 
                           windowSize = 1, 
                           thresholdEvents = 650,
                           downSampleRate = 25) {
  
  x = max(xtime) - windowSize
  indexLastWindow = max(which(xtime <= x)) + 1
  ind = seq(1, indexLastWindow, by = downSampleRate)
  
  timeMiddle <- xtime[ind] + windowSize/2 
  testStat = rep(NA, length(ind))
  
  for (i in 1:length(ind)) {
    Y_subset <- Y[xtime >= xtime[ind[i]] & xtime < xtime[ind[i]] + windowSize]
    testStat[i] <-  sd(Y_subset)
  }
  
  predictedEvent <- which(testStat > thresholdEvents)
  eventTimes <- timeMiddle[predictedEvent] # map back to the time of this 
  
  gaps <- which(diff(eventTimes) > mean(diff(eventTimes)))
  noiseInterval <- rbind(
    c(range(xtime)[1], min(eventTimes)),
    cbind(eventTimes[gaps], eventTimes[gaps+1]),
    c(max(eventTimes), range(xtime)[2])
  )
  
  eventsInterval <- cbind(noiseInterval[-nrow(noiseInterval),2], 
                          noiseInterval[-1,1])
  
  return(list(num_event = length(gaps) + 1, 
              predictedNoiseInterval = noiseInterval,
              predictedEventInterval = eventsInterval))
}

# wave_file = wave_file_short[[1]]
# Y = wave_file@left
# xtime = seq_len(length(wave_file))/wave_file@samp.rate 
# cut_result = identify_event_sd(Y, xtime)



########## Extract Signal for One ##########
extractSignal = function(limits, seq, xtime)
{
  index = (xtime > limits[1]) & (xtime < limits[2])
  return(seq[index])
}
# wave_seq_short = apply(cut_result$predictedEventInterval, 1, extractSignal, Y, xtime)



########## Extract Signal for All ##########
wave_seq_short = list()
for(i in 1:length(wave_file_short)){
  print(i)
  wave_file = wave_file_short[[i]]
  Y = wave_file@left
  xtime = seq_len(length(wave_file))/wave_file@samp.rate 
  cut_result = identify_event_time_zeroCrossing(Y, xtime)
  wave_seq_short[[i]] = apply(cut_result$predictedEventInterval, 1, extractSignal, Y, xtime)
}
# plot(wave_seq_short[[12]][[1]], type="l")
wave_seq_short[[12]] = wave_seq_short[[12]][1:3]
wave_seq_short[[11]] = wave_seq_short[[11]][1:3]



########## Left-Right Classifier ##########
LRclassify = function(waveseq)
{
  maxPos = which.max(waveseq) ## the position of the maximum value
  minPos = which.min(waveseq) ## the position of the minimum value
  call = ifelse(maxPos < minPos, "Left", "Right")
  return(call)
}



########## Update Data Frame Function ##########
update_df <- function(df,result) {
  df$event_type = result$predictedEventTimes
  for (i in 1:result$num_event) {
    t_idx = (df$time >= result$predictedInterval[i, 1]) & (df$time <= result$predictedInterval[i, 2])
    df$event_time[t_idx] <- seq_len(sum(t_idx))
    df$event_pos[t_idx] <- i
  }
  return(df = df)
}

df = update_df(df,res)
ggplot(df,aes(x=time,y=Y,col=event_type,group=1))+geom_line()



########## Feature Extraction ##########
Y_list = unlist(wave_seq_short, recursive=FALSE)
Y_lab = unlist(wave_label_short)
Y_features <- cbind(
  tsfeatures(Y_list,
             c("acf_features","entropy","lumpiness",
               "flat_spots","crossing_points")),
  tsfeatures(Y_list, "max_kl_shift", width=48),
  tsfeatures(Y_list,
             c("mean","var"), scale=FALSE, na.rm=TRUE),
  tsfeatures(Y_list,
             c("max_level_shift","max_var_shift"), trim=TRUE)) 
Y_features = Y_features[,-7]

saveRDS(Y_features,file='features.rds')
saveRDS(Y_lab,file='lab.rds')


########## Classification Models ##########


cvK = 5
ACC_knn = ACC_rf = ACC_svm = NA
F1_knn = F1_rf = F1_svm = NA
n = length(Y_lab)
for (i in 1:50) {
  cvSets = cvTools::cvFolds(n, cvK)  
  acc_knn = acc_rf = acc_svm = NA
  f1_knn = f1_rf = f1_svm = NA
  for (j in 1:cvK) {
    test_id = cvSets$subsets[cvSets$which == j]
    X_test = Y_features[test_id, ]
    X_train = Y_features[-test_id, ]
    y_test = Y_lab[test_id]
    y_train = Y_lab[-test_id]
    
    knn_fit = class::knn(train = X_train, test = X_test, cl = y_train, k = 3)
    acc_knn[j] = MLmetrics::Accuracy(y_pred = knn_fit, y_true = y_test)
    f1_knn[j] = MLmetrics::F1_Score(y_pred = knn_fit, y_true = y_test)
    
    rf_res = randomForest::randomForest(x = X_train, y = as.factor(y_train))
    rf_fit = predict(rf_res, X_test)
    acc_rf[j] = MLmetrics::Accuracy(y_pred = rf_fit, y_true = y_test)
    f1_rf[j] = MLmetrics::F1_Score(y_pred = rf_fit, y_true = y_test)
    
    svm_res = e1071::svm(x = X_train, y = as.factor(y_train))
    svm_fit = predict(svm_res, X_test)
    acc_svm[j] = MLmetrics::Accuracy(y_pred = svm_fit, y_true = y_test)
    f1_svm[j] = MLmetrics::F1_Score(y_pred = svm_fit, y_true = y_test)
  }
  ACC_knn[i] = mean(acc_knn)
  ACC_rf[i] = mean(acc_rf)
  ACC_svm[i] = mean(acc_svm)
  F1_knn[i] = mean(f1_knn)
  F1_rf[i] = mean(f1_rf)
  F1_svm[i] = mean(f1_svm)
}

ACC = data.frame(1:50,ACC_knn,ACC_rf,ACC_svm)
names(ACC) = c('id','KNN','RandomForest','SVM')
ACC = melt(ACC, id.vars = 'id', variable.name = 'Models', value.name = 'Accuracy')


F1Score = data.frame(1:50,F1_knn,F1_rf,F1_svm)
names(F1Score) = c('id','KNN','RandomForest','SVM')
F1Score = melt(F1Score, id.vars = 'id', variable.name = 'Models', value.name = 'F1_Score')

metrics = left_join(ACC,F1Score,by=c('id','Models'))

acc_hist = metrics %>% ggplot(aes(Accuracy)) + 
  geom_histogram(binwidth = 0.005) +
  facet_wrap(~Models) +
  ggtitle('Accuracy Distribution')
acc_box = metrics %>% ggplot() +
  geom_boxplot(aes(y=Accuracy)) +
  coord_flip() +
  facet_wrap(~Models)
grid.arrange(acc_hist,acc_box,nrow=2)

f1_hist = metrics %>% ggplot(aes(F1_Score)) + 
  geom_histogram(binwidth = 0.005) +
  facet_wrap(~Models) +
  ggtitle('F1 Score Distribution')
f1_box = metrics %>% ggplot() +
  geom_boxplot(aes(y=F1_Score)) +
  coord_flip() +
  facet_wrap(~Models)
grid.arrange(f1_hist,f1_box,nrow=2)
