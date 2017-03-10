setwd("~/Google Drive/UW/STAT441/data_challenge_2/")
library(R.matlab)
library(caret)
mat.data <- readMat.default("FacesDataChallenge.mat")
X.train <- mat.data$X.train
Y.train <- mat.data$Y.train
X.test <- mat.data$X.test
