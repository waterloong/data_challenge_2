setwd("~/Google Drive/UW/STAT441/data_challenge_2/")

r0 <- read.csv2('cnn_result4.csv', header = T, sep = ',')
r1 <- read.csv2('cnn_result.csv', header = T, sep = ',')
r2 <- read.csv2('cnn_result4.csv', header = T, sep = ',')
r3 <- read.csv2('eigen_result3.csv', header = T, sep = ',')
r4 <- read.csv2('svm_result6.csv', header = T, sep = ',')
r5 <- read.csv2('knn_result2.csv', header = T, sep = ',')

mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

r.ensemble <- NULL
for (i in 1:150) {
  l0 <- r0$ClassLabel[i]
  l1 <- r1$ClassLabel[i]
  l2 <- r2$ClassLabel[i]
  l3 <- r3$ClassLabel[i]
  l4 <- r4$ClassLabel[i]
  l5 <- r5$ClassLabel[i]

  r.ensemble <- c(r.ensemble, mode(c(l1, l2, l3)))
}
result <- data.frame(1:150, r.ensemble)
colnames(result) <- colnames(r1)
write.csv(result, file = 'ensemble_result2.csv', sep = ',', row.names = F, quote = F)
