observation = read.csv('./observations.csv', header = TRUE, sep=",")
observation2 = read.csv('./rand-observations.csv', header = TRUE, sep=",")
observation[1,2]
observation[,2]
observation[1,]
observation = round(observation, digits=4)
observation2 = round(observation2, digits=4)
wilcox.test(observation2[,1], observation2[,2], alternative="two.sided", paired = FALSE, conf.level = 0.95)
wilcox.test(observation[,1], observation[,2], alternative="two.sided", paired = TRUE, conf.level = 0.95)
list_observations = list(observation[,1], observation[,2], observation2[,1], observation2[,2])
kruskal.test(list_observations)
combined_observations <- data.frame(observation, observation2)
matrix_observations = data.matrix(combined_observations)
friedman.test(matrix_observations)

install.packages("PMCMR") 
library(PMCMR)

posthoc.kruskal.dunn.test(list_observations, p.adjust.method="holm")
posthoc.friedman.nemenyi.test(matrix_observations)

#may want to install PLUS

