# generate all models with bnlearn and save probabilities

library(dplyr)
library(bnlearn)
library(Rgraphviz)

# read in simplified compas data
df <- read.csv('data/compas-scores-two-years-short.csv', stringsAsFactors=TRUE)

df <- df %>%
  mutate(two_year_recid = factor(two_year_recid)) #%>%

nodes = c("race", "sex", "age_cat", "priors_count", "two_year_recid")
e = empty.graph(nodes)

# ==========================================================================================
# Model 1 - Indirect by Race and Gender

modelstring(e) = "[race][sex][age_cat][priors_count|race:sex:age_cat][two_year_recid|priors_count]"
dag = model2network(modelstring(e), ordering = nodes)
graphviz.plot(dag)

model1 = bn.fit(x = dag, data = df)

write.csv(model1$race$prob,"data/model1_race.csv", row.names = TRUE)
write.csv(model1$sex$prob,"data/model1_sex.csv", row.names = TRUE)
write.csv(model1$age_cat$prob,"data/model1_age_cat.csv", row.names = TRUE)
write.csv(model1$priors_count$prob,"data/model1_priors_count.csv", row.names = TRUE)
write.csv(model1$two_year_recid$prob,"data/model1_two_year_recid.csv", row.names = TRUE)

# ==========================================================================================
# Model 2 - Direct by Race, and Indirect by Race + Gender
modelstring(e) = "[race][sex][age_cat][priors_count|race:sex:age_cat][two_year_recid|priors_count:race]"
dag = model2network(modelstring(e), ordering = nodes)
graphviz.plot(dag)

model2 = bn.fit(x = dag, data = df)

write.csv(model2$race$prob,"data/model2_race.csv", row.names = TRUE)
write.csv(model2$sex$prob,"data/model2_sex.csv", row.names = TRUE)
write.csv(model2$age_cat$prob,"data/model2_age_cat.csv", row.names = TRUE)
write.csv(model2$priors_count$prob,"data/model2_priors_count.csv", row.names = TRUE)
write.csv(model2$two_year_recid$prob,"data/model2_two_year_recid.csv", row.names = TRUE)

# ==========================================================================================
# Model 3 - Unaware by Race
modelstring(e) = "[race][sex][age_cat][priors_count|sex:age_cat][two_year_recid|priors_count]"
dag = model2network(modelstring(e), ordering = nodes)
graphviz.plot(dag)

model3 = bn.fit(x = dag, data = df)

write.csv(model3$race$prob, "data/model3_race.csv", row.names = TRUE)
write.csv(model3$sex$prob, "data/model3_sex.csv", row.names = TRUE)
write.csv(model3$age_cat$prob, "data/model3_age_cat.csv", row.names = TRUE)
write.csv(model3$priors_count$prob, "data/model3_priors_count.csv", row.names = TRUE)
write.csv(model3$two_year_recid$prob, "data/model3_two_year_recid.csv", row.names = TRUE)