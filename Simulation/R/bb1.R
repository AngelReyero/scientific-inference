library(mlr3)
library(mlr3learners)
lgr::get_logger("mlr3")$set_threshold("warn")

setwd("/home/bischl/Downloads")
d = read.csv("extrapolation.csv")
d$X = NULL
sim_task = as_task_regr(d, target = "y")  

# LOCI: minbucket = minnodesize = 120 , ntree = 100, mtry = 1
# LOCO: minbucket = minnodesize = 3,  , ntree = 100, mtry = 1

loco = function(task, learner, resampling, measure) {
  if (!resampling$is_instantiated)
    resampling$instantiate(task)
  fnames = task$feature_names
  p = length(fnames)
  res = numeric(p); names(res) = fnames
  rr0 = resample(task = task, learner = learner, resampling = resampling)
  v0 = rr0$aggregate(measure)
  for (j in 1:p) {
    fnames2 = fnames[-j]
    task2 = task$clone()$select(fnames2)
    rr = resample(task = task2, learner = learner, resampling = resampling)
    res[j] = rr$aggregate(measure) - v0
  }
  return(res)
}

loci = function(task, learner, resampling, measure) {
  if (!resampling$is_instantiated)
    resampling$instantiate(task)
  fnames = task$feature_names
  p = length(fnames)
  res = numeric(p); names(res) = fnames
  # next line is slightly wrong, we should use loss optimal const,
  # but mlr3 doesnt provide this ---> lets add this to mlr3
  ll0 = lrn("regr.featureless") 
  rr0 = resample(task = task, learner = ll0, resampling = resampling)
  v0 = rr0$aggregate(measure)
  for (j in 1:p) {
    fnames2 = fnames[j]
    task2 = task$clone()$select(fnames2)
    rr = resample(task = task2, learner = learner, resampling = resampling)
    res[j] = v0 - rr$aggregate(measure)
  }
  return(res)
}

resa = rsmp("subsampling", ratio = 0.7, repeats = 20)
mm = msr("regr.mse")
lrn1 = lrn("regr.ranger", num.trees = 500, mtry.ratio = 1,  num.threads = 2,
  min.node.size = 3, min.bucket = 3)
res1 = loco(sim_task, lrn1, resa, mm)
lrn2 = lrn("regr.ranger", num.trees = 500, mtry.ratio = 1,  num.threads = 2,
  min.node.size = 120, min.bucket = 120)
res2 = loci(sim_task, lrn2, resa, mm)

res = as.data.frame(rbind(res1, res2))
rownames(res) = c("loco", "loci")
print(res)



# library(ggplot2)
# library(patchwork)

# n = 10000
# ids_all = 1:n
# ids_train = sample(1:n, n * 0.7)
# ids_test = setdiff(ids_all, ids_train)

# d_train = d[ids_train, ]
# d_test = d[ids_test, ]

# m4 = ranger(y~x4, data = d_train, num.trees = 500, mtry = 1, 
#   min.node.size = 5)
# yh4 = predict(m4, data = d_test)$predictions 
# m5 = ranger(y~x5, data = d_train, num.trees = 500, mtry = 1, 
#   min.node.size = 400, min.bucket = 400)
# yh5 = predict(m5, data = d_test)$predictions 


# ggd = d_test
# ggd$yh4 = yh4
# ggd$yh5 = yh5

# pl1 = ggplot(data = ggd, mapping = aes(x = x4, y = y))
# pl1 = pl1 + geom_point()
# pl1 = pl1 + geom_point(mapping = aes(y = yh4))
# pl1 = pl1 + geom_point(mapping = aes(y = yh4, col = "red"))

# pl2 = ggplot(data = ggd, mapping = aes(x = x5, y = y))
# pl2 = pl2 + geom_point()
# pl2 = pl2 + geom_point(mapping = aes(y = yh5, col = "red"))

# pl = pl1 + pl2
# print(pl)

# y0 = mean(d_train$y)
# e0 = mean((y0 - d_test$y)^2)
# e1 = mean((yh4 - d_test$y)^2)
# f1 = e1 / e0
# r1 = 1 - f1
# print(e0)
# print(e1)
# print(c(f1, r1, 1/f1))


# y0 = mean(d_test$y)
# e0 = mean((y0 - d_test$y)^2)
# e1 = mean((yh5 - d_test$y)^2)
# f1 = e1 / e0
# r1 = 1 - f1
# print(e0)
# print(e1)
# print(c(f1, r1, 1/f1))


