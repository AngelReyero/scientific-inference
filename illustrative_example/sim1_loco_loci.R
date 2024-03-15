library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(1)

d = read.csv("illustrative_example/extrapolation.csv")
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
lrn1 = lrn("regr.ranger", num.trees = 100, mtry.ratio = 1,  num.threads = 8,
  min.node.size = 3, min.bucket = 3)
res1 = loco(sim_task, lrn1, resa, mm)
lrn2 = lrn("regr.ranger", num.trees = 100, mtry.ratio = 1,  num.threads = 8,
  min.node.size = 120, min.bucket = 120)
res2 = loci(sim_task, lrn2, resa, mm)
lrn = lrn("regr.rsm", modelfun = "TWI")
res_rsm = loco(sim_task, lrn, resa, mm)
res_rsm2 = loci(sim_task, lrn, resa, mm)

res = rbind(data.frame(
  feature = c(names(res1), names(res2)),
  importance = c(res1, res2),
  type = rep(c("LOCO", "LOCI"), each = length(res1)),
  model = "RF"),
  data.frame(
    feature = c(names(res_rsm), names(res_rsm2)),
    importance = c(res_rsm, res_rsm2),
    type = rep(c("LOCO", "LOCI"), each = length(res_rsm)),
    model = "LM"
  )
)

write.csv(res, file = "illustrative_example/df_res_sim1_lm_rf_loco_loci.csv", row.names = FALSE)
#res = as.data.frame(rbind(res1, res2))
#rownames(res) = c("LOCO", "LOCI")
#print(res)
