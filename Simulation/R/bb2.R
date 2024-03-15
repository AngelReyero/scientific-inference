library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(ggplot2)
library(patchwork)

lgr::get_logger("mlr3")$set_threshold("warn")
options(width = 250)

setwd("/home/bischl/Downloads")
d = read.csv("extrapolation.csv")
d$X = NULL
sim_task = as_task_regr(d, target = "y")  

resa = rsmp("holdout", ratio = 0.7)
mm = msr("regr.mse")


mytune = function(feats, g_lim, reso) {
  task2 = sim_task$clone()$select(feats)
  lrn = lrn("regr.ranger", num.trees = 200, mtry.ratio = 1, num.threads = 2,
    min.node.size = to_tune(1, 10000), min.bucket = to_tune(1, 10000))
  grid = round(seq(g_lim[1], g_lim[2], length.out = reso))
  print(grid)
  #stop()
  des = data.table(min.node.size = grid, min.bucket = grid)
  tuner = tnr("design_points", design = des)
  ti = tune(task = task2, learner = lrn, resampling = resa,
    tuner = tuner, measure = mm)
  a = as.data.frame(as.data.table(ti$archive))
  a = a[, 1:3]
  return(a)
}

reso = 40
a1 = mytune(feats = "x4", g_lim = c(1, 250), reso = reso)
a2 = mytune(feat = c("x1", "x2", "x3", "x5"), g_lim = c(1, 10), reso = reso)

pl1 = ggplot(data = a1, mapping = aes(x = min.node.size, y = regr.mse))
pl1 = pl1 + geom_point()
pl2 = ggplot(data = a2, mapping = aes(x = min.node.size, y = regr.mse))
pl2 = pl2 + geom_point()

pl = pl1 + pl2

print(pl)