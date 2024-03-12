# load required functions and libraries
library("SuperLearner")
library("vimp")
library("xgboost")
library("glmnet")

# -------------------------------------------------------------
# problem setup
# -------------------------------------------------------------
# set up the data
n <- 100
p <- 2
s <- 1 # desire importance for X_1
x <- as.data.frame(replicate(p, runif(n, -1, 1)))
y <- (x[,1])^2*(x[,1]+7/5) + (25/9)*(x[,2])^2 + rnorm(n, 0, 1)

# -------------------------------------------------------------
# get variable importance!
# -------------------------------------------------------------
# set up the learner library, consisting of the mean, boosted trees,
# elastic net, and random forest
learner.lib <- c("SL.mean", "SL.xgboost", "SL.glmnet", "SL.randomForest")
# get the variable importance estimate, SE, and CI
# I'm using only 2 cross-validation folds to make things run quickly; in practice, you should use more
set.seed(20231213)
vimp <- vimp_rsquared(Y = y, X = x, indx = 1, V = 1, sample_splitting = TRUE, final_point_estimate = "split")



# -------------------------------------------------------------
# from vignette
# -------------------------------------------------------------

data("vrc01")
library("dplyr")
library("tidyselect")

y <- vrc01$ic50.censored
X <- vrc01 %>%
  select(starts_with("geog"), starts_with("subtype"), starts_with("length"))

subtype_01_AE_marg <- vimp_auc(Y = y, X = X[, 5, drop = FALSE], indx = 1, SL.library = learners, na.rm = TRUE, V = V, cvControl = sl_cvcontrol)
