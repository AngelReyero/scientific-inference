library("SuperLearner")
library("vimp")
theme_set(theme_bw())

set.seed(123)

# -------------------------------------------------------------
# problem setup
# -------------------------------------------------------------
# set up the data
data = read.csv("Simulation/Python/extrapolation.csv")
s <- 1 # desire importance for X_1

# -------------------------------------------------------------
# get variable importance!
# -------------------------------------------------------------
# set up the learner library, consisting of the mean, boosted trees,
# elastic net, and random forest
learner.lib <- c("SL.mean", "SL.xgboost", "SL.bayesglm", "SL.ranger", "SL.glm.interaction")
# get the variable importance estimate, SE, and CI
# I'm using only 2 cross-validation folds to make things run quickly; in practice, you should use more
vimp <- vimp_rsquared(Y = data$y, X = data[,-which(names(data) == "y")],
                      indx = 1, V = 2, SL.library = c("SL.xgboost", "SL.mean", "SL.glm.interaction"))


vimp <- cv_vim(type = "deviance", Y = data$y, X = data[,-which(names(data) == "y")],
               indx = 1, V = 10, run_regression = TRUE,
               SL.library = c("SL.mean", "SL.xgboost", "SL.bayesglm", "SL.ranger", "SL.glm.interaction"),
               alpha = 0.05, delta = 0, na.rm = FALSE, stratified = TRUE,
               final_point_estimate = "split", ipc_weights = ipc_weights,
               C = rep(1, length(data$y)), Z = NULL,
               ipc_weights = rep(1, length(Y)), scale = "logit",
               ipc_est_type = "aipw", scale_est = TRUE, cross_fitted_se = TRUE)
