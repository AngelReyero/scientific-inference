library(mlr3)
library("mlr3oml")
library("mlr3verse")
library(mlr3pipelines)
library(mlr3benchmark)
library(mlr3extralearners)
library("ggplot2")
library("iml")
library(future)

theme_set(theme_bw())

# setwd("~/paper_2022_feature_importance_guide/Motivating_example")

### Preprocessing ##############################################################
bike = read.csv("data/bike_sharing_dataset/day.csv", stringsAsFactors = FALSE)
bike$weekday = factor(bike$weekday, levels = 0:6, labels = c('SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'))
bike$holiday = factor(bike$holiday, levels = c(0,1), labels = c('NO', 'YES'))
bike$workingday = factor(bike$workingday, levels = c(0,1), labels = c('NO', 'YES'))
bike$season = factor(bike$season, levels = 1:4, labels = c('WINTER', 'SPRING', 'SUMMER', 'FALL'))
bike$weathersit = factor(bike$weathersit, levels = 1:3, labels = c('CLEAR', 'MISTY/CLOUDY', 'SNOW/RAIN+STORM'))
bike$mnth = factor(bike$mnth, levels = 1:12, labels = c('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'))
bike$yr = factor(bike$yr, levels = 0:1, labels = c('2011', "2012"))
# denormalize weather features:
# temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
bike$temp = bike$temp * (39 - (-8)) + (-8)
# atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
bike$atemp = bike$atemp * (50 - (16)) + (16)
#windspeed: Normalized wind speed. The values are divided to 67 (max)
bike$windspeed = 67 * bike$windspeed
#hum: Normalized humidity. The values are divided to 100 (max)
bike$hum = 100 * bike$hum
# Account for trend
bike$days_since_2011 = as.numeric(as.Date(bike$dteday)-min(as.Date(bike$dteday)))
# remove features
bike$instant = bike$atemp = bike$dteday = bike$casual = bike$registered = NULL

# library(ggplot2)
# ggplot(data = bike, aes(x = as.factor(hr), y = cnt)) + geom_boxplot(varwidth = T)
save(bike, file = "data/bike.RData")


### ML model estimation ########################################################
load("data/bike.RData")

set.seed(123)

# Parallelization
plan("multisession")

# Task
task_bike = as_task_regr(bike, target = "cnt")
split = partition(task_bike)

# Base-Learner
base_learner = lrn("regr.featureless")

# Learner
learner = lrn("regr.ranger",
              mtry            = to_tune(1, 10, logscale = TRUE),
              num.trees       = 500,
              importance      = "none",
              max.depth       = to_tune(100, 150, logscale = TRUE),
              min.node.size   = to_tune(1, 5, logscale = FALSE)
)
# learner$fallback = lrn("regr.featureless")

# Auto-Tuner
at = auto_tuner(
  tuner = tnr("random_search"),
  learner = learner,
  resampling = rsmp("repeated_cv", folds = 5, repeats = 2),
  measure = msr("regr.mse"),
  terminator = trm("evals", n_evals= 200),
)

# Outer resampling (-> Nested resampling)
outer_resampling = rsmp("repeated_cv", folds = 5, repeats = 2)

# Compute
base_rr = resample(task_bike$filter(split$train), base_learner, outer_resampling, store_models = TRUE)
start_time = Sys.time()
rr = resample(task_bike$filter(split$train), at, outer_resampling, store_models = TRUE)
duration = Sys.time()-start_time

# Evaluation
tab = as.data.table(rr)
outer_learners = mlr3misc::map(tab$learner, "learner")

# Performance
extract_inner_tuning_results(rr)[,1:5]
# compare to outer
rr$score()[,7:9]
# performance of learner
base_rr$aggregate()
rr$aggregate()

# Optimal Model
# minimize performance measure
ind = which.min(rr$score()$regr.mse)
tuned_model = outer_learners[[ind]]

# Save / load
save(base_rr,rr,learner,at,tuned_model,duration,split, file = "trained_model.RData")
# outer_resampling

# load("trained_model.RData")


### Interpretation #############################################################

## fi_fname_func for all features in X_test.
fi <- function(fi_fname_func, ...) {
  ### Iterate over all features in X_test and calculate their single feature PFI score.
  unlist(lapply(colnames(X_test), fi_fname_func, ...))
}

n_times <- function(func, n, return_raw, ...) {
  ### Apply the function n times.
  ### We need to take the transpose to get the result into the right shape.
  results <- t(sapply(1:n, function(i) func(...)))

  ### Return the mean_fi, the std_fi and if wanted the raw results contained in a list.
  list(colMeans(results), apply(results, 2, sd), if (return_raw) results)
}

barplot_results <- function(results, feature_names) {
  ### Create a data.frame to be able to use ggplot2 appropriately.
  results_mean_std <- data.frame(results[1], results[2])
  rownames(results_mean_std) <- feature_names
  colnames(results_mean_std) <- c('col_means', 'col_stds')

  ### Use ggplot2 to create the barplot.
  ggplot(cbind(Features = rownames(results_mean_std), results_mean_std[1:length(feature_names), ]),
         aes(x = reorder(Features, results_mean_std$col_means),
             y = results_mean_std$col_means)) +
    ### Plot the mean value bars.
    geom_bar(stat = "identity", fill = "steelblue") #+
  ### Plot the standard deviations.
  # geom_errorbar(aes(ymin = results_mean_std$col_means - results_mean_std$col_stds,
  #                   ymax = results_mean_std$col_means + results_mean_std$col_stds),
  #               width = .1) #+
  ### Set the labels correctly.
  #labs(y = "Mean Value", x = "Features")
}

barplot_top6 <- function(results, feature_names) {
  ### Create a data.frame to be able to use ggplot2 appropriately.
  results_mean_std <- data.frame(results[1], results[2])
  rownames(results_mean_std) <- feature_names
  colnames(results_mean_std) <- c('col_means', 'col_stds')
  d = cbind(Features = rownames(results_mean_std), results_mean_std[1:length(feature_names), ])
  d = d[order(results_mean_std$col_means),]
  d = d[(nrow(d)-5):nrow(d),]
  ### Use ggplot2 to create the barplot.
  ggplot(d,
         aes(x = Features,
             y = col_means)) +
    ### Plot the mean value bars.
    geom_bar(stat = "identity", fill = "steelblue") +
    scale_x_discrete(limits=d$Features) +
    ### Plot the standard deviations.
    # geom_errorbar(aes(ymin = d$col_means - d$col_stds,
    #                   ymax = d$col_means + d$col_stds),
    #               width = .1) +
    ### Set the labels correctly.
    labs(y = "", x = "")
}

### PFI -----------------------------------------------------------------------

pfi_fname <- function(fname, model, X_test, y_test, metric = "mse") {
  ### Permute the observations for feature fname.
  X_test_perm <- X_test
  X_test_perm[[fname]] <- sample(X_test_perm[[fname]])

  ### Predict on the original data situation as well as on the permuted one.
  preds_original <- predict(model, X_test)
  preds_perm <- predict(model,X_test_perm)

  if(metric == "mse"){
    ### Get the MSE for the model with all features.
    original_metric <- mean((y_test - preds_original) ^ 2)

    ### Get the MSE for the model without the feature of interest.
    loco_metric <- mean((y_test - preds_perm) ^ 2)
  } else if(metric == "ce") {
    ### Get the CE for the model with all features.
    original_metric <- ce(as.factor(as.numeric(y_test)), as.factor(round(preds_original)))

    ### Get the CE for the model without the feature of interest.
    loco_metric <- ce(as.factor(as.numeric(y_test)), as.factor(round(preds_perm)))
  } else {
    loco_metric = 0
    original_metric = 0
  }

  ### The PFI score is now defined as the increase in metric when permuting the feature.
  loco_metric - original_metric
}

### LOCO ----------------------------------------------------------------------

loco <- function(fname, original_model, X_test, y_test, original_df, y_name, metric = "mse") {
  ### Get the training data without the column with the feature of interest.
  remainder <- original_df[colnames(original_df) != fname]

  ### The usual training and testing split (with 70% training data).
  set.seed(100)
  inds <- sample(nrow(remainder), 0.7 * nrow(remainder))
  new_training_data <- remainder[inds, ]
  new_test_data <- remainder[-inds, ]

  ### Get the features and the target.
  loco_X_test <- new_test_data[ , colnames(new_test_data) != y_name]
  loco_y_test <- new_test_data[ , y_name]

  ### Generate the formula object we will give to the lm()-function.
  outcome <- names(new_training_data[y_name])
  variables <- names(loco_X_test)
  f <- as.formula(paste(outcome, paste(variables, collapse = " + "), sep = " ~ "))

  ### Train the OLS model.
  new_model <- glm(f, family="gaussian", data = new_training_data) ### change here if y is not binomial

  ### predict
  preds_for_original <- predict(original_model,X_test)
  predict_for_loco <- predict(new_model, loco_X_test)

  if(metric == "mse"){
    ### Get the MSE for the model with all features.
    original_metric <- mean((y_test - preds_for_original) ^ 2)

    ### Get the MSE for the model without the feature of interest.
    loco_metric <- mean((loco_y_test - predict_for_loco) ^ 2)
  } else if(metric == "ce") {
    ### Get the CE for the model with all features.
    original_metric <- ce(as.factor(as.numeric(y_test)), as.factor(round(preds_for_original)))

    ### Get the CE for the model without the feature of interest.
    loco_metric <- ce(as.factor(as.numeric(loco_y_test)), as.factor(round(predict_for_loco)))
  } else {
    loco_metric = 0
    original_metric = 0
  }

  ### The performance is given by the differences of the metrics.
  loco_metric - original_metric
}

### Results --------------------------------------------------------------------

### Create appropriate data sets to use our implemented functions.
X_test <- task_bike$data(rows = split$test,
                         cols = task_bike$feature_names)
# y_train <- training_data[ , which(names(training_data) == "y")]
y_test <- task_bike$data(rows = split$test,
                         cols = task_bike$target_names)


# loco
loco_results <- n_times(fi, 10, FALSE, loco, tuned_model, X_test, y_test$cnt,
                        bike, 'cnt', "mse")
p_loco = barplot_top6(loco_results, colnames(X_test)) + coord_flip()
ggsave('figures/obesity_loco.pdf', p_loco, width=3, height=2)

# pfi
pfi_results <- n_times(fi, 10, FALSE, pfi_fname, tuned_model, X_test, y_test$cnt, "mse")
p_pfi = barplot_top6(pfi_results, colnames(X_test)) + coord_flip()
# importance = FeatureImp$new(model, loss = "mse", n.repetitions = 100, compare = "difference")
# importance$plot()
ggsave('figures/obesity_pfi.pdf', p_pfi, width=3, height=2)
