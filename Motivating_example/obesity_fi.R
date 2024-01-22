library("mlr3")
library("mlr3oml")
library("mlr3verse")
library(mlr3pipelines)
library(mlr3benchmark)
library(mlr3extralearners)
library("ggplot2")
library("iml")
library("dplyr")
library("mvtnorm")
library("matrixStats")
library(checkmate)
library(data.table)
library("coda")
library(purrr)
library(stats)
library(Metrics)
# source("utils.R") # Conditional sampler
theme_set(theme_bw())

set.seed(123)
# setwd("~/paper_2022_feature_importance_guide/Motivating_example")

#### Data ---------------------------------------------------------------------
obesity_data = read_arff("data/ObesityDataSet_raw_and_data_sinthetic.arff")
## Variables
# Gender
# Age
# Height
# Weight
# family_history_with_overweight
# FAVC (frequent high caloric food)
# FCVC (amount of vegetables per meal)
# NCP (how many main meals a day)
# CAEC (eating any food between meals)
# SMOKE (smoking)
# CH2O (how much water someone's drinking)
# SCC (monitoring the calories daily)
# FAF (physical activity)
# TUE (time spend on technological devices)
# CALC (frequency of drinking alcohol)
# MTRANS (transportation method)
# NObeyesdad (TARGET)

## Obesity
# Underweight Less than 18.5
# Normal 18.5 to 24.9
# Overweight 25.0 to 29.9
# Obesity I 30.0 to 34.9
# Obesity II 35.0 to 39.9
# Obesity III Higher than 40

# Check for duplicates
sum(duplicated(obesity_data))
obesity_data = distinct(obesity_data)

# age
summary(obesity_data$Age)
# different calculation of obesity for children. Let's just use data for people above 18
obesity_data = filter(obesity_data, Age >= 19)

# bmi
bmi = obesity_data$Weight/(obesity_data$Height^2)
summary(bmi)
# no unrealistic BMI values

# plot
p = ggplot(data = obesity_data, aes(x = Height, y = Weight, color = NObeyesdad)) + geom_point()
ggsave("figures/overview.pdf", p, width = 6, height = 4)

### Transform in a binary problem
levels(obesity_data$NObeyesdad)
# "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III" => 1
# "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II" => 0
obesity_data$obese = obesity_data$NObeyesdad %in% c("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III")
obesity_data = select(obesity_data, -NObeyesdad)
table(obesity_data$obese)

# when svm is used, no factors are allowed
for(i in 1:length(obesity_data)){
  if(class(obesity_data[[i]]) == "factor") obesity_data[[i]] = as.numeric(obesity_data[[i]])
}

# rename family_history_with_overweight in obese_hist
colnames(obesity_data)[colnames(obesity_data)=="family_history_with_overweight"] = "obese_hist"

# save
save(obesity_data, file = "data/obesity_data.RData")

#### Model --------------------------------------------------------------------
# train/test split for model interpretation
n_row = dim(obesity_data)[1]
train_set  = sample(n_row, 0.8 * n_row)
test_set = setdiff(seq_len(n_row), train_set )

# data now just contains training data
data = obesity_data[train_set,]

# Task
task = as_task_classif(data, target = "obese", id = "obesity")

# Base-Learner
base_learner = lrn("classif.featureless", predict_type = "prob")

# Learner
# learner = lrn("classif.ranger", predict_type = "prob", # alternative: "response",
#               mtry        = to_tune(1, 16, logscale = TRUE),
#               num.trees   = to_tune(1, 1000, logscale = TRUE),
#               num.threads = to_tune(1, 10, logscale = TRUE),
#               importance  = "permutation"
# )
learner = lrn("classif.svm", predict_type = "prob", # alternative: "response",
              cachesize   = to_tune(30, 50, logscale = FALSE),
              gamma       = to_tune(1, 10, logscale = TRUE),
              cost        = to_tune(1, 10, logscale = TRUE),
              type        = "C-classification",
              kernel      = "radial"
)

# Auto-Tuner
at = auto_tuner(
  tuner = tnr("random_search"),
  learner = learner,
  resampling = rsmp("repeated_cv", folds = 10, repeats = 4),
  measure = msr("classif.ce"),
  terminator = trm("evals", n_evals= 5),
)

# Outer resampling (-> Nested resampling)
outer_resampling = rsmp("repeated_cv", folds = 10, repeats = 4)

# Compute
base_rr = resample(task, base_learner, outer_resampling, store_models = TRUE)
start_time = Sys.time()
rr = resample(task, at, outer_resampling, store_models = TRUE)
duration = Sys.time()-start_time

# Evaluation
tab = as.data.table(rr)
outer_learners = mlr3misc::map(tab$learner, "learner")

# Performance
extract_inner_tuning_results(rr)[,1:5]
# compare to outer
rr$score()[,7:9]
# performance of final model
base_rr$aggregate()
rr$aggregate()

# Optimal Model
# minimize classif_ce
ind = which.min(rr$score()$classif.ce)
tuned_model = outer_learners[[ind]]

# Save / load
save(obesity_data,train_set,test_set,tuned_model,duration, file = "trained_model.RData")
# also good to save: task,learner,at,outer_resampling,rr
# load("trained_model.RData")

#### Interpretation ------------------------------------------------------------

##### Feature Effects
data = obesity_data[test_set,]
x = data[,!"obese"]
model = Predictor$new(tuned_model, data = x, y = data$obese)

#str(data)
num_feat = c()

for(i in 1:length(names(data))) num_feat[i] = class(data[[i]]) == "numeric"
num_features = names(data)[num_feat]

effect_pdp_ice = FeatureEffects$new(model, method = "pdp+ice")
png(file="figures/feature_effects_pdp+ice_test.png", width=1500, height=1000)
plot(effect_pdp_ice, features = num_features)
dev.off()

effect_ale = FeatureEffects$new(model) # ale is default
png(file="figures/feature_effects_ale_test.png", width=1500, height=1000)
plot(effect_ale, features = num_features)
dev.off()
# Weight strongest effect, rest nearly no effect (age and height little effect)


##### Feature Importance (self made)

### fi_fname_func for all features in X_test.
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
  preds_original <- model$predict(X_test)[[2]]
  preds_perm <- model$predict(X_test_perm)[[2]]

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
  new_model <- glm(f, family="binomial", data = new_training_data) ### change here if y is not binomial

  ### predict
  preds_for_original <- original_model$predict(X_test)[[2]]
  predict_for_loco <- predict(new_model, loco_X_test, type = "response")

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
# X_train <- training_data[ , -which(names(training_data) == "y")]
X_test <- x
# y_train <- training_data[ , which(names(training_data) == "y")]
y_test <- data[ , "obese"]


# loco
loco_results <- n_times(fi, 10, FALSE, loco, model, X_test, y_test$obese,
                        as.data.frame(data), 'obese', "ce")
p_loco = barplot_top6(loco_results, colnames(X_test)) + coord_flip()
ggsave('figures/obesity_loco.pdf', p_loco, width=3, height=2)

# pfi
pfi_results <- n_times(fi, 10, FALSE, pfi_fname, model, X_test, y_test$obese, "ce")
p_pfi = barplot_top6(pfi_results, colnames(X_test)) + coord_flip()
ggsave('figures/obesity_pfi.pdf', p_pfi, width=3, height=2)

# random forest fi
# rffi_results = importance(tuned_model$model)
# p_rffi = barplot_top6(list(rffi_results,0), names(rffi_results)) + coord_flip()
# ggsave('figures/obesity_rffi.pdf', p_rffi, width=3, height=2)
