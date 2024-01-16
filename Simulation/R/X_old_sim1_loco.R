library("iml")
library("mlr3")
library("mlr3verse")
library("ggplot2")
library("iml")
library("dplyr")
library("glmnet")
theme_set(theme_bw())

set.seed(123)

data = read.csv("Python/extrapolation.csv")
lp = 'Python/'

df <- data[ , -which(names(data) == "X")]

### The classical training and testing data split (here 70% training, 30% testing data).
set.seed(100)
# train <- sample(nrow(df), 0.7 * nrow(df))
train <- 1:(0.7*nrow(df))
training_data <- df[train, ]
test_data <- df[-train, ]

### Fit a linear model using the training data.
model <- lm(y ~ ., data = training_data)
model$coefficients[7] <- model$coefficients[7] + model$coefficients[8]
model$coefficients[8] <- 0
### Fit LASSO
# model = glmnet(training_data[,1:5], training_data$y, alpha = 1, lambda = 0)

### Assess the MSE on the test data set.
preds <- predict(model, newdata = test_data[,1:7])
mse <- mean((test_data$y - preds) ^ 2)
print(paste("MSE:", mse))

### Calculate the PFI score for a single feature given by fname.
pfi_fname <- function(fname, model, X_test, y_test) {
  ### Permute the observations for feature fname.
  X_test_perm <- X_test
  X_test_perm[[fname]] <- sample(X_test_perm[[fname]])

  ### Predict on the original data situation as well as on the permuted one.
  preds_original <- predict(model, X_test)
  preds_perm <- predict(model, X_test_perm)

  ### Calculate the MSE on both data situations.
  mse_original <- mean((y_test - preds_original) ^ 2)
  mse_perm <- mean((y_test - preds_perm) ^ 2)

  ### The PFI score is now defined as the increase in MSE when permuting the feature.
  mse_perm - mse_original
}

### Calculate the PFI score defined via the function
### fi_fname_func for all features in X_test.
fi <- function(fi_fname_func, ...) {
  ### Iterate over all features in X_test and calculate their single feature PFI score.
  unlist(lapply(colnames(X_test), fi_fname_func, ...))
}

n_times <- function(func, n, return_raw, ...) {
  ### Apply the function n times.
  ### We need to take the transpose to get the result into the right shape.
  results <- t(sapply(1:n, function(i) func(...)))

  # quantiles
  q.05 <- apply(results , 2 , quantile , probs = c(0.05) , na.rm = TRUE )
  q.95 <- apply(results , 2 , quantile , probs = c(0.95) , na.rm = TRUE )

  ### Return the mean_fi, the std_fi and if wanted the raw results contained in a list.
  list(colMeans(results), apply(results, 2, sd), if (return_raw) results)
}

### Create appropriate data sets to use our implemented functions.
X_train <- training_data[ , -which(names(training_data) == "y")]
X_test <- test_data[ , -which(names(test_data) == "y")]
y_train <- training_data[ , which(names(training_data) == "y")]
y_test <- test_data[ , which(names(test_data) == "y")]

barplot_results <- function(results) {
  ### Create a data.frame to be able to use ggplot2 appropriately.
  results_mean_std <- data.frame(results[1], results[2])
  rownames(results_mean_std) <- c('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7')
  colnames(results_mean_std) <- c('col_means', 'col_stds')

  ### Use ggplot2 to create the barplot.
  ggplot(cbind(Features = rownames(results_mean_std), results_mean_std[1:7, ]),
         aes(x = reorder(Features, results_mean_std$col_means),
             y = results_mean_std$col_means)) +
    ### Plot the mean value bars.
    geom_bar(stat = "identity", fill = "steelblue") +
    ### Plot the standard deviations.
    geom_errorbar(aes(ymin = results_mean_std$col_means - results_mean_std$col_stds,
                      ymax = results_mean_std$col_means + results_mean_std$col_stds),
                  width = .1) +
    ### Set the labels correctly.
    labs(y = "Mean Value", x = "Features")
}


### LOCO ----------------------------------------------------------------------

loco <- function(fname, original_model, X_test, y_test, original_df, y_name) {
  ### Get the training data without the column with the feature of interest.
  remainder <- original_df[colnames(original_df) != fname]

  ### The usual training and testing split (with 70% training data).
  #set.seed(100)
  inds <- sample(nrow(remainder), 0.7 * nrow(remainder))
  new_training_data <- remainder[inds, ]
  new_test_data <- remainder[-inds, ]

  ### Get the features and the target.
  loco_X_test <- new_test_data[ , colnames(new_test_data) != y_name]
  loco_y_test <- new_test_data[ y_name]

  ### Generate the formula object we will give to the lm()-function.
  outcome <- names(new_training_data[y_name])
  variables <- names(loco_X_test)
  f <- as.formula(paste(outcome, paste(variables, collapse = " + "), sep = " ~ "))

  ### Train the OLS model.
  new_model <- lm(f, data = new_training_data)

  ### Get the MSE for the model with all features.
  preds_for_original <- predict(original_model, X_test)
  original_mse <- mean((y_test - preds_for_original) ^ 2)

  ### Get the MSE for the model without the feature of interest.
  predict_for_loco <- predict(new_model, loco_X_test)
  loco_mse <- mean((loco_y_test$y - predict_for_loco) ^ 2)

  ### The performance is given by the differences of the MSEs.
  loco_mse - original_mse
}

loco_naive <- function(original_df, original_model, X_test, y_test, y_name, ...) {
  ### Iterate over all features and apply the above implemented LOCO function.
  sapply(colnames(X_test),
         function(name)loco(name, original_model, X_test, y_test, original_df, y_name))
}

model$coef

loco_results <- n_times(fi, 100, TRUE, loco, model, X_test, y_test, df, 'y')
#loco_results <- n_times(fi, 10, FALSE, loco, model, as.matrix(X_test), y_test, df, 'y')

p = barplot_results(loco_results)
p$data

res3 = p$data
res3$col_stds = NULL
res3$q.05 = res3$col_means
res3$q.95 = res3$col_means
res3$type = "loco"
colnames(res3) = c("feature", "mean", "q.05", "q.95", "type")

write.csv(res3, paste0(lp, 'df_res3.csv'))


### LOCI ----------------------------------------------------------------------
# no updated

loci <- function(fname, original_model, X_test, y_test, original_df, y_name) {
  ### In LOCI the risk of the "mean model" is compared with a model only
  # considering the feature of interest

  # Model with feature of interest:
  foi <- original_df[colnames(original_df) %in% c(fname, 'y')]

  ### The usual training and testing split (with 70% training data).
  set.seed(100)
  inds <- sample(nrow(foi), 0.7 * nrow(foi))
  new_training_data <- foi[inds, ]
  new_test_data <- foi[-inds, ]

  ### Get the features and the target.
  loci_X_test <- data.frame(new_test_data[ , colnames(new_test_data) != y_name])
  loci_y_test <- data.frame(new_test_data[ , y_name])
  colnames(loci_X_test) <- fname
  colnames(loci_y_test) <- y_name

  ### Generate the formula object we will give to the lm()-function.
  outcome <- names(new_training_data[y_name])
  variables <- names(loci_X_test)
  f <- as.formula(paste(outcome, paste(variables, collapse = " + "), sep = " ~ "))

  ### Train the OLS model.
  new_model <- lm(f, data = new_training_data)

  ### Get the MSE for the 'zero'/'mean' model with no features.
  mean_pred <- mean(y_train)
  mean_mse <- mean((y_test - mean_pred) ^ 2)

  ### Get the MSE for the model without the feature of interest.
  predict_for_loci <- predict(new_model, loci_X_test)
  loci_mse <- mean((loci_y_test - predict_for_loci)$y ^ 2)

  ### The performance is given by the differences of the MSEs.
  mean_mse - loci_mse
}

loci_naive <- function(original_df, original_model, X_test, y_test, y_name, ...) {
  ### Iterate over all features and apply the above implemented LOCI function.
  sapply(colnames(X_test),
         function(name)loci(name, original_df, original_model, X_test, y_test, y_name))
}

loci_results <- n_times(fi, 10, FALSE, loci, model, as.matrix(X_test), y_test, df, 'y')

p = barplot_results(loci_results)
p$data

res4 = p$data
res4$col_stds = NULL
res4$q.05 = res4$col_means
res4$q.95 = res4$col_means
res4$type = "loci"
colnames(res4) = c("feature", "mean", "q.05", "q.95", "type")

write.csv(res4, paste0(lp, 'df_res4.csv'))
