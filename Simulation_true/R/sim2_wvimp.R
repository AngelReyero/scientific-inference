library("ggplot2")
library("dplyr")
library("gam")
library("mgcv")
theme_set(theme_bw())

set.seed(123)

data = read.csv("Simulation_true/Python/extrapolation.csv")
lp = 'Simulation_true/Python/'

df <- data[ , -which(names(data) == "X")]

### The classical training and testing data split (here 70% training, 30% testing data).
set.seed(100)
# train <- sample(nrow(df), 0.7 * nrow(df))
train <- 1:(0.7*nrow(df))
training_data <- df[train, ]
test_data <- df[-train, ]

X_train <- training_data[ , -which(names(training_data) == "y")]
X_test <- test_data[ , -which(names(test_data) == "y")]
Y_train <- training_data[ , which(names(training_data) == "y")]
Y_test <- test_data[ , which(names(test_data) == "y")]

f_len <- dim(X_train)[2]

### Fit a linear model using the training data.
forma <- eval(paste0("y ~ ", paste(names(df[4:f_len]), collapse=" + "),
                     " + ",paste0("I(",names(df[4]),"*",names(df[f_len]),")")))
model <- lm(forma, data = training_data)
model$coefficients


### LOCI base
loci <- function(original_model, FOI, X_test, Y_test, original_data, target){
  # data only containing feature of interest
  remainder <- original_data[colnames(original_data) %in% c(FOI, target)]

  # train / test split
  inds <- sample(nrow(remainder), 0.7 * nrow(remainder))
  new_training_data <- remainder[inds, ]
  new_test_data <- remainder[-inds, ]

  # feature and target split
  loci_X_test <- new_test_data[colnames(new_test_data) != target]
  loci_y_test <- new_test_data[target]

  # Train the refitted model
  variables <- names(loco_X_test)
  outcome <- names(loco_y_test)
  if(FOI %in% c("x4","x5")) {
    forma <- eval(paste0(outcome," ~ ",FOI))
    new_model <- lm(forma, data = new_training_data)
    var_y <- var(loci_y_test$y)
    preds_for_loci <- predict(new_model, loci_X_test)
    loci_mse <- mean((loci_y_test$y - preds_for_loci) ^ 2)
    result <- var_y - loci_mse
  } else {
    result <- 0
  }
  result
}

loci_naive <- function(original_data, original_model, X_test, Y_test, target, ...) {
  ### Iterate over all features and apply the above implemented LOCI function.
  sapply(colnames(X_test),
         function(name)loci(original_model, name, X_test, Y_test, original_data, target))
}

### Repeat n times
n_times <- function(func, n, return_raw, ...) {
  ### Apply the function n times.
  ### We need to take the transpose to get the result into the right shape.
  results <- t(sapply(1:n, function(i) func(...)))

  # quantiles
  q.05 <- apply(results , 2 , quantile , probs = c(0.05) , na.rm = TRUE )
  q.95 <- apply(results , 2 , quantile , probs = c(0.95) , na.rm = TRUE )

  ### Return the mean_fi, the std_fi and if wanted the raw results contained in a list.
  list(colMeans(results), q.05, q.95, if (return_raw) results)
  #list(colMeans(results), apply(results, 2, sd), if (return_raw) results)
}

### Nice plot
barplot_results <- function(results) {
  ### Create a data.frame to be able to use ggplot2 appropriately.
  results_mean <- data.frame(results[1], results[2], results[3])
  rownames(results_mean) <- c('x1', 'x2', 'x3', 'x4', 'x5')
  colnames(results_mean) <- c('col_means', 'q.05', 'q.95')

  ### Use ggplot2 to create the barplot.
  ggplot(cbind(Features = rownames(results_mean), results_mean[1:f_len, ]),
         aes(x = reorder(Features, results_mean$col_means),
             y = results_mean$col_means)) +
    ### Plot the mean value bars.
    geom_bar(stat = "identity", fill = "steelblue") +
    ### Plot the standard deviations.
    geom_errorbar(aes(ymin = results_mean$q.05,
                      ymax = results_mean$q.95),
                  width = .1) +
    ### Set the labels correctly.
    labs(y = "Mean Value", x = "Features")
}

loci_results <- n_times(loci_naive, 100, FALSE, df, model, X_test, Y_test, 'y')

p = barplot_results(loci_results)
p
p$data

res3 = p$data
res3$type = "loci"
colnames(res3) = c("feature", "mean", "q.05", "q.95", "type")

write.csv(res3, paste0(lp, 'df_res4.csv'))
