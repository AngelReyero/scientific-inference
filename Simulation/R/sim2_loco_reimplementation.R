library("iml")
library("mlr3")
library("mlr3verse")
library("ggplot2")
library("iml")
library("dplyr")
library("glmnet")
theme_set(theme_bw())

set.seed(123)

data = read.csv("Python/interaction.csv")
lp = 'Python/'

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

### Fit a linear model using the training data.
form <- paste(paste0("I(",names(df[1]), "*", names(df[2:(dim(df)[2]-1)]),")"), collapse=" + ")
i <- 2
while(i < (dim(df)[2]-1)){
  form <- paste(form,"+", paste(paste0("I(",names(df[i]), "*", names(df[(i+1):(dim(df)[2]-1)]),")"), collapse=" + "))
  i <- i+1
}
form2 <- paste(paste0("I(",names(df[-8]), "*", names(df[-8]),")"), collapse=" + ")
forma <- eval(paste("y ~", paste(paste0(names(df[-8])), collapse=" + "), "+",form2, "+",form))
model <- lm(forma, data = training_data)

### Performance
preds <- predict(model, newdata = test_data[,1:7])
rmse <- sqrt(mean((test_data$y - preds) ^ 2))
print(paste("RMSE:", rmse))
summary_mod <- summary(model)
summary_mod$r.squared
r_sq <- cor(test_data$y,preds)^2
print(paste("R-squared:", r_sq))

# original_model <- model
# FOI <- "x1"
# target <- "y"
# original_data <- df
# X_test <- original_data[,colnames(original_data) != target]
# Y_test <- original_data[,colnames(original_data) == target]

### LOCO base
loco <- function(original_model, FOI, X_test, Y_test, original_data, target){
  # data without feature of interest
  remainder <- original_data[,colnames(original_data) != FOI]

  # train / test split
  inds <- sample(nrow(remainder), 0.7 * nrow(remainder))
  new_training_data <- remainder[inds, ]
  new_test_data <- remainder[-inds, ]

  # feature and target split
  loco_X_test <- new_test_data[ , colnames(new_test_data) != target]
  loco_y_test <- new_test_data[target]

  # Generate the formula object we will give to the lm()-function.
  outcome <- names(loco_y_test)
  variables <- names(loco_X_test)
  form <- paste(paste0("I(",variables[1], "*", variables[2:(length(variables))],")"), collapse=" + ")
  i <- 2
  while(i < (length(variables))){
    form <- paste(form,"+", paste(paste0("I(",variables[i], "*", variables[(i+1):(length(variables))],")"), collapse=" + "))
    i <- i+1
  }
  form2 <- paste(paste0("I(",variables, "*", variables,")"), collapse=" + ")
  forma <- eval(paste(outcome," ~ ", paste(paste0(variables), collapse=" + "), "+",form2, "+",form))

  # Train the OLS model.
  new_model <- lm(forma, data = new_training_data)

  # Get the MSE for the model with all features.
  preds_for_original <- predict(original_model, X_test)
  original_mse <- mean((Y_test - preds_for_original) ^ 2)

  # Get the MSE for the model without the feature of interest.
  preds_for_loco <- predict(new_model, loco_X_test)
  loco_mse <- mean((loco_y_test$y - preds_for_loco) ^ 2)

  ### LOCO is given by the differences of the MSEs.
  loco_mse - original_mse
}

loco_naive <- function(original_data, original_model, X_test, Y_test, target, ...) {
  ### Iterate over all features and apply the above implemented LOCO function.
  sapply(colnames(X_test),
         function(name)loco(original_model, name, X_test, Y_test, original_data, target))
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
  rownames(results_mean) <- c('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7')
  colnames(results_mean) <- c('col_means', 'q.05', 'q.95')

  ### Use ggplot2 to create the barplot.
  ggplot(cbind(Features = rownames(results_mean), results_mean[1:7, ]),
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

loco_results <- n_times(loco_naive, 100, FALSE, df, model, X_test, Y_test, 'y')

p = barplot_results(loco_results)
p
p$data

res3 = p$data
res3$type = "loco"
colnames(res3) = c("feature", "mean", "q.05", "q.95", "type")

write.csv(res3, paste0(lp, 'df2_res3.csv'))
