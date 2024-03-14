library("iml")
library("mlr3")
library("mlr3verse")
library(corrplot)

### simulation 1
# generating extrapolation.csv

set.seed(123)

n = 10000
ntrain = 0.7 * n

x1 = rnorm(n)
x2 = x1 + rnorm(n, sd=0.001)
x3 = rnorm(n)
x4 = x3 + rnorm(n, sd=0.1)
x5 = rnorm(n)
# x6 = rnorm(n)
# x7 = x6 + rnorm(n, sd=0.1)
# y = x4 + x5 + x4*x5 + rnorm(n, sd=0.1)
y = x4 + x5 + x4*x5 + rnorm(n, sd=0.1)

data = data.frame(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, y=y)
# maxs = apply(data, 2, function(x) max(abs(x)))
# for(i in 1:ncol(data)) data[,i] = data[,i]/maxs[i]
# corrplot(cor(data), method="circle", type="lower")
#
# task = TaskRegr$new(id='correlated', backend=data, target='y')
# learner = lrn('regr.lm')
#
# train_set = sample(task$nrow, ntrain)
# test_set = setdiff(seq_len(task$nrow), train_set)
#
# learner$train(task, row_ids = train_set)
# learner$model
#
# predictor_test = Predictor$new(learner, data[test_set,], y='y')
#
# imp_test <- FeatureImp$new(predictor_test,loss = "mae", n.repetitions = 10, compare='difference')
#
# p_pfi = plot(imp_test)
# p_pfi

write.csv(data, file="Simulation/Python/extrapolation.csv")
