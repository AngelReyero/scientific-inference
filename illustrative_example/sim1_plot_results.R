library("iml")
library("mlr3")
library("mlr3verse")
library("ggplot2")
library("iml")
library("dplyr")
library("scales")
library(latex2exp)
theme_set(theme_bw())

# squared scale with negative values
S_sqrt = function(x){sign(x)*sqrt(abs(x))}
IS_sqrt = function(x){x^2*sign(x)}
S_sqrt_trans = function() trans_new("S_sqrt",S_sqrt,IS_sqrt)

lp = 'illustrative_example/'

df_lm = read.csv(paste0(lp, 'df_res_sim1_rsm.csv')) #
df_lm$model = "LM"
df_rf = read.csv(paste0(lp, 'df_res_sim1_rf.csv'))
df_rf$model = "RF"

df = rbind(df_lm, df_rf)
df$X = NULL
df$type[df$type == "pfi"] = "PFI"
df$type[df$type == "cfi"] = "CFI"
df$type[df$type == "rfi"] = "RFI"
df$type[df$type == "marginal v(j)"] = "mSAGEvf"
df$type[df$type == "conditional v(j)"] = "cSAGEvf"
df$type[df$type == "conditional v(-j u j) - v(-j)"] = "cSAGEvfs"

#df$type[df$type == "loco"] = "LOCO"
#df$type[df$type == "loci"] = "LOCI"
colnames(df)[3] = "importance"

df_refit = read.csv(paste0(lp, 'df_res_sim1_lm_rf_loco_loci.csv')) #
df = rbind(df[, names(df_refit)], df_refit)
df$feature = as.factor(df$feature)
df = df[!df$type == "marginal v(-j u j) - v(-j)",]

# names[names == "RFI"] = expression(RFI^paste("{", X[1]  , ", " , X[3], "}"))
# names[names == "RFI"] = expression(RFI^paste("{", X[1],"}"))
renameType = function(names) {
  names[names == "cSAGEvfs"] = expression(cSAGEvfs^paste("-j"))
  return(names)
}
df$X = length(df$importance):1
# expression(paste("RFI(", X[1], ", ", X[3], ")"))
df$model[df$model == "LM"] = "a) LM with pair-wise interactions"
df$model[df$model == "RF"] = "b) RF (untuned)"

p = ggplot(data = df, aes(x = reorder(type, X), y = importance, fill = reorder(feature, X))) +
  geom_bar(stat = 'identity', position = position_dodge()) +
  #geom_errorbar(aes(ymin = q.05, ymax = q.95), width = .2, position = position_dodge(.9)) +
  scale_x_discrete(labels = renameType) +
  labs(x = 'IML technique', y = 'importance', fill = 'feature') +
  coord_flip() + # scale_y_continuous(trans = "S_sqrt", breaks = seq(-0.1,0.5,0.05))+
  scale_fill_discrete(breaks = c('x1', 'x2', 'x3', 'x4', 'x5'),
    labels = c(expression(X[1]), expression(X[2]), expression(X[3]), expression(X[4]), expression(X[5]))) +
  facet_wrap(~model)
p

# save absolute values
# ggsave('Simulation/figures/cfi_pfi_SAGEvalueFunc_orig.pdf', width=4, height=3)

######
# the following creates relative values (relative to most important feat.)
df_max = df %>%
  group_by(type, model) %>%
  mutate(importance = importance/max(abs(importance)))
  #filter(abs(importance) == max(abs(importance)))

p = ggplot(data = df_max, aes(x = reorder(type, X), y = importance, fill = reorder(feature, X))) +
  geom_bar(stat = 'identity', position = position_dodge()) +
  #geom_errorbar(aes(ymin = q.05, ymax = q.95), width = .2, position = position_dodge(.9)) +
  scale_x_discrete(labels = renameType) +
  labs(x = '', y = 'importance', fill = 'feature') +
  coord_flip() + # scale_y_continuous(trans = "S_sqrt", breaks = seq(-0.1,0.5,0.05))+
  scale_fill_discrete(breaks = c('x1', 'x2', 'x3', 'x4', 'x5'),
    labels = c(expression(X[1]), expression(X[2]), expression(X[3]), expression(X[4]), expression(X[5]))) +
  facet_wrap(~model, scales = "free_x")
p + theme(legend.position = "bottom")

ggsave('illustrative_example/illustrative_example.pdf', width = 5, height = 3.4)
