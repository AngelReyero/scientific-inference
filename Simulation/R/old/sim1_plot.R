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
S_sqrt <- function(x){sign(x)*sqrt(abs(x))}
IS_sqrt <- function(x){x^2*sign(x)}
S_sqrt_trans <- function() trans_new("S_sqrt",S_sqrt,IS_sqrt)

set.seed(123)

# setwd("~/paper_2022_feature_importance_guide/Simulation")
lp = 'Python/'

df1 = read.csv(paste0(lp, 'df_res.csv')) #pfi,cfi,rfi
df2 = read.csv(paste0(lp, 'df_res2.csv')) #SAGEvf, SAGEvf surplus
df3 = read.csv(paste0(lp, 'df_res3.csv')) # loco
# df4 = read.csv(paste0(lp, 'df_res4.csv')) #loci
df5 = read.csv(paste0(lp, 'df_res_SAGE.csv')) # SAGE
df = rbind(df1, df2[c(15:21,1:7),], df5, df3) #, df2[c(22:28,8:14),])
df$type[df$type == "pfi"] <- "PFI"
df$type[df$type == "cfi"] <- "CFI"
df$type[df$type == "rfi"] <- "RFI"
df$type[df$type == "marginal v(j)"] <- "mSAGEvf"
df$type[df$type == "conditional v(j)"] <- "cSAGEvf"
df$type[df$type == "loco"] <- "LOCO"
colnames(df)[3] = "importance"
df$X = length(df$importance):1
names = rev(unique(df$type))
names[names == "RFI"] = expression(RFI^paste("{", X[1]  , ", " , X[3], "}"))

# expression(paste("RFI(", X[1], ", ", X[3], ")"))

p = ggplot(data=df, aes(x=reorder(type, X), y=importance, fill=reorder(feature, X))) +
  geom_bar(stat='identity', position=position_dodge()) +
  scale_x_discrete(labels=names)#+
#geom_errorbar(aes(ymin=q.05, ymax=q.95), width=.2, position=position_dodge(.9))

p = p + labs(x='IML technique', y='importance', fill='feature')
# comment out scale_y_continuous in the following if you want "normal" scales
p + coord_flip() + # scale_y_continuous(trans="S_sqrt",breaks=seq(-0.1,0.5,0.05))+
  scale_fill_discrete(breaks=c('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'),
                      labels = c(expression(X[1]), expression(X[2]), expression(X[3]), expression(X[4]), expression(X[5]), expression(X[6]), expression(X[7])))

# save absolute values
ggsave('figures/cfi_pfi_SAGEvalueFunc_orig.pdf', width=4, height=3.5)

######
# the following creates relative values (relative to most important feat.)
df_max = df %>%
  group_by(type) %>%
  filter(abs(importance) == max(abs(importance)))
for(typ in df_max$type){
  df$importance[df$type == typ] = df$importance[df$type == typ]/df_max$importance[df_max$type == typ]
  df$q.05[df$type == typ] = df$q.05[df$type == typ]/df_max$importance[df_max$type == typ]
  df$q.95[df$type == typ] = df$q.95[df$type == typ]/df_max$importance[df_max$type == typ]
}

p = ggplot(data=df, aes(x=reorder(type, X), y=importance, fill=reorder(feature, X))) +
  geom_bar(stat='identity', position=position_dodge()) +
  scale_x_discrete(labels=names)#+
  #geom_errorbar(aes(ymin=q.05, ymax=q.95), width=.2, position=position_dodge(.9))

p = p + labs(x='IML technique', y='importance', fill='feature')
# comment out scale_y_continuous in the following if you want "normal" scales
p + coord_flip() + # scale_y_continuous(trans="S_sqrt",breaks=seq(-0.1,0.5,0.05))+
  scale_fill_discrete(breaks=c('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'),
                      labels = c(expression(X[1]), expression(X[2]), expression(X[3]), expression(X[4]), expression(X[5]), expression(X[6]), expression(X[7])))


# save relative values
ggsave('figures/cfi_pfi_SAGEvalueFunc.pdf', width=4, height=3.5)

### error bars
p2 = ggplot(data=df, aes(x=reorder(type, X), y=importance, fill=reorder(feature, X))) +
  geom_bar(stat='identity', position=position_dodge()) +
  scale_x_discrete(labels=names)+
  geom_errorbar(aes(ymin=q.05, ymax=q.95), width=.2, position=position_dodge(.9))

p2 = p2 + labs(x='IML technique', y='importance', fill='feature')
# comment out scale_y_continuous in the following if you want "normal" scales
p2 + coord_flip() + # scale_y_continuous(trans="S_sqrt",breaks=seq(-0.1,0.5,0.05))+
  scale_fill_discrete(breaks=c('x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'),
                      labels = c(expression(X[1]), expression(X[2]), expression(X[3]), expression(X[4]), expression(X[5]), expression(X[6]), expression(X[7])))

ggsave('figures/cfi_pfi_SAGEvalueFunc_errorBars.pdf', width=4, height=3.5)
