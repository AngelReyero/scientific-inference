library("iml")
library("mlr3")
library("mlr3verse")
library("ggplot2")
library("iml")
library("dplyr")
library("scales")
theme_set(theme_bw())

# squared scale with negative values
S_sqrt <- function(x){sign(x)*sqrt(abs(x))}
IS_sqrt <- function(x){x^2*sign(x)}
S_sqrt_trans <- function() trans_new("S_sqrt",S_sqrt,IS_sqrt)

set.seed(123)

# read.csv(data, '../Python/extrapolation.csv')

lp = '../Python/'

df1 = read.csv(paste0(lp, 'df_res.csv'))
df2 = read.csv(paste0(lp, 'df_res2.csv'))
df3 = read.csv(paste0(lp, 'df_res3.csv')) # loco
# df4 = read.csv(paste0(lp, 'df_res4.csv')) #loci
df5 = read.csv(paste0(lp, 'df_res_SAGE.csv')) # SAGE
df = rbind(df1[c(1:5,11:15),], df1[6:10,], df3, df2[c(11:15,1:5),], df2[c(16:20,6:10),], df5)
colnames(df)[3] = "importance"
df$X = length(df$importance):1

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
  geom_bar(stat='identity', position=position_dodge()) #+
  #geom_errorbar(aes(ymin=q.05, ymax=q.95), width=.2, position=position_dodge(.9))

p = p + labs(x='IML technique', y='importance', fill='feature')
# comment out scale_y_continuous in the following if you want "normal" scales
p + coord_flip() + #scale_y_continuous(trans="S_sqrt",breaks=seq(-0.1,0.5,0.05))+
  scale_fill_discrete(breaks=c('x1', 'x2', 'x3', 'x4', 'x5', 'x6'))

# save absolute values
ggsave('../figures/cfi_pfi_SAGEvalueFunc_orig.pdf', width=6, height=5)

# save relative values
# ggsave('../figures/cfi_pfi_SAGEvalueFunc.pdf', width=6, height=5)

