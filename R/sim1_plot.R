library("iml")
library("mlr3")
library("mlr3verse")
library("ggplot2")
library("iml")
library("dplyr")
theme_set(theme_bw())

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
df_max = df %>%
  group_by(type) %>%
  filter(abs(importance) == max(abs(importance)))
for(typ in df_max$type){
  df$importance[df$type == typ] = df$importance[df$type == typ]#/df_max$importance[df_max$type == typ]
}

p = ggplot(data=df, aes(x=reorder(type, X), y=importance, fill=reorder(feature, X))) +
  geom_bar(stat='identity', position=position_dodge()) +
  geom_errorbar(aes(ymin=q.05, ymax=q.95), width=.2, position=position_dodge(.9))

p = p + labs(x='IML technique', y='importance', fill='feature')
p + coord_flip()+
  scale_fill_discrete(breaks=c('x1', 'x2', 'x3', 'x4', 'x5'))

ggsave('../figures/cfi_pfi_SAGEvalueFunc_orig.pdf', width=6, height=5)
# ggsave('../figures/cfi_pfi_SAGEvalueFunc.pdf', width=6, height=5)

