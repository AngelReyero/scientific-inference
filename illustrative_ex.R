source("Simulation_true/R/sim2_plot.R")
source("Simulation/R/sim1_plot.R")

library(patchwork)

res = (ptrue + ggtitle("(a) Correct specified model") ) + (p + ggtitle("(b) Misspecified model")) & theme(legend.position = "bottom")
res + plot_layout(guides = "collect", axes = "collect")

ggsave('illustrative_example.pdf', width=5.7, height=4)
