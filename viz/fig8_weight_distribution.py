"""Fig 8: weight-value distribution and class-vector geometry vs ratio."""
from common import *

args, df = setup(__doc__)
save(panel_grid(df, ["w_value_entropy", "w_kurtosis", "w_std", "w_mean", "w_mean_cos_sim"]), args.out, "fig8_weight_distribution")
