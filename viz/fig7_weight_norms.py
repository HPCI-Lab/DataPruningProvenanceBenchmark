"""Fig 7: weight-norm growth/decay vs ratio."""
from common import *

args, df = setup(__doc__)
save(panel_grid(df, ["w_frobenius_norm", "w_spectral_norm", "w_nuclear_norm", "w_rownorm_mean", "w_rownorm_std", "w_bias_norm"]),
     args.out, "fig7_weight_norms")
