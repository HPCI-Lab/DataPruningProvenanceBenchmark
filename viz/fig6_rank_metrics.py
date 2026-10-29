"""Fig 6: rank-style descriptors of classifier weights vs ratio."""
from common import *

args, df = setup(__doc__)
save(panel_grid(df, ["w_effective_rank", "w_stable_rank", "w_spectral_entropy", "w_top1_energy", "w_decay_rate", "w_condition_number"],
                logy=("w_condition_number",)), args.out, "fig6_rank_metrics")
