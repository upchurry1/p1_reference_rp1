from isaaclab.utils import configclass
from ..p_pace.agents.rsl_rl_true_lift_smooth_ppo_cfg import KneeLiftPPORunnerCfg


@configclass
class ReferenceRP1RunnerCfg(KneeLiftPPORunnerCfg):
    experiment_name = "uestc_p1_reference_rp1"
    run_name = "rp1_low_speed_start_finetune"

