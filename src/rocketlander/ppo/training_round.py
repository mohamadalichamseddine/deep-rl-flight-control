"""How training went in one round: the numbers behind metrics.csv and the training curves."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingRound:
    """What was measured in one round (None: not measured in that round).

    A round: every rocket flies n_steps steps, then the network is updated. The training values
    are over the last 100 training flights (with exploration noise), the validation values over
    the validation flights (without), measured every validate_every steps. The losses are from
    the network update just before this round's flights, so the first round has none.
    """

    step: int  # training steps so far, summed over all rockets
    training_reward: float | None  # average reward per flight
    training_landing_rate: float | None
    validation_reward: float | None
    validation_landing_rate: float | None
    actor_loss: float | None  # PPO's clipped loss: SB3's train/policy_gradient_loss
    critic_loss: float | None  # error of the reward predictions: SB3's train/value_loss
    total_loss: float | None  # actor + 0.5 × critic loss (ent_coef is 0): SB3's train/loss
