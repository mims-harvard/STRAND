import numpy as np


def draw_condition(step=None, Tw=400):
    """
    Schedule: Unconditional  ->  P_to_P  ->  Regular
    Phases within warmup window Tw:
      [0,   Tw/3):   uncond-heavy warmup
      [Tw/3,2Tw/3):  ramp to p_to_p priority
      [2Tw/3,Tw):    ramp from p_to_p to regular
      [>=Tw]:        regular-dominant steady state

    Returns one of: {"regular","unconditional","reverse","p_to_p","shuffling"}.
    """

    conditions = ["regular", "unconditional", "reverse", "p_to_p", "shuffling"]

    # default (no step): mild mix, but bias to unconditional
    if step is None:
        probs = np.array([0.20, 0.60, 0.00, 0.20, 0.00], dtype=np.float64)
    else:
        # clamp negative steps
        s = max(0, int(step))

        # ---- Phase A: strong unconditional
        if s < Tw/3:
            # uncond -> 0.80, p2p -> 0.15, regular -> 0.05
            uncond = 0.80
            p2p    = 0.15
            regular= 0.05

        # ---- Phase B: shift to p2p priority
        elif s < 2*Tw/3:
            # linearly ramp (A -> B):
            #   uncond: 0.80 -> 0.30
            #   p2p:    0.15 -> 0.55
            #   regular:0.05 -> 0.15
            t = (s - Tw/3) / (Tw/3)  # in [0,1]
            uncond = (1 - t)*0.80 + t*0.30
            p2p    = (1 - t)*0.15 + t*0.55
            regular= (1 - t)*0.05 + t*0.15

        # ---- Phase C (still within warmup): shift to regular
        elif s < Tw:
            # linearly ramp (B -> C):
            #   uncond: 0.30 -> 0.10
            #   p2p:    0.55 -> 0.30
            #   regular:0.15 -> 0.55
            t = (s - 2*Tw/3) / (Tw/3)  # in [0,1]
            uncond = (1 - t)*0.30 + t*0.10
            p2p    = (1 - t)*0.55 + t*0.35
            regular= (1 - t)*0.15 + t*0.55

        # ---- Post warmup: steady state (regular-dominant)
        else:
            uncond = 0.10
            p2p    = 0.35
            regular= 0.55

        probs = np.array([regular, uncond, 0.0, p2p, 0.0], dtype=np.float64)

    # hygiene
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    probs = probs / s if s > 0 else np.array([1,0,0,0,0], dtype=np.float64)

    # numerical check
    if not np.isfinite(probs).all():
        probs = np.array([1,0,0,0,0], dtype=np.float64)

    return np.random.choice(conditions, p=probs, size=1)[0]

