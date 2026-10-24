import numpy as np

def grpo_advantages(rewards, G, scale="group", eps=1e-4):
    r = np.asarray(rewards, dtype=float).reshape(-1, G)      # (prompts, G)
    mean_g = r.mean(axis=1, keepdims=True)
    adv = r - mean_g                                          # baseline: per prompt
    std_g = r.std(axis=1, ddof=1, keepdims=True)              # torch .std() is unbiased
    if scale == "group":
        adv = adv / (std_g + eps)
    elif scale == "batch":
        adv = adv / (r.std(ddof=1) + eps)                     # one ruler for the batch
    elif scale != "none":
        raise ValueError(scale)
    zero_std_groups = np.isclose(std_g, 0).mean()             # what you should log yourself
    return adv.ravel(), zero_std_groups

rewards = [1, 1, 1, 1,   1, 1, 0, 0,   0.6, 0.6, 0.4, 0.4]
for mode in ["group", "batch", "none"]:
    a, z = grpo_advantages(rewards, G=4, scale=mode)
    print(mode, np.round(a, 2), "zero-std groups:", round(z, 3))
