import pandas as pd
import numpy as np
from ab_bayes_test import ABBayesTest

np.random.seed(42)
group_a = np.random.binomial(1, 0.25, size=50)
group_b = np.random.binomial(1, 0.45, size=50)
df = pd.DataFrame({
    "group": ["A"]*50 + ["B"]*50,
    "value": np.concatenate([group_a, group_b])
})

ab_test = ABBayesTest(
    df=df,
    group_col="group",
    value_col="value",
    metric_type="proportion",
    inference_type="mcmc",
    prior_params={"alpha": 1, "beta": 1},
    sampling_size=5000,
    control_group="A",
    treatment_group="B"
)
ab_test.fit()

lift_summary = ab_test.lift_summary()
print(f"Lift mean={lift_summary['mean_lift']:.3f}, std={lift_summary['std_lift']:.3f}")
print(f"P(Treatment > Control)={lift_summary['prob_treatment_superior']:.3f}")