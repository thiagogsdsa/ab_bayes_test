# examples/3_proportion_conjugate_low_prob.py
import pandas as pd
import numpy as np
from ab_bayes_test import ABBayesTest

# 1. Simulate A/B test data
np.random.seed(42)

# Group A: success rate ~50%
group_a = np.random.binomial(1, 0.5, size=100)

# Group B: success rate ~30% (lower than A)
group_b = np.random.binomial(1, 0.3, size=100)

df = pd.DataFrame({
    "group": ["A"]*100 + ["B"]*100,
    "value": np.concatenate([group_a, group_b])
})

# 2. Initialize and fit the A/B test
ab_test = ABBayesTest(
    df=df,
    group_col="group",
    value_col="value",
    metric_type="proportion",
    inference_type="conjugate",
    prior_params={"alpha": 1, "beta": 1},
    sampling_size=10000,
    control_group="A",
    treatment_group="B"
)
ab_test.fit()

# 3. Print results
print("Group summaries:")
for group, summary in ab_test.results().items():
    print(f"{group}: mean={summary['mean']:.3f}, std={summary['std']:.3f}")

lift_summary = ab_test.lift_summary()
print(f"\nLift mean={lift_summary['mean_lift']:.3f}, std={lift_summary['std_lift']:.3f}")
print(f"P(Treatment > Control)={lift_summary['prob_treatment_superior']:.3f}")
