# examples/1_mean_conjugate.py
import pandas as pd
import numpy as np
from ab_bayes_test import ABBayesTest

# Simulate numeric A/B test data
np.random.seed(42)
group_a = np.random.normal(loc=50, scale=5, size=100)
group_b = np.random.normal(loc=55, scale=5, size=100)
df = pd.DataFrame({
    "group": ["A"]*100 + ["B"]*100,
    "value": np.concatenate([group_a, group_b])
})

# Fit test
ab_test = ABBayesTest(
    df=df,
    group_col="group",
    value_col="value",
    metric_type="mean",
    inference_type="conjugate",
    prior_params={"mean": 0, "var": 10},
    sampling_size=100000,
    control_group="A",
    treatment_group="B"
)
ab_test.fit()

# Summaries
for group, summary in ab_test.results().items():
    print(f"{group}: mean={summary['mean']:.2f}, std={summary['std']:.2f}")

lift_summary = ab_test.lift_summary()
print(f"\nLift mean={lift_summary['mean_lift']:.2f}, std={lift_summary['std_lift']:.2f}")
print(f"P(Treatment > Control)={lift_summary['prob_treatment_superior']:.3f}")
