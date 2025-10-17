import numpy as np
import pandas as pd
from ab_bayes_test.ab_bayestest import ABBayesTest

np.random.seed(42)

# -----------------------------
# Simulate data
# -----------------------------
n = 1000

# Control and treatment
mean_control = 50
mean_treatment = 55
std_control = 5
std_treatment = 5

df = pd.DataFrame({
    "group": ["control"]*n + ["treatment"]*n,
    "value": np.concatenate([
        np.random.normal(mean_control, std_control, n),
        np.random.normal(mean_treatment, std_treatment, n)
    ])
})

# -----------------------------
# Run Bayesian test (MCMC)
# -----------------------------
ab_test = ABBayesTest(
    df=df,
    group_col="group",
    value_col="value",
    metric_type="mean",
    inference_type="mcmc",  # <-- MCMC here
    prior_params={"mean":50, "var":25},  # prior mean=50, var=5^2
    sampling_size=5000,
    control_group="control",
    treatment_group="treatment"
)

ab_test.fit()

results = ab_test.results(ci=0.95)
lift = ab_test.lift_summary()

# -----------------------------
# Display
# -----------------------------
print(f"A: mean={results['control']['mean']:.2f}, std={results['control']['std']:.2f}")
print(f"B: mean={results['treatment']['mean']:.2f}, std={results['treatment']['std']:.2f}\n")
print(f"Lift mean={lift['mean_lift']:.2f}, std={lift['std_lift']:.2f}")
print(f"P(Treatment > Control)={lift['prob_treatment_superior']:.4f}")
