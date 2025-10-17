import numpy as np
import pandas as pd
from ab_bayes_test.ab_bayestest import ABBayesTest

np.random.seed(42)

# -----------------------------
# Simulation parameters
# -----------------------------
n = 1000

# Define two clear scenarios for means
scenarios_params = [
    # Large difference → high probability treatment > control
    {"desc": "High diff, treatment clearly better", "mu_control": 45, "mu_treatment": 55, "sigma": 5},
    # Tiny difference → probability treatment > control ~0.5
    {"desc": "Tiny diff, treatment ~ control", "mu_control": 50, "mu_treatment": 50.1, "sigma": 5},
]

# -----------------------------
# Create DataFrame for each scenario
# -----------------------------
dfs = {}
for s in scenarios_params:
    df_s = pd.DataFrame({
        "group": ["control"]*n + ["treatment"]*n,
        "value": np.concatenate([
            np.random.normal(s["mu_control"], s["sigma"], n),
            np.random.normal(s["mu_treatment"], s["sigma"], n)
        ])
    })
    dfs[s["desc"]] = df_s

# -----------------------------
# Test runner
# -----------------------------
def run_test(df_subset, description, inference_type="conjugate", sampling_size=20000):
    ab_test = ABBayesTest(
        df=df_subset,
        group_col="group",
        value_col="value",
        metric_type="mean",  # now mean metric
        inference_type=inference_type,
        prior_params={"mean":0, "var":10},  # wide prior
        sampling_size=sampling_size,
        control_group="control",
        treatment_group="treatment"
    )
    ab_test.fit()
    results = ab_test.results(ci=0.95)
    lift = ab_test.lift_summary()
    return {
        "description": description,
        "inference": inference_type,
        "control_mean": results["control"]["mean"],
        "treatment_mean": results["treatment"]["mean"],
        "lift_mean": lift["mean_lift"],
        "lift_std": lift["std_lift"],
        "prob_treatment_superior": lift["prob_treatment_superior"]
    }

# -----------------------------
# Run all scenarios with both inference methods
# -----------------------------
all_results = []

for desc, df_s in dfs.items():
    all_results.append(run_test(df_s, desc, "conjugate"))
    all_results.append(run_test(df_s, desc, "mcmc"))

# -----------------------------
# Display results
# -----------------------------
results_df = pd.DataFrame(all_results)
print("\nComparison table:")
print(results_df)
