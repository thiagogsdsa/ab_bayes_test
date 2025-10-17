import numpy as np
import pandas as pd
from ab_bayes_test.ab_bayestest import ABBayesTest

np.random.seed(42)

# -----------------------------
# Simulation parameters
# -----------------------------
n = 1000

# Define two clear scenarios
scenarios_params = [
    # Large difference → high probability
    {"desc": "High diff, treatment clearly better", "p_control": 0.05, "p_treatment": 0.15},
    # Tiny difference → probability near 0.5
    {"desc": "Tiny diff, treatment ~ control", "p_control": 0.05, "p_treatment": 0.051},
]

# -----------------------------
# Create DataFrame for each scenario
# -----------------------------
dfs = {}
for s in scenarios_params:
    df_s = pd.DataFrame({
        "group": ["control"]*n + ["treatment"]*n,
        "converted": np.concatenate([
            np.random.binomial(1, s["p_control"], n),
            np.random.binomial(1, s["p_treatment"], n)
        ])
    })
    dfs[s["desc"]] = df_s

# -----------------------------
# Test runner
# -----------------------------
def run_test(df_subset, description, inference_type="conjugate", sampling_size=5000):
    ab_test = ABBayesTest(
        df=df_subset,
        group_col="group",
        value_col="converted",
        metric_type="proportion",
        inference_type=inference_type,
        prior_params={"alpha":1, "beta":1},
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
