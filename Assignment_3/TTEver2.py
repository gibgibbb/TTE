import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

np.random.seed(42)
n = 200  # number of patients
data_censored = pd.DataFrame({
    'id': np.arange(1, n+1),
    'period': 1,
    'treatment': np.random.binomial(1, 0.5, n),
    'outcome': np.random.normal(0, 1, n),
    'eligible': np.random.binomial(1, 0.8, n),
    'age': np.random.normal(50, 10, n),
    'x1': np.random.normal(0, 1, n),
    'x2': np.random.normal(0, 1, n),
    'x3': np.random.normal(0, 1, n),
    'censored': np.random.binomial(1, 0.1, n)
})

# Create a binary indicator variable for logistic regression modeling (e.g., age > 50)
data_censored['age_binary'] = (data_censored['age'] > 50).astype(int)

print("Head of dummy data:")
print(data_censored.head())

cluster_features = data_censored[['age', 'x1', 'x2', 'x3']]
kmeans = KMeans(n_clusters=3, random_state=42)
data_censored['cluster'] = kmeans.fit_predict(cluster_features)

# Visualize the cluster distribution using a pairplot.
sns.pairplot(data_censored, vars=['age', 'x1', 'x2', 'x3'], hue='cluster', palette='viridis')
plt.suptitle("Pairplot of Baseline Covariates Colored by Cluster", y=1.02)
plt.show()

print("Cluster counts:")
print(data_censored['cluster'].value_counts())

class TrialSequence:
    def __init__(self, estimand):
        self.estimand = estimand  # "PP" for Per-Protocol or "ITT" for Intention-to-Treat
        self.data = None
        self.expanded_data = None

    def set_data(self, data, id_col, period, treatment, outcome, eligible):
        self.data = data.copy()
        self.id_col = id_col
        self.period = period
        self.treatment = treatment
        self.outcome = outcome
        self.eligible = eligible
        return self

    def set_switch_weight_model(self, numerator_formula, denominator_formula):
        if self.estimand != "PP":
            print("Switch weight model is only applicable for PP estimand")
            return self
        self.switch_numerator_formula = numerator_formula
        self.switch_denominator_formula = denominator_formula
        self.switch_num_model = smf.glm(formula=numerator_formula, data=self.data, 
                                        family=sm.families.Binomial()).fit()
        self.switch_den_model = smf.glm(formula=denominator_formula, data=self.data, 
                                        family=sm.families.Binomial()).fit()
        return self

    def set_censor_weight_model(self, censor_event, numerator_formula, denominator_formula, pool_models="none"):
        self.censor_event = censor_event
        self.censor_num_formula = numerator_formula
        self.censor_den_formula = denominator_formula
        self.pool_models = pool_models
        self.censor_num_model = smf.glm(formula=numerator_formula, data=self.data, 
                                        family=sm.families.Binomial()).fit()
        self.censor_den_model = smf.glm(formula=denominator_formula, data=self.data, 
                                        family=sm.families.Binomial()).fit()
        return self

    def calculate_weights(self):
        if self.estimand == "PP":
            pred_switch_num = self.switch_num_model.predict(self.data)
            pred_switch_den = self.switch_den_model.predict(self.data)
            switch_weight = pred_switch_num / pred_switch_den
        else:
            switch_weight = 1.0
        pred_censor_num = self.censor_num_model.predict(self.data)
        pred_censor_den = self.censor_den_model.predict(self.data)
        censor_weight = pred_censor_num / pred_censor_den
        self.data['weight'] = switch_weight * censor_weight
        return self

    def show_weight_models(self):
        print("Censor weight model summary:")
        print(self.censor_den_model.summary())
        if self.estimand == "PP":
            print("\nSwitch weight model summary:")
            print(self.switch_den_model.summary())

    def set_outcome_model(self, adjustment_terms=None):
        formula = f"{self.outcome} ~ {self.treatment}"
        if adjustment_terms:
            formula += " + " + adjustment_terms
        self.outcome_formula = formula
        self.outcome_model = smf.wls(formula=formula, data=self.data, weights=self.data['weight']).fit()
        return self

    def set_expansion_options(self, chunk_size=500):
        expanded = []
        for _, row in self.data.iterrows():
            for t in range(11):  # simulate follow-up times 0 through 10
                new_row = row.copy()
                new_row['followup_time'] = t
                expanded.append(new_row)
        self.expanded_data = pd.DataFrame(expanded)
        return self

    def expand_trials(self):
        return self

    def load_expanded_data(self, seed=1234, p_control=0.5):
        np.random.seed(seed)
        if self.expanded_data is not None:
            control = self.expanded_data[self.expanded_data[self.treatment] == 0]
            treated = self.expanded_data[self.expanded_data[self.treatment] == 1]
            control_sampled = control.sample(frac=p_control)
            self.expanded_data = pd.concat([treated, control_sampled], ignore_index=True)
        return self

    def fit_msm(self, weight_col='weight'):
        w = self.expanded_data[weight_col]
        q99 = w.quantile(0.99)
        self.expanded_data['w_mod'] = np.where(w > q99, q99, w)
        msm_formula = f"{self.outcome} ~ {self.treatment} + followup_time"
        self.msm_model = smf.wls(formula=msm_formula, data=self.expanded_data, 
                                 weights=self.expanded_data['w_mod']).fit()
        return self

    def predict_survival(self, predict_times):
        preds = []
        for t in predict_times:
            df_t = self.expanded_data[self.expanded_data['followup_time'] == t]
            treated_mean = df_t[df_t[self.treatment] == 1][self.outcome].mean()
            control_mean = df_t[df_t[self.treatment] == 0][self.outcome].mean()
            survival_diff = treated_mean - control_mean
            preds.append({'followup_time': t, 'survival_diff': survival_diff})
        self.preds = pd.DataFrame(preds)
        return self.preds

trial_pp = TrialSequence(estimand="PP")
trial_itt = TrialSequence(estimand="ITT")

# Both trial objects now use the dataset that includes the new 'cluster' variable.
trial_pp.set_data(data_censored, id_col="id", period="period",
                    treatment="treatment", outcome="outcome", eligible="eligible")
trial_itt.set_data(data_censored, id_col="id", period="period",
                   treatment="treatment", outcome="outcome", eligible="eligible")

trial_pp.set_switch_weight_model(numerator_formula="age_binary ~ age",
                                 denominator_formula="age_binary ~ age + x1 + x3")

# For both PP and ITT: set up a censoring weight model using 'censored'
trial_pp.set_censor_weight_model(censor_event="censored",
                                 numerator_formula="censored ~ x2",
                                 denominator_formula="censored ~ x2 + x1",
                                 pool_models="none")

trial_itt.set_censor_weight_model(censor_event="censored",
                                  numerator_formula="censored ~ x2",
                                  denominator_formula="censored ~ x2 + x1",
                                  pool_models="numerator")

trial_pp.calculate_weights()
trial_itt.calculate_weights()

print("=== Per-Protocol weight models ===")
trial_pp.show_weight_models()

print("\n=== ITT weight models ===")
trial_itt.show_weight_models()

trial_pp.set_expansion_options(chunk_size=500).expand_trials()
trial_itt.set_expansion_options(chunk_size=500).expand_trials()

# For ITT, sample expanded data (keep all treated and 50% of control patients)
trial_itt.load_expanded_data(seed=1234, p_control=0.5)

trial_itt.fit_msm(weight_col='weight')
print("MSM model summary (ITT):")
print(trial_itt.msm_model.summary())

preds = trial_itt.predict_survival(predict_times=range(11))
print("Predicted survival differences:")
print(preds)

plt.figure(figsize=(8,5))
plt.plot(preds['followup_time'], preds['survival_diff'], label='Survival difference', marker='o')
plt.xlabel("Follow-up Time")
plt.ylabel("Survival Difference")
plt.title("Survival Differences over Time (ITT)")
plt.legend()
plt.grid(True)
plt.show()