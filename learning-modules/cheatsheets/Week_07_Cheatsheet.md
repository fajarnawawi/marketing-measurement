# Week 7: Statistics Foundations - Quick Reference Cheatsheet

## 📋 Core Concepts

### Probability Basics
```python
import numpy as np
from scipy import stats

# Basic probability
P_A = 0.3                           # Probability of event A
P_B = 0.4                           # Probability of event B
P_A_and_B = 0.12                    # Joint probability

# Conditional probability: P(A|B) = P(A and B) / P(B)
P_A_given_B = P_A_and_B / P_B       # 0.3

# Independence check: P(A and B) = P(A) * P(B)
is_independent = np.isclose(P_A_and_B, P_A * P_B)

# Complement rule: P(not A) = 1 - P(A)
P_not_A = 1 - P_A                   # 0.7

# Addition rule: P(A or B) = P(A) + P(B) - P(A and B)
P_A_or_B = P_A + P_B - P_A_and_B    # 0.58
```

### Common Distributions

#### Normal Distribution
```python
from scipy.stats import norm

# Parameters
mu = 100        # mean
sigma = 15      # standard deviation

# Probability density function (PDF)
x = 110
pdf = norm.pdf(x, loc=mu, scale=sigma)

# Cumulative distribution function (CDF)
# P(X <= 110)
cdf = norm.cdf(110, loc=mu, scale=sigma)

# Inverse CDF (percentile/quantile)
# What value gives us 95th percentile?
percentile_95 = norm.ppf(0.95, loc=mu, scale=sigma)  # ~124.67

# Generate random samples
samples = norm.rvs(loc=mu, scale=sigma, size=1000)

# Marketing example: Daily revenue ~ N(10000, 1500)
daily_revenue_mean = 10000
daily_revenue_std = 1500

# Probability revenue exceeds $12,000
prob_exceed = 1 - norm.cdf(12000, loc=daily_revenue_mean, scale=daily_revenue_std)
```

#### Binomial Distribution
```python
from scipy.stats import binom

# Parameters
n = 100         # number of trials (clicks)
p = 0.05        # probability of success (conversion rate)

# Probability of exactly k conversions
k = 8
prob_k = binom.pmf(k, n, p)

# Probability of k or fewer conversions
prob_k_or_less = binom.cdf(k, n, p)

# Expected value and variance
expected = binom.mean(n, p)          # n * p = 5
variance = binom.var(n, p)           # n * p * (1-p) = 4.75

# Marketing example: 1000 email opens, 2% CTR
# What's probability of getting 25 or more clicks?
prob_25_plus = 1 - binom.cdf(24, n=1000, p=0.02)
```

#### Poisson Distribution
```python
from scipy.stats import poisson

# Parameter (lambda): average rate
lambda_param = 5.2  # average conversions per hour

# Probability of exactly k events
k = 7
prob_k = poisson.pmf(k, lambda_param)

# Probability of k or fewer events
prob_k_or_less = poisson.cdf(k, lambda_param)

# Expected value = variance = lambda
expected = poisson.mean(lambda_param)  # 5.2

# Marketing example: Average 3.5 orders per hour
# Probability of more than 5 orders in next hour
prob_more_than_5 = 1 - poisson.cdf(5, 3.5)
```

### Confidence Intervals

#### Mean - Known Population Std
```python
from scipy.stats import norm
import numpy as np

# Data
data = np.array([100, 105, 98, 110, 95, 102, 108, 99])
n = len(data)
sample_mean = np.mean(data)
known_sigma = 5  # known population std

# 95% confidence interval (z-interval)
confidence = 0.95
alpha = 1 - confidence
z_critical = norm.ppf(1 - alpha/2)  # 1.96 for 95%

margin_of_error = z_critical * (known_sigma / np.sqrt(n))
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

#### Mean - Unknown Population Std (t-interval)
```python
from scipy.stats import t
import numpy as np

# Data
conversion_rates = np.array([0.05, 0.048, 0.052, 0.049, 0.051])
n = len(conversion_rates)
sample_mean = np.mean(conversion_rates)
sample_std = np.std(conversion_rates, ddof=1)  # ddof=1 for sample std

# 95% confidence interval (t-interval)
confidence = 0.95
alpha = 1 - confidence
df = n - 1  # degrees of freedom
t_critical = t.ppf(1 - alpha/2, df)

margin_of_error = t_critical * (sample_std / np.sqrt(n))
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Quick method using scipy
ci = t.interval(confidence, df, loc=sample_mean, scale=sample_std/np.sqrt(n))
```

#### Proportion
```python
from scipy.stats import norm
import numpy as np

# Sample proportion
conversions = 250
visitors = 5000
p_hat = conversions / visitors  # 0.05

# 95% confidence interval for proportion
confidence = 0.95
z_critical = norm.ppf(1 - (1-confidence)/2)  # 1.96

# Standard error
se = np.sqrt(p_hat * (1 - p_hat) / visitors)

# CI
ci_lower = p_hat - z_critical * se
ci_upper = p_hat + z_critical * se

print(f"95% CI for CVR: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"95% CI for CVR: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
```

### Hypothesis Testing Framework

#### Five Steps of Hypothesis Testing
```python
"""
1. State hypotheses (H₀ and Hₐ)
2. Choose significance level (α)
3. Calculate test statistic
4. Find p-value
5. Make decision: Reject H₀ if p-value < α
"""

# Example framework
def hypothesis_test_template(data, null_value, alternative='two-sided'):
    """
    alternative: 'two-sided', 'greater', 'less'
    """
    # Step 1: Hypotheses already stated

    # Step 2: Significance level
    alpha = 0.05

    # Step 3: Calculate test statistic (example: t-test)
    from scipy.stats import ttest_1samp
    statistic, p_value = ttest_1samp(data, null_value, alternative=alternative)

    # Step 4: p-value calculated above

    # Step 5: Decision
    reject_null = p_value < alpha

    return {
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': reject_null,
        'alpha': alpha
    }
```

### T-Tests

#### One-Sample T-Test
```python
from scipy.stats import ttest_1samp
import numpy as np

# H₀: μ = 100 (population mean equals 100)
# Hₐ: μ ≠ 100 (two-sided)

data = np.array([102, 98, 105, 99, 103, 101, 97, 104])
null_mean = 100

# Perform test
statistic, p_value = ttest_1samp(data, null_mean)

print(f"t-statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print(f"Reject H₀: Mean is significantly different from {null_mean}")
else:
    print(f"Fail to reject H₀: No significant difference from {null_mean}")

# Marketing example: Is average CPA different from $50?
cpa_data = np.array([48, 52, 49, 51, 47, 53, 50, 48])
statistic, p_value = ttest_1samp(cpa_data, 50)
```

#### Two-Sample T-Test (Independent)
```python
from scipy.stats import ttest_ind
import numpy as np

# H₀: μ₁ = μ₂ (means are equal)
# Hₐ: μ₁ ≠ μ₂ (means are different)

# Two campaigns
campaign_a = np.array([45, 48, 42, 50, 46, 49, 44, 47])
campaign_b = np.array([52, 55, 50, 58, 53, 56, 51, 54])

# Perform test (equal_var=True assumes equal variances)
statistic, p_value = ttest_ind(campaign_a, campaign_b, equal_var=True)

print(f"t-statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# For unequal variances (Welch's t-test)
statistic, p_value = ttest_ind(campaign_a, campaign_b, equal_var=False)

# Marketing example: Compare CPAs between two ad groups
```

#### Paired T-Test
```python
from scipy.stats import ttest_rel
import numpy as np

# H₀: μ_diff = 0 (no difference before and after)
# Hₐ: μ_diff ≠ 0 (there is a difference)

# Before and after optimization
before = np.array([100, 105, 98, 110, 95, 102, 108, 99])
after = np.array([95, 98, 92, 102, 90, 97, 103, 94])

# Perform paired test
statistic, p_value = ttest_rel(before, after)

print(f"t-statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

# Calculate mean difference
mean_diff = np.mean(before - after)
print(f"Mean difference: {mean_diff:.2f}")

# Marketing example: Compare metrics before and after website redesign
```

### Chi-Square Tests

#### Chi-Square Test of Independence
```python
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

# H₀: Variables are independent
# Hₐ: Variables are dependent (associated)

# Contingency table: Device type vs Conversion
#                  Converted  Not Converted
# Mobile              45          955
# Desktop             80          920
# Tablet              25          475

observed = np.array([
    [45, 955],   # Mobile
    [80, 920],   # Desktop
    [25, 475]    # Tablet
])

# Perform test
chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"Expected frequencies:\n{expected}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject H₀: Device type and conversion are associated")
else:
    print("Fail to reject H₀: No significant association")
```

#### Chi-Square Goodness-of-Fit Test
```python
from scipy.stats import chisquare
import numpy as np

# H₀: Observed frequencies match expected distribution
# Hₐ: Observed frequencies differ from expected

# Observed traffic by channel
observed = np.array([450, 320, 180, 50])  # Google, Facebook, Instagram, Other

# Expected proportions: 40%, 30%, 20%, 10%
total = observed.sum()
expected = np.array([0.40, 0.30, 0.20, 0.10]) * total

# Perform test
chi2, p_value = chisquare(observed, expected)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
```

### P-Values and Significance Levels

```python
# Significance levels (alpha)
# α = 0.05 (95% confidence) - most common
# α = 0.01 (99% confidence) - more stringent
# α = 0.10 (90% confidence) - more lenient

# Interpreting p-values
def interpret_p_value(p_value, alpha=0.05):
    """
    Interpret p-value in context.
    """
    if p_value < 0.001:
        return f"p < 0.001: Very strong evidence against H₀"
    elif p_value < 0.01:
        return f"p = {p_value:.4f}: Strong evidence against H₀"
    elif p_value < alpha:
        return f"p = {p_value:.4f}: Evidence against H₀ (reject H₀)"
    elif p_value < 0.10:
        return f"p = {p_value:.4f}: Weak evidence against H₀ (borderline)"
    else:
        return f"p = {p_value:.4f}: Insufficient evidence against H₀"

# One-tailed vs Two-tailed
# Two-tailed: H₀: μ = μ₀  vs  Hₐ: μ ≠ μ₀
# One-tailed: H₀: μ ≤ μ₀  vs  Hₐ: μ > μ₀
#         or  H₀: μ ≥ μ₀  vs  Hₐ: μ < μ₀

# Example: Convert two-tailed p-value to one-tailed
p_value_two_tailed = 0.06
p_value_one_tailed = p_value_two_tailed / 2  # 0.03
```

### Type I and Type II Errors

```python
"""
Type I Error (False Positive, α):
- Reject H₀ when H₀ is true
- Probability = α (significance level)
- Example: Conclude campaign improved CVR when it actually didn't

Type II Error (False Negative, β):
- Fail to reject H₀ when Hₐ is true
- Probability = β
- Example: Miss detecting a real improvement in CVR

Power = 1 - β
- Probability of correctly rejecting H₀ when Hₐ is true
- Typically aim for power ≥ 0.80 (80%)
"""

# Error summary table
import pandas as pd

error_table = pd.DataFrame({
    'H₀ True': ['Correct (1-α)', 'Type I Error (α)'],
    'H₀ False': ['Type II Error (β)', 'Correct (Power = 1-β)']
}, index=['Fail to Reject H₀', 'Reject H₀'])

print(error_table)

# Trade-offs
# - Lower α → Lower Type I error but higher Type II error
# - Increase sample size → Reduces both types of errors
# - Increase effect size → Easier to detect, more power
```

### Effect Size and Power

#### Cohen's d (Effect Size for t-tests)
```python
import numpy as np

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - |d| = 0.2: small
    - |d| = 0.5: medium
    - |d| = 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return d

# Example
control = np.array([45, 48, 42, 50, 46, 49, 44, 47])
treatment = np.array([52, 55, 50, 58, 53, 56, 51, 54])

d = cohens_d(treatment, control)
print(f"Cohen's d: {d:.3f}")

if abs(d) >= 0.8:
    print("Large effect size")
elif abs(d) >= 0.5:
    print("Medium effect size")
elif abs(d) >= 0.2:
    print("Small effect size")
else:
    print("Negligible effect size")
```

#### Statistical Power Analysis
```python
from statsmodels.stats.power import ttest_power, tt_ind_solve_power

# Calculate power given sample size
effect_size = 0.5    # Cohen's d
alpha = 0.05
n_per_group = 50

power = ttest_power(effect_size, n_per_group, alpha, alternative='two-sided')
print(f"Power: {power:.3f}")

# Calculate required sample size for desired power
desired_power = 0.80
required_n = tt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=desired_power,
    alternative='two-sided'
)
print(f"Required sample size per group: {np.ceil(required_n):.0f}")

# Marketing example: Sample size for A/B test
# Want to detect 0.5% absolute lift in CVR with 80% power
baseline_cvr = 0.05
lift = 0.005  # 0.5 percentage points
effect_size_cvr = lift / np.sqrt(baseline_cvr * (1 - baseline_cvr))
n_required = tt_ind_solve_power(effect_size_cvr, alpha, desired_power,
                                  alternative='two-sided')
```

---

## 🎯 Common scipy.stats Functions

### Quick Reference Table

| Function | Purpose | Example |
|----------|---------|---------|
| `norm.pdf(x, loc, scale)` | Normal PDF | `norm.pdf(100, 100, 15)` |
| `norm.cdf(x, loc, scale)` | Normal CDF (percentile) | `norm.cdf(110, 100, 15)` |
| `norm.ppf(q, loc, scale)` | Inverse CDF (quantile) | `norm.ppf(0.95, 100, 15)` |
| `norm.rvs(loc, scale, size)` | Random samples | `norm.rvs(100, 15, 1000)` |
| `ttest_1samp(data, popmean)` | One-sample t-test | `ttest_1samp(data, 100)` |
| `ttest_ind(g1, g2)` | Independent t-test | `ttest_ind(control, treatment)` |
| `ttest_rel(before, after)` | Paired t-test | `ttest_rel(pre, post)` |
| `chi2_contingency(obs)` | Chi-square independence | `chi2_contingency(table)` |
| `chisquare(obs, exp)` | Chi-square goodness of fit | `chisquare(obs, exp)` |

### Complete Function Signatures
```python
from scipy import stats

# Normal distribution
stats.norm.pdf(x, loc=0, scale=1)          # Probability density
stats.norm.cdf(x, loc=0, scale=1)          # Cumulative probability
stats.norm.ppf(q, loc=0, scale=1)          # Quantile (inverse CDF)
stats.norm.rvs(loc=0, scale=1, size=1)     # Random variates
stats.norm.mean(loc=0, scale=1)            # Mean
stats.norm.var(loc=0, scale=1)             # Variance

# T-tests
stats.ttest_1samp(a, popmean, alternative='two-sided')
stats.ttest_ind(a, b, equal_var=True, alternative='two-sided')
stats.ttest_rel(a, b, alternative='two-sided')

# Chi-square
stats.chi2_contingency(observed, correction=True)
stats.chisquare(f_obs, f_exp=None)

# Other distributions
stats.binom.pmf(k, n, p)                   # Binomial probability
stats.poisson.pmf(k, mu)                   # Poisson probability
stats.t.ppf(q, df)                         # t-distribution quantile
```

---

## 💡 Marketing Statistics Patterns

### Pattern 1: Comparing Campaign Performance
```python
import numpy as np
from scipy.stats import ttest_ind

def compare_campaigns(campaign_a_metrics, campaign_b_metrics, metric_name="metric"):
    """
    Compare two campaigns using independent t-test.

    Returns statistical test results and interpretation.
    """
    # Perform t-test
    statistic, p_value = ttest_ind(campaign_a_metrics, campaign_b_metrics)

    # Calculate descriptive statistics
    mean_a = np.mean(campaign_a_metrics)
    mean_b = np.mean(campaign_b_metrics)
    std_a = np.std(campaign_a_metrics, ddof=1)
    std_b = np.std(campaign_b_metrics, ddof=1)

    # Effect size
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = (mean_b - mean_a) / pooled_std

    # Results
    results = {
        'campaign_a_mean': mean_a,
        'campaign_b_mean': mean_b,
        'difference': mean_b - mean_a,
        'percent_change': ((mean_b - mean_a) / mean_a) * 100,
        't_statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }

    # Interpretation
    print(f"Campaign A {metric_name}: {mean_a:.2f} (±{std_a:.2f})")
    print(f"Campaign B {metric_name}: {mean_b:.2f} (±{std_b:.2f})")
    print(f"Difference: {results['difference']:.2f} ({results['percent_change']:.1f}%)")
    print(f"p-value: {p_value:.4f}")

    if results['significant']:
        print(f"✓ Campaign B is significantly different (α=0.05)")
    else:
        print(f"✗ No significant difference detected (α=0.05)")

    return results

# Example usage
campaign_a_cpas = np.array([45, 48, 42, 50, 46, 49, 44, 47])
campaign_b_cpas = np.array([52, 55, 50, 58, 53, 56, 51, 54])

results = compare_campaigns(campaign_a_cpas, campaign_b_cpas, "CPA")
```

### Pattern 2: Before/After Analysis
```python
import numpy as np
from scipy.stats import ttest_rel

def before_after_analysis(before, after, metric_name="metric"):
    """
    Analyze before/after change using paired t-test.
    """
    # Paired t-test
    statistic, p_value = ttest_rel(before, after)

    # Calculate changes
    differences = after - before
    mean_diff = np.mean(differences)
    mean_before = np.mean(before)
    percent_change = (mean_diff / mean_before) * 100

    # Results
    results = {
        'mean_before': mean_before,
        'mean_after': np.mean(after),
        'mean_difference': mean_diff,
        'percent_change': percent_change,
        't_statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'improved': mean_diff > 0
    }

    # Interpretation
    print(f"Before: {results['mean_before']:.2f}")
    print(f"After: {results['mean_after']:.2f}")
    print(f"Change: {mean_diff:.2f} ({percent_change:+.1f}%)")
    print(f"p-value: {p_value:.4f}")

    if results['significant']:
        direction = "improvement" if results['improved'] else "decline"
        print(f"✓ Statistically significant {direction}")
    else:
        print(f"✗ No significant change detected")

    return results

# Example: Website redesign impact on conversion rate
before_redesign = np.array([0.048, 0.052, 0.049, 0.051, 0.050, 0.047, 0.053])
after_redesign = np.array([0.055, 0.058, 0.054, 0.057, 0.056, 0.053, 0.059])

results = before_after_analysis(before_redesign, after_redesign, "CVR")
```

### Pattern 3: Channel Performance Distribution
```python
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

def analyze_channel_performance(data):
    """
    Analyze if conversion rates differ by channel using chi-square test.

    data: DataFrame with columns ['channel', 'converted', 'not_converted']
    """
    # Create contingency table
    contingency = data[['converted', 'not_converted']].values

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Calculate conversion rates
    data['total'] = data['converted'] + data['not_converted']
    data['cvr'] = data['converted'] / data['total']

    # Results
    print("Conversion Rates by Channel:")
    print(data[['channel', 'cvr', 'total']])
    print(f"\nChi-square: {chi2:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")

    if p_value < 0.05:
        print("\n✓ Conversion rates differ significantly across channels")
    else:
        print("\n✗ No significant difference in conversion rates")

    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'data': data
    }

# Example
channel_data = pd.DataFrame({
    'channel': ['Google', 'Facebook', 'Instagram'],
    'converted': [450, 380, 210],
    'not_converted': [4550, 4620, 3790]
})

results = analyze_channel_performance(channel_data)
```

### Pattern 4: Confidence Interval Reporting
```python
import numpy as np
from scipy.stats import t

def calculate_metric_with_ci(conversions, total, metric_name="CVR", confidence=0.95):
    """
    Calculate a proportion metric with confidence interval.
    """
    # Point estimate
    estimate = conversions / total

    # Standard error for proportion
    se = np.sqrt(estimate * (1 - estimate) / total)

    # Confidence interval
    alpha = 1 - confidence
    z_critical = 1.96  # for 95% CI

    ci_lower = estimate - z_critical * se
    ci_upper = estimate + z_critical * se

    # Results
    print(f"{metric_name}: {estimate:.4f} ({estimate*100:.2f}%)")
    print(f"{int(confidence*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"{int(confidence*100)}% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

    return {
        'estimate': estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se
    }

# Example
conversions = 250
visitors = 5000
results = calculate_metric_with_ci(conversions, visitors, "Conversion Rate")
```

---

## 🚀 Quick Tips

### Statistical Testing Guidelines
- Always check assumptions (normality, independence, equal variance)
- Use two-tailed tests unless you have strong prior directional hypothesis
- Report effect sizes alongside p-values
- Consider practical significance, not just statistical significance
- Account for multiple testing when running many tests

### Sample Size Considerations
- Larger samples → more power to detect small effects
- Small samples → only detect large effects
- Calculate required sample size before running test
- For marketing: aim for n ≥ 30 per group minimum
- For conversion rate tests: often need 1000s of observations

### Common Mistakes
```python
# ❌ Wrong: Ignoring assumptions
ttest_ind(small_sample_a, small_sample_b)  # Without checking normality

# ✅ Correct: Check assumptions first
from scipy.stats import shapiro
_, p_a = shapiro(small_sample_a)
_, p_b = shapiro(small_sample_b)
if p_a > 0.05 and p_b > 0.05:
    # Normality assumed
    ttest_ind(small_sample_a, small_sample_b)
else:
    # Use non-parametric test
    from scipy.stats import mannwhitneyu
    mannwhitneyu(small_sample_a, small_sample_b)

# ❌ Wrong: Confusing statistical and practical significance
# p=0.001 but only 0.1% improvement in CVR might not matter

# ✅ Correct: Consider both
if p_value < 0.05 and abs(percent_change) >= 5:
    print("Statistically AND practically significant")

# ❌ Wrong: Multiple testing without correction
for channel in channels:
    _, p = ttest_1samp(channel_data, target_metric)
    if p < 0.05:
        print(f"{channel} is significant")  # Inflated Type I error!

# ✅ Correct: Apply Bonferroni correction
alpha = 0.05
n_tests = len(channels)
adjusted_alpha = alpha / n_tests  # Bonferroni
for channel in channels:
    _, p = ttest_1samp(channel_data, target_metric)
    if p < adjusted_alpha:
        print(f"{channel} is significant")
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Campaign Performance Test
```python
import numpy as np
from scipy.stats import ttest_ind

# Campaign A (control): 30 days of CPA data
campaign_a = np.array([
    45.2, 48.1, 42.8, 50.3, 46.7, 49.2, 44.5, 47.8, 43.9, 46.3,
    48.5, 45.8, 47.2, 44.1, 49.7, 46.9, 45.3, 48.9, 47.5, 44.8,
    46.1, 49.3, 45.7, 47.9, 48.3, 44.6, 46.5, 48.7, 45.1, 47.3
])

# Campaign B (new): 30 days of CPA data
campaign_b = np.array([
    42.1, 40.8, 43.5, 41.2, 39.8, 42.7, 41.5, 40.2, 43.1, 41.8,
    42.3, 40.5, 41.9, 42.8, 40.1, 43.2, 41.3, 42.5, 40.9, 41.7,
    42.9, 41.1, 40.3, 42.2, 41.6, 43.3, 40.7, 42.4, 41.4, 40.6
])

# Test: Is Campaign B's CPA significantly lower?
statistic, p_value = ttest_ind(campaign_a, campaign_b)

print(f"Campaign A mean CPA: ${np.mean(campaign_a):.2f}")
print(f"Campaign B mean CPA: ${np.mean(campaign_b):.2f}")
print(f"Difference: ${np.mean(campaign_a) - np.mean(campaign_b):.2f}")
print(f"t-statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    savings = np.mean(campaign_a) - np.mean(campaign_b)
    print(f"\n✓ Campaign B has significantly lower CPA")
    print(f"  Average savings: ${savings:.2f} per acquisition")
else:
    print(f"\n✗ No significant difference in CPA")
```

### Exercise 2: Conversion Rate Confidence Interval
```python
import numpy as np
from scipy.stats import norm

# Landing page data
visitors = 10000
conversions = 523

# Calculate CVR and CI
cvr = conversions / visitors
se = np.sqrt(cvr * (1 - cvr) / visitors)

# 95% CI
z_critical = 1.96
ci_lower = cvr - z_critical * se
ci_upper = cvr + z_critical * se

print(f"Conversion Rate: {cvr:.4f} ({cvr*100:.2f}%)")
print(f"95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

# If we get 10,000 more visitors, how many conversions do we expect?
expected_conversions = 10000 * cvr
expected_lower = 10000 * ci_lower
expected_upper = 10000 * ci_upper

print(f"\nExpected conversions from 10,000 visitors:")
print(f"Point estimate: {expected_conversions:.0f}")
print(f"95% CI: [{expected_lower:.0f}, {expected_upper:.0f}]")
```

### Exercise 3: Multi-Channel Chi-Square Test
```python
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

# Traffic and conversion by device
data = pd.DataFrame({
    'device': ['Mobile', 'Desktop', 'Tablet'],
    'conversions': [450, 820, 125],
    'non_conversions': [9550, 14180, 2875]
})

# Create contingency table
observed = data[['conversions', 'non_conversions']].values

# Chi-square test
chi2, p_value, dof, expected = chi2_contingency(observed)

# Calculate conversion rates
data['total'] = data['conversions'] + data['non_conversions']
data['cvr'] = data['conversions'] / data['total']

print("Device Performance:")
print(data[['device', 'cvr', 'conversions', 'total']])
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n✓ Conversion rates differ significantly by device")
    print("\nRecommendation: Optimize campaigns by device type")
    best_device = data.loc[data['cvr'].idxmax(), 'device']
    print(f"Best performing: {best_device} ({data['cvr'].max()*100:.2f}% CVR)")
else:
    print("\n✗ No significant difference by device")
```

### Exercise 4: Website Redesign Impact
```python
import numpy as np
from scipy.stats import ttest_rel

# 7 days of conversion rates before and after redesign
before = np.array([0.0478, 0.0512, 0.0493, 0.0508, 0.0501, 0.0487, 0.0533])
after = np.array([0.0545, 0.0589, 0.0567, 0.0578, 0.0591, 0.0556, 0.0612])

# Paired t-test
statistic, p_value = ttest_rel(before, after)

# Calculate improvement
mean_before = np.mean(before)
mean_after = np.mean(after)
absolute_lift = mean_after - mean_before
relative_lift = (absolute_lift / mean_before) * 100

print(f"Before redesign: {mean_before*100:.2f}%")
print(f"After redesign: {mean_after*100:.2f}%")
print(f"Absolute lift: {absolute_lift*100:.2f} percentage points")
print(f"Relative lift: {relative_lift:.1f}%")
print(f"\nt-statistic: {statistic:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"\n✓ Redesign significantly improved conversion rate")
    print(f"  Keep the new design!")
else:
    print(f"\n✗ No significant improvement detected")

# Calculate projected impact
monthly_visitors = 100000
additional_conversions = monthly_visitors * absolute_lift
print(f"\nProjected monthly impact:")
print(f"  Additional conversions: {additional_conversions:.0f}")
```

---

## 🔍 Statistical Interpretation Guide

### P-Value Interpretation
- **p < 0.001**: Very strong evidence against H₀
- **p < 0.01**: Strong evidence against H₀
- **p < 0.05**: Moderate evidence against H₀ (typical threshold)
- **p < 0.10**: Weak evidence against H₀
- **p ≥ 0.10**: Insufficient evidence against H₀

### Effect Size Interpretation (Cohen's d)
- **|d| < 0.2**: Negligible effect
- **|d| = 0.2**: Small effect
- **|d| = 0.5**: Medium effect
- **|d| = 0.8**: Large effect
- **|d| > 1.3**: Very large effect

### When to Use Which Test

| Scenario | Test | Example |
|----------|------|---------|
| Compare mean to known value | One-sample t-test | Is avg CPA = $50? |
| Compare two independent groups | Two-sample t-test | Campaign A vs B |
| Compare paired observations | Paired t-test | Before vs after |
| Test independence of categories | Chi-square test | Device vs conversion |
| Compare proportions | Two-proportion z-test | CVR A vs B (Week 8) |

---

**Quick Navigation:**
- [← Week 6 Cheatsheet](Week_06_Cheatsheet.md)
- [Week 8 Cheatsheet →](Week_08_Cheatsheet.md)
- [Back to Main README](../README.md)
