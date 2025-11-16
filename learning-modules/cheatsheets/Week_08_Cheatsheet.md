# Week 8: A/B Testing - Quick Reference Cheatsheet

## 📋 Core Concepts

### A/B Test Design Fundamentals

```python
"""
Essential Components of A/B Test:
1. Hypothesis (H₀ and Hₐ)
2. Primary metric (e.g., conversion rate)
3. Minimum detectable effect (MDE)
4. Significance level (α, typically 0.05)
5. Statistical power (1-β, typically 0.80)
6. Sample size calculation
7. Test duration
8. Randomization method
"""

# A/B Test Design Template
ab_test_design = {
    'test_name': 'Landing Page Redesign',
    'hypothesis': {
        'null': 'New design has no effect on conversion rate',
        'alternative': 'New design increases conversion rate'
    },
    'metric': 'conversion_rate',
    'baseline_rate': 0.05,      # 5% current CVR
    'mde': 0.005,                # Want to detect 0.5% absolute lift
    'alpha': 0.05,               # 5% significance level
    'power': 0.80,               # 80% power
    'variants': ['control', 'treatment'],
    'traffic_split': [0.5, 0.5] # 50/50 split
}
```

### Sample Size Calculation

#### For Conversion Rate Tests
```python
import numpy as np
from scipy.stats import norm

def calculate_sample_size_conversion(p1, p2, alpha=0.05, power=0.80):
    """
    Calculate required sample size per variant for conversion rate test.

    Parameters:
    - p1: baseline conversion rate (control)
    - p2: expected conversion rate (treatment)
    - alpha: significance level (default 0.05)
    - power: statistical power (default 0.80)

    Returns:
    - n: sample size needed per variant
    """
    # Z-scores
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = norm.ppf(power)

    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Sample size formula
    numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
                 z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2

    n = numerator / denominator

    return np.ceil(n)

# Example: Detect 0.5% absolute lift from 5% baseline
baseline_cvr = 0.05
target_cvr = 0.055  # 10% relative lift

n_per_variant = calculate_sample_size_conversion(baseline_cvr, target_cvr)
print(f"Sample size needed per variant: {n_per_variant:.0f}")
print(f"Total sample size needed: {n_per_variant * 2:.0f}")

# Calculate test duration
daily_visitors = 1000
days_needed = (n_per_variant * 2) / daily_visitors
print(f"Test duration: {np.ceil(days_needed):.0f} days")
```

#### Sample Size with statsmodels
```python
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import zt_ind_solve_power

# Calculate effect size
p1 = 0.05   # control
p2 = 0.055  # treatment
effect_size = proportion_effectsize(p1, p2)

print(f"Effect size (Cohen's h): {effect_size:.4f}")

# Calculate required sample size
n_per_variant = zt_ind_solve_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.80,
    alternative='two-sided'
)

print(f"Sample size per variant: {np.ceil(n_per_variant):.0f}")
```

#### Relative vs Absolute Lift
```python
# Baseline conversion rate: 5%
baseline = 0.05

# Absolute lift: 0.5 percentage points
absolute_lift = 0.005
new_rate_absolute = baseline + absolute_lift  # 5.5%

# Relative lift: 10%
relative_lift = 0.10
new_rate_relative = baseline * (1 + relative_lift)  # 5.5%

print(f"Baseline: {baseline*100:.1f}%")
print(f"After 0.5% absolute lift: {new_rate_absolute*100:.1f}%")
print(f"After 10% relative lift: {new_rate_relative*100:.1f}%")

# Relationship
relative_from_absolute = (absolute_lift / baseline) * 100
print(f"0.5% absolute lift = {relative_from_absolute:.1f}% relative lift")
```

### Two-Proportion Z-Test

#### Manual Calculation
```python
import numpy as np
from scipy.stats import norm

def two_proportion_ztest(conversions_a, n_a, conversions_b, n_b):
    """
    Perform two-proportion z-test.

    Returns: z-statistic and p-value (two-tailed)
    """
    # Conversion rates
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b

    # Pooled proportion
    p_pooled = (conversions_a + conversions_b) / (n_a + n_b)

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))

    # Z-statistic
    z_stat = (p_b - p_a) / se

    # P-value (two-tailed)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return z_stat, p_value, p_a, p_b

# Example
control_conversions = 500
control_visitors = 10000
treatment_conversions = 575
treatment_visitors = 10000

z_stat, p_value, p_control, p_treatment = two_proportion_ztest(
    control_conversions, control_visitors,
    treatment_conversions, treatment_visitors
)

print(f"Control CVR: {p_control*100:.2f}%")
print(f"Treatment CVR: {p_treatment*100:.2f}%")
print(f"Absolute lift: {(p_treatment - p_control)*100:.2f} percentage points")
print(f"Relative lift: {((p_treatment - p_control)/p_control)*100:.1f}%")
print(f"\nZ-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n✓ Statistically significant difference (α=0.05)")
else:
    print("\n✗ No statistically significant difference")
```

#### Using statsmodels
```python
from statsmodels.stats.proportion import proportions_ztest

# Test data
counts = np.array([500, 575])      # conversions
nobs = np.array([10000, 10000])    # total visitors

# Perform test
z_stat, p_value = proportions_ztest(counts, nobs)

print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# One-tailed test (if testing for improvement only)
z_stat, p_value_one_tailed = proportions_ztest(counts, nobs, alternative='larger')
print(f"P-value (one-tailed): {p_value_one_tailed:.4f}")
```

### Statistical vs Practical Significance

```python
def evaluate_ab_test(control_cvr, treatment_cvr, p_value,
                     min_practical_lift=0.05):
    """
    Evaluate both statistical and practical significance.

    min_practical_lift: minimum relative improvement that matters (default 5%)
    """
    # Statistical significance
    is_stat_sig = p_value < 0.05

    # Practical significance
    relative_lift = (treatment_cvr - control_cvr) / control_cvr
    is_prac_sig = abs(relative_lift) >= min_practical_lift

    # Interpret
    print(f"Control CVR: {control_cvr*100:.2f}%")
    print(f"Treatment CVR: {treatment_cvr*100:.2f}%")
    print(f"Relative lift: {relative_lift*100:+.1f}%")
    print(f"P-value: {p_value:.4f}")
    print()

    if is_stat_sig and is_prac_sig:
        print("✓ WINNER: Statistically AND practically significant")
        print("  → Implement the treatment")
    elif is_stat_sig and not is_prac_sig:
        print("⚠ Statistically significant but NOT practically significant")
        print("  → The effect is too small to matter")
    elif not is_stat_sig and is_prac_sig:
        print("⚠ Practically significant but NOT statistically significant")
        print("  → Need more data to confirm")
    else:
        print("✗ Neither statistically nor practically significant")
        print("  → Keep the control (no change)")

    return {
        'statistical_sig': is_stat_sig,
        'practical_sig': is_prac_sig,
        'relative_lift': relative_lift,
        'p_value': p_value
    }

# Example 1: Stat sig AND prac sig
evaluate_ab_test(0.05, 0.058, p_value=0.001, min_practical_lift=0.05)

print("\n" + "="*60 + "\n")

# Example 2: Stat sig but NOT prac sig
evaluate_ab_test(0.05, 0.0505, p_value=0.04, min_practical_lift=0.05)
```

### Confidence Intervals for Difference
```python
import numpy as np
from scipy.stats import norm

def ab_test_confidence_interval(conversions_a, n_a, conversions_b, n_b,
                                 confidence=0.95):
    """
    Calculate confidence interval for the difference in conversion rates.
    """
    # Conversion rates
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    diff = p_b - p_a

    # Standard error of difference
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)

    # Critical value
    z_critical = norm.ppf(1 - (1 - confidence) / 2)

    # Confidence interval
    ci_lower = diff - z_critical * se_diff
    ci_upper = diff + z_critical * se_diff

    print(f"Control CVR: {p_a*100:.2f}%")
    print(f"Treatment CVR: {p_b*100:.2f}%")
    print(f"Difference: {diff*100:+.2f} percentage points")
    print(f"{int(confidence*100)}% CI for difference: "
          f"[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

    # Check if CI includes zero
    if ci_lower > 0:
        print("\n✓ Treatment is significantly better (CI > 0)")
    elif ci_upper < 0:
        print("\n✓ Treatment is significantly worse (CI < 0)")
    else:
        print("\n✗ CI includes zero - no significant difference")

    return {'diff': diff, 'ci_lower': ci_lower, 'ci_upper': ci_upper}

# Example
ab_test_confidence_interval(500, 10000, 575, 10000)
```

---

## 🎯 Multiple Testing Corrections

### The Multiple Testing Problem
```python
"""
Problem: Running multiple A/B tests increases false positive rate.

Example:
- Run 20 independent tests at α = 0.05
- Expected false positives: 20 × 0.05 = 1
- Family-wise error rate (FWER) ≈ 1 - (1 - 0.05)^20 = 64%!

Solution: Adjust significance threshold
"""

import numpy as np

# Simulate the problem
def simulate_multiple_tests(n_tests=20, alpha=0.05, n_simulations=10000):
    """Simulate multiple testing to show inflated error rate."""
    false_positives = []

    for _ in range(n_simulations):
        # Generate random p-values (under null hypothesis)
        p_values = np.random.uniform(0, 1, n_tests)

        # Count how many "significant" results
        n_sig = np.sum(p_values < alpha)
        false_positives.append(n_sig > 0)  # At least one false positive

    fwer = np.mean(false_positives)
    print(f"Family-wise error rate: {fwer:.2%}")
    print(f"Expected: {1 - (1 - alpha)**n_tests:.2%}")

simulate_multiple_tests()
```

### Bonferroni Correction
```python
def bonferroni_correction(p_values, alpha=0.05):
    """
    Bonferroni correction for multiple testing.
    Most conservative approach.
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests

    print(f"Original alpha: {alpha}")
    print(f"Number of tests: {n_tests}")
    print(f"Bonferroni adjusted alpha: {adjusted_alpha:.4f}")
    print()

    significant = []
    for i, p in enumerate(p_values):
        is_sig = p < adjusted_alpha
        significant.append(is_sig)
        print(f"Test {i+1}: p={p:.4f} {'✓ Significant' if is_sig else '✗ Not significant'}")

    return significant

# Example: Testing 5 different page variants
p_values = [0.001, 0.03, 0.02, 0.15, 0.08]
bonferroni_correction(p_values)
```

### Benjamini-Hochberg (FDR Control)
```python
from statsmodels.stats.multitest import multipletests

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR control.
    Less conservative than Bonferroni, controls false discovery rate.
    """
    # Apply correction
    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha,
                                              method='fdr_bh')

    print(f"Alpha level: {alpha}")
    print(f"Number of tests: {len(p_values)}")
    print()

    for i, (p_orig, p_adj, rej) in enumerate(zip(p_values, p_adjusted, reject)):
        print(f"Test {i+1}:")
        print(f"  Original p-value: {p_orig:.4f}")
        print(f"  Adjusted p-value: {p_adj:.4f}")
        print(f"  Significant: {'✓ Yes' if rej else '✗ No'}")

    return reject, p_adjusted

# Example
p_values = [0.001, 0.03, 0.02, 0.15, 0.08]
reject, p_adjusted = benjamini_hochberg(p_values)
```

### When to Use Which Correction

| Method | Use Case | Conservative? |
|--------|----------|---------------|
| Bonferroni | Few tests, need strict control | Very |
| Benjamini-Hochberg | Many tests, exploratory | Moderate |
| No correction | Single pre-planned test | N/A |

---

## 💡 Sequential Testing

### The Peeking Problem
```python
"""
Problem: Checking A/B test results repeatedly increases false positive rate.

Bad practice: Check test daily and stop when p < 0.05
- Inflates Type I error rate significantly
- "Peeking" at results is like running multiple tests

Solutions:
1. Pre-determine sample size and only check once
2. Use sequential testing methods (e.g., mSPRT)
3. Apply alpha spending functions
"""

def simulate_peeking_problem(n_simulations=1000):
    """
    Simulate how peeking inflates false positive rate.
    Under null hypothesis (no real effect).
    """
    false_positives = 0

    for _ in range(n_simulations):
        # Simulate data collection (null hypothesis: no difference)
        daily_visitors = 100
        total_days = 30
        baseline_cvr = 0.05

        for day in range(1, total_days + 1):
            # Both groups same CVR (null hypothesis)
            control = np.random.binomial(daily_visitors * day, baseline_cvr)
            treatment = np.random.binomial(daily_visitors * day, baseline_cvr)

            # Peek at results
            from statsmodels.stats.proportion import proportions_ztest
            counts = np.array([control, treatment])
            nobs = np.array([daily_visitors * day, daily_visitors * day])

            _, p_value = proportions_ztest(counts, nobs)

            # Stop if significant
            if p_value < 0.05:
                false_positives += 1
                break

    fpr = false_positives / n_simulations
    print(f"False positive rate with peeking: {fpr:.2%}")
    print(f"Expected without peeking: 5.00%")

simulate_peeking_problem()
```

### Sequential Probability Ratio Test (SPRT)
```python
import numpy as np

def sequential_test_boundaries(n_observations, alpha=0.05, beta=0.20):
    """
    Calculate sequential test boundaries (simplified).

    Returns boundaries for stopping the test early.
    """
    # Log likelihood ratio boundaries
    A = np.log((1 - beta) / alpha)
    B = np.log(beta / (1 - alpha))

    return {'upper': A, 'lower': B}

# Example usage
boundaries = sequential_test_boundaries(alpha=0.05, beta=0.20)
print(f"Upper boundary (reject null): {boundaries['upper']:.4f}")
print(f"Lower boundary (accept null): {boundaries['lower']:.4f}")
print("\nContinue test if log-likelihood ratio is between boundaries")
```

### Always Valid P-values (Safe Testing)
```python
"""
Modern approach: Always valid p-values
- Can check test anytime without inflating error rate
- Requires different calculation than traditional p-values
- Growing adoption in industry (e.g., Optimizely, VWO)

Libraries:
- statsmodels (basic sequential methods)
- msprt (more advanced)
"""

# Placeholder - would use specialized library
def safe_test_example():
    """
    Safe testing allows peeking without penalty.
    Requires specialized implementation.
    """
    print("Safe testing example:")
    print("1. Can check results anytime")
    print("2. P-value remains valid")
    print("3. False positive rate controlled at α")
    print("\nRecommended for production A/B testing platforms")

safe_test_example()
```

---

## 🚀 Common A/B Testing Patterns

### Pattern 1: Complete A/B Test Analysis
```python
import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

class ABTest:
    """Complete A/B test analysis."""

    def __init__(self, name, control_conversions, control_total,
                 treatment_conversions, treatment_total):
        self.name = name
        self.control_conversions = control_conversions
        self.control_total = control_total
        self.treatment_conversions = treatment_conversions
        self.treatment_total = treatment_total

        # Calculate metrics
        self.control_cvr = control_conversions / control_total
        self.treatment_cvr = treatment_conversions / treatment_total
        self.absolute_lift = self.treatment_cvr - self.control_cvr
        self.relative_lift = self.absolute_lift / self.control_cvr

        # Run test
        self._run_test()

    def _run_test(self):
        """Run statistical test."""
        counts = np.array([self.control_conversions, self.treatment_conversions])
        nobs = np.array([self.control_total, self.treatment_total])

        self.z_stat, self.p_value = proportions_ztest(counts, nobs)

        # Confidence interval
        diff = self.treatment_cvr - self.control_cvr
        se = np.sqrt(
            self.control_cvr * (1 - self.control_cvr) / self.control_total +
            self.treatment_cvr * (1 - self.treatment_cvr) / self.treatment_total
        )
        z_critical = norm.ppf(0.975)  # 95% CI
        self.ci_lower = diff - z_critical * se
        self.ci_upper = diff + z_critical * se

    def report(self, alpha=0.05, min_practical_lift=0.05):
        """Generate comprehensive report."""
        print(f"{'='*60}")
        print(f"A/B Test Report: {self.name}")
        print(f"{'='*60}\n")

        # Sample sizes
        print("Sample Sizes:")
        print(f"  Control: {self.control_total:,} visitors")
        print(f"  Treatment: {self.treatment_total:,} visitors")
        print()

        # Conversion rates
        print("Conversion Rates:")
        print(f"  Control: {self.control_cvr*100:.2f}% "
              f"({self.control_conversions} conversions)")
        print(f"  Treatment: {self.treatment_cvr*100:.2f}% "
              f"({self.treatment_conversions} conversions)")
        print()

        # Lift
        print("Lift:")
        print(f"  Absolute: {self.absolute_lift*100:+.2f} percentage points")
        print(f"  Relative: {self.relative_lift*100:+.1f}%")
        print()

        # Statistical test
        print("Statistical Test:")
        print(f"  Z-statistic: {self.z_stat:.4f}")
        print(f"  P-value: {self.p_value:.4f}")
        print(f"  95% CI for lift: [{self.ci_lower*100:.2f}%, {self.ci_upper*100:.2f}%]")
        print()

        # Significance
        is_stat_sig = self.p_value < alpha
        is_prac_sig = abs(self.relative_lift) >= min_practical_lift

        print("Decision:")
        if is_stat_sig and is_prac_sig:
            print("  ✓ WINNER: Launch treatment")
            print(f"    - Statistically significant (p < {alpha})")
            print(f"    - Practically significant (>{min_practical_lift*100:.0f}% lift)")
        elif is_stat_sig and not is_prac_sig:
            print("  ⚠ INCONCLUSIVE: Effect too small")
            print(f"    - Statistically significant (p < {alpha})")
            print(f"    - NOT practically significant (<{min_practical_lift*100:.0f}% lift)")
        elif not is_stat_sig and is_prac_sig:
            print("  ⚠ INCONCLUSIVE: Need more data")
            print(f"    - NOT statistically significant (p >= {alpha})")
            print(f"    - Potentially meaningful effect size")
        else:
            print("  ✗ NO WINNER: Keep control")
            print(f"    - No significant difference detected")

# Example usage
test = ABTest(
    name="Homepage Redesign",
    control_conversions=500,
    control_total=10000,
    treatment_conversions=575,
    treatment_total=10000
)

test.report()
```

### Pattern 2: Multi-Variant Test (A/B/C/D)
```python
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

def multi_variant_test(variants_data):
    """
    Analyze multi-variant test.

    variants_data: dict with variant names as keys,
                   {'conversions': x, 'total': y} as values
    """
    # Create contingency table
    conversions = []
    non_conversions = []
    names = []

    for name, data in variants_data.items():
        conversions.append(data['conversions'])
        non_conversions.append(data['total'] - data['conversions'])
        names.append(name)

    # Overall chi-square test
    observed = np.array([conversions, non_conversions]).T
    chi2, p_value, dof, expected = chi2_contingency(observed)

    # Calculate CVRs
    results = pd.DataFrame({
        'variant': names,
        'conversions': conversions,
        'total': [variants_data[n]['total'] for n in names],
    })
    results['cvr'] = results['conversions'] / results['total']
    results['cvr_pct'] = results['cvr'] * 100

    # Report
    print("Multi-Variant Test Results")
    print("="*60)
    print(results[['variant', 'conversions', 'total', 'cvr_pct']].to_string(index=False))
    print(f"\nOverall Test:")
    print(f"  Chi-square: {chi2:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Degrees of freedom: {dof}")

    if p_value < 0.05:
        print("\n✓ Significant difference detected between variants")

        # Pairwise comparisons (with Bonferroni correction)
        print("\nPairwise Comparisons (Bonferroni corrected):")
        n_comparisons = len(names) * (len(names) - 1) / 2
        bonf_alpha = 0.05 / n_comparisons

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_i, name_j = names[i], names[j]
                counts = np.array([conversions[i], conversions[j]])
                nobs = np.array([variants_data[name_i]['total'],
                                 variants_data[name_j]['total']])

                from statsmodels.stats.proportion import proportions_ztest
                _, p = proportions_ztest(counts, nobs)

                sig = "✓" if p < bonf_alpha else "✗"
                print(f"  {name_i} vs {name_j}: p={p:.4f} {sig}")
    else:
        print("\n✗ No significant difference between variants")

    return results

# Example: Test 4 different CTAs
variants = {
    'Control (Sign Up)': {'conversions': 500, 'total': 10000},
    'Variant A (Get Started)': {'conversions': 575, 'total': 10000},
    'Variant B (Try Free)': {'conversions': 612, 'total': 10000},
    'Variant C (Join Now)': {'conversions': 545, 'total': 10000},
}

results = multi_variant_test(variants)
```

### Pattern 3: Segmented Analysis
```python
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

def segmented_ab_analysis(data, segment_col='segment'):
    """
    Analyze A/B test results by segment.

    data: DataFrame with columns [segment, variant, conversions, total]
    """
    segments = data[segment_col].unique()

    print("Segmented A/B Test Analysis")
    print("="*60)

    for segment in segments:
        print(f"\n{segment.upper()}")
        print("-"*60)

        segment_data = data[data[segment_col] == segment]

        # Get control and treatment
        control = segment_data[segment_data['variant'] == 'control'].iloc[0]
        treatment = segment_data[segment_data['variant'] == 'treatment'].iloc[0]

        # Calculate metrics
        control_cvr = control['conversions'] / control['total']
        treatment_cvr = treatment['conversions'] / treatment['total']
        lift = (treatment_cvr - control_cvr) / control_cvr

        # Test
        counts = np.array([control['conversions'], treatment['conversions']])
        nobs = np.array([control['total'], treatment['total']])
        _, p_value = proportions_ztest(counts, nobs)

        # Report
        print(f"Control CVR: {control_cvr*100:.2f}%")
        print(f"Treatment CVR: {treatment_cvr*100:.2f}%")
        print(f"Lift: {lift*100:+.1f}%")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("✓ Significant for this segment")
        else:
            print("✗ Not significant for this segment")

# Example: Analyze by device type
data = pd.DataFrame({
    'segment': ['mobile', 'mobile', 'desktop', 'desktop', 'tablet', 'tablet'],
    'variant': ['control', 'treatment'] * 3,
    'conversions': [200, 225, 250, 280, 50, 70],
    'total': [4000, 4000, 5000, 5000, 1000, 1000]
})

segmented_ab_analysis(data)
```

### Pattern 4: Continuous Metrics (Revenue, Time on Site)
```python
import numpy as np
from scipy.stats import ttest_ind

def ab_test_continuous_metric(control_values, treatment_values, metric_name="metric"):
    """
    A/B test for continuous metrics (e.g., revenue per visitor, time on site).
    """
    # Descriptive statistics
    control_mean = np.mean(control_values)
    control_std = np.std(control_values, ddof=1)
    treatment_mean = np.mean(treatment_values)
    treatment_std = np.std(treatment_values, ddof=1)

    lift = (treatment_mean - control_mean) / control_mean

    # T-test
    t_stat, p_value = ttest_ind(control_values, treatment_values)

    # Confidence interval for difference
    from scipy.stats import t
    n1, n2 = len(control_values), len(treatment_values)
    df = n1 + n2 - 2
    pooled_std = np.sqrt(((n1-1)*control_std**2 + (n2-1)*treatment_std**2) / df)
    se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
    t_critical = t.ppf(0.975, df)

    diff = treatment_mean - control_mean
    ci_lower = diff - t_critical * se_diff
    ci_upper = diff + t_critical * se_diff

    # Report
    print(f"A/B Test: {metric_name}")
    print("="*60)
    print(f"Control: {control_mean:.2f} (±{control_std:.2f})")
    print(f"Treatment: {treatment_mean:.2f} (±{treatment_std:.2f})")
    print(f"Difference: {diff:+.2f}")
    print(f"Lift: {lift*100:+.1f}%")
    print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"\nT-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\n✓ Statistically significant difference")
    else:
        print("\n✗ No significant difference")

    return {
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'diff': diff,
        'lift': lift,
        'p_value': p_value,
        'ci': (ci_lower, ci_upper)
    }

# Example: Test impact on revenue per visitor
control_revenue = np.random.gamma(shape=2, scale=15, size=5000)  # avg $30
treatment_revenue = np.random.gamma(shape=2, scale=16, size=5000)  # avg $32

ab_test_continuous_metric(control_revenue, treatment_revenue, "Revenue per Visitor")
```

---

## 🚀 Quick Tips

### A/B Test Best Practices
1. **Pre-register your test**: Define hypothesis, metric, and sample size before starting
2. **One primary metric**: Avoid HARKing (Hypothesizing After Results are Known)
3. **Run full weeks**: Account for day-of-week effects
4. **Check randomization**: Ensure traffic split is truly random
5. **Account for seasonality**: Don't run tests during unusual periods
6. **Document everything**: Keep test logs and decisions

### Sample Size Guidelines
- **Minimum**: At least 100-200 conversions per variant
- **Typical**: 350+ conversions per variant for reliable results
- **High traffic**: Can detect smaller effects
- **Low traffic**: Need larger effects or longer tests
- **Rule of thumb**: 2-4 weeks minimum duration

### Common Mistakes
```python
# ❌ Wrong: Testing too many variants with small sample
variants = ['A', 'B', 'C', 'D', 'E', 'F']  # 6 variants
visitors_per_variant = 500  # Too small!

# ✅ Correct: Focus on one challenger
variants = ['control', 'treatment']  # Test one thing at a time
visitors_per_variant = 5000  # Adequate sample size

# ❌ Wrong: Stopping test early because p < 0.05
if current_p_value < 0.05:
    stop_test()  # Inflates false positive rate!

# ✅ Correct: Run to pre-determined sample size
if current_sample_size >= required_sample_size:
    stop_test()

# ❌ Wrong: Ignoring practical significance
if p_value < 0.05:
    deploy_treatment()  # Might be 0.01% improvement!

# ✅ Correct: Check both statistical and practical significance
if p_value < 0.05 and abs(relative_lift) >= 0.05:  # At least 5% lift
    deploy_treatment()

# ❌ Wrong: Running multiple metrics without correction
metrics = ['cvr', 'revenue', 'engagement', 'retention']
for metric in metrics:
    if test_metric(metric) < 0.05:
        print(f"{metric} is significant")  # Multiple testing problem!

# ✅ Correct: Bonferroni correction or single primary metric
primary_metric = 'cvr'  # Pre-defined
if test_metric(primary_metric) < 0.05:
    print("Primary metric is significant")
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Sample Size Calculation
```python
import numpy as np
from scipy.stats import norm

# Current landing page: 5% CVR
# Want to detect: 10% relative improvement (5% → 5.5%)
# Power: 80%, Alpha: 0.05

baseline = 0.05
target = 0.055  # 10% relative lift

# Calculate sample size
p1, p2 = baseline, target
z_alpha = norm.ppf(0.975)  # two-tailed, alpha=0.05
z_beta = norm.ppf(0.80)    # power=0.80

p_pooled = (p1 + p2) / 2
numerator = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
denominator = (p2 - p1) ** 2
n_per_variant = np.ceil(numerator / denominator)

print(f"Required sample size per variant: {n_per_variant:.0f}")
print(f"Total sample size: {n_per_variant * 2:.0f}")

# With 2,000 daily visitors
daily_visitors = 2000
days_needed = (n_per_variant * 2) / daily_visitors

print(f"\nTest duration: {np.ceil(days_needed):.0f} days")
print(f"Recommended: {np.ceil(days_needed / 7) * 7:.0f} days (full weeks)")
```

### Exercise 2: Analyze A/B Test Results
```python
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm

# Test ran for 14 days
control_visitors = 14253
control_conversions = 712

treatment_visitors = 14301
treatment_conversions = 786

# Calculate CVRs
control_cvr = control_conversions / control_visitors
treatment_cvr = treatment_conversions / treatment_visitors

print(f"Control CVR: {control_cvr*100:.2f}% ({control_conversions}/{control_visitors})")
print(f"Treatment CVR: {treatment_cvr*100:.2f}% ({treatment_conversions}/{treatment_visitors})")

# Lift
absolute_lift = treatment_cvr - control_cvr
relative_lift = absolute_lift / control_cvr

print(f"\nAbsolute lift: {absolute_lift*100:+.2f} percentage points")
print(f"Relative lift: {relative_lift*100:+.1f}%")

# Statistical test
counts = np.array([control_conversions, treatment_conversions])
nobs = np.array([control_visitors, treatment_visitors])
z_stat, p_value = proportions_ztest(counts, nobs)

print(f"\nZ-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Confidence interval
se_diff = np.sqrt(
    control_cvr * (1 - control_cvr) / control_visitors +
    treatment_cvr * (1 - treatment_cvr) / treatment_visitors
)
z_critical = norm.ppf(0.975)
ci_lower = absolute_lift - z_critical * se_diff
ci_upper = absolute_lift + z_critical * se_diff

print(f"95% CI for lift: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

# Decision
if p_value < 0.05 and relative_lift >= 0.05:  # At least 5% lift
    print("\n✓ WINNER: Deploy treatment")
    print(f"  Expected improvement: {relative_lift*100:.1f}%")
elif p_value < 0.05:
    print("\n⚠ Statistically significant but small effect")
    print(f"  Consider if {relative_lift*100:.1f}% lift is worth it")
else:
    print("\n✗ No significant difference - keep control")
```

### Exercise 3: Multi-Variant CTA Test
```python
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import pandas as pd

# Test 3 CTA variants
data = pd.DataFrame({
    'variant': ['Control: Sign Up', 'Variant A: Get Started', 'Variant B: Try Free'],
    'conversions': [523, 589, 612],
    'visitors': [10000, 10000, 10000]
})

data['cvr'] = data['conversions'] / data['visitors']

print("CTA Variant Test Results")
print("="*60)
print(data.to_string(index=False))

# Overall test
observed = np.array([
    data['conversions'].values,
    (data['visitors'] - data['conversions']).values
]).T

chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"\nOverall Chi-square test:")
print(f"  Chi-square: {chi2:.4f}")
print(f"  P-value: {p_value:.4f}")

if p_value < 0.05:
    print("  ✓ Significant difference between variants")

    # Find best variant
    best_idx = data['cvr'].idxmax()
    best = data.iloc[best_idx]
    control = data.iloc[0]

    print(f"\nBest performer: {best['variant']}")
    print(f"  CVR: {best['cvr']*100:.2f}%")
    print(f"  Lift vs control: {(best['cvr']/control['cvr'] - 1)*100:+.1f}%")

    # Test best vs control
    counts = np.array([control['conversions'], best['conversions']])
    nobs = np.array([control['visitors'], best['visitors']])
    _, p = proportions_ztest(counts, nobs)

    print(f"  P-value vs control: {p:.4f}")

    if p < 0.05:
        print(f"  ✓ Significantly better than control - deploy!")
else:
    print("  ✗ No significant difference")
```

### Exercise 4: Revenue Per Visitor Test
```python
import numpy as np
from scipy.stats import ttest_ind, t

# 14 days of revenue per visitor data (simulated)
np.random.seed(42)
control_rpv = np.random.gamma(shape=2, scale=12, size=14253)  # avg ~$24
treatment_rpv = np.random.gamma(shape=2, scale=13, size=14301)  # avg ~$26

# Descriptive stats
control_mean = np.mean(control_rpv)
control_std = np.std(control_rpv, ddof=1)
treatment_mean = np.mean(treatment_rpv)
treatment_std = np.std(treatment_rpv, ddof=1)

print("Revenue Per Visitor Test")
print("="*60)
print(f"Control: ${control_mean:.2f} (±${control_std:.2f})")
print(f"Treatment: ${treatment_mean:.2f} (±${treatment_std:.2f})")

# Lift
diff = treatment_mean - control_mean
lift = diff / control_mean

print(f"\nAbsolute difference: ${diff:+.2f}")
print(f"Relative lift: {lift*100:+.1f}%")

# T-test
t_stat, p_value = ttest_ind(control_rpv, treatment_rpv)

print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Confidence interval
n1, n2 = len(control_rpv), len(treatment_rpv)
df = n1 + n2 - 2
pooled_std = np.sqrt(((n1-1)*control_std**2 + (n2-1)*treatment_std**2) / df)
se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
t_critical = t.ppf(0.975, df)

ci_lower = diff - t_critical * se_diff
ci_upper = diff + t_critical * se_diff

print(f"95% CI: [${ci_lower:.2f}, ${ci_upper:.2f}]")

# Decision
if p_value < 0.05 and lift >= 0.05:
    print("\n✓ WINNER: Treatment increases revenue")
    print(f"  Expected lift: {lift*100:.1f}%")

    # Calculate annual impact
    annual_visitors = 14000 * 365 / 14  # scale to annual
    annual_lift = annual_visitors * diff
    print(f"  Projected annual impact: ${annual_lift:,.0f}")
else:
    print("\n✗ No significant increase in revenue")
```

---

## 🔍 A/B Testing Checklist

### Pre-Test
- [ ] Hypothesis clearly defined
- [ ] Primary metric selected
- [ ] Sample size calculated
- [ ] Minimum detectable effect determined
- [ ] Test duration estimated
- [ ] Randomization method validated
- [ ] Tracking implementation verified
- [ ] Stakeholder alignment obtained

### During Test
- [ ] No peeking at results before planned end
- [ ] Traffic split remains consistent
- [ ] No other major changes to site/campaign
- [ ] Sample ratio check (50/50 split actually 50/50?)
- [ ] Monitor for technical issues

### Post-Test
- [ ] Full sample size reached
- [ ] Statistical significance checked
- [ ] Practical significance evaluated
- [ ] Confidence intervals calculated
- [ ] Segment analysis performed
- [ ] Results documented
- [ ] Decision made and communicated

---

**Quick Navigation:**
- [← Week 7 Cheatsheet](Week_07_Cheatsheet.md)
- [Week 9 Cheatsheet →](Week_09_Cheatsheet.md)
- [Back to Main README](../README.md)
