# Week 11: Incrementality & Lift Studies - Quick Reference Cheatsheet

## 📋 Core Concepts

### Incrementality Fundamentals

```python
"""
Incrementality:
- Measures causal impact of marketing
- Answers: "What sales would NOT have happened without this campaign?"
- Goes beyond correlation to causation

Key Question:
"What is the TRUE incremental impact of our marketing?"

Methods:
1. Randomized Controlled Trials (RCTs) / A/B Tests
2. Geo-based experiments (Geo-lift)
3. Difference-in-Differences (DiD)
4. Matched Market Testing
5. Synthetic Control

Formula:
Incremental Lift = (Treatment Outcome) - (Control Outcome)
Incrementality % = (Lift / Control Outcome) × 100
"""

# Simple incrementality calculation
def calculate_incrementality(treatment_sales, control_sales):
    """
    Calculate incremental lift and incrementality percentage.
    """
    lift = treatment_sales - control_sales
    incrementality_pct = (lift / control_sales) * 100 if control_sales > 0 else 0

    return {
        'lift': lift,
        'incrementality_pct': incrementality_pct,
        'treatment_sales': treatment_sales,
        'control_sales': control_sales
    }

# Example
treatment = 150000  # Sales in treated markets
control = 120000    # Sales in control markets

result = calculate_incrementality(treatment, control)

print("Incrementality Analysis:")
print(f"  Treatment Sales: ${result['treatment_sales']:,.0f}")
print(f"  Control Sales: ${result['control_sales']:,.0f}")
print(f"  Incremental Lift: ${result['lift']:,.0f}")
print(f"  Incrementality: {result['incrementality_pct']:.1f}%")
```

### Geo-Lift Study Design

```python
"""
Geo-Lift (Geographic Experiment):
- Randomize geographic markets (cities, states, DMAs)
- Treatment group: Run campaign
- Control group: No campaign (or different campaign)
- Compare outcomes

Requirements:
- Similar markets in treatment and control
- Sufficient number of markets (n ≥ 10 per group ideally)
- Pre-period data to establish baseline
- Stable test conditions
"""

import pandas as pd
import numpy as np

# Example: Geo-lift study setup
geo_lift_design = pd.DataFrame({
    'market': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
               'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
    'population': [8.3, 3.9, 2.7, 2.3, 1.7, 1.6, 1.5, 1.4, 1.3, 1.0],  # millions
    'avg_weekly_sales': [150000, 95000, 67000, 58000, 45000, 42000,
                         40000, 38000, 36000, 28000],
    'group': ['treatment', 'treatment', 'control', 'treatment', 'control',
              'treatment', 'control', 'control', 'treatment', 'control']
})

print("Geo-Lift Study Design:")
print(geo_lift_design)

# Check balance between groups
print("\nGroup Balance:")
treatment_summary = geo_lift_design[geo_lift_design['group'] == 'treatment'].agg({
    'population': 'mean',
    'avg_weekly_sales': 'mean'
})
control_summary = geo_lift_design[geo_lift_design['group'] == 'control'].agg({
    'population': 'mean',
    'avg_weekly_sales': 'mean'
})

print("\nTreatment Group:")
print(f"  Avg Population: {treatment_summary['population']:.2f}M")
print(f"  Avg Weekly Sales: ${treatment_summary['avg_weekly_sales']:,.0f}")

print("\nControl Group:")
print(f"  Avg Population: {control_summary['population']:.2f}M")
print(f"  Avg Weekly Sales: ${control_summary['avg_weekly_sales']:,.0f}")
```

### Simple Geo-Lift Analysis
```python
import numpy as np
from scipy.stats import ttest_ind

def analyze_geo_lift(treatment_markets, control_markets):
    """
    Analyze geo-lift experiment results.

    treatment_markets: array of sales in treatment markets
    control_markets: array of sales in control markets
    """
    # Descriptive statistics
    treatment_mean = np.mean(treatment_markets)
    control_mean = np.mean(control_markets)
    lift = treatment_mean - control_mean
    incrementality = (lift / control_mean) * 100 if control_mean > 0 else 0

    # Statistical test
    t_stat, p_value = ttest_ind(treatment_markets, control_markets)

    # Confidence interval for difference
    from scipy.stats import t
    n1, n2 = len(treatment_markets), len(control_markets)
    df = n1 + n2 - 2
    s1, s2 = np.std(treatment_markets, ddof=1), np.std(control_markets, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / df)
    se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
    t_critical = t.ppf(0.975, df)

    ci_lower = lift - t_critical * se_diff
    ci_upper = lift + t_critical * se_diff

    # Results
    print("Geo-Lift Analysis Results")
    print("="*60)
    print(f"Treatment Markets (n={n1}): ${treatment_mean:,.0f}")
    print(f"Control Markets (n={n2}): ${control_mean:,.0f}")
    print(f"\nIncremental Lift: ${lift:,.0f}")
    print(f"Incrementality: {incrementality:+.1f}%")
    print(f"95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]")
    print(f"\nT-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\n✓ Statistically significant lift (α=0.05)")
    else:
        print("\n✗ No statistically significant lift detected")

    return {
        'treatment_mean': treatment_mean,
        'control_mean': control_mean,
        'lift': lift,
        'incrementality': incrementality,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value
    }

# Example: Campaign ran for 4 weeks
# Weekly sales in treatment markets
treatment_markets = np.array([152000, 148000, 155000, 151000, 149000])

# Weekly sales in control markets
control_markets = np.array([121000, 119000, 123000, 120000, 118000])

results = analyze_geo_lift(treatment_markets, control_markets)
```

---

## 🎯 Difference-in-Differences (DiD)

### DiD Concept
```python
"""
Difference-in-Differences (DiD):
- Accounts for pre-existing trends
- Compares change over time between treatment and control
- Controls for time-invariant differences between groups

Formula:
DiD = (Treatment_After - Treatment_Before) - (Control_After - Control_Before)

Assumptions:
1. Parallel trends: Treatment and control follow similar trends pre-intervention
2. No spillover effects
3. Stable composition of groups
"""
```

### DiD Implementation
```python
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def difference_in_differences(treatment_before, treatment_after,
                               control_before, control_after):
    """
    Calculate Difference-in-Differences estimate.

    Parameters:
    - treatment_before: Treatment group pre-period values (array)
    - treatment_after: Treatment group post-period values (array)
    - control_before: Control group pre-period values (array)
    - control_after: Control group post-period values (array)

    Returns:
    - DiD estimate and statistical test
    """
    # Calculate means
    treatment_before_mean = np.mean(treatment_before)
    treatment_after_mean = np.mean(treatment_after)
    control_before_mean = np.mean(control_before)
    control_after_mean = np.mean(control_after)

    # Changes within each group
    treatment_change = treatment_after_mean - treatment_before_mean
    control_change = control_after_mean - control_before_mean

    # DiD estimate
    did_estimate = treatment_change - control_change

    # Statistical test using regression approach
    # Create dataset
    data = pd.DataFrame({
        'sales': np.concatenate([treatment_before, treatment_after,
                                 control_before, control_after]),
        'treated': np.concatenate([
            np.ones(len(treatment_before) + len(treatment_after)),
            np.zeros(len(control_before) + len(control_after))
        ]),
        'post': np.concatenate([
            np.zeros(len(treatment_before)),
            np.ones(len(treatment_after)),
            np.zeros(len(control_before)),
            np.ones(len(control_after))
        ])
    })

    # Interaction term
    data['treated_x_post'] = data['treated'] * data['post']

    # Regression: Sales = β0 + β1*Treated + β2*Post + β3*Treated×Post + ε
    # β3 is the DiD estimate
    from sklearn.linear_model import LinearRegression
    X = data[['treated', 'post', 'treated_x_post']]
    y = data['sales']

    model = LinearRegression()
    model.fit(X, y)

    did_coefficient = model.coef_[2]  # Interaction term

    # Results
    print("Difference-in-Differences Analysis")
    print("="*60)
    print("\nPre-Period:")
    print(f"  Treatment: ${treatment_before_mean:,.0f}")
    print(f"  Control: ${control_before_mean:,.0f}")

    print("\nPost-Period:")
    print(f"  Treatment: ${treatment_after_mean:,.0f}")
    print(f"  Control: ${control_after_mean:,.0f}")

    print("\nChanges:")
    print(f"  Treatment: ${treatment_change:,.0f}")
    print(f"  Control: ${control_change:,.0f}")

    print(f"\nDiD Estimate: ${did_estimate:,.0f}")
    print(f"Incrementality: {(did_estimate/treatment_before_mean)*100:+.1f}%")

    return {
        'did_estimate': did_estimate,
        'treatment_change': treatment_change,
        'control_change': control_change,
        'model': model
    }

# Example
# Pre-campaign (4 weeks)
treatment_before = np.array([100000, 102000, 98000, 101000])
control_before = np.array([80000, 82000, 79000, 81000])

# During campaign (4 weeks)
treatment_after = np.array([135000, 138000, 132000, 136000])
control_after = np.array([85000, 87000, 84000, 86000])

results = difference_in_differences(treatment_before, treatment_after,
                                    control_before, control_after)
```

### Parallel Trends Check
```python
import numpy as np
import matplotlib.pyplot as plt

def check_parallel_trends(treatment_pre, control_pre, periods):
    """
    Visualize parallel trends assumption.

    treatment_pre: Treatment group pre-period time series
    control_pre: Control group pre-period time series
    periods: List of period labels
    """
    plt.figure(figsize=(10, 6))

    plt.plot(periods, treatment_pre, marker='o', label='Treatment', linewidth=2)
    plt.plot(periods, control_pre, marker='s', label='Control', linewidth=2)

    plt.xlabel('Period')
    plt.ylabel('Sales ($)')
    plt.title('Parallel Trends Check (Pre-Period)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate trend similarity
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(treatment_pre, control_pre)

    plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # plt.show()

    print("\nParallel Trends Check:")
    print(f"  Correlation: {correlation:.3f}")
    if correlation > 0.8:
        print("  ✓ Trends appear parallel (high correlation)")
    else:
        print("  ⚠ Trends may not be parallel (low correlation)")

    return correlation

# Example
periods = ['Week -4', 'Week -3', 'Week -2', 'Week -1']
treatment_pre = [100000, 102000, 98000, 101000]
control_pre = [80000, 82000, 79000, 81000]

# check_parallel_trends(treatment_pre, control_pre, periods)
```

---

## 🎯 Matched Market Testing

### Market Matching
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

def match_markets(markets_df, treatment_markets, features, n_matches=1):
    """
    Find control markets that best match treatment markets.

    Parameters:
    - markets_df: DataFrame with market data
    - treatment_markets: List of market IDs to be treated
    - features: List of feature columns to match on
    - n_matches: Number of control markets to match per treatment market

    Returns:
    - DataFrame with matched pairs
    """
    # Separate treatment and potential controls
    treatment_df = markets_df[markets_df['market_id'].isin(treatment_markets)]
    control_pool = markets_df[~markets_df['market_id'].isin(treatment_markets)]

    # Standardize features
    scaler = StandardScaler()
    treatment_features = scaler.fit_transform(treatment_df[features])
    control_features = scaler.transform(control_pool[features])

    # Calculate distances
    distances = euclidean_distances(treatment_features, control_features)

    # Find matches
    matches = []
    for i, treatment_market in enumerate(treatment_df['market_id']):
        # Get n closest controls
        closest_indices = np.argsort(distances[i])[:n_matches]

        for j, idx in enumerate(closest_indices):
            control_market = control_pool.iloc[idx]['market_id']
            distance = distances[i][idx]

            matches.append({
                'treatment_market': treatment_market,
                'control_market': control_market,
                'match_rank': j + 1,
                'distance': distance
            })

    matches_df = pd.DataFrame(matches)

    return matches_df

# Example
markets = pd.DataFrame({
    'market_id': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix',
                  'Philadelphia', 'San Antonio', 'San Diego'],
    'population': [8.3, 3.9, 2.7, 2.3, 1.7, 1.6, 1.5, 1.4],
    'avg_income': [75000, 70000, 65000, 60000, 58000, 62000, 55000, 72000],
    'baseline_sales': [150000, 95000, 67000, 58000, 45000, 42000, 40000, 38000]
})

# Want to treat NYC and LA
treatment_markets = ['NYC', 'LA']
features = ['population', 'avg_income', 'baseline_sales']

matches = match_markets(markets, treatment_markets, features, n_matches=2)

print("Matched Markets:")
print(matches)
```

### Analyze Matched Market Test
```python
def analyze_matched_market_test(matched_pairs, treatment_results, control_results):
    """
    Analyze matched market test with multiple pairs.

    matched_pairs: DataFrame with treatment-control pairs
    treatment_results: Dict mapping treatment market to outcome
    control_results: Dict mapping control market to outcome
    """
    lifts = []

    print("Matched Market Test Results")
    print("="*60)

    for _, pair in matched_pairs.iterrows():
        treatment_market = pair['treatment_market']
        control_market = pair['control_market']

        treatment_value = treatment_results[treatment_market]
        control_value = control_results[control_market]

        lift = treatment_value - control_value
        incrementality = (lift / control_value) * 100 if control_value > 0 else 0

        lifts.append(lift)

        print(f"\nPair: {treatment_market} (T) vs {control_market} (C)")
        print(f"  Treatment: ${treatment_value:,.0f}")
        print(f"  Control: ${control_value:,.0f}")
        print(f"  Lift: ${lift:,.0f} ({incrementality:+.1f}%)")

    # Aggregate results
    avg_lift = np.mean(lifts)
    lift_se = np.std(lifts, ddof=1) / np.sqrt(len(lifts))

    # Confidence interval
    from scipy.stats import t
    t_critical = t.ppf(0.975, len(lifts) - 1)
    ci_lower = avg_lift - t_critical * lift_se
    ci_upper = avg_lift + t_critical * lift_se

    # Overall test
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp(lifts, 0)

    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Average Lift: ${avg_lift:,.0f}")
    print(f"95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\n✓ Statistically significant lift")
    else:
        print("\n✗ No statistically significant lift")

    return {
        'avg_lift': avg_lift,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value
    }

# Example
treatment_results = {'NYC': 165000, 'LA': 105000}
control_results = {'Chicago': 68000, 'Houston': 59000,
                   'Philadelphia': 43000, 'Phoenix': 46000}

# Using matches from previous example (assuming first 2 matches)
matched_pairs = pd.DataFrame([
    {'treatment_market': 'NYC', 'control_market': 'Chicago'},
    {'treatment_market': 'LA', 'control_market': 'Houston'}
])

# results = analyze_matched_market_test(matched_pairs, treatment_results, control_results)
```

---

## 🎯 Synthetic Control (Introduction)

### Synthetic Control Concept
```python
"""
Synthetic Control:
- Creates a "synthetic" control by weighting multiple control units
- Weighted combination of controls matches treatment unit pre-intervention
- More flexible than single matched control

Example:
Treatment: NYC
Synthetic Control: 0.4×LA + 0.3×Chicago + 0.2×Houston + 0.1×Phoenix

The weights are chosen so synthetic control's pre-period matches NYC's pre-period.
"""
```

### Simple Synthetic Control
```python
import numpy as np
from scipy.optimize import minimize

def create_synthetic_control(treatment_pre, control_units_pre):
    """
    Create synthetic control by finding optimal weights.

    treatment_pre: Pre-period values for treatment unit (array)
    control_units_pre: Pre-period values for control units (2D array)
                       Shape: (n_periods, n_control_units)

    Returns:
    - weights: Optimal weights for each control unit
    """
    n_controls = control_units_pre.shape[1]

    # Objective: Minimize squared difference between treatment and synthetic
    def objective(weights):
        synthetic = control_units_pre @ weights
        return np.sum((treatment_pre - synthetic) ** 2)

    # Constraints: weights sum to 1, all weights >= 0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_controls)]

    # Initial guess: equal weights
    initial_weights = np.ones(n_controls) / n_controls

    # Optimize
    result = minimize(objective, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result.x

# Example
# 8 weeks pre-period
treatment_pre = np.array([100, 102, 98, 101, 99, 103, 100, 102])

# 3 control markets
control_units_pre = np.array([
    [80, 82, 79, 81, 80, 83, 81, 82],   # Control 1
    [60, 61, 59, 62, 60, 63, 61, 62],   # Control 2
    [50, 51, 49, 50, 51, 52, 50, 51]    # Control 3
]).T  # Transpose to (n_periods, n_controls)

weights = create_synthetic_control(treatment_pre, control_units_pre)

print("Synthetic Control Weights:")
for i, w in enumerate(weights):
    print(f"  Control {i+1}: {w:.3f} ({w*100:.1f}%)")

# Create synthetic control
synthetic_pre = control_units_pre @ weights

print("\nPre-Period Fit:")
print("Week | Treatment | Synthetic | Difference")
for week, (t, s) in enumerate(zip(treatment_pre, synthetic_pre), 1):
    diff = t - s
    print(f"  {week}  |   {t:5.0f}   |   {s:5.1f}   |   {diff:+5.1f}")

# Now use these weights for post-period analysis
# treatment_post vs (control_units_post @ weights)
```

### Synthetic Control Analysis
```python
def analyze_synthetic_control(treatment_pre, treatment_post,
                               control_units_pre, control_units_post):
    """
    Full synthetic control analysis.
    """
    # Create synthetic control
    weights = create_synthetic_control(treatment_pre, control_units_pre)

    # Generate synthetic control for post-period
    synthetic_pre = control_units_pre @ weights
    synthetic_post = control_units_post @ weights

    # Calculate lift
    treatment_post_mean = np.mean(treatment_post)
    synthetic_post_mean = np.mean(synthetic_post)
    lift = treatment_post_mean - synthetic_post_mean

    # Pre-period RMSE (goodness of fit)
    pre_rmse = np.sqrt(np.mean((treatment_pre - synthetic_pre) ** 2))

    print("Synthetic Control Analysis")
    print("="*60)
    print(f"\nPre-Period RMSE: {pre_rmse:.2f}")
    print("(Lower is better - indicates good fit)")

    print(f"\nPost-Period:")
    print(f"  Treatment: ${treatment_post_mean:,.0f}")
    print(f"  Synthetic Control: ${synthetic_post_mean:,.0f}")
    print(f"  Lift: ${lift:,.0f}")

    # Statistical significance via permutation test (simplified)
    # In practice, would do full placebo test

    return {
        'weights': weights,
        'lift': lift,
        'pre_rmse': pre_rmse,
        'treatment_post_mean': treatment_post_mean,
        'synthetic_post_mean': synthetic_post_mean
    }

# Example with post-period
treatment_post = np.array([135, 138, 132, 136])  # 4 weeks post

control_units_post = np.array([
    [85, 87, 84, 86],   # Control 1
    [64, 65, 63, 65],   # Control 2
    [53, 54, 52, 53]    # Control 3
]).T

results = analyze_synthetic_control(treatment_pre, treatment_post,
                                    control_units_pre, control_units_post)
```

---

## 🎯 Incrementality Metrics

### iROAS (Incremental ROAS)
```python
def calculate_iroas(incremental_revenue, campaign_cost):
    """
    Calculate Incremental Return on Ad Spend.

    iROAS = Incremental Revenue / Campaign Cost
    """
    iroas = incremental_revenue / campaign_cost if campaign_cost > 0 else 0

    return iroas

# Example
# Campaign cost: $50,000
# Treatment sales: $650,000
# Control sales: $600,000
# Incremental revenue: $50,000

incremental_revenue = 650000 - 600000
campaign_cost = 50000

iroas = calculate_iroas(incremental_revenue, campaign_cost)

print(f"Incremental Revenue: ${incremental_revenue:,.0f}")
print(f"Campaign Cost: ${campaign_cost:,.0f}")
print(f"iROAS: {iroas:.2f}x")

if iroas > 1:
    print(f"✓ Campaign is profitable (iROAS > 1)")
    print(f"  ${iroas:.2f} in revenue for every $1 spent")
else:
    print(f"✗ Campaign is not profitable (iROAS < 1)")
```

### Incremental Contribution Margin
```python
def calculate_incremental_margin(incremental_revenue, contribution_margin_pct,
                                  campaign_cost):
    """
    Calculate incremental profit accounting for margins.

    contribution_margin_pct: Percentage of revenue that is profit (before marketing)
    """
    incremental_gross_profit = incremental_revenue * (contribution_margin_pct / 100)
    incremental_net_profit = incremental_gross_profit - campaign_cost

    incremental_roi = (incremental_net_profit / campaign_cost) * 100 if campaign_cost > 0 else 0

    print("Incremental Contribution Analysis")
    print("="*60)
    print(f"Incremental Revenue: ${incremental_revenue:,.0f}")
    print(f"Contribution Margin: {contribution_margin_pct}%")
    print(f"Incremental Gross Profit: ${incremental_gross_profit:,.0f}")
    print(f"Campaign Cost: ${campaign_cost:,.0f}")
    print(f"Incremental Net Profit: ${incremental_net_profit:,.0f}")
    print(f"Incremental ROI: {incremental_roi:+.1f}%")

    return {
        'incremental_revenue': incremental_revenue,
        'incremental_gross_profit': incremental_gross_profit,
        'incremental_net_profit': incremental_net_profit,
        'incremental_roi': incremental_roi
    }

# Example: 40% contribution margin
calculate_incremental_margin(
    incremental_revenue=50000,
    contribution_margin_pct=40,
    campaign_cost=50000
)
```

### Organic vs Incremental
```python
def decompose_sales(total_sales, incremental_sales):
    """
    Decompose total sales into organic vs incremental.
    """
    organic_sales = total_sales - incremental_sales
    incremental_pct = (incremental_sales / total_sales) * 100

    print("Sales Decomposition")
    print("="*60)
    print(f"Total Sales: ${total_sales:,.0f}")
    print(f"  Organic: ${organic_sales:,.0f} ({100-incremental_pct:.1f}%)")
    print(f"  Incremental: ${incremental_sales:,.0f} ({incremental_pct:.1f}%)")

    return {
        'total_sales': total_sales,
        'organic_sales': organic_sales,
        'incremental_sales': incremental_sales,
        'incremental_pct': incremental_pct
    }

# Example
# Treatment group total sales: $650,000
# Baseline (control) sales: $600,000
# Incremental: $50,000

decompose_sales(total_sales=650000, incremental_sales=50000)
```

---

## 🚀 Quick Tips

### Experimental Design Best Practices
1. **Randomization**: Truly random assignment to treatment/control
2. **Sample Size**: Sufficient for statistical power (use power analysis)
3. **Pre-Period**: Collect baseline data (ideally 2-4x test duration)
4. **Isolation**: Avoid contamination between groups
5. **Duration**: Run long enough to capture full effect (2-4 weeks minimum)

### Common Mistakes
```python
# ❌ Wrong: Cherry-picking favorable markets for treatment
treatment = ['NYC', 'LA']  # Best performing markets
control = ['Rural Town 1', 'Rural Town 2']  # Worst performing

# ✅ Correct: Random assignment or matched pairs
import random
all_markets = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Philly']
random.shuffle(all_markets)
treatment = all_markets[:3]
control = all_markets[3:]

# ❌ Wrong: Not checking parallel trends
# Assumes treatment and control are comparable without verification

# ✅ Correct: Verify parallel trends in pre-period
check_parallel_trends(treatment_pre, control_pre, periods)

# ❌ Wrong: Running test for too short a duration
test_duration_days = 3  # Too short!

# ✅ Correct: Sufficient test duration
test_duration_weeks = 4  # Minimum recommended
# Account for purchase cycles

# ❌ Wrong: Ignoring external factors
# Sales spike due to holiday, not campaign

# ✅ Correct: Control for external events
# Use DiD to account for time trends
# Document and adjust for known external shocks
```

### Statistical Power for Geo-Lift
```python
from statsmodels.stats.power import ttest_power
import numpy as np

def calculate_required_markets(baseline_mean, expected_lift_pct,
                                baseline_std, alpha=0.05, power=0.80):
    """
    Calculate required number of markets per group.

    baseline_mean: Average sales in pre-period
    expected_lift_pct: Expected lift as percentage (e.g., 15 for 15%)
    baseline_std: Standard deviation of sales across markets
    """
    # Expected lift in absolute terms
    expected_lift = baseline_mean * (expected_lift_pct / 100)

    # Effect size (Cohen's d)
    effect_size = expected_lift / baseline_std

    # Calculate required sample size
    from statsmodels.stats.power import tt_ind_solve_power
    n_per_group = tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )

    print("Sample Size Calculation for Geo-Lift")
    print("="*60)
    print(f"Baseline Mean: ${baseline_mean:,.0f}")
    print(f"Expected Lift: {expected_lift_pct}% (${expected_lift:,.0f})")
    print(f"Baseline Std Dev: ${baseline_std:,.0f}")
    print(f"Effect Size (Cohen's d): {effect_size:.3f}")
    print(f"\nRequired markets per group: {np.ceil(n_per_group):.0f}")
    print(f"Total markets needed: {np.ceil(n_per_group) * 2:.0f}")

    return np.ceil(n_per_group)

# Example
calculate_required_markets(
    baseline_mean=50000,
    expected_lift_pct=15,
    baseline_std=10000,
    alpha=0.05,
    power=0.80
)
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Geo-Lift Analysis
```python
import numpy as np
from scipy.stats import ttest_ind

# Campaign ran in 6 treatment markets and 6 control markets for 4 weeks
# Weekly sales data

# Treatment markets (6 markets × 4 weeks)
treatment_data = np.array([
    [52000, 53500, 51800, 52200],  # Market 1
    [48000, 49200, 47800, 48500],  # Market 2
    [55000, 56800, 54500, 55200],  # Market 3
    [45000, 46500, 44800, 45300],  # Market 4
    [50000, 51500, 49800, 50200],  # Market 5
    [47000, 48200, 46500, 47300],  # Market 6
])

# Control markets (6 markets × 4 weeks)
control_data = np.array([
    [42000, 42500, 41800, 42200],  # Market 1
    [38000, 38500, 37800, 38200],  # Market 2
    [45000, 45500, 44800, 45200],  # Market 3
    [35000, 35500, 34800, 35200],  # Market 4
    [40000, 40500, 39800, 40200],  # Market 5
    [37000, 37500, 36800, 37200],  # Market 6
])

# Calculate averages per market
treatment_avg = treatment_data.mean(axis=1)
control_avg = control_data.mean(axis=1)

print("Geo-Lift Study Results")
print("="*60)

# Descriptive statistics
print("\nTreatment Markets:")
for i, avg in enumerate(treatment_avg, 1):
    print(f"  Market {i}: ${avg:,.0f}")
print(f"  Average: ${treatment_avg.mean():,.0f}")

print("\nControl Markets:")
for i, avg in enumerate(control_avg, 1):
    print(f"  Market {i}: ${avg:,.0f}")
print(f"  Average: ${control_avg.mean():,.0f}")

# Statistical test
t_stat, p_value = ttest_ind(treatment_avg, control_avg)

lift = treatment_avg.mean() - control_avg.mean()
incrementality = (lift / control_avg.mean()) * 100

print(f"\nIncremental Lift: ${lift:,.0f}")
print(f"Incrementality: {incrementality:+.1f}%")
print(f"\nT-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n✓ Statistically significant lift at α=0.05")

    # Calculate iROAS
    campaign_cost = 75000
    incremental_revenue = lift * 6 * 4  # 6 markets × 4 weeks
    iroas = incremental_revenue / campaign_cost

    print(f"\nROI Analysis:")
    print(f"  Campaign Cost: ${campaign_cost:,.0f}")
    print(f"  Incremental Revenue: ${incremental_revenue:,.0f}")
    print(f"  iROAS: {iroas:.2f}x")

    if iroas > 2:
        print(f"  ✓ Strong ROI - Scale campaign!")
else:
    print("\n✗ No statistically significant lift detected")
```

### Exercise 2: Difference-in-Differences
```python
import numpy as np

# Email campaign test
# Pre-period: 6 weeks before campaign
# Post-period: 4 weeks during campaign

# Treatment group (received email campaign)
treatment_pre = np.array([12000, 12500, 11800, 12200, 12300, 11900])
treatment_post = np.array([15200, 15800, 14900, 15300])

# Control group (no email campaign)
control_pre = np.array([10000, 10400, 9800, 10200, 10300, 9900])
control_post = np.array([10500, 10900, 10300, 10600])

# Calculate DiD
treatment_pre_mean = treatment_pre.mean()
treatment_post_mean = treatment_post.mean()
control_pre_mean = control_pre.mean()
control_post_mean = control_post.mean()

treatment_change = treatment_post_mean - treatment_pre_mean
control_change = control_post_mean - control_pre_mean
did_estimate = treatment_change - control_change

print("Difference-in-Differences Analysis")
print("="*60)

print("\nPre-Period Averages:")
print(f"  Treatment: ${treatment_pre_mean:,.0f}")
print(f"  Control: ${control_pre_mean:,.0f}")

print("\nPost-Period Averages:")
print(f"  Treatment: ${treatment_post_mean:,.0f}")
print(f"  Control: ${control_post_mean:,.0f}")

print("\nWithin-Group Changes:")
print(f"  Treatment: ${treatment_change:+,.0f}")
print(f"  Control: ${control_change:+,.0f}")

print(f"\nDiD Estimate (Incremental Impact): ${did_estimate:+,.0f}")
incrementality = (did_estimate / treatment_pre_mean) * 100
print(f"Incrementality: {incrementality:+.1f}%")

# What if we only looked at treatment group?
naive_estimate = treatment_change
print(f"\nNaive Estimate (ignoring control): ${naive_estimate:+,.0f}")
print(f"Difference: ${did_estimate - naive_estimate:+,.0f}")
print("↑ This is the bias we avoid by using DiD!")
```

### Exercise 3: Synthetic Control
```python
from scipy.optimize import minimize
import numpy as np

# NYC (treatment) vs creating synthetic NYC from other cities

# Pre-period: 8 weeks
nyc_pre = np.array([150000, 152000, 148000, 151000, 149000, 153000, 150000, 152000])

# Control cities pre-period
control_cities_pre = np.array([
    [95000, 96000, 94000, 96500, 95500, 97000, 95000, 96000],   # LA
    [67000, 68000, 66000, 68500, 67500, 69000, 67000, 68000],   # Chicago
    [58000, 59000, 57000, 59500, 58500, 60000, 58000, 59000],   # Houston
    [45000, 46000, 44000, 46500, 45500, 47000, 45000, 46000],   # Phoenix
]).T

# Create synthetic control
weights = create_synthetic_control(nyc_pre, control_cities_pre)

print("Synthetic NYC Construction")
print("="*60)
print("\nWeights:")
cities = ['LA', 'Chicago', 'Houston', 'Phoenix']
for city, weight in zip(cities, weights):
    print(f"  {city}: {weight:.3f} ({weight*100:.1f}%)")

# Check pre-period fit
synthetic_nyc_pre = control_cities_pre @ weights
pre_rmse = np.sqrt(np.mean((nyc_pre - synthetic_nyc_pre) ** 2))

print(f"\nPre-Period Fit RMSE: ${pre_rmse:,.0f}")

# Post-period (4 weeks during campaign)
nyc_post = np.array([168000, 172000, 165000, 170000])

control_cities_post = np.array([
    [96500, 97000, 96000, 97500],   # LA
    [68500, 69000, 68000, 69500],   # Chicago
    [59500, 60000, 59000, 60500],   # Houston
    [46500, 47000, 46000, 47500],   # Phoenix
]).T

# Synthetic NYC post-period
synthetic_nyc_post = control_cities_post @ weights

# Calculate lift
nyc_post_mean = nyc_post.mean()
synthetic_post_mean = synthetic_nyc_post.mean()
lift = nyc_post_mean - synthetic_post_mean

print(f"\nPost-Period Analysis:")
print(f"  Actual NYC: ${nyc_post_mean:,.0f}")
print(f"  Synthetic NYC: ${synthetic_post_mean:,.0f}")
print(f"  Incremental Lift: ${lift:,.0f}")

incrementality = (lift / synthetic_post_mean) * 100
print(f"  Incrementality: {incrementality:+.1f}%")
```

### Exercise 4: iROAS Calculation
```python
# Multi-channel campaign analysis

campaigns = {
    'TV': {
        'cost': 150000,
        'treatment_sales': 2200000,
        'control_sales': 2000000
    },
    'Digital': {
        'cost': 80000,
        'treatment_sales': 950000,
        'control_sales': 850000
    },
    'Radio': {
        'cost': 40000,
        'treatment_sales': 420000,
        'control_sales': 400000
    }
}

print("Multi-Channel Incrementality Analysis")
print("="*60)

total_cost = 0
total_incremental_revenue = 0

for channel, data in campaigns.items():
    incremental_revenue = data['treatment_sales'] - data['control_sales']
    iroas = incremental_revenue / data['cost']

    total_cost += data['cost']
    total_incremental_revenue += incremental_revenue

    print(f"\n{channel}:")
    print(f"  Cost: ${data['cost']:,.0f}")
    print(f"  Treatment Sales: ${data['treatment_sales']:,.0f}")
    print(f"  Control Sales: ${data['control_sales']:,.0f}")
    print(f"  Incremental Revenue: ${incremental_revenue:,.0f}")
    print(f"  iROAS: {iroas:.2f}x")

    if iroas > 2:
        print(f"  ✓ Strong performer - Scale up!")
    elif iroas > 1:
        print(f"  ✓ Profitable - Maintain")
    else:
        print(f"  ✗ Unprofitable - Reduce or cut")

# Overall
overall_iroas = total_incremental_revenue / total_cost

print(f"\n{'='*60}")
print("OVERALL CAMPAIGN:")
print(f"  Total Cost: ${total_cost:,.0f}")
print(f"  Total Incremental Revenue: ${total_incremental_revenue:,.0f}")
print(f"  Overall iROAS: {overall_iroas:.2f}x")

# With contribution margin (assume 40%)
contribution_margin = 0.40
incremental_gross_profit = total_incremental_revenue * contribution_margin
incremental_net_profit = incremental_gross_profit - total_cost
incremental_roi = (incremental_net_profit / total_cost) * 100

print(f"\nWith 40% Contribution Margin:")
print(f"  Incremental Gross Profit: ${incremental_gross_profit:,.0f}")
print(f"  Incremental Net Profit: ${incremental_net_profit:,.0f}")
print(f"  Incremental ROI: {incremental_roi:+.1f}%")
```

---

## 🔍 Incrementality Testing Checklist

### Pre-Test
- [ ] Clear hypothesis defined
- [ ] Primary metric selected (sales, conversions, etc.)
- [ ] Treatment and control groups defined
- [ ] Sufficient sample size (power analysis)
- [ ] Pre-period baseline established
- [ ] Parallel trends verified (for DiD)
- [ ] Randomization protocol defined

### During Test
- [ ] No contamination between groups
- [ ] Consistent treatment delivery
- [ ] Monitor for external shocks
- [ ] Data collection ongoing
- [ ] No mid-test changes

### Post-Test
- [ ] Statistical significance assessed
- [ ] Effect size calculated
- [ ] Confidence intervals reported
- [ ] iROAS/iROI calculated
- [ ] Results validated
- [ ] Decision made (scale/maintain/cut)

---

**Quick Navigation:**
- [← Week 10 Cheatsheet](Week_10_Cheatsheet.md)
- [Week 12 Cheatsheet →](Week_12_Cheatsheet.md)
- [Back to Main README](../README.md)
