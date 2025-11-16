# Week 10: Marketing Mix Modeling (MMM) - Quick Reference Cheatsheet

## 📋 Core Concepts

### Marketing Mix Modeling Fundamentals

```python
"""
Marketing Mix Modeling (MMM):
- Statistical analysis of marketing spend impact on sales
- Regression-based approach
- Accounts for:
  * Marketing channels (TV, digital, print, etc.)
  * Adstock (carryover effects)
  * Saturation (diminishing returns)
  * Seasonality
  * External factors (economy, weather, competitors)

Basic MMM Equation:
Sales = β₀ + β₁×TV_adstocked + β₂×Digital_adstocked + ... + ε

Where:
- β coefficients measure channel contribution
- Adstocked variables account for carryover
- ε is error term
"""
```

### Simple Linear Regression
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Sample data: TV spend → Sales
tv_spend = np.array([100, 150, 200, 250, 300, 350, 400]).reshape(-1, 1)
sales = np.array([1200, 1500, 1800, 2000, 2150, 2250, 2300])

# Fit model
model = LinearRegression()
model.fit(tv_spend, sales)

# Coefficients
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient (TV): {model.coef_[0]:.2f}")
print(f"Interpretation: Each $1 in TV spend → ${model.coef_[0]:.2f} in sales")

# Predictions
predictions = model.predict(tv_spend)

# Model performance
r2 = r2_score(sales, predictions)
rmse = np.sqrt(mean_squared_error(sales, predictions))

print(f"\nR²: {r2:.3f} ({r2*100:.1f}% of variance explained)")
print(f"RMSE: ${rmse:.2f}")

# Predict future sales
future_spend = np.array([[450]])
future_sales = model.predict(future_spend)
print(f"\nPredicted sales at $450 TV spend: ${future_sales[0]:.2f}")
```

### Multiple Linear Regression
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Multiple channels
data = pd.DataFrame({
    'tv_spend': [100, 150, 200, 250, 300, 350, 400],
    'digital_spend': [50, 75, 100, 125, 150, 175, 200],
    'print_spend': [30, 40, 50, 60, 70, 80, 90],
    'sales': [1200, 1650, 2100, 2500, 2850, 3100, 3300]
})

# Features and target
X = data[['tv_spend', 'digital_spend', 'print_spend']]
y = data['sales']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Coefficients
print("MMM Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"  Intercept: {model.intercept_:.2f}")

# Model performance
predictions = model.predict(X)
r2 = r2_score(y, predictions)
print(f"\nR²: {r2:.3f}")

# Interpretation: ROI
print("\nROI Interpretation:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: ${coef:.2f} sales per $1 spent → ROI = {(coef-1)*100:.1f}%")
```

---

## 🎯 Adstock Transformations

### Adstock Concept
```python
"""
Adstock (Carryover Effect):
- Marketing impact doesn't stop immediately
- Effect "decays" over time
- TV ad today impacts sales for weeks
- Email campaign has shorter carryover

Geometric Adstock Formula:
Adstock_t = Spend_t + λ × Adstock_(t-1)

Where:
- λ (lambda) = retention rate (0 to 1)
- Higher λ = longer carryover
- λ = 0.5 means 50% of impact carries to next period
"""
```

### Geometric Adstock
```python
import numpy as np

def geometric_adstock(spend, decay_rate):
    """
    Apply geometric adstock transformation.

    Parameters:
    - spend: array of marketing spend by period
    - decay_rate (λ): retention rate (0 to 1)

    Returns:
    - adstocked: transformed spend with carryover
    """
    adstocked = np.zeros(len(spend))
    adstocked[0] = spend[0]

    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay_rate * adstocked[t-1]

    return adstocked

# Example
weekly_tv_spend = np.array([100, 0, 0, 0, 100, 0, 0, 0])

# No adstock
print("Original spend:", weekly_tv_spend)

# 50% decay (short carryover)
adstock_50 = geometric_adstock(weekly_tv_spend, decay_rate=0.5)
print("Adstock (λ=0.5):", adstock_50.round(1))

# 80% decay (long carryover)
adstock_80 = geometric_adstock(weekly_tv_spend, decay_rate=0.8)
print("Adstock (λ=0.8):", adstock_80.round(1))
```

### Visualize Adstock Effect
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_adstock(spend, decay_rates=[0.3, 0.5, 0.7, 0.9]):
    """
    Visualize adstock with different decay rates.
    """
    plt.figure(figsize=(12, 6))

    # Original spend
    plt.subplot(1, 2, 1)
    plt.bar(range(len(spend)), spend, alpha=0.6, label='Original Spend')
    plt.xlabel('Time Period')
    plt.ylabel('Spend ($)')
    plt.title('Original Spend')
    plt.legend()

    # Adstocked versions
    plt.subplot(1, 2, 2)
    for decay in decay_rates:
        adstocked = geometric_adstock(spend, decay)
        plt.plot(adstocked, marker='o', label=f'λ={decay}')

    plt.xlabel('Time Period')
    plt.ylabel('Adstocked Value')
    plt.title('Adstock Effect (Different Decay Rates)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.show()  # Uncomment to display

# Example: Burst spending
spend = np.array([0, 0, 500, 0, 0, 0, 300, 0, 0, 0])
# visualize_adstock(spend)  # Uncomment to visualize
```

### Optimal Decay Rate
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def find_optimal_decay(spend, sales, decay_range=np.arange(0, 1, 0.05)):
    """
    Find optimal decay rate by minimizing RMSE.
    """
    best_decay = None
    best_rmse = float('inf')
    results = []

    for decay in decay_range:
        # Apply adstock
        adstocked = geometric_adstock(spend, decay)

        # Fit model
        X = adstocked.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, sales)

        # Evaluate
        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(sales, predictions))

        results.append({'decay': decay, 'rmse': rmse})

        if rmse < best_rmse:
            best_rmse = rmse
            best_decay = decay

    print(f"Optimal decay rate: {best_decay:.2f}")
    print(f"Best RMSE: ${best_rmse:.2f}")

    return best_decay, results

# Example
weeks = 20
np.random.seed(42)
tv_spend = np.random.uniform(50, 200, weeks)
# Simulate sales with known decay of 0.6
adstocked_true = geometric_adstock(tv_spend, 0.6)
sales = 500 + 3 * adstocked_true + np.random.normal(0, 50, weeks)

optimal_decay, results = find_optimal_decay(tv_spend, sales)
```

---

## 🎯 Saturation Curves

### Diminishing Returns Concept
```python
"""
Saturation (Diminishing Returns):
- First $100 in spend → High return
- Next $100 → Lower return
- Eventually plateaus

Common Functions:
1. Logarithmic: Sales = β × log(Spend)
2. Power: Sales = β × Spend^α (where 0 < α < 1)
3. Sigmoid: Sales = L / (1 + e^(-k(Spend-x₀)))

Used to model:
- Channel capacity limits
- Audience saturation
- Creative fatigue
"""
```

### Logarithmic Saturation
```python
import numpy as np
from sklearn.linear_model import LinearRegression

def log_transform(spend, shift=1):
    """
    Apply logarithmic transformation.
    shift prevents log(0)
    """
    return np.log(spend + shift)

# Example
spend = np.array([10, 50, 100, 200, 500, 1000, 2000])
sales = np.array([100, 350, 550, 750, 950, 1050, 1100])

# Linear model on log-transformed spend
X_log = log_transform(spend).reshape(-1, 1)
model = LinearRegression()
model.fit(X_log, sales)

# Predictions
predictions = model.predict(X_log)

print("Logarithmic Saturation Model")
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Interpretation
print("\nIncremental Returns:")
for i in range(1, len(spend)):
    spend_increase = spend[i] - spend[i-1]
    sales_increase = predictions[i] - predictions[i-1]
    roi = sales_increase / spend_increase
    print(f"  ${spend[i-1]} → ${spend[i]}: ROI = {roi:.2f}")
```

### Power Function Saturation
```python
import numpy as np
from scipy.optimize import curve_fit

def power_curve(spend, alpha, beta):
    """
    Power curve: Sales = beta * Spend^alpha
    alpha < 1 for diminishing returns
    """
    return beta * (spend ** alpha)

# Fit power curve
spend = np.array([10, 50, 100, 200, 500, 1000, 2000])
sales = np.array([100, 350, 550, 750, 950, 1050, 1100])

# Fit curve
params, _ = curve_fit(power_curve, spend, sales, p0=[0.5, 50])
alpha, beta = params

print("Power Curve Model")
print(f"Alpha: {alpha:.3f} (diminishing returns)")
print(f"Beta: {beta:.2f}")

# Predictions
predictions = power_curve(spend, alpha, beta)

# Marginal ROI at different spend levels
print("\nMarginal ROI:")
for s in [100, 500, 1000, 2000]:
    # Derivative: d(Sales)/d(Spend) = alpha * beta * Spend^(alpha-1)
    marginal_return = alpha * beta * (s ** (alpha - 1))
    print(f"  At ${s} spend: ${marginal_return:.2f} per additional dollar")
```

### Hill Function (S-Curve)
```python
import numpy as np
from scipy.optimize import curve_fit

def hill_function(spend, max_sales, k, s):
    """
    Hill function (Sigmoid/S-curve).

    Parameters:
    - max_sales: maximum achievable sales
    - k: half-saturation point (spend at 50% max)
    - s: slope parameter (higher = steeper curve)
    """
    return max_sales * (spend ** s) / (k ** s + spend ** s)

# Example data with S-curve pattern
spend = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000])
sales = np.array([50, 180, 300, 480, 750, 900, 980, 1020, 1040])

# Fit Hill function
params, _ = curve_fit(hill_function, spend, sales, p0=[1100, 50, 1.5])
max_sales, k, s = params

print("Hill Function (S-Curve) Model")
print(f"Max Sales: ${max_sales:.2f}")
print(f"Half-Saturation (K): ${k:.2f}")
print(f"Slope (S): {s:.2f}")

# Predictions
predictions = hill_function(spend, max_sales, k, s)

print("\nSpend Efficiency:")
for spend_level in [50, 100, 200, 500]:
    sales_at_level = hill_function(spend_level, max_sales, k, s)
    pct_of_max = (sales_at_level / max_sales) * 100
    print(f"  ${spend_level}: ${sales_at_level:.0f} ({pct_of_max:.1f}% of max)")
```

---

## 🎯 Complete MMM Implementation

### Full MMM with Adstock and Saturation
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

class MarketingMixModel:
    """
    Complete Marketing Mix Model with adstock and saturation.
    """

    def __init__(self, data, target_col='sales', date_col='date'):
        """
        Initialize MMM.

        data: DataFrame with marketing channels and sales
        target_col: name of sales column
        date_col: name of date column (optional)
        """
        self.data = data.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.model = None
        self.feature_cols = []
        self.adstock_params = {}
        self.saturation_params = {}

    def apply_adstock(self, channel, decay_rate):
        """Apply geometric adstock to a channel."""
        spend = self.data[channel].values
        adstocked = geometric_adstock(spend, decay_rate)
        adstocked_col = f"{channel}_adstocked"
        self.data[adstocked_col] = adstocked
        self.adstock_params[channel] = decay_rate
        return adstocked_col

    def apply_saturation(self, channel, saturation_type='log'):
        """Apply saturation transformation."""
        if saturation_type == 'log':
            self.data[f"{channel}_saturated"] = np.log(self.data[channel] + 1)
            saturated_col = f"{channel}_saturated"
        # Can add other saturation types
        return saturated_col

    def add_seasonality(self, freq='weekly'):
        """Add seasonal features."""
        if freq == 'weekly' and self.date_col in self.data.columns:
            self.data['week_of_year'] = pd.to_datetime(self.data[self.date_col]).dt.isocalendar().week
            # Create dummy variables
            week_dummies = pd.get_dummies(self.data['week_of_year'], prefix='week')
            self.data = pd.concat([self.data, week_dummies], axis=1)

    def prepare_features(self, channel_configs):
        """
        Prepare features with adstock and saturation.

        channel_configs: dict like {'tv': {'decay': 0.5, 'saturation': 'log'}}
        """
        self.feature_cols = []

        for channel, config in channel_configs.items():
            # Apply transformations
            col = channel

            if 'decay' in config:
                col = self.apply_adstock(channel, config['decay'])

            if 'saturation' in config:
                col = self.apply_saturation(col, config['saturation'])

            self.feature_cols.append(col)

    def fit(self):
        """Fit the MMM."""
        X = self.data[self.feature_cols]
        y = self.data[self.target_col]

        self.model = LinearRegression()
        self.model.fit(X, y)

        # Store predictions
        self.data['predicted_sales'] = self.model.predict(X)
        self.data['residuals'] = y - self.data['predicted_sales']

    def get_coefficients(self):
        """Get model coefficients."""
        coef_df = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': self.model.coef_
        })
        return coef_df

    def evaluate(self):
        """Evaluate model performance."""
        y_true = self.data[self.target_col]
        y_pred = self.data['predicted_sales']

        metrics = {
            'R²': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        return metrics

    def channel_contribution(self):
        """Calculate contribution of each channel to total sales."""
        contributions = {}

        for i, col in enumerate(self.feature_cols):
            contribution = self.model.coef_[i] * self.data[col].sum()
            contributions[col] = contribution

        baseline = self.model.intercept_ * len(self.data)
        contributions['baseline'] = baseline

        return pd.Series(contributions)

    def optimize_budget(self, total_budget, channels):
        """
        Simple budget optimization (placeholder).
        In practice, use scipy.optimize or similar.
        """
        # This is a simplified version
        # Real optimization would use marginal ROI curves
        print(f"Optimizing ${total_budget:,.0f} across {len(channels)} channels...")
        print("(Advanced optimization requires numerical methods)")

# Example usage
data = pd.DataFrame({
    'week': range(1, 53),
    'tv': np.random.uniform(1000, 5000, 52),
    'digital': np.random.uniform(500, 2000, 52),
    'print': np.random.uniform(200, 800, 52),
    'sales': np.random.uniform(10000, 30000, 52)
})

# Initialize MMM
mmm = MarketingMixModel(data, target_col='sales')

# Configure channels
channel_configs = {
    'tv': {'decay': 0.7, 'saturation': 'log'},
    'digital': {'decay': 0.3, 'saturation': 'log'},
    'print': {'decay': 0.5, 'saturation': 'log'}
}

# Prepare and fit
mmm.prepare_features(channel_configs)
mmm.fit()

# Results
print("Model Coefficients:")
print(mmm.get_coefficients())

print("\nModel Performance:")
metrics = mmm.evaluate()
for metric, value in metrics.items():
    print(f"  {metric}: {value:.2f}")

print("\nChannel Contributions:")
contributions = mmm.channel_contribution()
print(contributions.round(2))
```

---

## 🎯 Model Diagnostics

### R² and Adjusted R²
```python
from sklearn.metrics import r2_score

def adjusted_r2(r2, n, k):
    """
    Calculate adjusted R².

    Parameters:
    - r2: R² value
    - n: number of observations
    - k: number of predictors (excluding intercept)
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Example
n_observations = 52  # weeks
n_predictors = 3     # channels
r2 = 0.75

adj_r2 = adjusted_r2(r2, n_observations, n_predictors)

print(f"R²: {r2:.3f}")
print(f"Adjusted R²: {adj_r2:.3f}")
print(f"\nAdjusted R² penalizes adding more predictors")
```

### RMSE and MAPE
```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # R²
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

    return metrics

# Example
y_true = np.array([1000, 1500, 2000, 2500, 3000])
y_pred = np.array([950, 1600, 1900, 2600, 2950])

metrics = calculate_metrics(y_true, y_pred)

print("Model Performance Metrics:")
for metric, value in metrics.items():
    if metric == 'MAPE':
        print(f"  {metric}: {value:.2f}%")
    elif metric == 'R²':
        print(f"  {metric}: {value:.3f}")
    else:
        print(f"  {metric}: ${value:.2f}")
```

### Residual Analysis
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def analyze_residuals(y_true, y_pred):
    """
    Analyze residuals for model diagnostics.
    """
    residuals = y_true - y_pred

    print("Residual Analysis")
    print("="*60)

    # Basic stats
    print(f"Mean of residuals: {np.mean(residuals):.2f} (should be ~0)")
    print(f"Std of residuals: {np.std(residuals):.2f}")

    # Normality test
    _, p_value = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk test p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("  ✓ Residuals appear normally distributed")
    else:
        print("  ✗ Residuals may not be normally distributed")

    # Autocorrelation (Durbin-Watson)
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(residuals)
    print(f"\nDurbin-Watson: {dw:.2f}")
    print("  (Values 1.5-2.5 indicate no autocorrelation)")

    # Heteroscedasticity check
    # Plot residuals vs predicted (visual check)
    # plt.scatter(y_pred, residuals)
    # plt.xlabel('Predicted')
    # plt.ylabel('Residuals')
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.title('Residual Plot')
    # plt.show()

    return residuals

# Example
y_true = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000])
y_pred = np.array([950, 1550, 1980, 2520, 2950, 3600, 3950])

residuals = analyze_residuals(y_true, y_pred)
```

### VIF (Multicollinearity)
```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor to detect multicollinearity.

    VIF > 10: High multicollinearity
    VIF > 5: Moderate multicollinearity
    VIF < 5: Low multicollinearity
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]

    return vif_data

# Example
data = pd.DataFrame({
    'tv': [100, 150, 200, 250, 300],
    'digital': [50, 75, 100, 125, 150],
    'print': [30, 45, 60, 75, 90]
})

vif = calculate_vif(data)
print("Multicollinearity Check (VIF):")
print(vif)
print("\nVIF < 5: Low multicollinearity ✓")
print("VIF 5-10: Moderate multicollinearity ⚠")
print("VIF > 10: High multicollinearity ✗")
```

---

## 🎯 Budget Optimization

### Marginal ROI
```python
import numpy as np

def calculate_marginal_roi(spend_levels, model, feature_col, current_spend):
    """
    Calculate marginal ROI at current spend level.
    """
    # Small increment
    delta = current_spend * 0.01  # 1% increase

    # Create feature vectors
    base_features = spend_levels.copy()
    incremented_features = spend_levels.copy()
    incremented_features[feature_col] += delta

    # Predict
    base_sales = model.predict(base_features.reshape(1, -1))[0]
    new_sales = model.predict(incremented_features.reshape(1, -1))[0]

    # Marginal return
    marginal_return = new_sales - base_sales
    marginal_roi = marginal_return / delta

    return marginal_roi

# Example: Simple optimization rule
def optimize_budget_simple(model, feature_cols, total_budget, current_allocation):
    """
    Shift budget toward channels with highest marginal ROI.
    """
    print("Budget Optimization (Simplified)")
    print("="*60)

    marginal_rois = {}

    for i, col in enumerate(feature_cols):
        spend_levels = np.array(current_allocation)
        mroi = calculate_marginal_roi(spend_levels, model, i, current_allocation[i])
        marginal_rois[col] = mroi

    print("Current Marginal ROI:")
    for channel, mroi in marginal_rois.items():
        print(f"  {channel}: ${mroi:.2f} per $1")

    # Recommendation
    best_channel = max(marginal_rois, key=marginal_rois.get)
    worst_channel = min(marginal_rois, key=marginal_rois.get)

    print(f"\nRecommendation:")
    print(f"  ↑ Increase: {best_channel} (highest marginal ROI)")
    print(f"  ↓ Decrease: {worst_channel} (lowest marginal ROI)")

# Would implement with actual model and data
```

### Response Curves
```python
import numpy as np
import matplotlib.pyplot as plt

def generate_response_curve(model, feature_idx, spend_range, baseline_spend):
    """
    Generate response curve for a channel.
    Shows sales at different spend levels (holding others constant).
    """
    sales_predictions = []

    for spend in spend_range:
        # Create feature vector with varying spend
        features = baseline_spend.copy()
        features[feature_idx] = spend

        # Predict
        sales = model.predict(features.reshape(1, -1))[0]
        sales_predictions.append(sales)

    return np.array(sales_predictions)

# Visualization
def plot_response_curves(model, feature_cols, spend_ranges, baseline_spends):
    """
    Plot response curves for all channels.
    """
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(15, 4))

    for i, (col, spend_range) in enumerate(zip(feature_cols, spend_ranges)):
        sales = generate_response_curve(model, i, spend_range, baseline_spends)

        axes[i].plot(spend_range, sales)
        axes[i].set_xlabel(f'{col} Spend ($)')
        axes[i].set_ylabel('Sales ($)')
        axes[i].set_title(f'{col} Response Curve')
        axes[i].grid(True, alpha=0.3)

        # Mark current spend
        current_spend = baseline_spends[i]
        current_idx = np.argmin(np.abs(spend_range - current_spend))
        axes[i].plot(current_spend, sales[current_idx], 'ro', markersize=10,
                     label='Current')
        axes[i].legend()

    plt.tight_layout()
    # plt.show()

# Would use with actual model
```

---

## 🚀 Quick Tips

### MMM Best Practices
1. **Data Requirements**: Minimum 2 years of weekly data (100+ observations)
2. **Granularity**: Weekly is typical, daily can be noisy
3. **Variables**: Include all major marketing channels + external factors
4. **Validation**: Hold out last 10-20% for testing
5. **Iteration**: Refine adstock/saturation parameters iteratively

### Common Mistakes
```python
# ❌ Wrong: Using raw spend without adstock
X = data[['tv_spend', 'digital_spend']]  # Ignores carryover!

# ✅ Correct: Apply adstock transformation
data['tv_adstocked'] = geometric_adstock(data['tv_spend'], decay=0.7)
data['digital_adstocked'] = geometric_adstock(data['digital_spend'], decay=0.3)
X = data[['tv_adstocked', 'digital_adstocked']]

# ❌ Wrong: Assuming linear returns
# $1M spend → 2x the return of $500K? Usually not!

# ✅ Correct: Apply saturation transformation
X['tv_saturated'] = np.log(X['tv_adstocked'] + 1)

# ❌ Wrong: Ignoring seasonality
# Sales vary by month/season, not just marketing

# ✅ Correct: Include seasonal controls
data['month'] = pd.to_datetime(data['date']).dt.month
month_dummies = pd.get_dummies(data['month'], prefix='month')

# ❌ Wrong: Not checking residuals
# Model might violate assumptions

# ✅ Correct: Diagnostic checks
analyze_residuals(y_true, y_pred)
```

### Interpretation Guidelines
- **Coefficient**: Marginal impact of 1 unit increase in transformed spend
- **R² > 0.7**: Good fit for MMM
- **MAPE < 10%**: Excellent accuracy
- **Check residuals**: Should be randomly distributed
- **VIF < 10**: Acceptable multicollinearity

---

## 📚 Practice Exercises Solutions

### Exercise 1: Build Simple MMM
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Generate sample data (52 weeks)
np.random.seed(42)
n_weeks = 52

data = pd.DataFrame({
    'week': range(1, n_weeks + 1),
    'tv_spend': np.random.uniform(1000, 5000, n_weeks),
    'digital_spend': np.random.uniform(500, 2500, n_weeks),
    'radio_spend': np.random.uniform(200, 1000, n_weeks),
})

# Simulate sales with known relationships
data['sales'] = (
    5000 +  # baseline
    2.5 * data['tv_spend'] +
    3.5 * data['digital_spend'] +
    1.8 * data['radio_spend'] +
    np.random.normal(0, 1000, n_weeks)  # noise
)

# Build MMM
X = data[['tv_spend', 'digital_spend', 'radio_spend']]
y = data['sales']

model = LinearRegression()
model.fit(X, y)

# Results
print("Marketing Mix Model Results")
print("="*60)
print("\nCoefficients:")
for channel, coef in zip(X.columns, model.coef_):
    print(f"  {channel}: ${coef:.2f} per $1 spent")
print(f"  Baseline: ${model.intercept_:.2f}")

# Performance
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\nModel Performance:")
print(f"  R²: {r2:.3f}")
print(f"  RMSE: ${rmse:.2f}")

# ROI Analysis
print(f"\nROI Analysis:")
for channel, coef in zip(X.columns, model.coef_):
    roi = ((coef - 1) / 1) * 100
    print(f"  {channel}: {roi:.1f}% ROI")
```

### Exercise 2: Apply Adstock
```python
# Using data from Exercise 1

# Apply adstock to each channel
decay_rates = {
    'tv_spend': 0.7,      # Long carryover (7 weeks)
    'digital_spend': 0.3,  # Short carryover (3 weeks)
    'radio_spend': 0.5     # Medium carryover (5 weeks)
}

for channel, decay in decay_rates.items():
    adstocked_col = f"{channel}_adstocked"
    data[adstocked_col] = geometric_adstock(data[channel].values, decay)

# Build MMM with adstocked variables
X_adstocked = data[['tv_spend_adstocked', 'digital_spend_adstocked', 'radio_spend_adstocked']]
model_adstocked = LinearRegression()
model_adstocked.fit(X_adstocked, y)

# Compare models
y_pred_adstocked = model_adstocked.predict(X_adstocked)
r2_adstocked = r2_score(y, y_pred_adstocked)

print("\nModel Comparison:")
print(f"  Without adstock R²: {r2:.3f}")
print(f"  With adstock R²: {r2_adstocked:.3f}")
print(f"  Improvement: {(r2_adstocked - r2):.3f}")
```

### Exercise 3: Optimize Budget
```python
# Current budget allocation
current_budget = {
    'tv_spend': 3000,
    'digital_spend': 1500,
    'radio_spend': 600
}

total_budget = sum(current_budget.values())

print(f"Current Budget: ${total_budget:,.0f}")
print("\nCurrent Allocation:")
for channel, spend in current_budget.items():
    pct = (spend / total_budget) * 100
    print(f"  {channel}: ${spend:,.0f} ({pct:.1f}%)")

# Calculate current sales
current_features = np.array([
    data[f'{ch}_adstocked'].iloc[-1]  # Use last week's adstocked values as baseline
    for ch in ['tv_spend', 'digital_spend', 'radio_spend']
]).reshape(1, -1)

current_sales = model_adstocked.predict(current_features)[0]
print(f"\nPredicted Sales: ${current_sales:,.0f}")

# Test alternative allocations
print("\n" + "="*60)
print("Testing Alternative Allocations:")
print("="*60)

alternatives = [
    {'tv_spend': 2500, 'digital_spend': 2000, 'radio_spend': 600},  # Shift TV → Digital
    {'tv_spend': 3500, 'digital_spend': 1000, 'radio_spend': 600},  # Shift Digital → TV
    {'tv_spend': 3000, 'digital_spend': 1800, 'radio_spend': 300},  # Shift Radio → Digital
]

for i, alt in enumerate(alternatives, 1):
    # Apply adstock to alternative spend
    alt_adstocked = []
    for channel in ['tv_spend', 'digital_spend', 'radio_spend']:
        decay = decay_rates[channel]
        # Simplified: just transform current value
        adstocked_val = alt[channel]  # In practice, would use full history
        alt_adstocked.append(adstocked_val)

    alt_features = np.array(alt_adstocked).reshape(1, -1)
    alt_sales = model_adstocked.predict(alt_features)[0]
    sales_change = alt_sales - current_sales

    print(f"\nAlternative {i}:")
    for channel, spend in alt.items():
        change = spend - current_budget[channel]
        print(f"  {channel}: ${spend:,.0f} ({change:+,.0f})")
    print(f"  → Predicted Sales: ${alt_sales:,.0f} ({sales_change:+,.0f})")
```

### Exercise 4: Diagnostic Analysis
```python
# Using model from previous exercises

# Predictions and residuals
predictions = model_adstocked.predict(X_adstocked)
residuals = y - predictions

print("Model Diagnostics")
print("="*60)

# 1. Residual statistics
print("\n1. RESIDUAL ANALYSIS")
print(f"  Mean: {np.mean(residuals):.2f} (should be ~0)")
print(f"  Std Dev: {np.std(residuals):.2f}")
print(f"  Min: {np.min(residuals):.2f}")
print(f"  Max: {np.max(residuals):.2f}")

# 2. Performance metrics
from sklearn.metrics import mean_absolute_percentage_error

rmse = np.sqrt(mean_squared_error(y, predictions))
mae = np.mean(np.abs(residuals))
mape = mean_absolute_percentage_error(y, predictions) * 100

print("\n2. PERFORMANCE METRICS")
print(f"  R²: {r2_adstocked:.3f}")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAE: ${mae:.2f}")
print(f"  MAPE: {mape:.2f}%")

# 3. Normality of residuals
from scipy import stats
_, p_value = stats.shapiro(residuals)

print("\n3. NORMALITY TEST")
print(f"  Shapiro-Wilk p-value: {p_value:.4f}")
if p_value > 0.05:
    print("  ✓ Residuals appear normally distributed")
else:
    print("  ⚠ Residuals may not be normal (p < 0.05)")

# 4. Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X_adstocked.columns
vif_data["VIF"] = [variance_inflation_factor(X_adstocked.values, i)
                   for i in range(len(X_adstocked.columns))]

print("\n4. MULTICOLLINEARITY (VIF)")
print(vif_data.to_string(index=False))
print("  Note: VIF < 5 is good, 5-10 is moderate, >10 is high")
```

---

## 🔍 MMM Validation Checklist

### Data Quality
- [ ] Sufficient history (2+ years)
- [ ] Consistent granularity (weekly/monthly)
- [ ] All major channels included
- [ ] External factors captured (seasonality, events)
- [ ] No missing data or properly imputed

### Model Fit
- [ ] R² > 0.7 (for MMM)
- [ ] MAPE < 10-15%
- [ ] Residuals normally distributed
- [ ] No autocorrelation (Durbin-Watson 1.5-2.5)
- [ ] VIF < 10 for all variables

### Business Logic
- [ ] Coefficients have correct signs (positive for sales drivers)
- [ ] Magnitude of coefficients makes business sense
- [ ] Adstock decay rates reasonable for each channel
- [ ] Saturation patterns aligned with expectations
- [ ] Holds on holdout set

---

**Quick Navigation:**
- [← Week 9 Cheatsheet](Week_09_Cheatsheet.md)
- [Week 11 Cheatsheet →](Week_11_Cheatsheet.md)
- [Back to Main README](../README.md)
