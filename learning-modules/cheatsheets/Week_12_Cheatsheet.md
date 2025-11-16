# Week 12: CLV & Comprehensive Measurement - Quick Reference Cheatsheet

## 📋 Core Concepts

### Customer Lifetime Value (CLV) Fundamentals

```python
"""
Customer Lifetime Value (CLV):
- Total profit expected from a customer over their lifetime
- Critical for understanding long-term marketing ROI
- Guides customer acquisition spending

Key Components:
1. Average Order Value (AOV)
2. Purchase Frequency
3. Customer Lifespan
4. Retention Rate
5. Profit Margin

Basic Formula:
CLV = (Average Order Value × Purchase Frequency × Customer Lifespan) × Profit Margin
"""

# Simple CLV calculation
def calculate_simple_clv(avg_order_value, purchases_per_year,
                          avg_customer_lifespan_years, profit_margin_pct):
    """
    Calculate basic CLV.

    profit_margin_pct: Percentage (e.g., 40 for 40%)
    """
    annual_revenue = avg_order_value * purchases_per_year
    total_lifetime_revenue = annual_revenue * avg_customer_lifespan_years
    clv = total_lifetime_revenue * (profit_margin_pct / 100)

    return clv

# Example
clv = calculate_simple_clv(
    avg_order_value=75,
    purchases_per_year=4,
    avg_customer_lifespan_years=3,
    profit_margin_pct=40
)

print(f"Customer Lifetime Value: ${clv:.2f}")
```

### Historical CLV
```python
import pandas as pd
import numpy as np

def calculate_historical_clv(customer_transactions):
    """
    Calculate historical CLV from actual transaction data.

    customer_transactions: DataFrame with columns ['customer_id', 'date', 'revenue']
    """
    # Total revenue per customer
    clv_data = customer_transactions.groupby('customer_id').agg({
        'revenue': 'sum',
        'date': ['min', 'max', 'count']
    }).reset_index()

    clv_data.columns = ['customer_id', 'total_revenue', 'first_purchase',
                        'last_purchase', 'num_purchases']

    # Calculate lifespan (days)
    clv_data['lifespan_days'] = (
        pd.to_datetime(clv_data['last_purchase']) -
        pd.to_datetime(clv_data['first_purchase'])
    ).dt.days

    # Historical CLV = total revenue (or could adjust for margin)
    clv_data['historical_clv'] = clv_data['total_revenue']

    return clv_data

# Example
transactions = pd.DataFrame({
    'customer_id': ['C001', 'C001', 'C001', 'C002', 'C002', 'C003'],
    'date': ['2024-01-01', '2024-03-15', '2024-06-20', '2024-02-10', '2024-05-15', '2024-03-01'],
    'revenue': [100, 120, 90, 150, 200, 80]
})

historical_clv = calculate_historical_clv(transactions)
print("\nHistorical CLV:")
print(historical_clv)
```

### Predictive CLV
```python
import numpy as np

def calculate_predictive_clv(avg_order_value, purchase_frequency,
                              retention_rate, discount_rate=0.10):
    """
    Calculate predictive CLV using retention rate.

    Formula: CLV = Margin × (Retention Rate / (1 + Discount Rate - Retention Rate))

    Simplified version:
    CLV = (AOV × Purchase Frequency × Margin) / (1 + Discount Rate - Retention Rate)

    retention_rate: Percentage as decimal (e.g., 0.70 for 70%)
    discount_rate: Annual discount rate (e.g., 0.10 for 10%)
    """
    # Annual revenue per customer
    annual_revenue = avg_order_value * purchase_frequency

    # CLV formula (simplified, assuming constant retention)
    clv = annual_revenue * (retention_rate / (1 + discount_rate - retention_rate))

    return clv

# Example
clv_predictive = calculate_predictive_clv(
    avg_order_value=75,
    purchase_frequency=4,     # 4 purchases per year
    retention_rate=0.70,      # 70% annual retention
    discount_rate=0.10        # 10% discount rate
)

print(f"Predictive CLV: ${clv_predictive:.2f}")

# With different retention rates
print("\nCLV by Retention Rate:")
for retention in [0.60, 0.70, 0.80, 0.90]:
    clv = calculate_predictive_clv(75, 4, retention, 0.10)
    print(f"  {retention*100:.0f}% retention → CLV = ${clv:.2f}")
```

---

## 🎯 Cohort-Based CLV Analysis

### Cohort CLV
```python
import pandas as pd
import numpy as np

def cohort_clv_analysis(transactions_df, cohort_col='cohort_month'):
    """
    Calculate CLV by customer cohort.

    cohort_col: Column defining cohort (e.g., signup month)
    """
    # Assume transactions_df has: customer_id, cohort_month, transaction_date, revenue

    # Total revenue per customer
    customer_totals = transactions_df.groupby(['customer_id', cohort_col])['revenue'].sum().reset_index()
    customer_totals.columns = ['customer_id', 'cohort', 'total_revenue']

    # CLV by cohort
    cohort_clv = customer_totals.groupby('cohort').agg({
        'total_revenue': ['mean', 'median', 'std', 'count']
    }).round(2)

    cohort_clv.columns = ['mean_clv', 'median_clv', 'std_clv', 'num_customers']

    return cohort_clv

# Example
transactions = pd.DataFrame({
    'customer_id': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4'],
    'cohort_month': ['2024-01', '2024-01', '2024-01', '2024-01',
                     '2024-02', '2024-02', '2024-02', '2024-02'],
    'transaction_date': ['2024-01-15', '2024-02-10', '2024-01-20', '2024-03-15',
                         '2024-02-05', '2024-03-20', '2024-02-25', '2024-04-10'],
    'revenue': [100, 120, 150, 180, 90, 110, 200, 220]
})

cohort_results = cohort_clv_analysis(transactions, 'cohort_month')
print("Cohort CLV Analysis:")
print(cohort_results)
```

### Cohort Retention Analysis
```python
import pandas as pd
import numpy as np

def cohort_retention(transactions_df, cohort_period='M'):
    """
    Calculate cohort retention rates over time.

    cohort_period: 'M' for monthly, 'Q' for quarterly
    """
    # Convert dates
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

    # Determine cohort (first purchase month)
    cohort_data = transactions_df.groupby('customer_id')['transaction_date'].min().reset_index()
    cohort_data.columns = ['customer_id', 'cohort_date']
    cohort_data['cohort'] = cohort_data['cohort_date'].dt.to_period(cohort_period)

    # Merge back
    transactions_df = transactions_df.merge(cohort_data[['customer_id', 'cohort']], on='customer_id')

    # Order period
    transactions_df['order_period'] = transactions_df['transaction_date'].dt.to_period(cohort_period)

    # Calculate period number (0, 1, 2, ...)
    transactions_df['period_number'] = (
        transactions_df['order_period'] - transactions_df['cohort']
    ).apply(lambda x: x.n)

    # Count unique customers per cohort per period
    cohort_pivot = transactions_df.groupby(['cohort', 'period_number'])['customer_id'].nunique().reset_index()
    cohort_pivot = cohort_pivot.pivot(index='cohort', columns='period_number', values='customer_id')

    # Calculate retention rates
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0) * 100

    return retention.round(1)

# Example usage would require proper transaction data
# retention_rates = cohort_retention(transactions_df)
```

---

## 🎯 LTV:CAC Ratio

### LTV:CAC Calculation
```python
def calculate_ltv_cac_ratio(avg_clv, avg_cac):
    """
    Calculate LTV:CAC ratio.

    Interpretation:
    - < 1: Losing money on customers
    - 1-3: Acceptable but could be better
    - 3-5: Good, healthy business
    - > 5: Excellent, room to invest more in acquisition
    """
    ratio = avg_clv / avg_cac if avg_cac > 0 else 0

    return ratio

# Example
avg_clv = 600
avg_cac = 150

ratio = calculate_ltv_cac_ratio(avg_clv, avg_cac)

print(f"Average CLV: ${avg_clv:.2f}")
print(f"Average CAC: ${avg_cac:.2f}")
print(f"LTV:CAC Ratio: {ratio:.2f}:1")

if ratio > 5:
    print("✓ Excellent - Can invest more in acquisition")
elif ratio >= 3:
    print("✓ Good - Healthy unit economics")
elif ratio >= 1:
    print("⚠ Acceptable - But needs improvement")
else:
    print("✗ Poor - Losing money on customers")
```

### Payback Period
```python
def calculate_payback_period(avg_cac, monthly_margin):
    """
    Calculate time to recover customer acquisition cost.

    monthly_margin: Average profit per customer per month
    """
    if monthly_margin <= 0:
        return float('inf')

    payback_months = avg_cac / monthly_margin

    return payback_months

# Example
cac = 150
monthly_margin = 25  # Customer generates $25 profit/month

payback = calculate_payback_period(cac, monthly_margin)

print(f"CAC: ${cac:.2f}")
print(f"Monthly Margin: ${monthly_margin:.2f}")
print(f"Payback Period: {payback:.1f} months")

if payback <= 12:
    print("✓ Good - Payback within 1 year")
elif payback <= 24:
    print("⚠ Acceptable - Payback within 2 years")
else:
    print("✗ Concerning - Long payback period")
```

### Unit Economics Dashboard
```python
import pandas as pd

def unit_economics_dashboard(avg_order_value, purchase_frequency,
                               retention_rate, profit_margin_pct, avg_cac):
    """
    Comprehensive unit economics analysis.
    """
    # Calculate CLV
    annual_revenue = avg_order_value * purchase_frequency
    clv = calculate_predictive_clv(avg_order_value, purchase_frequency,
                                    retention_rate, discount_rate=0.10)
    clv_with_margin = clv * (profit_margin_pct / 100)

    # Metrics
    ltv_cac = clv_with_margin / avg_cac if avg_cac > 0 else 0

    monthly_margin = (annual_revenue * (profit_margin_pct / 100)) / 12
    payback_months = avg_cac / monthly_margin if monthly_margin > 0 else float('inf')

    # Customer lifetime (from retention)
    avg_lifetime_months = 1 / (1 - retention_rate) if retention_rate < 1 else float('inf')

    print("="*60)
    print("UNIT ECONOMICS DASHBOARD")
    print("="*60)

    print("\nREVENUE METRICS:")
    print(f"  Average Order Value: ${avg_order_value:.2f}")
    print(f"  Purchase Frequency: {purchase_frequency:.1f} / year")
    print(f"  Annual Revenue per Customer: ${annual_revenue:.2f}")

    print("\nRETENTION:")
    print(f"  Retention Rate: {retention_rate*100:.1f}%")
    print(f"  Avg Customer Lifetime: {avg_lifetime_months:.1f} months")

    print("\nPROFITABILITY:")
    print(f"  Profit Margin: {profit_margin_pct:.1f}%")
    print(f"  Monthly Margin per Customer: ${monthly_margin:.2f}")
    print(f"  Customer Lifetime Value: ${clv_with_margin:.2f}")

    print("\nACQUISITION:")
    print(f"  Customer Acquisition Cost: ${avg_cac:.2f}")
    print(f"  LTV:CAC Ratio: {ltv_cac:.2f}:1")
    print(f"  Payback Period: {payback_months:.1f} months")

    print("\nHEALTH CHECK:")
    if ltv_cac >= 3 and payback_months <= 12:
        print("  ✓ Excellent unit economics")
    elif ltv_cac >= 1 and payback_months <= 24:
        print("  ✓ Acceptable unit economics")
    else:
        print("  ✗ Unit economics need improvement")

    return {
        'clv': clv_with_margin,
        'cac': avg_cac,
        'ltv_cac': ltv_cac,
        'payback_months': payback_months
    }

# Example
unit_economics_dashboard(
    avg_order_value=75,
    purchase_frequency=4,
    retention_rate=0.70,
    profit_margin_pct=40,
    avg_cac=150
)
```

---

## 🎯 Retention & Churn Analysis

### Retention Rate
```python
import numpy as np

def calculate_retention_rate(customers_start, customers_end, new_customers):
    """
    Calculate retention rate for a period.

    Formula: ((Customers_End - New_Customers) / Customers_Start) × 100
    """
    retained_customers = customers_end - new_customers
    retention_rate = (retained_customers / customers_start) * 100 if customers_start > 0 else 0

    return retention_rate

# Example
customers_start_month = 1000
customers_end_month = 950
new_customers_month = 100

retention = calculate_retention_rate(customers_start_month,
                                      customers_end_month,
                                      new_customers_month)

print(f"Customers at Start: {customers_start_month}")
print(f"New Customers: {new_customers_month}")
print(f"Customers at End: {customers_end_month}")
print(f"Retention Rate: {retention:.1f}%")

# Retained customers
retained = customers_end_month - new_customers_month
print(f"Retained Customers: {retained} out of {customers_start_month}")
```

### Churn Rate
```python
def calculate_churn_rate(customers_start, churned_customers):
    """
    Calculate churn rate.

    Churn Rate = (Churned Customers / Customers at Start) × 100
    """
    churn_rate = (churned_customers / customers_start) * 100 if customers_start > 0 else 0

    return churn_rate

# Relationship between retention and churn
# Retention Rate + Churn Rate = 100%

customers_start = 1000
churned = 150

churn = calculate_churn_rate(customers_start, churned)
retention = 100 - churn

print(f"Churn Rate: {churn:.1f}%")
print(f"Retention Rate: {retention:.1f}%")

# Impact of churn on CLV
print("\nImpact on Customer Lifetime:")
for churn_rate in [0.10, 0.20, 0.30, 0.40]:
    retention_rate = 1 - churn_rate
    avg_lifetime = 1 / churn_rate if churn_rate > 0 else float('inf')
    print(f"  {churn_rate*100:.0f}% monthly churn → {avg_lifetime:.1f} months lifetime")
```

### Retention Curve
```python
import numpy as np
import matplotlib.pyplot as plt

def generate_retention_curve(initial_customers, monthly_retention_rate, months=12):
    """
    Generate retention curve over time.

    monthly_retention_rate: Retention as decimal (e.g., 0.90 for 90%)
    """
    retained = [initial_customers]

    for month in range(1, months + 1):
        retained.append(retained[-1] * monthly_retention_rate)

    retention_pct = [(r / initial_customers) * 100 for r in retained]

    return retention_pct

# Example
initial = 1000
retention_rate = 0.90  # 90% monthly retention = 10% monthly churn

retention_curve = generate_retention_curve(initial, retention_rate, months=12)

print("Retention Curve (% of initial cohort):")
for month, pct in enumerate(retention_curve):
    print(f"  Month {month}: {pct:.1f}%")

# Visualize
# plt.figure(figsize=(10, 6))
# plt.plot(range(13), retention_curve, marker='o')
# plt.xlabel('Months')
# plt.ylabel('Retention %')
# plt.title('Customer Retention Curve')
# plt.grid(True, alpha=0.3)
# plt.show()
```

### Churn Prediction (Simple)
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def simple_churn_model(customer_data):
    """
    Simple churn prediction model.

    customer_data: DataFrame with features and 'churned' column (0/1)
    """
    # Features
    features = ['months_active', 'total_purchases', 'avg_order_value',
                'days_since_last_purchase']

    X = customer_data[features]
    y = customer_data['churned']

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Feature importance
    print("Churn Drivers (Feature Importance):")
    for feature, coef in zip(features, model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")

    # Predict churn probability
    customer_data['churn_probability'] = model.predict_proba(X)[:, 1]

    # Identify high-risk customers
    high_risk = customer_data[customer_data['churn_probability'] > 0.7]

    print(f"\nHigh-risk customers (>70% churn probability): {len(high_risk)}")

    return model, customer_data

# Example usage would require customer data
# model, predictions = simple_churn_model(customer_df)
```

---

## 🎯 Comprehensive Measurement Framework

### Marketing Measurement Hierarchy
```python
"""
Comprehensive Marketing Measurement Framework:

LEVEL 1: Basic Metrics (Tactical)
├─ Impressions, Clicks, CTR
├─ Cost per Click (CPC)
└─ Cost per Acquisition (CPA)

LEVEL 2: Attribution & Journey (Strategic)
├─ Multi-touch attribution
├─ Customer journey analysis
└─ Channel contribution

LEVEL 3: Incrementality (Causal)
├─ Geo-lift tests
├─ Difference-in-Differences
└─ True incremental impact

LEVEL 4: Long-term Value (Business Impact)
├─ Customer Lifetime Value (CLV)
├─ LTV:CAC ratio
└─ Retention & Churn

LEVEL 5: Mix Optimization (Holistic)
├─ Marketing Mix Modeling (MMM)
├─ Budget optimization
└─ Channel mix recommendations
"""
```

### Integrated Measurement Dashboard
```python
import pandas as pd
import numpy as np

class MarketingMeasurementDashboard:
    """
    Comprehensive marketing measurement dashboard.
    Integrates all measurement techniques learned.
    """

    def __init__(self, campaign_data):
        """
        campaign_data: Dict with all campaign metrics
        """
        self.data = campaign_data

    def basic_metrics(self):
        """Level 1: Basic tactical metrics."""
        print("="*60)
        print("LEVEL 1: BASIC METRICS")
        print("="*60)

        impressions = self.data.get('impressions', 0)
        clicks = self.data.get('clicks', 0)
        conversions = self.data.get('conversions', 0)
        cost = self.data.get('cost', 0)
        revenue = self.data.get('revenue', 0)

        # Calculate
        ctr = clicks / impressions if impressions > 0 else 0
        cvr = conversions / clicks if clicks > 0 else 0
        cpc = cost / clicks if clicks > 0 else 0
        cpa = cost / conversions if conversions > 0 else 0
        roas = revenue / cost if cost > 0 else 0

        print(f"Impressions: {impressions:,}")
        print(f"Clicks: {clicks:,} (CTR: {ctr:.2%})")
        print(f"Conversions: {conversions:,} (CVR: {cvr:.2%})")
        print(f"Cost: ${cost:,.2f}")
        print(f"Revenue: ${revenue:,.2f}")
        print(f"CPC: ${cpc:.2f}")
        print(f"CPA: ${cpa:.2f}")
        print(f"ROAS: {roas:.2f}x")

    def attribution_analysis(self):
        """Level 2: Attribution insights."""
        print("\n" + "="*60)
        print("LEVEL 2: ATTRIBUTION ANALYSIS")
        print("="*60)

        # Would use actual attribution model results
        attribution = self.data.get('attribution_results', {})

        print("\nChannel Attribution (Last-Touch):")
        for channel, value in attribution.items():
            print(f"  {channel}: ${value:,.2f}")

        print("\nNote: Compare with multi-touch attribution for full picture")

    def incrementality_metrics(self):
        """Level 3: Incrementality."""
        print("\n" + "="*60)
        print("LEVEL 3: INCREMENTALITY")
        print("="*60)

        incremental_revenue = self.data.get('incremental_revenue', 0)
        campaign_cost = self.data.get('cost', 0)

        iroas = incremental_revenue / campaign_cost if campaign_cost > 0 else 0

        print(f"Incremental Revenue: ${incremental_revenue:,.2f}")
        print(f"Campaign Cost: ${campaign_cost:,.2f}")
        print(f"iROAS: {iroas:.2f}x")

        if iroas > 1:
            print("✓ Positive incremental impact")
        else:
            print("✗ Negative or no incremental impact")

    def lifetime_value_metrics(self):
        """Level 4: CLV and retention."""
        print("\n" + "="*60)
        print("LEVEL 4: CUSTOMER LIFETIME VALUE")
        print("="*60)

        avg_clv = self.data.get('avg_clv', 0)
        avg_cac = self.data.get('avg_cac', 0)
        retention_rate = self.data.get('retention_rate', 0)

        ltv_cac = avg_clv / avg_cac if avg_cac > 0 else 0

        print(f"Average CLV: ${avg_clv:.2f}")
        print(f"Average CAC: ${avg_cac:.2f}")
        print(f"LTV:CAC Ratio: {ltv_cac:.2f}:1")
        print(f"Retention Rate: {retention_rate*100:.1f}%")

        if ltv_cac >= 3:
            print("✓ Healthy unit economics")
        else:
            print("⚠ Unit economics need improvement")

    def optimization_recommendations(self):
        """Level 5: Optimization insights."""
        print("\n" + "="*60)
        print("LEVEL 5: OPTIMIZATION RECOMMENDATIONS")
        print("="*60)

        # Would use MMM results, marginal ROI, etc.
        print("\nBudget Allocation Recommendations:")
        print("  • Increase: Channels with high iROAS (>3x)")
        print("  • Maintain: Channels with moderate iROAS (1-3x)")
        print("  • Decrease: Channels with low iROAS (<1x)")

        print("\nNext Steps:")
        print("  1. Run geo-lift test on underperforming channels")
        print("  2. Implement multi-touch attribution")
        print("  3. Build MMM for long-term optimization")
        print("  4. Improve retention to increase CLV")

    def generate_report(self):
        """Generate complete measurement report."""
        print("\n" + "="*70)
        print("COMPREHENSIVE MARKETING MEASUREMENT REPORT")
        print("="*70)

        self.basic_metrics()
        self.attribution_analysis()
        self.incrementality_metrics()
        self.lifetime_value_metrics()
        self.optimization_recommendations()

# Example usage
campaign_data = {
    'impressions': 1000000,
    'clicks': 50000,
    'conversions': 2500,
    'cost': 125000,
    'revenue': 375000,
    'attribution_results': {
        'Google Ads': 150000,
        'Facebook': 125000,
        'Email': 75000,
        'Organic': 25000
    },
    'incremental_revenue': 100000,
    'avg_clv': 600,
    'avg_cac': 150,
    'retention_rate': 0.70
}

dashboard = MarketingMeasurementDashboard(campaign_data)
dashboard.generate_report()
```

---

## 🎯 Predictive Modeling Basics

### Simple RFM Model
```python
import pandas as pd
import numpy as np

def rfm_analysis(transactions_df, analysis_date='2024-12-31'):
    """
    RFM (Recency, Frequency, Monetary) analysis.

    transactions_df: DataFrame with columns ['customer_id', 'transaction_date', 'revenue']
    """
    analysis_date = pd.to_datetime(analysis_date)

    # Calculate RFM metrics
    rfm = transactions_df.groupby('customer_id').agg({
        'transaction_date': lambda x: (analysis_date - pd.to_datetime(x.max())).days,
        'customer_id': 'count',
        'revenue': 'sum'
    }).reset_index()

    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    # Score each dimension (1-5)
    rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # Overall RFM score
    rfm['RFM_score'] = (
        rfm['R_score'].astype(str) +
        rfm['F_score'].astype(str) +
        rfm['M_score'].astype(str)
    )

    # Segment customers
    def segment_customer(row):
        if row['R_score'] >= 4 and row['F_score'] >= 4:
            return 'Champions'
        elif row['R_score'] >= 3 and row['F_score'] >= 3:
            return 'Loyal Customers'
        elif row['R_score'] >= 4:
            return 'Recent Customers'
        elif row['F_score'] >= 4:
            return 'Frequent Buyers'
        elif row['R_score'] <= 2 and row['F_score'] >= 3:
            return 'At Risk'
        elif row['R_score'] <= 2 and row['F_score'] <= 2:
            return 'Lost'
        else:
            return 'Others'

    rfm['segment'] = rfm.apply(segment_customer, axis=1)

    # Summary by segment
    segment_summary = rfm.groupby('segment').agg({
        'customer_id': 'count',
        'monetary': 'mean'
    }).round(2)
    segment_summary.columns = ['num_customers', 'avg_monetary_value']

    print("RFM Segmentation:")
    print(segment_summary)

    return rfm

# Example usage would require transaction data
# rfm_results = rfm_analysis(transactions_df)
```

### Next Purchase Prediction
```python
import numpy as np
from scipy import stats

def predict_next_purchase(days_since_last_purchase, avg_days_between_purchases,
                           std_days_between_purchases):
    """
    Predict probability of purchase in next N days.

    Simplified model assuming normal distribution of inter-purchase times.
    """
    # Probability customer hasn't churned yet
    z_score = (days_since_last_purchase - avg_days_between_purchases) / std_days_between_purchases
    prob_not_churned = 1 - stats.norm.cdf(z_score)

    # Expected days until next purchase
    expected_days = max(0, avg_days_between_purchases - days_since_last_purchase)

    print(f"Days since last purchase: {days_since_last_purchase}")
    print(f"Average inter-purchase time: {avg_days_between_purchases:.1f} days")
    print(f"Probability still active: {prob_not_churned:.2%}")
    print(f"Expected days until next purchase: {expected_days:.0f}")

    return prob_not_churned, expected_days

# Example
predict_next_purchase(
    days_since_last_purchase=45,
    avg_days_between_purchases=30,
    std_days_between_purchases=10
)
```

---

## 🚀 Quick Tips

### CLV Best Practices
1. **Segment CLV**: Different customer segments have different values
2. **Update regularly**: CLV changes as business evolves
3. **Include costs**: Factor in cost of goods, not just revenue
4. **Test assumptions**: Validate retention rates and margins
5. **Use for decisions**: Guide acquisition spending and retention investments

### Common Mistakes
```python
# ❌ Wrong: Ignoring time value of money
clv_simple = annual_revenue * 5  # Assumes $1 today = $1 in 5 years

# ✅ Correct: Apply discount rate
discount_rate = 0.10
clv_discounted = sum([
    annual_revenue / ((1 + discount_rate) ** year)
    for year in range(1, 6)
])

# ❌ Wrong: Using revenue instead of profit
clv = avg_order_value * purchases_per_year * years  # Revenue!

# ✅ Correct: Use profit margin
clv = (avg_order_value * purchases_per_year * years) * profit_margin

# ❌ Wrong: Treating all customers the same
overall_clv = 500  # One size fits all

# ✅ Correct: Segment-specific CLV
segment_clv = {
    'high_value': 1200,
    'medium_value': 500,
    'low_value': 200
}

# ❌ Wrong: Calculating LTV:CAC with inconsistent timeframes
clv_lifetime = 1000  # Total lifetime
cac_first_month = 50  # Just first month
ratio = clv_lifetime / cac_first_month  # Inconsistent!

# ✅ Correct: Align timeframes
clv_lifetime = 1000
cac_total = 100  # Total acquisition cost
ratio = clv_lifetime / cac_total
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: CLV Calculation
```python
# Calculate CLV for different customer segments

segments = {
    'Premium': {
        'avg_order_value': 150,
        'purchases_per_year': 8,
        'retention_rate': 0.80,
        'profit_margin': 0.45
    },
    'Standard': {
        'avg_order_value': 75,
        'purchases_per_year': 4,
        'retention_rate': 0.70,
        'profit_margin': 0.40
    },
    'Budget': {
        'avg_order_value': 30,
        'purchases_per_year': 6,
        'retention_rate': 0.60,
        'profit_margin': 0.35
    }
}

print("CLV by Customer Segment")
print("="*60)

for segment_name, params in segments.items():
    # Annual revenue
    annual_revenue = params['avg_order_value'] * params['purchases_per_year']

    # Annual profit
    annual_profit = annual_revenue * params['profit_margin']

    # CLV (predictive, with discount rate)
    discount_rate = 0.10
    retention = params['retention_rate']

    clv = annual_profit * (retention / (1 + discount_rate - retention))

    print(f"\n{segment_name}:")
    print(f"  AOV: ${params['avg_order_value']:.2f}")
    print(f"  Purchase Frequency: {params['purchases_per_year']}/year")
    print(f"  Annual Revenue: ${annual_revenue:.2f}")
    print(f"  Profit Margin: {params['profit_margin']*100:.0f}%")
    print(f"  Retention Rate: {retention*100:.0f}%")
    print(f"  → CLV: ${clv:.2f}")

    # How much can we spend to acquire?
    max_cac_3x = clv / 3  # Target 3:1 LTV:CAC
    print(f"  → Max CAC (for 3:1 ratio): ${max_cac_3x:.2f}")
```

### Exercise 2: Cohort Analysis
```python
import pandas as pd
import numpy as np

# Monthly cohorts over 6 months
cohort_data = pd.DataFrame({
    'cohort_month': ['Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Feb', 'Mar', 'Mar', 'Mar'],
    'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
    'total_revenue': [450, 320, 550, 380, 420, 490, 350, 280, 410]
})

# CLV by cohort
cohort_summary = cohort_data.groupby('cohort_month').agg({
    'total_revenue': ['mean', 'sum', 'count']
}).round(2)

cohort_summary.columns = ['avg_clv', 'total_revenue', 'num_customers']

print("Cohort Analysis:")
print(cohort_summary)

# CAC by cohort (assume we know CAC)
cac_by_cohort = {'Jan': 120, 'Feb': 135, 'Mar': 110}

cohort_summary['avg_cac'] = cohort_summary.index.map(cac_by_cohort)
cohort_summary['ltv_cac_ratio'] = cohort_summary['avg_clv'] / cohort_summary['avg_cac']

print("\nLTV:CAC by Cohort:")
print(cohort_summary[['avg_clv', 'avg_cac', 'ltv_cac_ratio']])

# Which cohort is healthiest?
best_cohort = cohort_summary['ltv_cac_ratio'].idxmax()
print(f"\nBest performing cohort: {best_cohort}")
print(f"LTV:CAC ratio: {cohort_summary.loc[best_cohort, 'ltv_cac_ratio']:.2f}:1")
```

### Exercise 3: Retention Impact on CLV
```python
import numpy as np

# Show impact of retention improvements on CLV

base_params = {
    'avg_order_value': 75,
    'purchases_per_year': 4,
    'profit_margin': 0.40,
    'discount_rate': 0.10
}

print("Impact of Retention Rate on CLV")
print("="*60)

retention_rates = [0.50, 0.60, 0.70, 0.80, 0.90]

results = []
for retention in retention_rates:
    annual_revenue = base_params['avg_order_value'] * base_params['purchases_per_year']
    annual_profit = annual_revenue * base_params['profit_margin']

    clv = annual_profit * (retention / (1 + base_params['discount_rate'] - retention))

    results.append({
        'retention_rate': retention,
        'clv': clv
    })

    print(f"Retention: {retention*100:>3.0f}% → CLV: ${clv:>7.2f}")

# Calculate incremental value of retention improvements
print("\nValue of Retention Improvements:")
for i in range(1, len(results)):
    prev_retention = results[i-1]['retention_rate']
    curr_retention = results[i]['retention_rate']
    retention_improvement = (curr_retention - prev_retention) * 100

    prev_clv = results[i-1]['clv']
    curr_clv = results[i]['clv']
    clv_increase = curr_clv - prev_clv
    clv_increase_pct = (clv_increase / prev_clv) * 100

    print(f"{prev_retention*100:.0f}% → {curr_retention*100:.0f}% (+{retention_improvement:.0f}pp): "
          f"CLV increases ${clv_increase:.2f} (+{clv_increase_pct:.1f}%)")

print("\n💡 Key Insight: Small improvements in retention have large CLV impact!")
```

### Exercise 4: Unit Economics Optimization
```python
# Current state
current_metrics = {
    'avg_order_value': 60,
    'purchases_per_year': 3,
    'retention_rate': 0.65,
    'profit_margin': 0.35,
    'avg_cac': 150
}

# Calculate current CLV and LTV:CAC
def calc_clv(aov, freq, retention, margin, discount=0.10):
    annual_revenue = aov * freq
    annual_profit = annual_revenue * margin
    return annual_profit * (retention / (1 + discount - retention))

current_clv = calc_clv(
    current_metrics['avg_order_value'],
    current_metrics['purchases_per_year'],
    current_metrics['retention_rate'],
    current_metrics['profit_margin']
)

current_ltv_cac = current_clv / current_metrics['avg_cac']

print("Current Unit Economics:")
print("="*60)
print(f"CLV: ${current_clv:.2f}")
print(f"CAC: ${current_metrics['avg_cac']:.2f}")
print(f"LTV:CAC: {current_ltv_cac:.2f}:1")

if current_ltv_cac < 3:
    print("⚠ Below target of 3:1")

# Test improvement scenarios
print("\n\nImprovement Scenarios:")
print("="*60)

scenarios = [
    {'name': 'Increase AOV by 20%', 'avg_order_value': 72, 'purchases_per_year': 3,
     'retention_rate': 0.65, 'profit_margin': 0.35, 'avg_cac': 150},

    {'name': 'Increase frequency by 1 purchase/year', 'avg_order_value': 60,
     'purchases_per_year': 4, 'retention_rate': 0.65, 'profit_margin': 0.35, 'avg_cac': 150},

    {'name': 'Improve retention to 75%', 'avg_order_value': 60, 'purchases_per_year': 3,
     'retention_rate': 0.75, 'profit_margin': 0.35, 'avg_cac': 150},

    {'name': 'Reduce CAC by 20%', 'avg_order_value': 60, 'purchases_per_year': 3,
     'retention_rate': 0.65, 'profit_margin': 0.35, 'avg_cac': 120},
]

for scenario in scenarios:
    new_clv = calc_clv(
        scenario['avg_order_value'],
        scenario['purchases_per_year'],
        scenario['retention_rate'],
        scenario['profit_margin']
    )

    new_ltv_cac = new_clv / scenario['avg_cac']
    clv_change = new_clv - current_clv
    ratio_change = new_ltv_cac - current_ltv_cac

    print(f"\n{scenario['name']}:")
    print(f"  New CLV: ${new_clv:.2f} ({clv_change:+.2f})")
    print(f"  New LTV:CAC: {new_ltv_cac:.2f}:1 ({ratio_change:+.2f})")

    if new_ltv_cac >= 3:
        print(f"  ✓ Reaches 3:1 target!")

print("\n\n💡 Recommendation: Focus on retention improvement for biggest CLV impact")
```

---

## 🔍 Final Project Integration

### Putting It All Together
```python
"""
Comprehensive Marketing Measurement Project:

1. DATA COLLECTION
   - Campaign metrics (impressions, clicks, conversions, cost)
   - Customer journey data
   - Transaction history
   - Retention/churn data

2. BASIC ANALYSIS (Week 1-6)
   - Calculate funnel metrics (CTR, CVR, CPA, ROAS)
   - Data visualization and exploration
   - Statistical analysis

3. ADVANCED STATISTICS (Week 7-8)
   - Hypothesis testing for campaign performance
   - A/B test analysis with proper sample sizing

4. ATTRIBUTION (Week 9)
   - Multi-touch attribution model
   - Compare last-touch, first-touch, linear, position-based
   - Calculate channel contribution

5. MARKETING MIX MODELING (Week 10)
   - Build MMM with adstock and saturation
   - Estimate channel coefficients
   - Generate response curves

6. INCREMENTALITY (Week 11)
   - Design and analyze geo-lift test
   - Calculate iROAS
   - Measure true incremental impact

7. CLV & OPTIMIZATION (Week 12)
   - Calculate CLV by segment
   - Analyze LTV:CAC ratios
   - Build retention/churn model
   - Optimize budget allocation

8. COMPREHENSIVE DASHBOARD
   - Integrate all measurements
   - Generate recommendations
   - Present to stakeholders
"""
```

---

**Quick Navigation:**
- [← Week 11 Cheatsheet](Week_11_Cheatsheet.md)
- [Back to Main README](../README.md)
- [Final Project →](../projects/final-project.md)
