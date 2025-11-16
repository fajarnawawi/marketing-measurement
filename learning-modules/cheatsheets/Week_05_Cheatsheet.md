# Week 5: EDA Fundamentals - Quick Reference Cheatsheet

## 📋 Descriptive Statistics

### Central Tendency
```python
import pandas as pd
import numpy as np

# Mean (average)
df['cost'].mean()
np.mean(df['cost'])

# Median (middle value)
df['cost'].median()
np.median(df['cost'])

# Mode (most common value)
df['channel'].mode()
df['cost'].mode()[0]  # Get first mode if multiple

# Weighted average
weights = df['conversions']
np.average(df['cpa'], weights=weights)
```

### Dispersion & Spread
```python
# Variance
df['cost'].var()
np.var(df['cost'])

# Standard deviation
df['cost'].std()
np.std(df['cost'])

# Range
df['cost'].max() - df['cost'].min()
np.ptp(df['cost'])  # Peak to peak

# Interquartile range (IQR)
Q1 = df['cost'].quantile(0.25)
Q3 = df['cost'].quantile(0.75)
IQR = Q3 - Q1

# Mean Absolute Deviation
np.mean(np.abs(df['cost'] - df['cost'].mean()))

# Coefficient of Variation (relative variability)
cv = df['cost'].std() / df['cost'].mean()
```

### Quantiles & Percentiles
```python
# Quartiles
df['cost'].quantile(0.25)  # Q1 (25th percentile)
df['cost'].quantile(0.50)  # Q2 (median, 50th percentile)
df['cost'].quantile(0.75)  # Q3 (75th percentile)

# Multiple quantiles at once
df['cost'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# Deciles
df['cost'].quantile(np.arange(0, 1.1, 0.1))

# Specific percentile
df['cost'].quantile(0.95)  # 95th percentile
```

### Summary Statistics
```python
# Comprehensive summary
df.describe()

# Include all columns (including non-numeric)
df.describe(include='all')

# Only specific statistics
df.describe(percentiles=[.1, .25, .5, .75, .9])

# Custom aggregations
df.agg({
    'cost': ['count', 'mean', 'std', 'min', 'max'],
    'revenue': ['sum', 'mean', 'median'],
    'conversions': ['sum', 'count']
})

# Summary by group
df.groupby('channel').describe()
df.groupby('channel')['cost'].agg(['count', 'mean', 'std', 'min', 'max'])
```

### Shape Statistics
```python
# Skewness (asymmetry of distribution)
df['cost'].skew()
# > 0: Right-skewed (tail on right)
# < 0: Left-skewed (tail on left)
# ≈ 0: Symmetric

# Kurtosis (tailedness)
df['cost'].kurtosis()
# > 0: Heavy tails (leptokurtic)
# < 0: Light tails (platykurtic)
# ≈ 0: Normal tails (mesokurtic)

# Using scipy for more options
from scipy import stats
stats.skew(df['cost'])
stats.kurtosis(df['cost'])
```

---

## 📊 Distribution Analysis

### Frequency Distributions
```python
# Value counts
df['channel'].value_counts()
df['channel'].value_counts(normalize=True)  # Proportions
df['channel'].value_counts(sort=False)      # Unsorted

# Binning continuous data
df['cost_bin'] = pd.cut(df['cost'], bins=5)
df['cost_bin'].value_counts()

# Custom bins
bins = [0, 1000, 5000, 10000, float('inf')]
labels = ['Low', 'Medium', 'High', 'Very High']
df['cost_category'] = pd.cut(df['cost'], bins=bins, labels=labels)

# Equal-frequency bins (quantile-based)
df['cost_quartile'] = pd.qcut(df['cost'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### Histogram Data
```python
# Get histogram counts and bins
counts, bins = np.histogram(df['cost'], bins=10)

# Pandas histogram (returns counts)
df['cost'].hist(bins=20)

# Value counts for binned data
pd.cut(df['cost'], bins=10).value_counts().sort_index()
```

### Cumulative Distribution
```python
# Empirical CDF
sorted_data = np.sort(df['cost'])
cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

# Percentile rank
from scipy import stats
df['cost_percentile'] = df['cost'].rank(pct=True)

# Alternative using scipy
percentile_rank = stats.percentileofscore(df['cost'], value=5000)
```

### Normality Testing
```python
from scipy import stats

# Shapiro-Wilk test
statistic, p_value = stats.shapiro(df['cost'])
# p_value < 0.05: Reject normality

# Anderson-Darling test
result = stats.anderson(df['cost'])

# Q-Q plot data
stats.probplot(df['cost'], dist="norm")

# Kolmogorov-Smirnov test
statistic, p_value = stats.kstest(df['cost'], 'norm')
```

---

## 🔍 Outlier Detection

### Z-Score Method
```python
# Calculate z-scores
z_scores = np.abs(stats.zscore(df['cost']))

# Identify outliers (|z| > 3)
outliers = df[z_scores > 3]

# Add z-score column
df['cost_zscore'] = stats.zscore(df['cost'])
df['is_outlier_z'] = np.abs(df['cost_zscore']) > 3
```

### IQR Method
```python
# Calculate IQR
Q1 = df['cost'].quantile(0.25)
Q3 = df['cost'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['cost'] < lower_bound) | (df['cost'] > upper_bound)]

# Add outlier flag
df['is_outlier_iqr'] = (df['cost'] < lower_bound) | (df['cost'] > upper_bound)

# Remove outliers
df_clean = df[(df['cost'] >= lower_bound) & (df['cost'] <= upper_bound)]
```

### Modified Z-Score (Robust)
```python
# Using median absolute deviation (MAD)
median = df['cost'].median()
mad = np.median(np.abs(df['cost'] - median))
modified_z_scores = 0.6745 * (df['cost'] - median) / mad

# Outliers (modified z-score > 3.5)
df['is_outlier_mad'] = np.abs(modified_z_scores) > 3.5
```

### Isolation Forest
```python
from sklearn.ensemble import IsolationForest

# Fit isolation forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['outlier_if'] = iso_forest.fit_predict(df[['cost', 'revenue']])
# -1 = outlier, 1 = inlier

outliers = df[df['outlier_if'] == -1]
```

### Local Outlier Factor (LOF)
```python
from sklearn.neighbors import LocalOutlierFactor

# Fit LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
df['outlier_lof'] = lof.fit_predict(df[['cost', 'revenue', 'conversions']])
# -1 = outlier, 1 = inlier

# Get outlier scores
outlier_scores = lof.negative_outlier_factor_
```

### Handling Outliers
```python
# 1. Remove outliers
df_clean = df[~df['is_outlier_iqr']]

# 2. Cap outliers (winsorization)
df['cost_capped'] = df['cost'].clip(lower=lower_bound, upper=upper_bound)

# 3. Transform data
df['cost_log'] = np.log1p(df['cost'])  # Log transformation
df['cost_sqrt'] = np.sqrt(df['cost'])   # Square root

# 4. Replace with bounds
df['cost_cleaned'] = df['cost'].apply(
    lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x)
)
```

---

## 📈 Correlation Analysis

### Pearson Correlation
```python
# Correlation between two variables
df['cost'].corr(df['revenue'])

# Correlation matrix
correlation_matrix = df[['cost', 'revenue', 'conversions', 'clicks']].corr()

# All correlations with target variable
df.corr()['revenue'].sort_values(ascending=False)

# Correlation with p-values
from scipy.stats import pearsonr
corr, p_value = pearsonr(df['cost'], df['revenue'])
```

### Spearman Correlation (Rank-based)
```python
# Spearman correlation (for non-linear monotonic relationships)
df['cost'].corr(df['revenue'], method='spearman')

# Correlation matrix
df.corr(method='spearman')

# With p-values
from scipy.stats import spearmanr
corr, p_value = spearmanr(df['cost'], df['revenue'])
```

### Kendall's Tau
```python
# Kendall correlation (robust to outliers)
df['cost'].corr(df['revenue'], method='kendall')

# With p-values
from scipy.stats import kendalltau
corr, p_value = kendalltau(df['cost'], df['revenue'])
```

### Correlation Analysis
```python
# Find highly correlated pairs
def get_high_correlations(df, threshold=0.7):
    corr_matrix = df.corr()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    return pd.DataFrame(high_corr)

high_corr_pairs = get_high_correlations(df[numeric_cols], threshold=0.7)

# Correlation with target, sorted
target_corr = df.corr()['revenue'].abs().sort_values(ascending=False)
print(target_corr[target_corr > 0.3])  # Show correlations > 0.3
```

### Multicollinearity Detection
```python
# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
features = ['cost', 'clicks', 'impressions', 'conversions']
vif_data = pd.DataFrame()
vif_data['feature'] = features
vif_data['VIF'] = [variance_inflation_factor(df[features].values, i)
                    for i in range(len(features))]

# VIF > 10 indicates high multicollinearity
print(vif_data[vif_data['VIF'] > 10])
```

---

## 🔬 Data Profiling

### Missing Data Analysis
```python
# Count and percentage of missing values
missing_count = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    'count': missing_count,
    'percentage': missing_pct
}).sort_values('percentage', ascending=False)

# Visualize missing patterns
import missingno as msno
msno.matrix(df)
msno.heatmap(df)  # Correlation of missingness
```

### Data Types & Cardinality
```python
# Data type summary
df.dtypes

# Cardinality (unique values) per column
cardinality = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes,
    'unique_count': [df[col].nunique() for col in df.columns],
    'unique_pct': [df[col].nunique() / len(df) * 100 for col in df.columns]
})

# High cardinality columns (potential IDs)
high_cardinality = cardinality[cardinality['unique_pct'] > 90]

# Low cardinality columns (potential categories)
low_cardinality = cardinality[cardinality['unique_count'] < 10]
```

### Value Distribution
```python
# Check distribution of categorical variables
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}:")
    print(df[col].value_counts())
    print(f"Unique values: {df[col].nunique()}")

# Check for constant columns (no variation)
constant_cols = [col for col in df.columns if df[col].nunique() == 1]

# Check for duplicates
duplicate_count = df.duplicated().sum()
duplicate_rows = df[df.duplicated(keep=False)]
```

### Data Quality Checks
```python
# Negative values check (for cost, revenue, etc.)
negative_cost = df[df['cost'] < 0]
negative_revenue = df[df['revenue'] < 0]

# Impossible relationships
invalid_ctr = df[df['clicks'] > df['impressions']]
invalid_cvr = df[df['conversions'] > df['clicks']]

# Zero values
zero_cost = df[df['cost'] == 0]
zero_conversions = df[df['conversions'] == 0]

# Data range checks
cost_range_issues = df[(df['cost'] < 0) | (df['cost'] > 1000000)]
```

---

## 👥 Segmentation Analysis

### RFM Analysis (Recency, Frequency, Monetary)
```python
# Calculate RFM metrics
import datetime as dt

snapshot_date = df['order_date'].max() + dt.timedelta(days=1)

rfm = df.groupby('customer_id').agg({
    'order_date': lambda x: (snapshot_date - x.max()).days,  # Recency
    'order_id': 'count',                                      # Frequency
    'revenue': 'sum'                                          # Monetary
})

rfm.columns = ['recency', 'frequency', 'monetary']

# Create RFM scores (1-5)
rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1])
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1,2,3,4,5])

# Combined RFM score
rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

# Segment customers
def rfm_segment(row):
    if row['rfm_score'] in ['555', '554', '544', '545']:
        return 'Champions'
    elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345']:
        return 'Loyal'
    elif row['rfm_score'] in ['553', '551', '552', '541', '542']:
        return 'Potential Loyalist'
    elif row['rfm_score'] in ['525', '524', '523', '522', '521']:
        return 'New Customers'
    elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411']:
        return 'At Risk'
    else:
        return 'Other'

rfm['segment'] = rfm.apply(rfm_segment, axis=1)
```

### Customer Segmentation with K-Means
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare features
features = ['total_spend', 'order_count', 'avg_order_value', 'days_since_last_order']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters (Elbow method)
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Fit final model
kmeans = KMeans(n_clusters=4, random_state=42)
df['segment'] = kmeans.fit_predict(X_scaled)

# Analyze segments
segment_profile = df.groupby('segment')[features].mean()
```

### Behavioral Segmentation
```python
# Create behavioral features
df['is_high_spender'] = df['total_spend'] > df['total_spend'].quantile(0.75)
df['is_frequent_buyer'] = df['order_count'] > df['order_count'].median()
df['is_recent'] = df['days_since_last_order'] < 30

# Combine into segments
def behavioral_segment(row):
    if row['is_high_spender'] and row['is_frequent_buyer'] and row['is_recent']:
        return 'VIP'
    elif row['is_high_spender'] and row['is_frequent_buyer']:
        return 'Loyal High Value'
    elif row['is_frequent_buyer'] and row['is_recent']:
        return 'Frequent Buyers'
    elif row['is_high_spender']:
        return 'Big Spenders'
    elif row['is_recent']:
        return 'New/Recent'
    else:
        return 'Occasional'

df['segment'] = df.apply(behavioral_segment, axis=1)

# Segment summary
segment_summary = df.groupby('segment').agg({
    'customer_id': 'count',
    'total_spend': ['sum', 'mean'],
    'order_count': 'mean',
    'days_since_last_order': 'mean'
})
```

---

## 📊 Cohort Analysis

### Time-Based Cohorts
```python
# Create cohort based on first purchase month
df['cohort'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')

# Calculate cohort period (months since first purchase)
df['cohort_period'] = (df['order_date'].dt.to_period('M') - df['cohort']).apply(lambda x: x.n)

# Cohort analysis table
cohort_data = df.groupby(['cohort', 'cohort_period']).agg({
    'customer_id': 'nunique',
    'revenue': 'sum'
}).rename(columns={'customer_id': 'customers', 'revenue': 'revenue'})

# Pivot for easier viewing
cohort_size = cohort_data.groupby('cohort')['customers'].first()
retention = cohort_data['customers'].unstack(fill_value=0)

# Calculate retention rate
retention_rate = retention.divide(cohort_size, axis=0) * 100

# Revenue per cohort
cohort_revenue = cohort_data['revenue'].unstack(fill_value=0)
```

### Cohort LTV
```python
# Cumulative revenue per cohort
cumulative_revenue = cohort_revenue.cumsum(axis=1)

# Average LTV per customer
avg_ltv = cumulative_revenue.divide(cohort_size, axis=0)

# Cohort-specific metrics
cohort_summary = df.groupby('cohort').agg({
    'customer_id': 'nunique',
    'order_id': 'count',
    'revenue': 'sum'
})
cohort_summary['orders_per_customer'] = cohort_summary['order_id'] / cohort_summary['customer_id']
cohort_summary['revenue_per_customer'] = cohort_summary['revenue'] / cohort_summary['customer_id']
```

---

## 📉 Common Pandas/NumPy Functions

### Pandas Aggregations
```python
# Basic aggregations
df['cost'].sum()
df['cost'].mean()
df['cost'].median()
df['cost'].std()
df['cost'].var()
df['cost'].min()
df['cost'].max()
df['cost'].count()

# Multiple aggregations
df.agg({
    'cost': ['sum', 'mean', 'std'],
    'revenue': ['sum', 'mean']
})

# GroupBy aggregations
df.groupby('channel').agg({
    'cost': 'sum',
    'revenue': 'sum',
    'conversions': 'sum'
})
```

### NumPy Statistical Functions
```python
# Central tendency
np.mean(df['cost'])
np.median(df['cost'])
np.average(df['cost'], weights=df['conversions'])

# Dispersion
np.std(df['cost'])
np.var(df['cost'])
np.ptp(df['cost'])  # Range (peak to peak)

# Quantiles
np.percentile(df['cost'], 25)   # 25th percentile
np.percentile(df['cost'], [25, 50, 75])  # Multiple percentiles
np.quantile(df['cost'], 0.95)   # 95th percentile

# Correlation
np.corrcoef(df['cost'], df['revenue'])
```

### Window Functions
```python
# Rolling statistics
df['cost_ma7'] = df['cost'].rolling(window=7).mean()
df['cost_ma30'] = df['cost'].rolling(window=30).mean()
df['revenue_std7'] = df['revenue'].rolling(window=7).std()

# Expanding window (cumulative)
df['cost_cumsum'] = df['cost'].expanding().sum()
df['revenue_cummean'] = df['revenue'].expanding().mean()

# Exponential weighted moving average
df['cost_ewma'] = df['cost'].ewm(span=7).mean()
```

### Transform Functions
```python
# Apply function to groups
df['cost_pct_of_channel'] = df.groupby('channel')['cost'].transform(
    lambda x: x / x.sum() * 100
)

# Z-score normalization by group
df['cost_zscore_by_channel'] = df.groupby('channel')['cost'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Rank within groups
df['cost_rank_by_channel'] = df.groupby('channel')['cost'].transform('rank')
```

---

## 💡 Quick Tips & Best Practices

### EDA Workflow
```python
# 1. Initial inspection
print(df.shape)
print(df.info())
print(df.head())

# 2. Summary statistics
print(df.describe())

# 3. Missing values
print(df.isnull().sum())

# 4. Data types
print(df.dtypes)

# 5. Unique values
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# 6. Distributions
df.hist(figsize=(15, 10), bins=30)

# 7. Correlations
print(df.corr())

# 8. Outliers
for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers")
```

### Data Validation
```python
# Check for data quality issues
def validate_data(df):
    issues = []

    # Duplicates
    if df.duplicated().sum() > 0:
        issues.append(f"Found {df.duplicated().sum()} duplicate rows")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values:\n{missing[missing > 0]}")

    # Negative values in cost/revenue
    if (df['cost'] < 0).any():
        issues.append("Found negative cost values")

    # Invalid relationships
    if (df['clicks'] > df['impressions']).any():
        issues.append("Found clicks > impressions")

    return issues

# Run validation
validation_issues = validate_data(df)
for issue in validation_issues:
    print(issue)
```

### Performance Tips
```python
# ✅ Use vectorized operations
df['roas'] = df['revenue'] / df['cost']

# ❌ Avoid loops
for i in range(len(df)):
    df.loc[i, 'roas'] = df.loc[i, 'revenue'] / df.loc[i, 'cost']

# ✅ Use .loc for setting values
df.loc[df['cost'] > 1000, 'category'] = 'high'

# ❌ Avoid chained indexing
df[df['cost'] > 1000]['category'] = 'high'  # Warning!

# ✅ Use category dtype for repeated strings
df['channel'] = df['channel'].astype('category')
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Comprehensive EDA
```python
# Problem: Perform complete EDA on campaign data
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('campaigns.csv')

# 1. Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.info())

# 2. Summary statistics
print(df.describe())

# 3. Missing values
missing = df.isnull().sum()
print(f"\nMissing values:\n{missing[missing > 0]}")

# 4. Distribution of key metrics
metrics = ['cost', 'revenue', 'conversions', 'roas']
for metric in metrics:
    print(f"\n{metric.upper()} Distribution:")
    print(f"Mean: {df[metric].mean():.2f}")
    print(f"Median: {df[metric].median():.2f}")
    print(f"Std: {df[metric].std():.2f}")
    print(f"Skewness: {df[metric].skew():.2f}")
    print(f"Kurtosis: {df[metric].kurtosis():.2f}")

# 5. Outlier detection
for col in metrics:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"\n{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

# 6. Correlation analysis
corr_matrix = df[metrics].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)
```

### Exercise 2: Advanced Outlier Analysis
```python
# Problem: Compare different outlier detection methods
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

# Z-score method
z_scores = np.abs(stats.zscore(df['cost']))
outliers_z = df[z_scores > 3]

# IQR method
Q1 = df['cost'].quantile(0.25)
Q3 = df['cost'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['cost'] < Q1 - 1.5*IQR) | (df['cost'] > Q3 + 1.5*IQR)]

# Modified Z-score
median = df['cost'].median()
mad = np.median(np.abs(df['cost'] - median))
modified_z = 0.6745 * (df['cost'] - median) / mad
outliers_mad = df[np.abs(modified_z) > 3.5]

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(df[['cost', 'revenue']])
outliers_if = df[outlier_labels == -1]

# Compare results
print(f"Z-score outliers: {len(outliers_z)}")
print(f"IQR outliers: {len(outliers_iqr)}")
print(f"MAD outliers: {len(outliers_mad)}")
print(f"Isolation Forest outliers: {len(outliers_if)}")

# Consensus outliers (detected by multiple methods)
df['outlier_count'] = (
    (z_scores > 3).astype(int) +
    ((df['cost'] < Q1 - 1.5*IQR) | (df['cost'] > Q3 + 1.5*IQR)).astype(int) +
    (np.abs(modified_z) > 3.5).astype(int) +
    (outlier_labels == -1).astype(int)
)
consensus_outliers = df[df['outlier_count'] >= 2]
print(f"\nConsensus outliers (2+ methods): {len(consensus_outliers)}")
```

### Exercise 3: Correlation Deep Dive
```python
# Problem: Analyze correlations and identify multicollinearity
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate different correlation types
features = ['cost', 'clicks', 'impressions', 'conversions']

# Pearson correlation
pearson_corr = df[features].corr(method='pearson')

# Spearman correlation
spearman_corr = df[features].corr(method='spearman')

# Compare correlations
print("Pearson vs Spearman Correlation Differences:")
diff = pearson_corr - spearman_corr
print(diff)

# Statistical significance testing
print("\nCorrelations with p-values:")
for i, var1 in enumerate(features):
    for var2 in features[i+1:]:
        corr, p_value = pearsonr(df[var1], df[var2])
        print(f"{var1} vs {var2}: r={corr:.3f}, p={p_value:.4f}")

# VIF for multicollinearity
vif_data = pd.DataFrame()
vif_data['feature'] = features
vif_data['VIF'] = [variance_inflation_factor(df[features].values, i)
                    for i in range(len(features))]
print("\nVariance Inflation Factors:")
print(vif_data)
print("\nHigh multicollinearity (VIF > 10):")
print(vif_data[vif_data['VIF'] > 10])
```

### Exercise 4: Cohort Analysis
```python
# Problem: Perform cohort retention and revenue analysis
import pandas as pd
import numpy as np

# Prepare data
df['order_date'] = pd.to_datetime(df['order_date'])
df['cohort'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
df['cohort_period'] = (df['order_date'].dt.to_period('M') - df['cohort']).apply(lambda x: x.n)

# Cohort size
cohort_size = df.groupby('cohort')['customer_id'].nunique()

# Retention analysis
retention_data = df.groupby(['cohort', 'cohort_period'])['customer_id'].nunique()
retention_table = retention_data.unstack(fill_value=0)
retention_rate = retention_table.divide(cohort_size, axis=0) * 100

print("Cohort Retention Rates (%):")
print(retention_rate.round(1))

# Revenue analysis
revenue_data = df.groupby(['cohort', 'cohort_period'])['revenue'].sum()
revenue_table = revenue_data.unstack(fill_value=0)
cumulative_revenue = revenue_table.cumsum(axis=1)

# LTV by cohort
ltv = cumulative_revenue.divide(cohort_size, axis=0)

print("\nCumulative LTV by Cohort:")
print(ltv.round(2))

# Cohort performance summary
cohort_summary = pd.DataFrame({
    'cohort_size': cohort_size,
    'total_revenue': revenue_table.sum(axis=1),
    'avg_ltv': ltv.iloc[:, -1],  # Last period LTV
    'retention_3m': retention_rate[3] if 3 in retention_rate.columns else np.nan,
    'retention_6m': retention_rate[6] if 6 in retention_rate.columns else np.nan
})

print("\nCohort Summary:")
print(cohort_summary)
```

### Exercise 5: Customer Segmentation
```python
# Problem: Segment customers using multiple methods
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Calculate customer features
customer_features = df.groupby('customer_id').agg({
    'order_id': 'count',
    'revenue': 'sum',
    'order_date': lambda x: (pd.Timestamp.now() - x.max()).days
}).rename(columns={
    'order_id': 'order_count',
    'revenue': 'total_revenue',
    'order_date': 'days_since_last_order'
})

customer_features['avg_order_value'] = (
    customer_features['total_revenue'] / customer_features['order_count']
)

# Method 1: RFM Segmentation
customer_features['r_score'] = pd.qcut(
    customer_features['days_since_last_order'],
    q=5,
    labels=[5,4,3,2,1]
)
customer_features['f_score'] = pd.qcut(
    customer_features['order_count'].rank(method='first'),
    q=5,
    labels=[1,2,3,4,5]
)
customer_features['m_score'] = pd.qcut(
    customer_features['total_revenue'],
    q=5,
    labels=[1,2,3,4,5]
)

# Method 2: K-Means Clustering
features_for_clustering = ['order_count', 'total_revenue', 'avg_order_value', 'days_since_last_order']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(customer_features[features_for_clustering])

kmeans = KMeans(n_clusters=4, random_state=42)
customer_features['kmeans_segment'] = kmeans.fit_predict(features_scaled)

# Method 3: Rule-based segmentation
def rule_based_segment(row):
    if row['total_revenue'] > customer_features['total_revenue'].quantile(0.75) and \
       row['order_count'] > customer_features['order_count'].median():
        return 'VIP'
    elif row['total_revenue'] > customer_features['total_revenue'].median():
        return 'High Value'
    elif row['order_count'] > customer_features['order_count'].median():
        return 'Frequent'
    else:
        return 'Regular'

customer_features['rule_segment'] = customer_features.apply(rule_based_segment, axis=1)

# Compare segmentation methods
print("Segment Sizes - RFM:")
print(customer_features['r_score'].value_counts())

print("\nSegment Sizes - K-Means:")
print(customer_features['kmeans_segment'].value_counts())

print("\nSegment Sizes - Rule-Based:")
print(customer_features['rule_segment'].value_counts())

# Segment profiles
print("\nK-Means Segment Profiles:")
print(customer_features.groupby('kmeans_segment')[features_for_clustering].mean())
```

---

## 🔍 Quick Reference Table

| Metric | Function | Description |
|--------|----------|-------------|
| Mean | `df['col'].mean()` | Average value |
| Median | `df['col'].median()` | Middle value |
| Std Dev | `df['col'].std()` | Standard deviation |
| Variance | `df['col'].var()` | Variance |
| Quantile | `df['col'].quantile(0.75)` | 75th percentile |
| IQR | `Q3 - Q1` | Interquartile range |
| Skewness | `df['col'].skew()` | Distribution asymmetry |
| Kurtosis | `df['col'].kurtosis()` | Distribution tailedness |
| Correlation | `df['col1'].corr(df['col2'])` | Linear relationship |
| Z-score | `stats.zscore(df['col'])` | Standardized score |

---

**Quick Navigation:**
- [← Week 4 Cheatsheet](Week_04_Cheatsheet.md)
- [Week 6 Cheatsheet →](Week_06_Cheatsheet.md)
- [Back to Main README](../README.md)
