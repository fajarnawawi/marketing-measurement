# Week 2: Pandas & Data Manipulation - Quick Reference Cheatsheet

## 📋 DataFrame Basics

### Creating DataFrames
```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'campaign': ['Search', 'Social', 'Display'],
    'cost': [5000, 3500, 2000],
    'revenue': [20000, 14000, 10000],
    'conversions': [250, 175, 125]
}
df = pd.DataFrame(data)

# From lists
campaigns = ['Search', 'Social', 'Display']
costs = [5000, 3500, 2000]
df = pd.DataFrame({'campaign': campaigns, 'cost': costs})

# From CSV file
df = pd.read_csv('campaigns.csv')
df = pd.read_csv('campaigns.csv', parse_dates=['date'])

# Creating empty DataFrame
df = pd.DataFrame(columns=['campaign', 'cost', 'revenue'])

# From list of dictionaries
data = [
    {'campaign': 'Search', 'cost': 5000, 'revenue': 20000},
    {'campaign': 'Social', 'cost': 3500, 'revenue': 14000}
]
df = pd.DataFrame(data)
```

### DataFrame Info & Inspection
```python
# Basic information
df.head()                    # First 5 rows
df.head(10)                  # First 10 rows
df.tail()                    # Last 5 rows
df.info()                    # Column types and non-null counts
df.describe()                # Statistical summary
df.shape                     # (rows, columns)
df.columns                   # Column names
df.dtypes                    # Data types of each column
len(df)                      # Number of rows

# Quick statistics
df['cost'].sum()             # Sum of cost column
df['cost'].mean()            # Mean
df['cost'].median()          # Median
df['cost'].std()             # Standard deviation
df['cost'].min()             # Minimum
df['cost'].max()             # Maximum
df['cost'].quantile(0.75)    # 75th percentile

# Value counts
df['campaign'].value_counts()              # Count of each unique value
df['campaign'].nunique()                   # Number of unique values
df['campaign'].unique()                    # Array of unique values
```

---

## 🎯 Selecting Columns & Rows

### Selecting Columns
```python
# Single column (returns Series)
df['campaign']
df.campaign                  # Dot notation (only for valid Python names)

# Multiple columns (returns DataFrame)
df[['campaign', 'cost', 'revenue']]

# Select columns by position
df.iloc[:, 0]                # First column
df.iloc[:, [0, 2, 3]]        # Columns 0, 2, 3

# Select columns by condition
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols]
```

### Selecting Rows with loc (label-based)
```python
# Single row by index label
df.loc[0]                    # Row with index 0

# Multiple rows
df.loc[0:4]                  # Rows 0 through 4 (inclusive!)
df.loc[[0, 2, 5]]            # Rows 0, 2, and 5

# Rows and columns
df.loc[0:4, 'campaign']                           # Rows 0-4, campaign column
df.loc[0:4, ['campaign', 'cost']]                 # Rows 0-4, specific columns
df.loc[:, 'campaign':'revenue']                    # All rows, column range
```

### Selecting Rows with iloc (position-based)
```python
# Single row by position
df.iloc[0]                   # First row
df.iloc[-1]                  # Last row

# Multiple rows
df.iloc[0:5]                 # First 5 rows (0-4, exclusive end!)
df.iloc[[0, 2, 5]]           # Rows at positions 0, 2, 5

# Rows and columns by position
df.iloc[0:5, 0]              # First 5 rows, first column
df.iloc[0:5, [0, 2]]         # First 5 rows, columns 0 and 2
df.iloc[:, 0:3]              # All rows, first 3 columns

# Slicing
df.iloc[::2]                 # Every other row
df.iloc[::-1]                # Reverse row order
```

---

## 🔍 Filtering with Boolean Indexing

### Basic Filtering
```python
# Single condition
high_cost = df[df['cost'] > 3000]
search_campaigns = df[df['campaign'] == 'Search']
active = df[df['is_active'] == True]

# Multiple conditions with & (and) or | (or)
good_performers = df[(df['roas'] > 3.0) & (df['conversions'] > 100)]
needs_attention = df[(df['roas'] < 2.0) | (df['cpa'] > 50)]

# NOT condition with ~
not_search = df[~(df['campaign'] == 'Search')]
low_cost = df[~(df['cost'] > 3000)]

# Complex conditions
df[(df['cost'] > 1000) & (df['cost'] < 5000) & (df['channel'] == 'Google')]
```

### Advanced Filtering
```python
# Using isin() for multiple values
selected_campaigns = df[df['campaign'].isin(['Search', 'Social'])]
excluded = df[~df['campaign'].isin(['Display', 'Email'])]

# String methods
google_campaigns = df[df['campaign'].str.contains('Google')]
starts_with_s = df[df['campaign'].str.startswith('S')]
case_insensitive = df[df['campaign'].str.contains('SEARCH', case=False)]

# Between values
mid_range = df[df['cost'].between(1000, 5000)]

# Null/Not null filtering
has_revenue = df[df['revenue'].notna()]
missing_revenue = df[df['revenue'].isna()]

# Query method (alternative syntax)
df.query('cost > 3000 and conversions > 100')
df.query('roas > @target_roas')  # Using variable with @
```

---

## 📊 Sorting & Ranking

### Sorting
```python
# Sort by single column
df.sort_values('cost')                              # Ascending
df.sort_values('cost', ascending=False)             # Descending

# Sort by multiple columns
df.sort_values(['channel', 'cost'], ascending=[True, False])

# Sort by index
df.sort_index()                                     # Sort by row index
df.sort_index(axis=1)                              # Sort by column names

# Reset index after sorting
df.sort_values('roas', ascending=False).reset_index(drop=True)
```

### Ranking
```python
# Add rank column
df['cost_rank'] = df['cost'].rank()                # Default: average method
df['cost_rank'] = df['cost'].rank(ascending=False) # Highest cost = rank 1
df['cost_rank'] = df['cost'].rank(method='first')  # No ties

# Rank methods
df['rank'] = df['roas'].rank(method='average')     # Ties get average rank
df['rank'] = df['roas'].rank(method='min')         # Ties get minimum rank
df['rank'] = df['roas'].rank(method='max')         # Ties get maximum rank
df['rank'] = df['roas'].rank(method='first')       # Ties broken by order
df['rank'] = df['roas'].rank(method='dense')       # Consecutive ranks

# Get top N
top_5 = df.nlargest(5, 'revenue')
bottom_5 = df.nsmallest(5, 'cpa')
```

---

## 🔄 GroupBy & Aggregations

### Basic GroupBy
```python
# Group by single column
df.groupby('channel')['cost'].sum()
df.groupby('channel')['conversions'].mean()

# Group by multiple columns
df.groupby(['channel', 'campaign_type'])['cost'].sum()

# Multiple aggregations
df.groupby('channel').agg({
    'cost': 'sum',
    'revenue': 'sum',
    'conversions': 'sum'
})

# Different aggregations per column
df.groupby('channel').agg({
    'cost': ['sum', 'mean', 'count'],
    'revenue': ['sum', 'mean'],
    'conversions': 'sum'
})

# Named aggregations (pandas >= 0.25)
df.groupby('channel').agg(
    total_cost=('cost', 'sum'),
    avg_cost=('cost', 'mean'),
    total_conversions=('conversions', 'sum'),
    campaign_count=('campaign', 'count')
)
```

### Advanced Aggregations
```python
# Custom aggregation functions
df.groupby('channel').agg({
    'cost': lambda x: x.max() - x.min(),  # Range
    'revenue': ['sum', 'mean', np.std]
})

# Apply custom function
def calculate_roas(group):
    return group['revenue'].sum() / group['cost'].sum()

df.groupby('channel').apply(calculate_roas)

# Multiple group operations
channel_stats = df.groupby('channel').agg({
    'cost': ['sum', 'mean'],
    'revenue': 'sum',
    'conversions': 'sum'
})

# Flatten multi-level columns
channel_stats.columns = ['_'.join(col).strip() for col in channel_stats.columns]

# Reset index to make groupby columns regular columns
channel_summary = df.groupby('channel')['cost'].sum().reset_index()
```

### Common Aggregation Functions
```python
# Built-in aggregations
df.groupby('channel').agg({
    'cost': 'sum',          # Total
    'revenue': 'mean',      # Average
    'conversions': 'count', # Count
    'cpa': 'min',          # Minimum
    'roas': 'max',         # Maximum
    'clicks': 'std',       # Standard deviation
    'impressions': 'var',  # Variance
    'ctr': 'median'        # Median
})

# First and last
df.groupby('channel').first()
df.groupby('channel').last()

# Size (count of rows in each group)
df.groupby('channel').size()

# Nunique (count unique values)
df.groupby('channel')['campaign'].nunique()
```

---

## 🔗 Merging & Joining

### Merge (SQL-style joins)
```python
# Inner join (default)
merged = pd.merge(campaigns_df, conversions_df, on='campaign_id')
merged = pd.merge(campaigns_df, conversions_df,
                  left_on='campaign_id', right_on='id')

# Left join (keep all from left)
merged = pd.merge(campaigns_df, conversions_df,
                  on='campaign_id', how='left')

# Right join (keep all from right)
merged = pd.merge(campaigns_df, conversions_df,
                  on='campaign_id', how='right')

# Outer join (keep all from both)
merged = pd.merge(campaigns_df, conversions_df,
                  on='campaign_id', how='outer')

# Join on multiple columns
merged = pd.merge(df1, df2, on=['campaign_id', 'date'])

# Handle duplicate column names
merged = pd.merge(df1, df2, on='id', suffixes=('_campaign', '_conversion'))
```

### Concat (Stack DataFrames)
```python
# Vertical concatenation (stack rows)
combined = pd.concat([df1, df2, df3])
combined = pd.concat([df1, df2], ignore_index=True)  # Reset index

# Horizontal concatenation (add columns)
combined = pd.concat([df1, df2], axis=1)

# Add identifier for source
combined = pd.concat([df1, df2], keys=['Q1', 'Q2'])
```

### Join (Index-based)
```python
# Join on index
df1.join(df2)                    # Default: left join
df1.join(df2, how='inner')       # Inner join
df1.join(df2, how='outer')       # Outer join

# Join with column name
df1.set_index('campaign_id').join(df2.set_index('campaign_id'))
```

---

## 🧹 Handling Missing Data

### Detecting Missing Data
```python
# Check for nulls
df.isnull()                      # Boolean DataFrame
df.isnull().sum()                # Count nulls per column
df.isnull().sum().sum()          # Total null count

# Check for non-nulls
df.notnull()
df.notnull().sum()

# Any/all nulls
df.isnull().any()                # True if any null in column
df.isnull().all()                # True if all null in column

# Rows with any null
df[df.isnull().any(axis=1)]

# Percentage of nulls
(df.isnull().sum() / len(df)) * 100
```

### Handling Missing Data
```python
# Drop nulls
df.dropna()                      # Drop rows with any null
df.dropna(how='all')            # Drop only if all values are null
df.dropna(subset=['cost'])      # Drop if null in specific column
df.dropna(axis=1)               # Drop columns with nulls

# Fill nulls
df.fillna(0)                     # Fill with constant
df['cost'].fillna(df['cost'].mean())           # Fill with mean
df['cost'].fillna(df['cost'].median())         # Fill with median
df.fillna(method='ffill')        # Forward fill
df.fillna(method='bfill')        # Backward fill

# Fill with different values per column
df.fillna({
    'cost': 0,
    'revenue': 0,
    'campaign': 'Unknown'
})

# Replace specific values
df.replace(0, np.nan)            # Replace 0 with NaN
df.replace([0, -1], np.nan)      # Replace multiple values
df['campaign'].replace('', 'Unknown')
```

---

## 📅 Time Series Basics

### Working with Dates
```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0
df['day_name'] = df['date'].dt.day_name()     # 'Monday', 'Tuesday'
df['week'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Date arithmetic
df['date'] + pd.Timedelta(days=7)             # Add 7 days
df['date'] - pd.Timedelta(weeks=1)            # Subtract 1 week

# Create date range
date_range = pd.date_range('2024-01-01', '2024-12-31', freq='D')
date_range = pd.date_range('2024-01-01', periods=365, freq='D')
```

### Time-based Operations
```python
# Set date as index
df = df.set_index('date')

# Resample (aggregate by time period)
df.resample('D').sum()           # Daily
df.resample('W').sum()           # Weekly
df.resample('M').sum()           # Monthly
df.resample('Q').sum()           # Quarterly

# Rolling windows
df['cost'].rolling(window=7).mean()           # 7-day moving average
df['cost'].rolling(window=7).sum()            # 7-day rolling sum
df['revenue'].rolling(window=30).std()        # 30-day rolling std

# Filter by date
df[df['date'] >= '2024-01-01']
df[df['date'].between('2024-01-01', '2024-12-31')]

# Sort by date
df.sort_values('date')
```

---

## 🎯 Common Marketing Analysis Patterns

### Calculate Metrics
```python
# Add calculated columns
df['roas'] = df['revenue'] / df['cost']
df['cpa'] = df['cost'] / df['conversions']
df['ctr'] = df['clicks'] / df['impressions']
df['cvr'] = df['conversions'] / df['clicks']
df['profit'] = df['revenue'] - df['cost']
df['roi'] = (df['revenue'] - df['cost']) / df['cost']

# Conditional calculations (avoid division by zero)
df['cpa'] = df.apply(
    lambda row: row['cost'] / row['conversions'] if row['conversions'] > 0 else 0,
    axis=1
)

# Or using np.where
df['cpa'] = np.where(df['conversions'] > 0, df['cost'] / df['conversions'], 0)
```

### Channel Performance Analysis
```python
# Group by channel
channel_performance = df.groupby('channel').agg({
    'cost': 'sum',
    'revenue': 'sum',
    'conversions': 'sum',
    'impressions': 'sum',
    'clicks': 'sum'
})

# Calculate channel-level metrics
channel_performance['roas'] = (
    channel_performance['revenue'] / channel_performance['cost']
)
channel_performance['cpa'] = (
    channel_performance['cost'] / channel_performance['conversions']
)

# Sort by performance
channel_performance.sort_values('roas', ascending=False)

# Get top performers
top_channels = channel_performance.nlargest(5, 'roas')
```

### Campaign Comparison
```python
# Pivot table for comparison
pivot = df.pivot_table(
    values=['cost', 'revenue', 'conversions'],
    index='campaign',
    columns='channel',
    aggfunc='sum',
    fill_value=0
)

# Cross-tab
ct = pd.crosstab(df['channel'], df['campaign_type'],
                 values=df['revenue'], aggfunc='sum')
```

### Performance Scoring
```python
# Percentile ranking
df['cost_percentile'] = df['cost'].rank(pct=True)
df['roas_percentile'] = df['roas'].rank(pct=True)

# Performance categories
df['performance'] = pd.cut(
    df['roas'],
    bins=[0, 2, 3, 5, float('inf')],
    labels=['Poor', 'Fair', 'Good', 'Excellent']
)

# Or using qcut for equal-sized bins
df['performance_quartile'] = pd.qcut(
    df['roas'],
    q=4,
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)
```

---

## 💡 Advanced Pandas Techniques

### Apply Custom Functions
```python
# Apply to single column
df['campaign'].apply(len)                     # Length of each string
df['cost'].apply(lambda x: x * 1.1)          # 10% markup

# Apply to rows
df.apply(lambda row: row['revenue'] / row['cost'], axis=1)

# Apply to DataFrame
def calculate_efficiency(row):
    if row['conversions'] > 0:
        return row['revenue'] / row['conversions']
    return 0

df['aov'] = df.apply(calculate_efficiency, axis=1)
```

### Creating New Columns
```python
# Simple assignment
df['profit'] = df['revenue'] - df['cost']

# Using assign (returns new DataFrame)
df2 = df.assign(
    profit=df['revenue'] - df['cost'],
    roas=df['revenue'] / df['cost']
)

# Insert at specific position
df.insert(2, 'profit', df['revenue'] - df['cost'])

# Conditional column
df['status'] = np.where(df['roas'] >= 3, 'Good', 'Poor')

# Multiple conditions with np.select
conditions = [
    df['roas'] >= 4,
    df['roas'] >= 2.5,
    df['roas'] >= 1.5
]
choices = ['Excellent', 'Good', 'Fair']
df['status'] = np.select(conditions, choices, default='Poor')
```

### Renaming & Transforming
```python
# Rename columns
df.rename(columns={'old_name': 'new_name'})
df.columns = ['campaign', 'cost', 'revenue']  # Replace all names

# Rename with function
df.rename(columns=str.lower)                  # Lowercase all
df.rename(columns=lambda x: x.replace(' ', '_'))

# Change data types
df['cost'] = df['cost'].astype(float)
df['date'] = pd.to_datetime(df['date'])

# Round numbers
df['roas'] = df['roas'].round(2)
df = df.round({'roas': 2, 'cpa': 2})
```

---

## 🚀 Quick Tips & Best Practices

### Performance Optimization
```python
# Use vectorized operations instead of loops
# ❌ Slow
for i in range(len(df)):
    df.loc[i, 'roas'] = df.loc[i, 'revenue'] / df.loc[i, 'cost']

# ✅ Fast
df['roas'] = df['revenue'] / df['cost']

# Use categorical for repeated strings
df['channel'] = df['channel'].astype('category')

# Chain operations
result = (df
    .query('cost > 1000')
    .groupby('channel')['revenue'].sum()
    .sort_values(ascending=False)
    .head(5)
)
```

### Data Quality Checks
```python
# Check for duplicates
df.duplicated().sum()                         # Count duplicates
df[df.duplicated()]                          # Show duplicates
df.drop_duplicates()                         # Remove duplicates
df.drop_duplicates(subset=['campaign_id'])   # Based on specific column

# Data types
df.dtypes
df.select_dtypes(include=[np.number])        # Numeric columns only
df.select_dtypes(include=['object'])         # String columns only

# Value ranges
df.describe()                                 # Statistical summary
df['cost'].min(), df['cost'].max()           # Min and max
```

### Common Mistakes to Avoid
```python
# ❌ Wrong: Modifying without assignment
df.sort_values('cost')                        # Doesn't modify df

# ✅ Correct: Assign or use inplace
df = df.sort_values('cost')
df.sort_values('cost', inplace=True)

# ❌ Wrong: Chained assignment warning
df[df['cost'] > 1000]['revenue'] = 0

# ✅ Correct: Use loc
df.loc[df['cost'] > 1000, 'revenue'] = 0

# ❌ Wrong: Comparing DataFrames with ==
if df1 == df2:  # Raises error

# ✅ Correct: Use equals()
if df1.equals(df2):
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Campaign Analysis
```python
# Given campaign data, calculate performance metrics
data = {
    'campaign': ['Search_A', 'Search_B', 'Social_A', 'Display_A'],
    'impressions': [100000, 85000, 120000, 200000],
    'clicks': [5000, 4250, 3600, 2000],
    'conversions': [250, 200, 150, 80],
    'cost': [5000, 4500, 3000, 2500],
    'revenue': [20000, 16000, 9000, 6000]
}
df = pd.DataFrame(data)

# Solution: Add all metrics
df['ctr'] = df['clicks'] / df['impressions']
df['cvr'] = df['conversions'] / df['clicks']
df['cpa'] = df['cost'] / df['conversions']
df['roas'] = df['revenue'] / df['cost']

# Show top performers
print(df.sort_values('roas', ascending=False))
```

### Exercise 2: Channel Performance
```python
# Group campaigns by channel (extract from campaign name)
df['channel'] = df['campaign'].str.split('_').str[0]

# Calculate channel totals
channel_summary = df.groupby('channel').agg({
    'cost': 'sum',
    'revenue': 'sum',
    'conversions': 'sum'
}).reset_index()

# Calculate channel ROAS
channel_summary['roas'] = channel_summary['revenue'] / channel_summary['cost']

# Find best channel
best_channel = channel_summary.loc[channel_summary['roas'].idxmax()]
print(f"Best channel: {best_channel['channel']} with ROAS: {best_channel['roas']:.2f}x")
```

### Exercise 3: Time Series Analysis
```python
# Daily campaign data
daily_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=90),
    'cost': np.random.uniform(1000, 3000, 90),
    'revenue': np.random.uniform(3000, 12000, 90)
})

# Calculate daily ROAS
daily_data['roas'] = daily_data['revenue'] / daily_data['cost']

# 7-day moving average
daily_data['roas_7d_ma'] = daily_data['roas'].rolling(window=7).mean()

# Weekly aggregation
weekly = daily_data.resample('W', on='date').agg({
    'cost': 'sum',
    'revenue': 'sum'
})
weekly['roas'] = weekly['revenue'] / weekly['cost']

print(weekly.head())
```

### Exercise 4: Merging Campaign and Conversion Data
```python
# Campaign data
campaigns = pd.DataFrame({
    'campaign_id': [101, 102, 103, 104],
    'campaign_name': ['Search_A', 'Social_B', 'Display_C', 'Video_D'],
    'cost': [5000, 3500, 2000, 4000]
})

# Conversion data
conversions = pd.DataFrame({
    'campaign_id': [101, 102, 103, 105],  # Note: 105 doesn't exist in campaigns
    'conversions': [250, 175, 100, 50],
    'revenue': [20000, 14000, 8000, 4000]
})

# Left join to keep all campaigns
merged = pd.merge(campaigns, conversions, on='campaign_id', how='left')

# Fill missing conversions with 0
merged['conversions'] = merged['conversions'].fillna(0)
merged['revenue'] = merged['revenue'].fillna(0)

# Calculate metrics
merged['roas'] = np.where(
    merged['cost'] > 0,
    merged['revenue'] / merged['cost'],
    0
)

print(merged)
```

### Exercise 5: Advanced Filtering & Ranking
```python
# Filter for high-performing, low-cost campaigns
good_campaigns = df[
    (df['roas'] > 3.0) &
    (df['cpa'] < 25) &
    (df['conversions'] > 100)
]

# Rank campaigns by multiple criteria
df['roas_rank'] = df['roas'].rank(ascending=False)
df['cpa_rank'] = df['cpa'].rank(ascending=True)  # Lower is better

# Combined score (simple average of ranks)
df['combined_rank'] = (df['roas_rank'] + df['cpa_rank']) / 2

# Get top 5 overall
top_5 = df.nsmallest(5, 'combined_rank')[
    ['campaign', 'roas', 'cpa', 'combined_rank']
]

print(top_5)
```

---

## 🔍 Quick Reference Table

| Task | Code |
|------|------|
| Read CSV | `pd.read_csv('file.csv')` |
| Write CSV | `df.to_csv('file.csv', index=False)` |
| First N rows | `df.head(N)` |
| Filter rows | `df[df['col'] > value]` |
| Select columns | `df[['col1', 'col2']]` |
| Group and sum | `df.groupby('col')['value'].sum()` |
| Sort | `df.sort_values('col', ascending=False)` |
| Add column | `df['new'] = df['a'] + df['b']` |
| Drop column | `df.drop('col', axis=1)` |
| Drop nulls | `df.dropna()` |
| Fill nulls | `df.fillna(value)` |
| Merge | `pd.merge(df1, df2, on='key')` |
| Concat | `pd.concat([df1, df2])` |
| Unique values | `df['col'].unique()` |
| Value counts | `df['col'].value_counts()` |
| Reset index | `df.reset_index(drop=True)` |

---

**Quick Navigation:**
- [← Week 1 Cheatsheet](Week_01_Cheatsheet.md)
- [Week 3 Cheatsheet →](Week_03_Cheatsheet.md)
- [Back to Main README](../README.md)
