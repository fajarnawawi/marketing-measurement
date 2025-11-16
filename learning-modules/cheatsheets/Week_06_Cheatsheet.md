# Week 6: Data Visualization - Quick Reference Cheatsheet

## 📋 Matplotlib Basics

### Figure & Axes Setup
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simple plot (implicit figure/axes)
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()

# Explicit figure and axes
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()

# Figure with specific size
fig, ax = plt.subplots(figsize=(10, 6))

# Multiple subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplots with shared axes
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
```

### Basic Plot Types
```python
# Line plot
plt.plot(x, y, label='Revenue')
plt.plot(x, y, marker='o', linestyle='--', color='blue', linewidth=2)

# Scatter plot
plt.scatter(x, y, s=100, c='red', alpha=0.5, marker='o')

# Bar plot
plt.bar(categories, values, color='steelblue')
plt.barh(categories, values)  # Horizontal bars

# Histogram
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)

# Box plot
plt.boxplot(data, labels=['Group 1', 'Group 2'])

# Pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
```

### Customization
```python
# Labels and title
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.title('Daily Revenue Trend', fontsize=14, fontweight='bold')

# Legend
plt.legend(loc='upper left')
plt.legend(loc='best', frameon=True, shadow=True)

# Grid
plt.grid(True, alpha=0.3)
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Axis limits
plt.xlim(0, 100)
plt.ylim(0, 1000)

# Axis ticks
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 1000, 100))

# Tight layout (prevent label cutoff)
plt.tight_layout()

# Save figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

### Colors and Styles
```python
# Named colors
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Hex colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# Colormap
plt.plot(x, y, c=values, cmap='viridis')
plt.colorbar(label='Value')

# Style presets
plt.style.use('seaborn-v0_8-darkgrid')
plt.style.use('ggplot')
plt.style.available  # List all available styles

# Alpha (transparency)
plt.plot(x, y, alpha=0.7)
```

---

## 🎨 Seaborn Fundamentals

### Setup & Basic Plots
```python
import seaborn as sns

# Set theme/style
sns.set_theme(style='whitegrid')
sns.set_style('darkgrid')
sns.set_context('notebook')  # notebook, paper, talk, poster

# Set color palette
sns.set_palette('husl')
sns.set_palette(['#FF6B6B', '#4ECDC4', '#45B7D1'])

# Distribution plot (histogram + KDE)
sns.histplot(data=df, x='cost', bins=30, kde=True)

# KDE plot only
sns.kdeplot(data=df, x='cost', fill=True)

# Multiple distributions
sns.kdeplot(data=df, x='cost', hue='channel', fill=True, alpha=0.5)
```

### Statistical Plots
```python
# Box plot
sns.boxplot(data=df, x='channel', y='revenue')

# Violin plot (box plot + KDE)
sns.violinplot(data=df, x='channel', y='revenue')

# Swarm plot (categorical scatter)
sns.swarmplot(data=df, x='channel', y='revenue', size=3)

# Strip plot
sns.stripplot(data=df, x='channel', y='revenue', jitter=True)

# Combined plots
sns.violinplot(data=df, x='channel', y='revenue', inner=None)
sns.swarmplot(data=df, x='channel', y='revenue', color='black', size=2)
```

### Relational Plots
```python
# Scatter plot
sns.scatterplot(data=df, x='cost', y='revenue', hue='channel', size='conversions')

# Line plot
sns.lineplot(data=df, x='date', y='revenue', hue='channel')

# Line plot with confidence interval
sns.lineplot(data=df, x='date', y='revenue', ci=95)

# Regression plot
sns.regplot(data=df, x='cost', y='revenue', scatter_kws={'alpha':0.5})

# LM plot (regression with facets)
sns.lmplot(data=df, x='cost', y='revenue', hue='channel', height=6, aspect=1.5)
```

### Categorical Plots
```python
# Bar plot (with error bars)
sns.barplot(data=df, x='channel', y='revenue', estimator=np.mean, ci=95)

# Count plot
sns.countplot(data=df, x='channel')

# Point plot
sns.pointplot(data=df, x='date', y='revenue', hue='channel')
```

### Matrix Plots
```python
# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=1)

# Clustermap (hierarchical clustering)
sns.clustermap(correlation_matrix, cmap='coolwarm', center=0,
               figsize=(10, 10), dendrogram_ratio=0.1)
```

---

## 📊 Time Series Visualization

### Line Plots for Time Series
```python
# Basic time series
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['date'], df['revenue'], marker='o', linewidth=2)
ax.set_xlabel('Date')
ax.set_ylabel('Revenue ($)')
ax.set_title('Daily Revenue Trend')
plt.xticks(rotation=45)
plt.tight_layout()

# Multiple time series
fig, ax = plt.subplots(figsize=(14, 7))
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]
    ax.plot(channel_data['date'], channel_data['revenue'], label=channel, marker='o')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Revenue ($)')
ax.set_title('Revenue by Channel Over Time')
plt.xticks(rotation=45)
plt.tight_layout()

# With Seaborn
sns.lineplot(data=df, x='date', y='revenue', hue='channel', marker='o')
plt.xticks(rotation=45)
plt.tight_layout()
```

### Moving Averages
```python
# Calculate moving average
df['revenue_ma7'] = df['revenue'].rolling(window=7).mean()
df['revenue_ma30'] = df['revenue'].rolling(window=30).mean()

# Plot with moving averages
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['date'], df['revenue'], label='Daily Revenue', alpha=0.5, linewidth=1)
ax.plot(df['date'], df['revenue_ma7'], label='7-Day MA', linewidth=2)
ax.plot(df['date'], df['revenue_ma30'], label='30-Day MA', linewidth=2)
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Revenue ($)')
ax.set_title('Revenue with Moving Averages')
plt.xticks(rotation=45)
plt.tight_layout()
```

### Area Plots
```python
# Stacked area plot
fig, ax = plt.subplots(figsize=(14, 7))
df_pivot = df.pivot_table(values='revenue', index='date', columns='channel', aggfunc='sum')
ax.stackplot(df_pivot.index, df_pivot.T, labels=df_pivot.columns, alpha=0.8)
ax.legend(loc='upper left')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue ($)')
ax.set_title('Revenue by Channel (Stacked)')
plt.xticks(rotation=45)
plt.tight_layout()

# Filled area
plt.fill_between(df['date'], df['revenue'], alpha=0.3)
```

### Date Formatting
```python
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['date'], df['revenue'])

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

plt.xticks(rotation=45)
plt.tight_layout()
```

---

## 📈 Distribution Plots

### Histograms
```python
# Basic histogram
plt.hist(df['cost'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Cost Distribution')

# Multiple histograms
fig, ax = plt.subplots(figsize=(10, 6))
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]['cost']
    ax.hist(channel_data, bins=20, alpha=0.5, label=channel)
ax.legend()
ax.set_xlabel('Cost')
ax.set_ylabel('Frequency')

# Seaborn histogram with KDE
sns.histplot(data=df, x='cost', bins=30, kde=True, color='steelblue')

# By category
sns.histplot(data=df, x='cost', hue='channel', bins=30, kde=True, alpha=0.5)
```

### Box Plots
```python
# Single box plot
plt.boxplot(df['revenue'])
plt.ylabel('Revenue')

# Multiple box plots
fig, ax = plt.subplots(figsize=(10, 6))
data_to_plot = [df[df['channel'] == ch]['revenue'] for ch in df['channel'].unique()]
ax.boxplot(data_to_plot, labels=df['channel'].unique())
ax.set_ylabel('Revenue')
ax.set_xlabel('Channel')

# Seaborn box plot
sns.boxplot(data=df, x='channel', y='revenue', palette='Set2')
plt.xticks(rotation=45)

# Horizontal box plot
sns.boxplot(data=df, y='channel', x='revenue', orient='h')

# With swarm overlay
sns.boxplot(data=df, x='channel', y='revenue', palette='Set2')
sns.swarmplot(data=df, x='channel', y='revenue', color='black', alpha=0.5, size=3)
```

### Violin Plots
```python
# Basic violin plot
sns.violinplot(data=df, x='channel', y='revenue', palette='muted')

# Split violin (compare two groups)
sns.violinplot(data=df, x='channel', y='revenue', hue='campaign_type', split=True)

# With inner quartiles
sns.violinplot(data=df, x='channel', y='revenue', inner='quartile')
```

### KDE Plots
```python
# Single KDE
sns.kdeplot(data=df, x='cost', fill=True, alpha=0.5)

# Multiple KDEs
sns.kdeplot(data=df, x='cost', hue='channel', fill=True, alpha=0.5)

# 2D KDE (joint distribution)
sns.kdeplot(data=df, x='cost', y='revenue', cmap='Blues', fill=True)

# Ridge plot (multiple KDEs stacked)
import matplotlib.pyplot as plt
channels = df['channel'].unique()
fig, axes = plt.subplots(len(channels), 1, figsize=(10, 8), sharex=True)
for i, channel in enumerate(channels):
    channel_data = df[df['channel'] == channel]['cost']
    sns.kdeplot(channel_data, ax=axes[i], fill=True, alpha=0.7)
    axes[i].set_ylabel(channel)
plt.tight_layout()
```

---

## 🔥 Heatmaps & Correlation

### Correlation Heatmap
```python
# Calculate correlation
correlation = df[['cost', 'revenue', 'conversions', 'clicks']].corr()

# Basic heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix')

# With mask for upper triangle
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm',
            center=0, fmt='.2f', square=True, linewidths=1)

# Custom color scheme
sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0,
            vmin=-1, vmax=1, fmt='.2f', square=True)
```

### Pivot Table Heatmap
```python
# Create pivot table
pivot = df.pivot_table(values='revenue', index='date', columns='channel', aggfunc='sum')

# Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pivot, cmap='YlOrRd', annot=False, fmt='.0f', cbar_kws={'label': 'Revenue'})
plt.title('Revenue by Date and Channel')
plt.tight_layout()

# Normalized heatmap (percentage)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
sns.heatmap(pivot_pct, cmap='Blues', annot=True, fmt='.1f', cbar_kws={'label': '% of Daily Revenue'})
```

### Cohort Retention Heatmap
```python
# Cohort retention data (from Week 5)
plt.figure(figsize=(14, 8))
sns.heatmap(retention_rate, annot=True, fmt='.1f', cmap='RdYlGn',
            vmin=0, vmax=100, cbar_kws={'label': 'Retention Rate (%)'})
plt.title('Cohort Retention Rate (%)')
plt.xlabel('Months Since First Purchase')
plt.ylabel('Cohort Month')
plt.tight_layout()
```

---

## 📐 Multi-Panel Layouts

### Subplots
```python
# 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot in each subplot
axes[0, 0].plot(df['date'], df['revenue'])
axes[0, 0].set_title('Revenue Over Time')

axes[0, 1].hist(df['cost'], bins=30)
axes[0, 1].set_title('Cost Distribution')

axes[1, 0].scatter(df['cost'], df['revenue'])
axes[1, 0].set_title('Cost vs Revenue')

axes[1, 1].boxplot([df[df['channel'] == ch]['revenue'] for ch in df['channel'].unique()])
axes[1, 1].set_title('Revenue by Channel')

plt.tight_layout()

# Uneven grid
fig = plt.figure(figsize=(14, 10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 1, 2)  # Bottom half

ax1.plot(df['date'], df['revenue'])
ax2.hist(df['cost'], bins=30)
ax3.scatter(df['cost'], df['revenue'])
plt.tight_layout()
```

### GridSpec for Complex Layouts
```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 3, figure=fig)

# Large plot spanning multiple cells
ax1 = fig.add_subplot(gs[0:2, 0:2])  # Top-left 2x2
ax1.plot(df['date'], df['revenue'])

# Smaller plots
ax2 = fig.add_subplot(gs[0, 2])  # Top-right
ax2.hist(df['cost'], bins=20)

ax3 = fig.add_subplot(gs[1, 2])  # Middle-right
ax3.scatter(df['cost'], df['revenue'])

ax4 = fig.add_subplot(gs[2, :])  # Bottom row (full width)
ax4.boxplot([df[df['channel'] == ch]['revenue'] for ch in df['channel'].unique()])

plt.tight_layout()
```

### Seaborn FacetGrid
```python
# Create grid based on categorical variable
g = sns.FacetGrid(df, col='channel', height=5, aspect=1.2)
g.map(sns.histplot, 'revenue', bins=30, kde=True)
g.set_titles("{col_name}")
g.set_axis_labels("Revenue", "Count")

# Multiple rows and columns
g = sns.FacetGrid(df, col='channel', row='campaign_type', height=4)
g.map(plt.scatter, 'cost', 'revenue', alpha=0.5)
g.add_legend()

# With hue
g = sns.FacetGrid(df, col='channel', hue='campaign_type', height=5)
g.map(sns.scatterplot, 'cost', 'revenue', alpha=0.5)
g.add_legend()
```

### PairGrid (Pairwise Relationships)
```python
# Pairplot (quick version)
sns.pairplot(df[['cost', 'revenue', 'conversions', 'channel']], hue='channel')

# PairGrid (more control)
g = sns.PairGrid(df, vars=['cost', 'revenue', 'conversions'], hue='channel')
g.map_upper(sns.scatterplot, alpha=0.5)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
```

---

## 🎨 Color Schemes & Accessibility

### Color Palettes
```python
# Seaborn palettes
sns.color_palette("deep")           # Default
sns.color_palette("husl", 8)        # Evenly spaced hues
sns.color_palette("Set2")           # Qualitative
sns.color_palette("Paired")         # Paired colors
sns.color_palette("tab10")          # Tableau colors

# Sequential (for continuous data)
sns.color_palette("Blues")
sns.color_palette("YlOrRd")
sns.color_palette("viridis")

# Diverging (for data with meaningful center)
sns.color_palette("RdBu")
sns.color_palette("coolwarm")
sns.color_palette("vlag")

# Set palette globally
sns.set_palette("husl")

# View palette
sns.palplot(sns.color_palette("viridis", 10))
```

### Colorblind-Friendly Palettes
```python
# Colorblind-safe palettes
colorblind_palette = sns.color_palette("colorblind")
sns.set_palette("colorblind")

# Custom colorblind-safe palette
safe_colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#949494']
sns.set_palette(safe_colors)

# Use patterns for additional differentiation
fig, ax = plt.subplots()
bars = ax.bar(categories, values, color=safe_colors[:len(categories)])
patterns = ['/', '\\', '|', '-', '+', 'x']
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)
```

### Custom Color Maps
```python
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['#FF6B6B', '#FFA500', '#4ECDC4']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Use custom colormap
sns.heatmap(correlation, cmap=cmap)

# Discrete colormap
from matplotlib.colors import ListedColormap
discrete_cmap = ListedColormap(['#FF6B6B', '#4ECDC4', '#45B7D1'])
```

---

## 📊 Marketing Dashboard Examples

### Performance Dashboard
```python
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Revenue over time (large)
ax1 = fig.add_subplot(gs[0, :])
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]
    ax1.plot(channel_data['date'], channel_data['revenue'], marker='o', label=channel)
ax1.set_title('Daily Revenue by Channel', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Revenue ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Channel distribution (pie)
ax2 = fig.add_subplot(gs[1, 0])
channel_revenue = df.groupby('channel')['revenue'].sum()
ax2.pie(channel_revenue, labels=channel_revenue.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Revenue Share by Channel')

# 3. ROAS by channel (bar)
ax3 = fig.add_subplot(gs[1, 1])
channel_metrics = df.groupby('channel').agg({'revenue': 'sum', 'cost': 'sum'})
channel_metrics['roas'] = channel_metrics['revenue'] / channel_metrics['cost']
channel_metrics['roas'].plot(kind='bar', ax=ax3, color='steelblue')
ax3.set_title('ROAS by Channel')
ax3.set_ylabel('ROAS')
ax3.axhline(y=3.0, color='r', linestyle='--', label='Target')
ax3.legend()
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 4. Cost distribution (box)
ax4 = fig.add_subplot(gs[1, 2])
df.boxplot(column='cost', by='channel', ax=ax4)
ax4.set_title('Cost Distribution by Channel')
ax4.set_xlabel('Channel')
ax4.set_ylabel('Cost ($)')
plt.suptitle('')  # Remove automatic title

# 5. Cost vs Revenue (scatter)
ax5 = fig.add_subplot(gs[2, :2])
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]
    ax5.scatter(channel_data['cost'], channel_data['revenue'], label=channel, alpha=0.6, s=50)
ax5.set_title('Cost vs Revenue by Channel')
ax5.set_xlabel('Cost ($)')
ax5.set_ylabel('Revenue ($)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Key metrics (text)
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
total_revenue = df['revenue'].sum()
total_cost = df['cost'].sum()
overall_roas = total_revenue / total_cost
total_conversions = df['conversions'].sum()

metrics_text = f"""
KEY METRICS

Total Revenue: ${total_revenue:,.0f}
Total Cost: ${total_cost:,.0f}
Overall ROAS: {overall_roas:.2f}x
Total Conversions: {total_conversions:,.0f}
Avg CPA: ${total_cost/total_conversions:.2f}
"""
ax6.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Marketing Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Channel Comparison Report
```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Channel Performance Comparison', fontsize=16, fontweight='bold')

channels = df['channel'].unique()

# 1. Revenue trends
for channel in channels:
    channel_data = df[df['channel'] == channel]
    axes[0, 0].plot(channel_data['date'], channel_data['revenue'], marker='o', label=channel)
axes[0, 0].set_title('Revenue Trend')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Revenue ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. ROAS comparison
channel_roas = df.groupby('channel').apply(lambda x: x['revenue'].sum() / x['cost'].sum())
axes[0, 1].bar(channel_roas.index, channel_roas.values, color='steelblue')
axes[0, 1].axhline(y=3.0, color='r', linestyle='--', label='Target')
axes[0, 1].set_title('ROAS by Channel')
axes[0, 1].set_ylabel('ROAS')
axes[0, 1].legend()
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

# 3. Conversion rate
channel_cvr = df.groupby('channel').apply(lambda x: x['conversions'].sum() / x['clicks'].sum() * 100)
axes[0, 2].bar(channel_cvr.index, channel_cvr.values, color='green', alpha=0.7)
axes[0, 2].set_title('Conversion Rate by Channel')
axes[0, 2].set_ylabel('CVR (%)')
plt.setp(axes[0, 2].xaxis.get_majorticklabels(), rotation=45)

# 4. Cost distribution
df.boxplot(column='cost', by='channel', ax=axes[1, 0])
axes[1, 0].set_title('Cost Distribution')
axes[1, 0].set_xlabel('Channel')
axes[1, 0].set_ylabel('Cost ($)')
plt.suptitle('')

# 5. Revenue distribution
df.boxplot(column='revenue', by='channel', ax=axes[1, 1])
axes[1, 1].set_title('Revenue Distribution')
axes[1, 1].set_xlabel('Channel')
axes[1, 1].set_ylabel('Revenue ($)')
plt.suptitle('')

# 6. Spend vs Revenue
for channel in channels:
    channel_data = df[df['channel'] == channel]
    axes[1, 2].scatter(channel_data['cost'], channel_data['revenue'], label=channel, alpha=0.6, s=50)
axes[1, 2].set_title('Cost vs Revenue')
axes[1, 2].set_xlabel('Cost ($)')
axes[1, 2].set_ylabel('Revenue ($)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('channel_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 💡 Best Practices

### Chart Selection Guide
```python
# Relationship between two continuous variables → Scatter plot
plt.scatter(df['cost'], df['revenue'])

# Trend over time → Line plot
plt.plot(df['date'], df['revenue'])

# Distribution of single variable → Histogram or KDE
sns.histplot(df['cost'], kde=True)

# Compare categories → Bar plot or Box plot
sns.barplot(data=df, x='channel', y='revenue')

# Part-to-whole relationship → Pie chart or Stacked bar
plt.pie(channel_revenue, labels=channels, autopct='%1.1f%%')

# Correlation between multiple variables → Heatmap
sns.heatmap(correlation_matrix, annot=True)

# Distribution across categories → Violin or Box plot
sns.violinplot(data=df, x='channel', y='revenue')
```

### Styling Guidelines
```python
# Clear and descriptive titles
plt.title('Daily Revenue Trend - Q1 2024', fontsize=14, fontweight='bold')

# Always label axes with units
plt.xlabel('Date')
plt.ylabel('Revenue ($)')

# Use consistent color schemes
company_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
sns.set_palette(company_colors)

# Appropriate font sizes
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# Grid for easier reading
plt.grid(True, alpha=0.3, linestyle='--')

# Legend placement
plt.legend(loc='best', frameon=True, shadow=False)
```

### Common Patterns
```python
# Time series with annotations
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['date'], df['revenue'])
# Annotate important events
ax.annotate('Campaign Launch', xy=('2024-03-15', 50000),
            xytext=('2024-03-20', 55000),
            arrowprops=dict(arrowstyle='->', color='red'))
ax.axvline(x='2024-03-15', color='red', linestyle='--', alpha=0.5)

# Dual y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df['date'], df['revenue'], color='blue', label='Revenue')
ax1.set_ylabel('Revenue ($)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(df['date'], df['roas'], color='red', label='ROAS')
ax2.set_ylabel('ROAS', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Reference lines
plt.axhline(y=target_value, color='r', linestyle='--', label='Target')
plt.axvline(x=cutoff_date, color='g', linestyle=':', label='Cutoff')

# Error bars
means = df.groupby('channel')['revenue'].mean()
stds = df.groupby('channel')['revenue'].std()
plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Complete Marketing Dashboard
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Prepare data
df['date'] = pd.to_datetime(df['date'])
df['roas'] = df['revenue'] / df['cost']

# Create dashboard
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. Revenue time series with moving average
ax1 = fig.add_subplot(gs[0, :])
daily_revenue = df.groupby('date')['revenue'].sum()
ma7 = daily_revenue.rolling(window=7).mean()
ma30 = daily_revenue.rolling(window=30).mean()

ax1.plot(daily_revenue.index, daily_revenue.values, alpha=0.3, label='Daily', linewidth=1)
ax1.plot(ma7.index, ma7.values, label='7-Day MA', linewidth=2)
ax1.plot(ma30.index, ma30.values, label='30-Day MA', linewidth=2)
ax1.set_title('Daily Revenue with Moving Averages', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Revenue ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Channel performance metrics
ax2 = fig.add_subplot(gs[1, 0])
channel_metrics = df.groupby('channel').agg({
    'revenue': 'sum',
    'cost': 'sum',
    'conversions': 'sum'
})
channel_metrics['roas'] = channel_metrics['revenue'] / channel_metrics['cost']
channel_metrics['roas'].sort_values().plot(kind='barh', ax=ax2, color='steelblue')
ax2.axvline(x=3.0, color='r', linestyle='--', linewidth=2, label='Target')
ax2.set_title('ROAS by Channel')
ax2.set_xlabel('ROAS')
ax2.legend()

# 3. Revenue distribution heatmap
ax3 = fig.add_subplot(gs[1, 1:])
pivot = df.pivot_table(values='revenue', index=df['date'].dt.week, columns='channel', aggfunc='sum')
sns.heatmap(pivot, cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Revenue'}, fmt='.0f')
ax3.set_title('Weekly Revenue Heatmap')
ax3.set_xlabel('Channel')
ax3.set_ylabel('Week')

# 4. Cost vs Revenue scatter
ax4 = fig.add_subplot(gs[2, 0])
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]
    ax4.scatter(channel_data['cost'], channel_data['revenue'],
                label=channel, alpha=0.6, s=50)
# Add ROAS reference line
max_val = max(df['cost'].max(), df['revenue'].max())
ax4.plot([0, max_val], [0, max_val*3], 'r--', label='ROAS=3.0', linewidth=2)
ax4.set_title('Cost vs Revenue')
ax4.set_xlabel('Cost ($)')
ax4.set_ylabel('Revenue ($)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Funnel metrics
ax5 = fig.add_subplot(gs[2, 1])
funnel_data = df.agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'conversions': 'sum'
})
funnel_pct = [100,
              (funnel_data['clicks'] / funnel_data['impressions']) * 100,
              (funnel_data['conversions'] / funnel_data['clicks']) * 100]
stages = ['Impressions', 'Clicks', 'Conversions']
ax5.barh(stages, funnel_pct, color=['lightblue', 'skyblue', 'steelblue'])
ax5.set_xlabel('% of Previous Stage')
ax5.set_title('Conversion Funnel')
for i, (stage, pct) in enumerate(zip(stages, funnel_pct)):
    ax5.text(pct + 2, i, f'{pct:.1f}%', va='center')

# 6. Performance summary
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
summary_stats = f"""
PERFORMANCE SUMMARY

Total Revenue: ${df['revenue'].sum():,.0f}
Total Cost: ${df['cost'].sum():,.0f}
Overall ROAS: {df['revenue'].sum()/df['cost'].sum():.2f}x

Total Conversions: {df['conversions'].sum():,.0f}
Avg CPA: ${df['cost'].sum()/df['conversions'].sum():.2f}

CTR: {(df['clicks'].sum()/df['impressions'].sum())*100:.2f}%
CVR: {(df['conversions'].sum()/df['clicks'].sum())*100:.2f}%

Date Range: {df['date'].min().date()} to {df['date'].max().date()}
"""
ax6.text(0.05, 0.5, summary_stats, fontsize=11, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Marketing Performance Dashboard', fontsize=18, fontweight='bold', y=0.995)
plt.savefig('marketing_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Exercise 2: Distribution Analysis
```python
# Compare distributions across channels
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histograms with KDE
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]
    sns.histplot(channel_data['roas'], bins=30, kde=True, label=channel,
                 alpha=0.4, ax=axes[0, 0])
axes[0, 0].set_title('ROAS Distribution by Channel')
axes[0, 0].set_xlabel('ROAS')
axes[0, 0].legend()

# 2. Violin plots
sns.violinplot(data=df, x='channel', y='roas', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('ROAS Distribution (Violin)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Box plots with swarm
sns.boxplot(data=df, x='channel', y='roas', ax=axes[1, 0], palette='Set2')
sns.swarmplot(data=df, x='channel', y='roas', ax=axes[1, 0],
              color='black', alpha=0.3, size=2)
axes[1, 0].set_title('ROAS Distribution (Box + Swarm)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Cumulative distribution
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]['roas'].sort_values()
    cumulative = np.arange(1, len(channel_data) + 1) / len(channel_data)
    axes[1, 1].plot(channel_data, cumulative, label=channel, linewidth=2)
axes[1, 1].set_title('Cumulative Distribution Function')
axes[1, 1].set_xlabel('ROAS')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('ROAS Distribution Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Exercise 3: Time Series Analysis
```python
# Multi-panel time series visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# 1. Revenue by channel (stacked area)
pivot_revenue = df.pivot_table(values='revenue', index='date', columns='channel', aggfunc='sum')
pivot_revenue.plot.area(ax=axes[0], alpha=0.7, stacked=True)
axes[0].set_title('Revenue by Channel (Stacked)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Revenue ($)')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# 2. ROAS trend by channel
for channel in df['channel'].unique():
    channel_daily = df[df['channel'] == channel].groupby('date').agg({
        'revenue': 'sum',
        'cost': 'sum'
    })
    channel_daily['roas'] = channel_daily['revenue'] / channel_daily['cost']
    # 7-day moving average
    roas_ma = channel_daily['roas'].rolling(window=7).mean()
    axes[1].plot(roas_ma.index, roas_ma.values, label=channel, linewidth=2, marker='o', markersize=3)

axes[1].axhline(y=3.0, color='r', linestyle='--', linewidth=2, label='Target')
axes[1].set_title('ROAS Trend (7-Day MA)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('ROAS')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Daily conversion rate
daily_funnel = df.groupby('date').agg({
    'impressions': 'sum',
    'clicks': 'sum',
    'conversions': 'sum'
})
daily_funnel['ctr'] = (daily_funnel['clicks'] / daily_funnel['impressions']) * 100
daily_funnel['cvr'] = (daily_funnel['conversions'] / daily_funnel['clicks']) * 100

axes[2].plot(daily_funnel.index, daily_funnel['ctr'], label='CTR', linewidth=2)
axes[2].plot(daily_funnel.index, daily_funnel['cvr'], label='CVR', linewidth=2)
axes[2].set_title('Funnel Metrics Over Time', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Rate (%)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Time Series Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
```

### Exercise 4: Correlation Analysis Visualization
```python
# Comprehensive correlation analysis
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Correlation heatmap
ax1 = fig.add_subplot(gs[0, :])
metrics = ['cost', 'revenue', 'conversions', 'clicks', 'impressions', 'roas']
corr = df[metrics].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=1, ax=ax1,
            cbar_kws={'label': 'Correlation'})
ax1.set_title('Correlation Matrix', fontsize=14, fontweight='bold')

# 2. Pairwise scatter plots
ax2 = fig.add_subplot(gs[1, 0])
for channel in df['channel'].unique():
    channel_data = df[df['channel'] == channel]
    ax2.scatter(channel_data['cost'], channel_data['revenue'],
                label=channel, alpha=0.6, s=50)
ax2.set_title('Cost vs Revenue')
ax2.set_xlabel('Cost ($)')
ax2.set_ylabel('Revenue ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add regression line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['cost'], df['revenue'])
line = slope * df['cost'] + intercept
ax2.plot(df['cost'], line, 'r--', linewidth=2,
         label=f'R²={r_value**2:.3f}')
ax2.legend()

# 3. Correlation bar chart
ax3 = fig.add_subplot(gs[1, 1])
target_corr = corr['revenue'].drop('revenue').sort_values()
colors = ['red' if x < 0 else 'green' for x in target_corr]
target_corr.plot(kind='barh', ax=ax3, color=colors, alpha=0.7)
ax3.set_title('Correlation with Revenue')
ax3.set_xlabel('Correlation Coefficient')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.suptitle('Correlation Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Exercise 5: Cohort Visualization
```python
# Cohort analysis visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Assuming cohort data from Week 5
# 1. Retention heatmap
sns.heatmap(retention_rate, annot=True, fmt='.0f', cmap='RdYlGn',
            vmin=0, vmax=100, ax=axes[0, 0],
            cbar_kws={'label': 'Retention Rate (%)'})
axes[0, 0].set_title('Cohort Retention Rate (%)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Months Since First Purchase')
axes[0, 0].set_ylabel('Cohort Month')

# 2. Cumulative revenue heatmap
cumulative_revenue = cohort_revenue.cumsum(axis=1)
sns.heatmap(cumulative_revenue, annot=True, fmt='.0f', cmap='Blues',
            ax=axes[0, 1], cbar_kws={'label': 'Cumulative Revenue ($)'})
axes[0, 1].set_title('Cumulative Revenue by Cohort', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Months Since First Purchase')
axes[0, 1].set_ylabel('Cohort Month')

# 3. LTV curve by cohort
ltv_by_cohort = cumulative_revenue.divide(cohort_size, axis=0)
for cohort in ltv_by_cohort.index:
    axes[1, 0].plot(ltv_by_cohort.columns, ltv_by_cohort.loc[cohort],
                    marker='o', label=str(cohort))
axes[1, 0].set_title('LTV Curve by Cohort', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Months Since First Purchase')
axes[1, 0].set_ylabel('Cumulative LTV ($)')
axes[1, 0].legend(title='Cohort')
axes[1, 0].grid(True, alpha=0.3)

# 4. Cohort size and performance
cohort_summary = pd.DataFrame({
    'size': cohort_size,
    'ltv': ltv_by_cohort.iloc[:, -1]  # Latest LTV
})

ax_twin = axes[1, 1]
ax_twin2 = ax_twin.twinx()

cohort_summary['size'].plot(kind='bar', ax=ax_twin, color='steelblue', alpha=0.7, label='Cohort Size')
cohort_summary['ltv'].plot(kind='line', ax=ax_twin2, color='red',
                            marker='o', linewidth=2, label='LTV')

axes[1, 1].set_title('Cohort Size and LTV', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Cohort Month')
ax_twin.set_ylabel('Cohort Size', color='steelblue')
ax_twin2.set_ylabel('LTV ($)', color='red')
ax_twin.legend(loc='upper left')
ax_twin2.legend(loc='upper right')
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

plt.suptitle('Cohort Analysis Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('cohort_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🔍 Quick Reference Table

| Chart Type | Use Case | Code |
|------------|----------|------|
| Line Plot | Time trends | `plt.plot(x, y)` |
| Scatter | Relationships | `plt.scatter(x, y)` |
| Bar Chart | Category comparison | `plt.bar(x, y)` |
| Histogram | Distribution | `plt.hist(data, bins=30)` |
| Box Plot | Distribution + outliers | `sns.boxplot(x, y)` |
| Heatmap | Matrix data | `sns.heatmap(data)` |
| Violin | Distribution by category | `sns.violinplot(x, y)` |
| Pie Chart | Parts of whole | `plt.pie(values, labels)` |

---

**Quick Navigation:**
- [← Week 5 Cheatsheet](Week_05_Cheatsheet.md)
- [Back to Main README](../README.md)
