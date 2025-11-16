# Kaggle Datasets Guide for Marketing Analytics Learning

## Table of Contents
1. [Introduction](#1-introduction)
2. [Setting Up Kaggle API](#2-setting-up-kaggle-api)
3. [Recommended Marketing Datasets](#3-recommended-marketing-datasets)
4. [Loading and Exploring Datasets](#4-loading-and-exploring-datasets)
5. [Integration with Course Modules](#5-integration-with-course-modules)
6. [Best Practices](#6-best-practices)
7. [Project Ideas](#7-project-ideas)

---

## 1. Introduction

### Why Use Real-World Datasets?

Learning marketing analytics with real-world data provides several critical advantages:

- **Realistic complexity**: Real datasets contain the messiness, missing values, and inconsistencies you'll encounter in actual marketing roles
- **Industry relevance**: Working with authentic marketing data helps you understand common data structures, metrics, and business questions
- **Portfolio building**: Projects with real datasets demonstrate practical skills to potential employers
- **Problem-solving practice**: Real data requires creative solutions to data quality issues and analytical challenges

### Benefits of Kaggle Datasets for Learning

Kaggle is an ideal platform for learners because it offers:

- **Free access**: Thousands of datasets available at no cost
- **Community support**: Active discussions, kernels (notebooks), and shared insights
- **Variety**: Datasets spanning all marketing channels and business types
- **Documentation**: Most datasets include context, column definitions, and use cases
- **Competition data**: High-quality datasets from real companies and competitions
- **Easy download**: Simple API and web interface for acquiring data
- **Legal clarity**: Clear licensing terms for educational use

### How to Access Kaggle Datasets

You can access Kaggle datasets in three ways:

1. **Web Interface**: Browse and download directly from kaggle.com/datasets
2. **Kaggle API**: Download programmatically using Python CLI
3. **Kaggle Kernels**: Work with datasets directly in cloud notebooks

For this course, we'll focus on the API method to integrate seamlessly with your local Jupyter environment.

---

## 2. Setting Up Kaggle API

### Creating a Kaggle Account

1. Go to [kaggle.com](https://www.kaggle.com)
2. Click "Register" in the top right
3. Sign up with your Google account or email
4. Verify your email address
5. Complete your profile (optional but recommended)

### Getting API Credentials

1. Log in to Kaggle
2. Click on your profile picture (top right) → "Settings"
3. Scroll down to the "API" section
4. Click "Create New API Token"
5. This downloads a `kaggle.json` file containing your credentials

**Important**: Keep this file secure! It contains your API username and key.

### Installing Kaggle Package

Install the official Kaggle API package:

```bash
# Using pip
pip install kaggle

# Using conda
conda install -c conda-forge kaggle
```

Verify installation:

```bash
kaggle --version
```

### Authentication Setup

#### On Linux/Mac:

```bash
# Create .kaggle directory in your home folder
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set restrictive permissions (required)
chmod 600 ~/.kaggle/kaggle.json
```

#### On Windows:

```bash
# Create .kaggle directory
mkdir %USERPROFILE%\.kaggle

# Move kaggle.json to the directory
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

#### Verify Setup:

```bash
# This should list available competitions
kaggle competitions list
```

If you see a list of competitions, you're all set!

### Downloading Datasets via CLI

#### Basic Download Command:

```bash
# Download a dataset
kaggle datasets download -d username/dataset-name

# Download and unzip
kaggle datasets download -d username/dataset-name --unzip

# Download to specific directory
kaggle datasets download -d username/dataset-name --path ./data/
```

#### Search for Datasets:

```bash
# Search for marketing datasets
kaggle datasets list -s marketing

# Search with more results
kaggle datasets list -s "e-commerce" --max-size 100000000
```

### Downloading Datasets via Python

```python
import os
from pathlib import Path

# Method 1: Using subprocess
import subprocess

def download_kaggle_dataset(dataset_name, output_path):
    """
    Download a Kaggle dataset using the API.

    Parameters:
    -----------
    dataset_name : str
        Format: 'username/dataset-name'
    output_path : str
        Directory to save the dataset
    """
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Download and unzip
    cmd = f"kaggle datasets download -d {dataset_name} --unzip --path {output_path}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Dataset downloaded to {output_path}")

# Example usage
download_kaggle_dataset(
    'olistbr/brazilian-ecommerce',
    './learning-modules/resources/kaggle-datasets/olist-ecommerce/'
)
```

```python
# Method 2: Using kaggle package directly
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset_api(dataset_name, output_path):
    """
    Download using Kaggle API object.
    """
    api = KaggleApi()
    api.authenticate()

    # Create directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Download and unzip
    api.dataset_download_files(
        dataset_name,
        path=output_path,
        unzip=True
    )
    print(f"Downloaded {dataset_name} to {output_path}")

# Example
download_dataset_api(
    'carrie1/ecommerce-data',
    './learning-modules/resources/kaggle-datasets/ecommerce-uci/'
)
```

---

## 3. Recommended Marketing Datasets

### Dataset 1: Brazilian E-Commerce (Olist)

**Kaggle URL**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

**Description**:
- **Size**: ~50MB, 100k orders from 2016-2018
- Real commercial data from Brazilian e-commerce marketplace
- Multiple tables: orders, customers, products, reviews, payments, sellers

**What You Can Learn**:
- Relational database design and SQL joins
- Customer journey analysis
- RFM segmentation
- Order fulfillment metrics
- Review sentiment impact on sales

**Relevant Weeks**:
- Week 2-4: SQL and data manipulation
- Week 5: Exploratory data analysis
- Week 6: Visualization of e-commerce metrics
- Week 12: Customer Lifetime Value

**Sample Analysis Ideas**:
- Calculate average order value by category
- Analyze delivery time impact on reviews
- Segment customers by purchase frequency
- Create seller performance dashboard

**Download Command**:
```bash
kaggle datasets download -d olistbr/brazilian-ecommerce --unzip \
  --path ./learning-modules/resources/kaggle-datasets/olist/
```

---

### Dataset 2: Online Retail Dataset (UCI)

**Kaggle URL**: https://www.kaggle.com/datasets/carrie1/ecommerce-data

**Description**:
- **Size**: ~45MB, 500k transactions
- UK-based online retail from 2010-2011
- Transactional data: invoice, stock code, quantity, price, customer ID

**What You Can Learn**:
- Market basket analysis
- Customer segmentation (RFM)
- Time series analysis of sales
- Product affinity analysis

**Relevant Weeks**:
- Week 2: Pandas manipulation
- Week 5-6: EDA and visualization
- Week 7: Statistics (cohort analysis)
- Week 12: CLV calculation

**Sample Analysis Ideas**:
- Identify best-selling products and trends
- Perform RFM segmentation
- Calculate monthly retention rates
- Analyze seasonal patterns

**Download Command**:
```bash
kaggle datasets download -d carrie1/ecommerce-data --unzip \
  --path ./learning-modules/resources/kaggle-datasets/online-retail/
```

---

### Dataset 3: Digital Advertising Campaign Performance

**Kaggle URL**: https://www.kaggle.com/datasets/loveall/clicks-conversion-tracking

**Description**:
- **Size**: ~2MB, 1 million ad impressions
- Ad campaign data with impressions, clicks, conversions
- Multiple campaigns and ad groups

**What You Can Learn**:
- Campaign performance metrics (CTR, CVR, CPA)
- A/B testing fundamentals
- Statistical significance testing
- Budget allocation optimization

**Relevant Weeks**:
- Week 5-6: Metrics calculation and visualization
- Week 7: Statistical foundations
- Week 8: A/B testing
- Week 10: Marketing mix modeling basics

**Sample Analysis Ideas**:
- Calculate and compare CTR across campaigns
- Analyze conversion funnel drop-offs
- Test significance of performance differences
- Optimize budget allocation by campaign

**Download Command**:
```bash
kaggle datasets download -d loveall/clicks-conversion-tracking --unzip \
  --path ./learning-modules/resources/kaggle-datasets/ad-campaigns/
```

---

### Dataset 4: Customer Churn Prediction Dataset

**Kaggle URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Description**:
- **Size**: <1MB, 7k customers
- Telco customer data with churn labels
- Demographics, services, billing info

**What You Can Learn**:
- Churn analysis and prediction
- Feature engineering for ML
- Logistic regression
- Customer retention strategies

**Relevant Weeks**:
- Week 5: EDA to understand churn drivers
- Week 7: Statistical analysis of churn factors
- Week 8: Hypothesis testing
- Week 11: Incrementality concepts

**Sample Analysis Ideas**:
- Identify key churn drivers
- Calculate churn rate by segment
- Build simple churn prediction model
- Estimate retention program impact

**Download Command**:
```bash
kaggle datasets download -d blastchar/telco-customer-churn --unzip \
  --path ./learning-modules/resources/kaggle-datasets/telco-churn/
```

---

### Dataset 5: Facebook Advertising Data

**Kaggle URL**: https://www.kaggle.com/datasets/madislemsalu/facebook-ad-campaign

**Description**:
- **Size**: <1MB, 1k ad campaigns
- Real Facebook ad campaign performance
- Impressions, clicks, conversions, spend by campaign

**What You Can Learn**:
- Social media advertising metrics
- ROAS calculation
- Campaign optimization
- Performance benchmarking

**Relevant Weeks**:
- Week 5-6: Metric calculation and visualization
- Week 8: A/B testing campaigns
- Week 9: Attribution basics
- Week 10: Marketing mix modeling

**Sample Analysis Ideas**:
- Calculate ROAS by campaign objective
- Analyze performance by ad placement
- Compare campaign types effectiveness
- Optimize cost per conversion

**Download Command**:
```bash
kaggle datasets download -d madislemsalu/facebook-ad-campaign --unzip \
  --path ./learning-modules/resources/kaggle-datasets/facebook-ads/
```

---

### Dataset 6: Google Merchandise Store (GA360)

**Kaggle URL**: https://www.kaggle.com/datasets/bigquery/google-analytics-sample

**Description**:
- **Size**: Varies (BigQuery export)
- Google Analytics 360 sample data
- Sessions, pageviews, transactions, user behavior

**What You Can Learn**:
- Web analytics fundamentals
- User journey analysis
- Conversion funnel optimization
- Traffic source attribution

**Relevant Weeks**:
- Week 3-4: SQL for analytics queries
- Week 5-6: Session-level analysis
- Week 9: Multi-channel attribution
- Week 10: Channel performance

**Sample Analysis Ideas**:
- Analyze traffic sources and conversion rates
- Build conversion funnels
- Calculate assisted conversions by channel
- Segment users by behavior

**Download Command**:
```bash
# Note: This is a BigQuery dataset, access via BigQuery or exported CSV
# For CSV export version:
kaggle datasets download -d bigquery/google-analytics-sample --unzip \
  --path ./learning-modules/resources/kaggle-datasets/ga-sample/
```

---

### Dataset 7: Marketing Campaign Dataset

**Kaggle URL**: https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign

**Description**:
- **Size**: <1MB, 2.2k customers
- Customer profiles and campaign responses
- Demographics, spending, campaign acceptance

**What You Can Learn**:
- Campaign response modeling
- Customer profiling
- Segmentation strategies
- Response rate analysis

**Relevant Weeks**:
- Week 5: Customer profile EDA
- Week 6: Segmentation visualization
- Week 7-8: Response rate testing
- Week 11: Lift analysis

**Sample Analysis Ideas**:
- Identify high-response customer segments
- Analyze spending patterns by demographics
- Calculate campaign ROI
- Build response prediction model

**Download Command**:
```bash
kaggle datasets download -d rodsaldanha/arketing-campaign --unzip \
  --path ./learning-modules/resources/kaggle-datasets/marketing-campaign/
```

---

### Dataset 8: Email Campaign Performance

**Kaggle URL**: https://www.kaggle.com/datasets/janilbenzamin/email-marketing-campaign

**Description**:
- **Size**: <1MB, email campaign metrics
- Send, open, click, conversion data
- Segmentation and personalization flags

**What You Can Learn**:
- Email marketing KPIs
- A/B testing in email
- Segmentation impact
- Personalization effectiveness

**Relevant Weeks**:
- Week 5-6: Email metric analysis
- Week 8: A/B test email campaigns
- Week 9: Email attribution value
- Week 11: Incrementality of email

**Sample Analysis Ideas**:
- Calculate open, click, conversion rates
- Test subject line effectiveness
- Analyze send time impact
- Measure email channel incrementality

**Download Command**:
```bash
kaggle datasets download -d janilbenzamin/email-marketing-campaign --unzip \
  --path ./learning-modules/resources/kaggle-datasets/email-campaign/
```

---

### Dataset 9: Multi-Touch Attribution Dataset

**Kaggle URL**: https://www.kaggle.com/datasets/rishikumarjayakumar/multi-touch-attribution

**Description**:
- **Size**: ~5MB, customer touchpoint journeys
- Multi-channel customer paths
- Conversion outcomes

**What You Can Learn**:
- Multi-touch attribution models
- Customer journey mapping
- Channel path analysis
- Attribution modeling techniques

**Relevant Weeks**:
- Week 6: Journey visualization
- Week 7: Statistical path analysis
- Week 9: Attribution modeling (CRITICAL)
- Week 10: Channel contribution

**Sample Analysis Ideas**:
- Implement first-touch, last-touch attribution
- Build linear and time-decay models
- Analyze common conversion paths
- Calculate channel-specific conversion rates

**Download Command**:
```bash
kaggle datasets download -d rishikumarjayakumar/multi-touch-attribution --unzip \
  --path ./learning-modules/resources/kaggle-datasets/attribution/
```

---

### Dataset 10: Social Media Influencer Dataset

**Kaggle URL**: https://www.kaggle.com/datasets/ramjasmaurya/top-1000-social-media-channels

**Description**:
- **Size**: <1MB, influencer metrics
- Followers, engagement, platform data
- Performance metrics across platforms

**What You Can Learn**:
- Social media analytics
- Engagement rate calculation
- Platform comparison
- Influencer performance benchmarking

**Relevant Weeks**:
- Week 5-6: Social metrics analysis
- Week 7: Statistical comparisons
- Week 8: Platform A/B testing
- Week 10: Social media mix contribution

**Sample Analysis Ideas**:
- Calculate engagement rates by platform
- Identify top-performing content types
- Compare platform effectiveness
- Analyze audience growth patterns

**Download Command**:
```bash
kaggle datasets download -d ramjasmaurya/top-1000-social-media-channels --unzip \
  --path ./learning-modules/resources/kaggle-datasets/social-media/
```

---

### Dataset 11: Superstore Sales Dataset

**Kaggle URL**: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

**Description**:
- **Size**: ~1MB, 10k retail transactions
- Product, customer, sales data
- Geographic and temporal information

**What You Can Learn**:
- Retail analytics fundamentals
- Product performance analysis
- Geographic segmentation
- Discount impact analysis

**Relevant Weeks**:
- Week 2-3: Data manipulation and SQL
- Week 5-6: Retail metrics and viz
- Week 7: Discount effectiveness testing
- Week 10: Product mix analysis

**Sample Analysis Ideas**:
- Analyze sales trends by region
- Evaluate discount strategies
- Identify top product categories
- Calculate customer segments by value

**Download Command**:
```bash
kaggle datasets download -d vivek468/superstore-dataset-final --unzip \
  --path ./learning-modules/resources/kaggle-datasets/superstore/
```

---

### Dataset 12: Instagram Influencer Analysis

**Kaggle URL**: https://www.kaggle.com/datasets/surajjha101/social-media-influencer-data

**Description**:
- **Size**: <1MB, influencer profiles
- Engagement metrics, follower counts
- Content categories

**What You Can Learn**:
- Influencer marketing metrics
- Engagement analysis
- ROI estimation for influencer campaigns
- Platform-specific performance

**Relevant Weeks**:
- Week 5-6: Influencer metrics
- Week 7: Engagement statistical analysis
- Week 11: Incrementality of influencer marketing
- Week 12: LTV of influencer-acquired customers

**Sample Analysis Ideas**:
- Calculate cost per engagement
- Analyze engagement rate distributions
- Compare micro vs macro influencers
- Estimate influencer marketing ROI

**Download Command**:
```bash
kaggle datasets download -d surajjha101/social-media-influencer-data --unzip \
  --path ./learning-modules/resources/kaggle-datasets/instagram/
```

---

### Dataset 13: SEO and Website Traffic

**Kaggle URL**: https://www.kaggle.com/datasets/konradb/seo-keyword-research-dataset

**Description**:
- **Size**: ~10MB, SEO keyword data
- Search volume, competition, CPC
- Keyword performance metrics

**What You Can Learn**:
- SEO analytics fundamentals
- Keyword research and analysis
- Organic vs paid search comparison
- Content optimization strategies

**Relevant Weeks**:
- Week 5-6: SEO metrics analysis
- Week 9: Organic channel attribution
- Week 10: SEO in marketing mix
- Week 11: Incrementality of SEO

**Sample Analysis Ideas**:
- Analyze keyword difficulty vs opportunity
- Calculate estimated traffic value
- Compare paid vs organic strategies
- Identify content gap opportunities

**Download Command**:
```bash
kaggle datasets download -d konradb/seo-keyword-research-dataset --unzip \
  --path ./learning-modules/resources/kaggle-datasets/seo/
```

---

### Dataset 14: Retail Basket Analysis

**Kaggle URL**: https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis

**Description**:
- **Size**: ~15MB, transaction data
- Product baskets and combinations
- Store and temporal information

**What You Can Learn**:
- Market basket analysis
- Association rules (Apriori algorithm)
- Cross-sell opportunities
- Product bundling strategies

**Relevant Weeks**:
- Week 2: Data transformation
- Week 5-6: Basket composition analysis
- Week 7: Statistical product associations
- Week 10: Product mix optimization

**Sample Analysis Ideas**:
- Identify frequently bought together items
- Calculate lift for product pairs
- Recommend product bundles
- Optimize store layout based on associations

**Download Command**:
```bash
kaggle datasets download -d aslanahmedov/market-basket-analysis --unzip \
  --path ./learning-modules/resources/kaggle-datasets/basket-analysis/
```

---

### Dataset 15: Customer Lifetime Value Dataset

**Kaggle URL**: https://www.kaggle.com/datasets/uttamp/customer-lifetime-value-prediction

**Description**:
- **Size**: ~1MB, customer transaction history
- Purchase frequency, monetary value, recency
- Customer demographics and behavior

**What You Can Learn**:
- CLV calculation methods
- Cohort analysis
- Retention rate calculation
- Customer value segmentation

**Relevant Weeks**:
- Week 5: CLV exploratory analysis
- Week 7: Statistical customer behavior
- Week 11: Incrementality impact on CLV
- Week 12: CLV capstone (CRITICAL)

**Sample Analysis Ideas**:
- Calculate historical CLV
- Build predictive CLV model
- Segment customers by lifetime value
- Analyze cohort retention patterns

**Download Command**:
```bash
kaggle datasets download -d uttamp/customer-lifetime-value-prediction --unzip \
  --path ./learning-modules/resources/kaggle-datasets/clv/
```

---

## 4. Loading and Exploring Datasets

### Code Examples for Loading with Pandas

#### Basic CSV Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Define base path
BASE_PATH = Path('./learning-modules/resources/kaggle-datasets/')

# Load a single CSV file
df = pd.read_csv(BASE_PATH / 'online-retail/data.csv')

# Load with specific encoding (common for international data)
df = pd.read_csv(
    BASE_PATH / 'online-retail/data.csv',
    encoding='ISO-8859-1'
)

# Load with date parsing
df = pd.read_csv(
    BASE_PATH / 'online-retail/data.csv',
    parse_dates=['InvoiceDate'],
    encoding='ISO-8859-1'
)

# Load large files in chunks
chunk_iter = pd.read_csv(
    BASE_PATH / 'large-dataset/data.csv',
    chunksize=10000
)

# Process chunks
results = []
for chunk in chunk_iter:
    # Process each chunk
    processed = chunk.groupby('category')['sales'].sum()
    results.append(processed)

# Combine results
final_result = pd.concat(results).groupby(level=0).sum()
```

#### Loading Multiple Related Tables

```python
# Example: Olist E-commerce dataset with multiple tables
olist_path = BASE_PATH / 'olist/'

# Load all tables
orders = pd.read_csv(olist_path / 'olist_orders_dataset.csv')
order_items = pd.read_csv(olist_path / 'olist_order_items_dataset.csv')
customers = pd.read_csv(olist_path / 'olist_customers_dataset.csv')
products = pd.read_csv(olist_path / 'olist_products_dataset.csv')
sellers = pd.read_csv(olist_path / 'olist_sellers_dataset.csv')
payments = pd.read_csv(olist_path / 'olist_order_payments_dataset.csv')
reviews = pd.read_csv(olist_path / 'olist_order_reviews_dataset.csv')

# Parse dates across tables
date_columns = {
    'orders': ['order_purchase_timestamp', 'order_approved_at',
               'order_delivered_carrier_date', 'order_delivered_customer_date',
               'order_estimated_delivery_date'],
    'reviews': ['review_creation_date', 'review_answer_timestamp']
}

for col in date_columns['orders']:
    orders[col] = pd.to_datetime(orders[col])

for col in date_columns['reviews']:
    reviews[col] = pd.to_datetime(reviews[col])

# Create a data loading function
def load_olist_data(base_path):
    """
    Load and prepare all Olist e-commerce tables.

    Returns dict of DataFrames with parsed dates.
    """
    tables = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv'
    }

    data = {}
    for key, filename in tables.items():
        data[key] = pd.read_csv(base_path / filename)

    # Parse dates
    date_cols_orders = ['order_purchase_timestamp', 'order_approved_at',
                        'order_delivered_carrier_date',
                        'order_delivered_customer_date',
                        'order_estimated_delivery_date']

    for col in date_cols_orders:
        data['orders'][col] = pd.to_datetime(data['orders'][col])

    data['reviews']['review_creation_date'] = pd.to_datetime(
        data['reviews']['review_creation_date']
    )

    return data

# Use the function
olist_data = load_olist_data(olist_path)
```

### Initial Exploration Checklist

Use this systematic approach for every new dataset:

```python
def explore_dataset(df, name="Dataset"):
    """
    Comprehensive initial exploration of a dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to explore
    name : str
        Name of the dataset for reporting
    """
    print(f"{'='*60}")
    print(f"EXPLORATION REPORT: {name}")
    print(f"{'='*60}\n")

    # 1. Basic Information
    print("1. BASIC INFORMATION")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    # 2. Column Information
    print("2. COLUMN INFORMATION")
    print(df.dtypes)
    print()

    # 3. Missing Values
    print("3. MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percent': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    )

    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("   No missing values detected!")
    print()

    # 4. Duplicate Rows
    print("4. DUPLICATE ROWS")
    duplicates = df.duplicated().sum()
    print(f"   Total duplicates: {duplicates:,} ({duplicates/len(df)*100:.2f}%)\n")

    # 5. Numerical Columns Summary
    print("5. NUMERICAL COLUMNS SUMMARY")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("   No numerical columns found.")
    print()

    # 6. Categorical Columns Summary
    print("6. CATEGORICAL COLUMNS SUMMARY")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # Show first 5
        unique_count = df[col].nunique()
        print(f"   {col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"      Values: {df[col].value_counts().to_dict()}")
    print()

    # 7. Date Columns
    print("7. DATE COLUMNS")
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        for col in date_cols:
            print(f"   {col}:")
            print(f"      Range: {df[col].min()} to {df[col].max()}")
            print(f"      Missing: {df[col].isnull().sum()}")
    else:
        print("   No datetime columns found.")
    print()

    # 8. First Few Rows
    print("8. SAMPLE DATA (First 5 rows)")
    print(df.head())
    print()

    print(f"{'='*60}\n")

# Example usage
explore_dataset(df, "Online Retail Dataset")
```

### Data Quality Assessment

```python
def assess_data_quality(df, id_column=None):
    """
    Assess data quality issues and provide recommendations.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to assess
    id_column : str, optional
        Name of the ID/primary key column
    """
    issues = []
    recommendations = []

    # Check 1: Missing values
    missing_pct = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 50]

    if len(high_missing) > 0:
        issues.append(f"High missing values: {list(high_missing.index)}")
        recommendations.append(
            "Consider dropping columns with >50% missing or investigate why"
        )

    # Check 2: Duplicates
    if id_column and df[id_column].duplicated().any():
        issues.append(f"Duplicate IDs found in {id_column}")
        recommendations.append("Investigate and remove duplicate records")

    # Check 3: Data types
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Check if column should be datetime
        if any(word in col.lower() for word in ['date', 'time', 'timestamp']):
            issues.append(f"{col} appears to be a date but stored as object")
            recommendations.append(f"Convert {col} to datetime using pd.to_datetime()")

        # Check for numeric values stored as strings
        sample = df[col].dropna().head(100)
        if sample.apply(lambda x: str(x).replace('.','').replace('-','').isdigit()).mean() > 0.8:
            issues.append(f"{col} appears to contain numbers stored as strings")
            recommendations.append(f"Convert {col} to numeric using pd.to_numeric()")

    # Check 4: Outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
        outlier_pct = outliers / len(df) * 100

        if outlier_pct > 5:
            issues.append(f"{col} has {outliers} extreme outliers ({outlier_pct:.1f}%)")
            recommendations.append(
                f"Investigate outliers in {col} - may be data errors or true extremes"
            )

    # Check 5: Low cardinality
    for col in df.columns:
        if df[col].nunique() == 1:
            issues.append(f"{col} has only one unique value")
            recommendations.append(f"Consider removing {col} as it provides no information")

    # Print report
    print("DATA QUALITY ASSESSMENT")
    print("="*60)

    if issues:
        print(f"\n⚠️  {len(issues)} ISSUES FOUND:\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

        print(f"\n💡 RECOMMENDATIONS:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\n✓ No major data quality issues detected!")

    print("\n" + "="*60)

# Example usage
assess_data_quality(df, id_column='InvoiceNo')
```

### Adapting Datasets to Course Exercises

```python
def prepare_for_course(df, week_number):
    """
    Prepare dataset based on course week requirements.

    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset
    week_number : int
        Week number (1-12)

    Returns:
    --------
    pd.DataFrame
        Prepared dataset for the week
    """
    df_prepared = df.copy()

    if week_number <= 2:
        # Weeks 1-2: Basic Python/Pandas
        # Keep dataset simple, clean missing values
        df_prepared = df_prepared.dropna(thresh=len(df_prepared.columns) * 0.7, axis=1)
        df_prepared = df_prepared.fillna(method='ffill').fillna(0)

    elif week_number <= 4:
        # Weeks 3-4: SQL
        # Ensure proper data types for SQL-like operations
        for col in df_prepared.select_dtypes(include=['object']).columns:
            if 'date' in col.lower() or 'time' in col.lower():
                df_prepared[col] = pd.to_datetime(df_prepared[col], errors='coerce')

    elif week_number <= 6:
        # Weeks 5-6: EDA and Visualization
        # Calculate derived metrics if applicable
        if 'sales' in df_prepared.columns and 'quantity' in df_prepared.columns:
            df_prepared['avg_price'] = df_prepared['sales'] / df_prepared['quantity']

    elif week_number == 7:
        # Week 7: Statistics
        # Ensure numeric columns for statistical analysis
        numeric_cols = df_prepared.select_dtypes(include=[np.number]).columns
        df_prepared = df_prepared[numeric_cols]

    elif week_number == 8:
        # Week 8: A/B Testing
        # Create test/control groups if not present
        if 'group' not in df_prepared.columns:
            df_prepared['group'] = np.random.choice(['control', 'test'], size=len(df_prepared))

    elif week_number == 9:
        # Week 9: Attribution
        # Ensure timestamp and channel columns
        # (dataset-specific preparation)
        pass

    elif week_number >= 12:
        # Week 12: CLV
        # Ensure customer_id, purchase_date, revenue columns
        # (dataset-specific preparation)
        pass

    return df_prepared

# Example usage
df_week5 = prepare_for_course(df, week_number=5)
```

---

## 5. Integration with Course Modules

### Weeks 1-4: Foundations (Python, Pandas, SQL)

#### Week 1: Python Foundations
**Focus**: Basic Python operations on marketing data

**Recommended Datasets**:
- Superstore Sales (simple structure)
- Marketing Campaign (small, clean)

**Example Integration**:
```python
# Week_01_Python_Foundations.ipynb addition

# Load Superstore data
import pandas as pd

df = pd.read_csv('../resources/kaggle-datasets/superstore/Sample-Superstore.csv')

# Exercise: Calculate basic metrics using Python
orders = df['Order ID'].unique()
print(f"Total unique orders: {len(orders)}")

# Exercise: Filter data
high_value_orders = df[df['Sales'] > 1000]
print(f"Orders above $1000: {len(high_value_orders)}")

# Exercise: Group and aggregate
category_sales = {}
for category in df['Category'].unique():
    category_sales[category] = df[df['Category'] == category]['Sales'].sum()

print("Sales by Category:", category_sales)
```

#### Week 2: Pandas Data Manipulation
**Focus**: Data cleaning and transformation

**Recommended Datasets**:
- Online Retail Dataset (needs cleaning)
- Olist E-commerce (multiple tables)

**Example Integration**:
```python
# Week_02_Pandas_Data_Manipulation.ipynb addition

# Load Online Retail data
df = pd.read_csv(
    '../resources/kaggle-datasets/online-retail/data.csv',
    encoding='ISO-8859-1'
)

# Exercise: Handle missing values
print("Missing values before:")
print(df.isnull().sum())

# Remove rows where CustomerID is missing
df = df.dropna(subset=['CustomerID'])

# Exercise: Create new features
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['Month'] = df['InvoiceDate'].dt.month
df['Year'] = df['InvoiceDate'].dt.year

# Exercise: Remove invalid transactions
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

print(f"\nCleaned dataset: {len(df)} rows")

# Exercise: Aggregations
monthly_sales = df.groupby(['Year', 'Month'])['TotalPrice'].sum()
print("\nMonthly Sales:")
print(monthly_sales.head(10))
```

#### Week 3: SQL Basics
**Focus**: SQL queries on marketing data

**Recommended Datasets**:
- Olist E-commerce (relational structure)
- Google Analytics Sample

**Example Integration**:
```python
# Week_03_SQL_Basics.ipynb addition

import sqlite3
import pandas as pd

# Load Olist data into SQLite
conn = sqlite3.connect(':memory:')

# Load tables
orders = pd.read_csv('../resources/kaggle-datasets/olist/olist_orders_dataset.csv')
customers = pd.read_csv('../resources/kaggle-datasets/olist/olist_customers_dataset.csv')
order_items = pd.read_csv('../resources/kaggle-datasets/olist/olist_order_items_dataset.csv')

# Write to SQL
orders.to_sql('orders', conn, index=False, if_exists='replace')
customers.to_sql('customers', conn, index=False, if_exists='replace')
order_items.to_sql('order_items', conn, index=False, if_exists='replace')

# Exercise: Basic SELECT
query = """
SELECT
    customer_state,
    COUNT(DISTINCT customer_id) as customer_count
FROM customers
GROUP BY customer_state
ORDER BY customer_count DESC
LIMIT 10
"""
top_states = pd.read_sql(query, conn)
print("Top 10 States by Customer Count:")
print(top_states)

# Exercise: JOIN operations
query = """
SELECT
    o.order_id,
    o.order_status,
    c.customer_state,
    SUM(oi.price) as order_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_status = 'delivered'
GROUP BY o.order_id, o.order_status, c.customer_state
"""
order_details = pd.read_sql(query, conn)
print(f"\nOrder details: {len(order_details)} delivered orders")
```

#### Week 4: Advanced SQL
**Focus**: Window functions, CTEs, complex queries

**Recommended Datasets**:
- Olist E-commerce
- Brazilian E-commerce

**Example Integration**:
```python
# Week_04_SQL_Advanced.ipynb addition

# Exercise: Calculate running totals (window function)
query = """
WITH daily_orders AS (
    SELECT
        DATE(order_purchase_timestamp) as order_date,
        COUNT(*) as daily_order_count,
        SUM(oi.price) as daily_revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY DATE(order_purchase_timestamp)
)
SELECT
    order_date,
    daily_order_count,
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY order_date) as cumulative_revenue
FROM daily_orders
ORDER BY order_date
"""
cumulative = pd.read_sql(query, conn)
print("Cumulative Revenue Over Time:")
print(cumulative.head())

# Exercise: Customer cohort analysis
query = """
WITH first_purchase AS (
    SELECT
        customer_id,
        MIN(DATE(order_purchase_timestamp)) as cohort_date
    FROM orders
    GROUP BY customer_id
),
customer_orders AS (
    SELECT
        o.customer_id,
        fp.cohort_date,
        DATE(o.order_purchase_timestamp) as order_date,
        julianday(DATE(o.order_purchase_timestamp)) - julianday(fp.cohort_date) as days_since_first
    FROM orders o
    JOIN first_purchase fp ON o.customer_id = fp.customer_id
)
SELECT
    cohort_date,
    CAST(days_since_first / 30 AS INTEGER) as month_number,
    COUNT(DISTINCT customer_id) as customers
FROM customer_orders
GROUP BY cohort_date, month_number
ORDER BY cohort_date, month_number
LIMIT 50
"""
cohorts = pd.read_sql(query, conn)
print("\nCohort Analysis:")
print(cohorts.head(20))
```

---

### Weeks 5-7: Analysis and Visualization

#### Week 5: EDA Fundamentals
**Focus**: Exploratory analysis of marketing data

**Recommended Datasets**:
- All datasets work well
- Start with E-commerce datasets

**Example Integration**:
```python
# Week_05_EDA_Fundamentals.ipynb addition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
    '../resources/kaggle-datasets/online-retail/data.csv',
    encoding='ISO-8859-1',
    parse_dates=['InvoiceDate']
)

# Clean data
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Exercise: Distribution analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sales distribution
axes[0,0].hist(df['TotalPrice'], bins=50, edgecolor='black')
axes[0,0].set_xlabel('Transaction Value')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Distribution of Transaction Values')
axes[0,0].set_xlim(0, 100)  # Focus on typical range

# Log scale for better visibility
axes[0,1].hist(np.log10(df['TotalPrice']), bins=50, edgecolor='black')
axes[0,1].set_xlabel('Log10(Transaction Value)')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Distribution of Transaction Values (Log Scale)')

# Quantity distribution
axes[1,0].hist(df['Quantity'], bins=50, edgecolor='black')
axes[1,0].set_xlabel('Quantity')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Distribution of Order Quantities')
axes[1,0].set_xlim(0, 50)

# Unit price distribution
axes[1,1].hist(df['UnitPrice'], bins=50, edgecolor='black')
axes[1,1].set_xlabel('Unit Price')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Unit Prices')
axes[1,1].set_xlim(0, 20)

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Exercise: Calculate summary statistics
print("Summary Statistics:")
print(df[['Quantity', 'UnitPrice', 'TotalPrice']].describe())

# Exercise: Customer-level metrics
customer_metrics = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',  # Number of orders
    'TotalPrice': 'sum',     # Total spend
    'InvoiceDate': ['min', 'max']  # First and last purchase
}).reset_index()

customer_metrics.columns = ['CustomerID', 'Orders', 'TotalSpend', 'FirstPurchase', 'LastPurchase']
customer_metrics['AvgOrderValue'] = customer_metrics['TotalSpend'] / customer_metrics['Orders']

print("\nCustomer Metrics:")
print(customer_metrics.describe())
```

#### Week 6: Data Visualization
**Focus**: Creating marketing dashboards

**Recommended Datasets**:
- Facebook Ads
- Digital Advertising Campaign
- E-commerce datasets

**Example Integration**:
```python
# Week_06_Data_Visualization.ipynb addition

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Load advertising data
ads = pd.read_csv('../resources/kaggle-datasets/facebook-ads/KAG_conversion_data.csv')

# Exercise: Campaign performance dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Impressions by campaign
ax1 = fig.add_subplot(gs[0, :2])
campaign_impressions = ads.groupby('campaign_id')['Impressions'].sum().sort_values(ascending=False)
campaign_impressions.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Total Impressions by Campaign', fontsize=14, fontweight='bold')
ax1.set_xlabel('Campaign ID')
ax1.set_ylabel('Impressions')

# 2. CTR by campaign
ax2 = fig.add_subplot(gs[0, 2])
ads['CTR'] = ads['Clicks'] / ads['Impressions']
campaign_ctr = ads.groupby('campaign_id')['CTR'].mean().sort_values(ascending=False)
campaign_ctr.plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('Avg CTR by Campaign', fontsize=12, fontweight='bold')
ax2.set_xlabel('CTR')

# 3. Conversion funnel
ax3 = fig.add_subplot(gs[1, 0])
funnel_data = {
    'Stage': ['Impressions', 'Clicks', 'Conversions'],
    'Count': [
        ads['Impressions'].sum(),
        ads['Clicks'].sum(),
        ads['Total_Conversion'].sum()
    ]
}
funnel_df = pd.DataFrame(funnel_data)
ax3.bar(funnel_df['Stage'], funnel_df['Count'], color=['#3498db', '#2ecc71', '#e74c3c'])
ax3.set_title('Conversion Funnel', fontsize=12, fontweight='bold')
ax3.set_ylabel('Count')
for i, v in enumerate(funnel_df['Count']):
    ax3.text(i, v, f'{v:,.0f}', ha='center', va='bottom')

# 4. Cost metrics
ax4 = fig.add_subplot(gs[1, 1:])
campaign_costs = ads.groupby('campaign_id').agg({
    'Spent': 'sum',
    'Total_Conversion': 'sum'
}).reset_index()
campaign_costs['CPA'] = campaign_costs['Spent'] / campaign_costs['Total_Conversion']

x = np.arange(len(campaign_costs))
width = 0.35

ax4_twin = ax4.twinx()
bars1 = ax4.bar(x - width/2, campaign_costs['Spent'], width, label='Total Spend', color='#3498db')
bars2 = ax4_twin.bar(x + width/2, campaign_costs['CPA'], width, label='CPA', color='#e74c3c')

ax4.set_xlabel('Campaign ID')
ax4.set_ylabel('Total Spend ($)', color='#3498db')
ax4_twin.set_ylabel('Cost Per Acquisition ($)', color='#e74c3c')
ax4.set_title('Spend and CPA by Campaign', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(campaign_costs['campaign_id'])
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')

# 5. Daily performance
ax5 = fig.add_subplot(gs[2, :])
# Assuming there's a date column, otherwise create sample dates
if 'date' not in ads.columns:
    ads['date'] = pd.date_range(start='2020-01-01', periods=len(ads), freq='H')
else:
    ads['date'] = pd.to_datetime(ads['date'])

daily_perf = ads.groupby(ads['date'].dt.date).agg({
    'Impressions': 'sum',
    'Clicks': 'sum',
    'Total_Conversion': 'sum',
    'Spent': 'sum'
}).reset_index()

ax5.plot(daily_perf['date'], daily_perf['Impressions'], label='Impressions', marker='o')
ax5_twin = ax5.twinx()
ax5_twin.plot(daily_perf['date'], daily_perf['Clicks'], label='Clicks', color='green', marker='s')
ax5.set_xlabel('Date')
ax5.set_ylabel('Impressions', color='blue')
ax5_twin.set_ylabel('Clicks', color='green')
ax5.set_title('Daily Performance Trends', fontsize=12, fontweight='bold')
ax5.legend(loc='upper left')
ax5_twin.legend(loc='upper right')
plt.xticks(rotation=45)

plt.suptitle('Facebook Advertising Campaign Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('campaign_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Week 7: Statistics Foundations
**Focus**: Statistical analysis for marketing

**Recommended Datasets**:
- Marketing Campaign (response analysis)
- Telco Churn (hypothesis testing)
- Digital Advertising (significance testing)

**Example Integration**:
```python
# Week_07_Statistics_Foundations.ipynb addition

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load marketing campaign data
campaign = pd.read_csv('../resources/kaggle-datasets/marketing-campaign/marketing_campaign.csv', sep='\t')

# Exercise: Compare response rates across customer segments
# Clean data
campaign = campaign.dropna()
campaign['TotalSpent'] = (
    campaign['MntWines'] + campaign['MntFruits'] +
    campaign['MntMeatProducts'] + campaign['MntFishProducts'] +
    campaign['MntSweetProducts'] + campaign['MntGoldProds']
)

# Create age groups
campaign['Age'] = 2024 - campaign['Year_Birth']
campaign['AgeGroup'] = pd.cut(
    campaign['Age'],
    bins=[0, 35, 50, 65, 100],
    labels=['18-35', '36-50', '51-65', '65+']
)

# Exercise: Hypothesis test - Do different age groups respond differently?
response_by_age = campaign.groupby('AgeGroup')['Response'].agg(['mean', 'count', 'sum'])
print("Response Rate by Age Group:")
print(response_by_age)

# Chi-square test
contingency_table = pd.crosstab(campaign['AgeGroup'], campaign['Response'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-square test:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("Result: REJECT null hypothesis - Age groups have significantly different response rates")
else:
    print("Result: FAIL TO REJECT null hypothesis - No significant difference")

# Exercise: T-test - Do responders spend more than non-responders?
responders = campaign[campaign['Response'] == 1]['TotalSpent']
non_responders = campaign[campaign['Response'] == 0]['TotalSpent']

t_stat, t_p_value = stats.ttest_ind(responders, non_responders)

print(f"\n\nT-test: Spending comparison")
print(f"Responders mean spend: ${responders.mean():.2f}")
print(f"Non-responders mean spend: ${non_responders.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {t_p_value:.4f}")

if t_p_value < 0.05:
    print("Result: REJECT null hypothesis - Significant spending difference")
else:
    print("Result: FAIL TO REJECT null hypothesis - No significant difference")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot
campaign.boxplot(column='TotalSpent', by='Response', ax=axes[0])
axes[0].set_title('Spending Distribution by Response')
axes[0].set_xlabel('Response (0=No, 1=Yes)')
axes[0].set_ylabel('Total Spent ($)')

# Histogram
axes[1].hist(responders, alpha=0.5, label='Responders', bins=30)
axes[1].hist(non_responders, alpha=0.5, label='Non-Responders', bins=30)
axes[1].set_xlabel('Total Spent ($)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Spending Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig('statistical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Weeks 8-12: Advanced Marketing Measurement

#### Week 8: A/B Testing
**Focus**: Campaign experimentation

**Recommended Datasets**:
- Digital Advertising Campaign
- Email Campaign Performance
- Facebook Ads

**Example Integration**:
```python
# Week_08_AB_Testing.ipynb addition

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load advertising data
ads = pd.read_csv('../resources/kaggle-datasets/ad-campaigns/clicks_conversion.csv')

# Create A/B test scenario
# Let's say campaigns 1-2 are control, 3-4 are test
ads['test_group'] = ads['campaign_id'].apply(
    lambda x: 'Control' if x in [1, 2] else 'Test'
)

# Exercise: Calculate key metrics
test_results = ads.groupby('test_group').agg({
    'Impressions': 'sum',
    'Clicks': 'sum',
    'Conversions': 'sum',
    'Cost': 'sum'
}).reset_index()

test_results['CTR'] = test_results['Clicks'] / test_results['Impressions']
test_results['CVR'] = test_results['Conversions'] / test_results['Clicks']
test_results['CPC'] = test_results['Cost'] / test_results['Clicks']
test_results['CPA'] = test_results['Cost'] / test_results['Conversions']

print("A/B Test Results:")
print(test_results)

# Exercise: Statistical significance test for CTR
control = ads[ads['test_group'] == 'Control']
test = ads[ads['test_group'] == 'Test']

control_clicks = control['Clicks'].sum()
control_impressions = control['Impressions'].sum()
test_clicks = test['Clicks'].sum()
test_impressions = test['Impressions'].sum()

# Z-test for proportions
p1 = control_clicks / control_impressions
p2 = test_clicks / test_impressions
p_pool = (control_clicks + test_clicks) / (control_impressions + test_impressions)

se = np.sqrt(p_pool * (1 - p_pool) * (1/control_impressions + 1/test_impressions))
z_score = (p2 - p1) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f"\n\nZ-test for CTR difference:")
print(f"Control CTR: {p1:.4%}")
print(f"Test CTR: {p2:.4%}")
print(f"Lift: {(p2-p1)/p1:.2%}")
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: STATISTICALLY SIGNIFICANT difference")
else:
    print("Result: NOT statistically significant")

# Exercise: Calculate required sample size for future tests
def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """
    Calculate required sample size per group.

    Parameters:
    -----------
    baseline_rate : float
        Current conversion/click rate
    mde : float
        Minimum detectable effect (e.g., 0.10 for 10% relative lift)
    alpha : float
        Significance level (default 0.05)
    power : float
        Statistical power (default 0.80)
    """
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    # Expected test rate
    test_rate = baseline_rate * (1 + mde)

    # Pooled standard deviation
    p_avg = (baseline_rate + test_rate) / 2

    # Sample size calculation
    n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta)**2) / (test_rate - baseline_rate)**2

    return int(np.ceil(n))

baseline_ctr = p1
mde = 0.10  # Want to detect 10% lift

sample_size = calculate_sample_size(baseline_ctr, mde)
print(f"\n\nSample Size Calculation:")
print(f"To detect a {mde:.0%} lift with 80% power:")
print(f"Required sample size per group: {sample_size:,} impressions")
```

#### Week 9: Attribution Modeling
**Focus**: Multi-touch attribution

**Recommended Datasets**:
- Multi-Touch Attribution Dataset (PRIMARY)
- Google Analytics Sample

**Example Integration**:
```python
# Week_09_Attribution_Modeling.ipynb addition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load attribution data
attribution = pd.read_csv(
    '../resources/kaggle-datasets/attribution/attribution_data.csv'
)

# Data structure: user_id, timestamp, channel, conversion (1/0)

# Exercise: Implement different attribution models

def first_touch_attribution(df):
    """First touch attribution: 100% credit to first interaction"""
    first_touch = df.groupby('user_id').first().reset_index()
    conversions = first_touch[first_touch['conversion'] == 1]
    attribution = conversions['channel'].value_counts()
    return attribution

def last_touch_attribution(df):
    """Last touch attribution: 100% credit to last interaction"""
    last_touch = df.groupby('user_id').last().reset_index()
    conversions = last_touch[last_touch['conversion'] == 1]
    attribution = conversions['channel'].value_counts()
    return attribution

def linear_attribution(df):
    """Linear attribution: Equal credit to all touchpoints"""
    # Get converters
    converters = df[df['conversion'] == 1]['user_id'].unique()

    # Get all touchpoints for converters
    converter_paths = df[df['user_id'].isin(converters)]

    # Count touchpoints per user
    touchpoint_counts = converter_paths.groupby('user_id').size()

    # Assign equal credit
    attribution = {}
    for user_id, count in touchpoint_counts.items():
        user_touches = converter_paths[converter_paths['user_id'] == user_id]
        credit_per_touch = 1 / count

        for channel in user_touches['channel']:
            attribution[channel] = attribution.get(channel, 0) + credit_per_touch

    return pd.Series(attribution).sort_values(ascending=False)

def time_decay_attribution(df, half_life_days=7):
    """Time decay attribution: More recent touchpoints get more credit"""
    converters = df[df['conversion'] == 1]['user_id'].unique()
    converter_paths = df[df['user_id'].isin(converters)].copy()

    # Ensure timestamp is datetime
    converter_paths['timestamp'] = pd.to_datetime(converter_paths['timestamp'])

    # Get conversion timestamp for each user
    conversion_times = converter_paths[converter_paths['conversion'] == 1].groupby('user_id')['timestamp'].max()

    attribution = {}

    for user_id in converters:
        user_touches = converter_paths[converter_paths['user_id'] == user_id].copy()
        conversion_time = conversion_times[user_id]

        # Calculate days before conversion
        user_touches['days_before'] = (conversion_time - user_touches['timestamp']).dt.total_seconds() / 86400

        # Time decay weight (exponential decay)
        user_touches['weight'] = np.exp(-user_touches['days_before'] * np.log(2) / half_life_days)

        # Normalize weights
        total_weight = user_touches['weight'].sum()
        user_touches['credit'] = user_touches['weight'] / total_weight

        # Assign credit
        for _, row in user_touches.iterrows():
            attribution[row['channel']] = attribution.get(row['channel'], 0) + row['credit']

    return pd.Series(attribution).sort_values(ascending=False)

# Calculate all attribution models
first_touch = first_touch_attribution(attribution)
last_touch = last_touch_attribution(attribution)
linear = linear_attribution(attribution)
time_decay = time_decay_attribution(attribution)

# Create comparison DataFrame
attribution_comparison = pd.DataFrame({
    'First Touch': first_touch,
    'Last Touch': last_touch,
    'Linear': linear,
    'Time Decay': time_decay
}).fillna(0)

print("Attribution Model Comparison:")
print(attribution_comparison)

# Normalize to percentages
attribution_pct = attribution_comparison.div(attribution_comparison.sum(axis=0), axis=1) * 100

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = ['First Touch', 'Last Touch', 'Linear', 'Time Decay']
for idx, model in enumerate(models):
    ax = axes[idx // 2, idx % 2]
    attribution_pct[model].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'{model} Attribution', fontweight='bold')
    ax.set_ylabel('Attribution %')
    ax.set_xlabel('Channel')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('attribution_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate attribution shift
attribution_shift = attribution_pct['Last Touch'] - attribution_pct['First Touch']
print("\n\nAttribution Shift (Last Touch - First Touch):")
print(attribution_shift.sort_values(ascending=False))
```

#### Week 10: Marketing Mix Modeling
**Focus**: Channel contribution analysis

**Recommended Datasets**:
- Facebook Ads
- Digital Advertising Campaign
- E-commerce data with multiple channels

**Example Integration**:
```python
# Week_10_Marketing_Mix_Modeling.ipynb addition

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulate or load marketing mix data
# For demo, let's create synthetic data
np.random.seed(42)
weeks = 52

mmm_data = pd.DataFrame({
    'week': range(1, weeks + 1),
    'tv_spend': np.random.uniform(5000, 15000, weeks),
    'digital_spend': np.random.uniform(3000, 10000, weeks),
    'social_spend': np.random.uniform(2000, 8000, weeks),
    'email_spend': np.random.uniform(500, 2000, weeks),
    'seasonality': np.sin(np.linspace(0, 4*np.pi, weeks)) * 0.2 + 1
})

# Generate sales with known relationships (for teaching purposes)
mmm_data['sales'] = (
    50000 +  # Baseline
    0.8 * mmm_data['tv_spend'] +
    1.2 * mmm_data['digital_spend'] +
    0.9 * mmm_data['social_spend'] +
    1.5 * mmm_data['email_spend'] +
    20000 * mmm_data['seasonality'] +
    np.random.normal(0, 5000, weeks)  # Noise
)

# Exercise: Build basic MMM model

# Prepare features
features = ['tv_spend', 'digital_spend', 'social_spend', 'email_spend', 'seasonality']
X = mmm_data[features]
y = mmm_data['sales']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Get coefficients
coefficients = pd.DataFrame({
    'Channel': features,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("Marketing Mix Model Coefficients:")
print(coefficients)

print(f"\nModel R-squared: {model.score(X, y):.4f}")
print(f"Intercept (Baseline Sales): ${model.intercept_:,.2f}")

# Exercise: Calculate channel contribution
mmm_data['predicted_sales'] = model.predict(X)

# Calculate contribution of each channel
contributions = {}
for channel in ['tv_spend', 'digital_spend', 'social_spend', 'email_spend']:
    idx = features.index(channel)
    contributions[channel] = model.coef_[idx] * mmm_data[channel].sum()

total_contribution = sum(contributions.values())

contribution_df = pd.DataFrame({
    'Channel': contributions.keys(),
    'Total_Contribution': contributions.values()
})
contribution_df['Contribution_%'] = (
    contribution_df['Total_Contribution'] / total_contribution * 100
)
contribution_df = contribution_df.sort_values('Contribution_%', ascending=False)

print("\n\nChannel Contribution to Sales:")
print(contribution_df)

# Exercise: Calculate ROI by channel
spend_data = mmm_data[['tv_spend', 'digital_spend', 'social_spend', 'email_spend']].sum()
roi_df = pd.DataFrame({
    'Channel': spend_data.index,
    'Total_Spend': spend_data.values
})

roi_df['Sales_Contribution'] = roi_df['Channel'].map(contributions)
roi_df['ROI'] = (roi_df['Sales_Contribution'] - roi_df['Total_Spend']) / roi_df['Total_Spend']
roi_df['ROAS'] = roi_df['Sales_Contribution'] / roi_df['Total_Spend']

roi_df = roi_df.sort_values('ROAS', ascending=False)

print("\n\nROI by Channel:")
print(roi_df)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(mmm_data['sales'], mmm_data['predicted_sales'], alpha=0.5)
axes[0, 0].plot([mmm_data['sales'].min(), mmm_data['sales'].max()],
                [mmm_data['sales'].min(), mmm_data['sales'].max()],
                'r--', lw=2)
axes[0, 0].set_xlabel('Actual Sales')
axes[0, 0].set_ylabel('Predicted Sales')
axes[0, 0].set_title('Actual vs Predicted Sales')

# 2. Channel Contribution
axes[0, 1].bar(contribution_df['Channel'], contribution_df['Contribution_%'])
axes[0, 1].set_xlabel('Channel')
axes[0, 1].set_ylabel('Contribution %')
axes[0, 1].set_title('Channel Contribution to Sales')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. ROAS by Channel
axes[1, 0].barh(roi_df['Channel'], roi_df['ROAS'], color='green')
axes[1, 0].set_xlabel('ROAS')
axes[1, 0].set_ylabel('Channel')
axes[1, 0].set_title('Return on Ad Spend by Channel')
axes[1, 0].axvline(x=1, color='r', linestyle='--', label='Break-even')
axes[1, 0].legend()

# 4. Time series
axes[1, 1].plot(mmm_data['week'], mmm_data['sales'], label='Actual', marker='o')
axes[1, 1].plot(mmm_data['week'], mmm_data['predicted_sales'], label='Predicted', marker='s')
axes[1, 1].set_xlabel('Week')
axes[1, 1].set_ylabel('Sales')
axes[1, 1].set_title('Sales Over Time')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('mmm_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Week 11: Incrementality and Lift Testing
**Focus**: Measuring true impact

**Recommended Datasets**:
- Marketing Campaign (for lift)
- Telco Churn (for treatment effects)
- Any dataset where you can create test/control

**Example Integration**:
```python
# Week_11_Incrementality_Lift.ipynb addition

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load campaign data
campaign = pd.read_csv(
    '../resources/kaggle-datasets/marketing-campaign/marketing_campaign.csv',
    sep='\t'
)

# Clean and prepare
campaign = campaign.dropna()
campaign['TotalSpent'] = (
    campaign['MntWines'] + campaign['MntFruits'] +
    campaign['MntMeatProducts'] + campaign['MntFishProducts'] +
    campaign['MntSweetProducts'] + campaign['MntGoldProds']
)

# Create synthetic holdout group for incrementality test
np.random.seed(42)
campaign['holdout_group'] = np.random.choice(
    ['control', 'treatment'],
    size=len(campaign),
    p=[0.5, 0.5]
)

# Assume 'Response' only applies to treatment group
# For control group, they didn't receive campaign
campaign.loc[campaign['holdout_group'] == 'control', 'Response'] = 0

# Exercise: Calculate incrementality

incrementality_analysis = campaign.groupby('holdout_group').agg({
    'Response': ['sum', 'mean'],
    'customer_id': 'count'
})

incrementality_analysis.columns = ['Conversions', 'Conversion_Rate', 'Customers']
incrementality_analysis = incrementality_analysis.reset_index()

print("Incrementality Analysis:")
print(incrementality_analysis)

# Calculate lift
treatment_rate = incrementality_analysis[
    incrementality_analysis['holdout_group'] == 'treatment'
]['Conversion_Rate'].values[0]

control_rate = incrementality_analysis[
    incrementality_analysis['holdout_group'] == 'control'
]['Conversion_Rate'].values[0]

absolute_lift = treatment_rate - control_rate
relative_lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else np.inf

print(f"\nLift Metrics:")
print(f"Treatment conversion rate: {treatment_rate:.2%}")
print(f"Control conversion rate: {control_rate:.2%}")
print(f"Absolute lift: {absolute_lift:.2%}")
print(f"Relative lift: {relative_lift:.2%}")

# Statistical significance test
treatment_conversions = incrementality_analysis[
    incrementality_analysis['holdout_group'] == 'treatment'
]['Conversions'].values[0]
treatment_total = incrementality_analysis[
    incrementality_analysis['holdout_group'] == 'treatment'
]['Customers'].values[0]

control_conversions = incrementality_analysis[
    incrementality_analysis['holdout_group'] == 'control'
]['Conversions'].values[0]
control_total = incrementality_analysis[
    incrementality_analysis['holdout_group'] == 'control'
]['Customers'].values[0]

# Z-test for proportions
p_pool = (treatment_conversions + control_conversions) / (treatment_total + control_total)
se = np.sqrt(p_pool * (1 - p_pool) * (1/treatment_total + 1/control_total))
z_score = (treatment_rate - control_rate) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f"\nStatistical Significance:")
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Lift is STATISTICALLY SIGNIFICANT")
else:
    print("Result: Lift is NOT statistically significant")

# Exercise: Calculate incremental revenue
treatment_revenue = campaign[campaign['holdout_group'] == 'treatment']['TotalSpent'].sum()
control_revenue = campaign[campaign['holdout_group'] == 'control']['TotalSpent'].sum()

# Normalize by group size
treatment_avg_revenue = treatment_revenue / treatment_total
control_avg_revenue = control_revenue / control_total

incremental_revenue_per_customer = treatment_avg_revenue - control_avg_revenue
total_incremental_revenue = incremental_revenue_per_customer * treatment_total

print(f"\n\nIncremental Revenue Analysis:")
print(f"Treatment avg revenue: ${treatment_avg_revenue:,.2f}")
print(f"Control avg revenue: ${control_avg_revenue:,.2f}")
print(f"Incremental revenue per customer: ${incremental_revenue_per_customer:,.2f}")
print(f"Total incremental revenue: ${total_incremental_revenue:,.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Conversion rate comparison
axes[0, 0].bar(['Control', 'Treatment'],
               [control_rate, treatment_rate],
               color=['#e74c3c', '#2ecc71'])
axes[0, 0].set_ylabel('Conversion Rate')
axes[0, 0].set_title('Conversion Rate: Control vs Treatment')
axes[0, 0].axhline(y=control_rate, color='r', linestyle='--', alpha=0.5)

# Add labels
for i, (label, value) in enumerate([('Control', control_rate), ('Treatment', treatment_rate)]):
    axes[0, 0].text(i, value, f'{value:.2%}', ha='center', va='bottom')

# 2. Revenue comparison
axes[0, 1].bar(['Control', 'Treatment'],
               [control_avg_revenue, treatment_avg_revenue],
               color=['#e74c3c', '#2ecc71'])
axes[0, 1].set_ylabel('Avg Revenue per Customer ($)')
axes[0, 1].set_title('Revenue: Control vs Treatment')

# 3. Distribution of spending
campaign.boxplot(column='TotalSpent', by='holdout_group', ax=axes[1, 0])
axes[1, 0].set_title('Spending Distribution by Group')
axes[1, 0].set_xlabel('Group')
axes[1, 0].set_ylabel('Total Spent ($)')

# 4. Lift visualization
axes[1, 1].barh(['Absolute Lift', 'Relative Lift'],
                [absolute_lift * 100, relative_lift * 100],
                color='steelblue')
axes[1, 1].set_xlabel('Lift (%)')
axes[1, 1].set_title('Campaign Lift Metrics')

plt.suptitle('Incrementality Test Results', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('incrementality_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Week 12: Customer Lifetime Value (Capstone)
**Focus**: CLV calculation and optimization

**Recommended Datasets**:
- CLV Prediction Dataset (PRIMARY)
- Online Retail Dataset
- Olist E-commerce

**Example Integration**:
```python
# Week_12_CLV_Capstone.ipynb addition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Load Online Retail data
df = pd.read_csv(
    '../resources/kaggle-datasets/online-retail/data.csv',
    encoding='ISO-8859-1',
    parse_dates=['InvoiceDate']
)

# Clean data
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int)

# Set analysis date
analysis_date = df['InvoiceDate'].max()

# Exercise: Calculate RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("RFM Summary:")
print(rfm.describe())

# Exercise: Calculate Historical CLV
customer_history = df.groupby('CustomerID').agg({
    'InvoiceDate': ['min', 'max'],
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

customer_history.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase',
                             'TotalOrders', 'TotalRevenue']

# Calculate customer age in days
customer_history['CustomerAge_Days'] = (
    customer_history['LastPurchase'] - customer_history['FirstPurchase']
).dt.days

# Avoid division by zero
customer_history['CustomerAge_Days'] = customer_history['CustomerAge_Days'].replace(0, 1)

# Calculate metrics
customer_history['AvgOrderValue'] = (
    customer_history['TotalRevenue'] / customer_history['TotalOrders']
)

customer_history['PurchaseFrequency'] = (
    customer_history['TotalOrders'] / customer_history['CustomerAge_Days'] * 365
)

# Historical CLV (actual value delivered so far)
customer_history['HistoricalCLV'] = customer_history['TotalRevenue']

print("\n\nCustomer Metrics:")
print(customer_history[['CustomerID', 'TotalOrders', 'TotalRevenue',
                         'AvgOrderValue', 'PurchaseFrequency', 'HistoricalCLV']].head(10))

# Exercise: Predict Future CLV (simple method)
# Using historical patterns to project forward

# Calculate average customer lifespan
avg_lifespan_days = customer_history['CustomerAge_Days'].median()
avg_lifespan_years = avg_lifespan_days / 365

# Calculate retention rate (simplified)
# Customers who purchased in last 6 months vs total
recent_cutoff = analysis_date - timedelta(days=180)
active_customers = df[df['InvoiceDate'] >= recent_cutoff]['CustomerID'].nunique()
total_customers = df['CustomerID'].nunique()
retention_rate = active_customers / total_customers

print(f"\n\nCohort Metrics:")
print(f"Average customer lifespan: {avg_lifespan_years:.2f} years")
print(f"6-month retention rate: {retention_rate:.2%}")

# Predictive CLV (simplified formula)
# CLV = (Avg Order Value × Purchase Frequency × Customer Lifespan)

# For customers with enough history (>1 purchase)
active_customers_df = customer_history[customer_history['TotalOrders'] > 1].copy()

active_customers_df['PredictedLifespan_Years'] = avg_lifespan_years
active_customers_df['PredictedCLV'] = (
    active_customers_df['AvgOrderValue'] *
    active_customers_df['PurchaseFrequency'] *
    active_customers_df['PredictedLifespan_Years']
)

# Apply retention rate
active_customers_df['PredictedCLV_Adjusted'] = (
    active_customers_df['PredictedCLV'] * retention_rate
)

print("\n\nPredicted CLV (Top 10 Customers):")
print(active_customers_df.nlargest(10, 'PredictedCLV_Adjusted')[
    ['CustomerID', 'AvgOrderValue', 'PurchaseFrequency',
     'PredictedCLV_Adjusted']
])

# Exercise: Customer Segmentation by CLV
# Segment into quintiles
active_customers_df['CLV_Segment'] = pd.qcut(
    active_customers_df['PredictedCLV_Adjusted'],
    q=5,
    labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
)

segment_summary = active_customers_df.groupby('CLV_Segment').agg({
    'CustomerID': 'count',
    'TotalRevenue': 'sum',
    'AvgOrderValue': 'mean',
    'PurchaseFrequency': 'mean',
    'PredictedCLV_Adjusted': 'mean'
}).reset_index()

segment_summary.columns = ['Segment', 'CustomerCount', 'TotalRevenue',
                            'AvgOrderValue', 'AvgPurchaseFreq', 'AvgPredictedCLV']

print("\n\nCLV Segmentation:")
print(segment_summary)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. CLV Distribution
axes[0, 0].hist(active_customers_df['PredictedCLV_Adjusted'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Predicted CLV ($)')
axes[0, 0].set_ylabel('Number of Customers')
axes[0, 0].set_title('Distribution of Predicted CLV')
axes[0, 0].set_xlim(0, active_customers_df['PredictedCLV_Adjusted'].quantile(0.95))

# 2. CLV by Segment
segment_summary.plot(x='Segment', y='AvgPredictedCLV', kind='bar', ax=axes[0, 1],
                     color='steelblue', legend=False)
axes[0, 1].set_ylabel('Avg Predicted CLV ($)')
axes[0, 1].set_title('Average CLV by Segment')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Customer Count by Segment
segment_summary.plot(x='Segment', y='CustomerCount', kind='bar', ax=axes[0, 2],
                     color='coral', legend=False)
axes[0, 2].set_ylabel('Number of Customers')
axes[0, 2].set_title('Customers by Segment')
axes[0, 2].tick_params(axis='x', rotation=45)

# 4. Recency vs Monetary (RFM)
scatter = axes[1, 0].scatter(rfm['Recency'], rfm['Monetary'],
                             alpha=0.5, c=rfm['Frequency'], cmap='viridis')
axes[1, 0].set_xlabel('Recency (days)')
axes[1, 0].set_ylabel('Monetary ($)')
axes[1, 0].set_title('RFM Analysis: Recency vs Monetary')
axes[1, 0].set_xlim(0, rfm['Recency'].quantile(0.95))
axes[1, 0].set_ylim(0, rfm['Monetary'].quantile(0.95))
plt.colorbar(scatter, ax=axes[1, 0], label='Frequency')

# 5. Purchase Frequency Distribution
axes[1, 1].hist(active_customers_df['PurchaseFrequency'], bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Annual Purchase Frequency')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].set_title('Distribution of Purchase Frequency')
axes[1, 1].set_xlim(0, active_customers_df['PurchaseFrequency'].quantile(0.95))

# 6. Revenue Concentration
top_20_pct = active_customers_df.nlargest(int(len(active_customers_df) * 0.2),
                                           'TotalRevenue')
revenue_concentration = {
    'Top 20%': top_20_pct['TotalRevenue'].sum(),
    'Bottom 80%': active_customers_df['TotalRevenue'].sum() - top_20_pct['TotalRevenue'].sum()
}

axes[1, 2].pie(revenue_concentration.values(), labels=revenue_concentration.keys(),
               autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
axes[1, 2].set_title('Revenue Concentration (Pareto Analysis)')

plt.suptitle('Customer Lifetime Value Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('clv_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Exercise: Calculate CAC payback period
# Assume average CAC (you would get this from your actual data)
assumed_cac = 50  # dollars

active_customers_df['CAC_Payback_Orders'] = assumed_cac / active_customers_df['AvgOrderValue']
active_customers_df['CAC_Payback_Days'] = (
    active_customers_df['CAC_Payback_Orders'] / active_customers_df['PurchaseFrequency'] * 365
)

print(f"\n\nCAC Payback Analysis (assuming ${assumed_cac} CAC):")
print(f"Median payback period: {active_customers_df['CAC_Payback_Days'].median():.0f} days")
print(f"Mean payback period: {active_customers_df['CAC_Payback_Days'].mean():.0f} days")

# LTV/CAC Ratio
active_customers_df['LTV_CAC_Ratio'] = active_customers_df['PredictedCLV_Adjusted'] / assumed_cac

print(f"\nMedian LTV/CAC Ratio: {active_customers_df['LTV_CAC_Ratio'].median():.2f}")
print(f"Mean LTV/CAC Ratio: {active_customers_df['LTV_CAC_Ratio'].mean():.2f}")

# Healthy ratio is typically 3:1 or higher
healthy_ratio_customers = (active_customers_df['LTV_CAC_Ratio'] >= 3).sum()
print(f"\nCustomers with healthy LTV/CAC (≥3): {healthy_ratio_customers} ({healthy_ratio_customers/len(active_customers_df)*100:.1f}%)")
```

---

## 6. Best Practices

### Dataset Selection Criteria

When choosing a Kaggle dataset for learning, evaluate it based on:

#### 1. **Data Quality**
- Minimal missing values (or well-documented missingness)
- Consistent formatting
- Clear column names and definitions
- Reasonable data types

```python
def evaluate_dataset_quality(df, name="Dataset"):
    """
    Quick quality check for potential learning datasets.
    """
    score = 0
    max_score = 5
    issues = []

    # 1. Missing values (max 20%)
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_pct <= 0.20:
        score += 1
    else:
        issues.append(f"High missing values: {missing_pct:.1%}")

    # 2. Documentation (column names)
    if not any(col.startswith('Unnamed') for col in df.columns):
        score += 1
    else:
        issues.append("Unnamed columns present")

    # 3. Size (not too small, not too large)
    if 1000 <= len(df) <= 1000000:
        score += 1
    else:
        issues.append(f"Size concern: {len(df)} rows")

    # 4. Variety of data types
    type_count = len(df.dtypes.unique())
    if type_count >= 2:
        score += 1
    else:
        issues.append("Limited data type variety")

    # 5. Low duplication
    dup_pct = df.duplicated().sum() / len(df)
    if dup_pct <= 0.10:
        score += 1
    else:
        issues.append(f"High duplication: {dup_pct:.1%}")

    print(f"\n{name} Quality Score: {score}/{max_score}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")

    return score >= 3  # Acceptable if >= 3/5

# Example usage
is_good = evaluate_dataset_quality(df, "Online Retail Dataset")
```

#### 2. **Relevance to Marketing**
- Contains marketing-relevant metrics (sales, conversions, campaigns, etc.)
- Represents realistic business scenarios
- Includes temporal data for time-series analysis
- Has customer/transaction identifiers

#### 3. **Size and Scope**
- **For beginners (Weeks 1-4)**: 1K - 100K rows
- **For intermediate (Weeks 5-8)**: 10K - 500K rows
- **For advanced (Weeks 9-12)**: 100K - 1M+ rows
- Not so large that it requires special hardware

#### 4. **Educational Value**
- Can demonstrate multiple concepts
- Has interesting patterns to discover
- Allows for progressive complexity
- Supports various analysis types

### Data Cleaning Considerations

#### Standard Cleaning Workflow

```python
def clean_marketing_dataset(df):
    """
    Standard cleaning workflow for marketing datasets.
    """
    print(f"Original shape: {df.shape}")

    # 1. Remove completely empty rows/columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    print(f"After removing empty rows/cols: {df.shape}")

    # 2. Handle date columns
    date_keywords = ['date', 'time', 'timestamp', 'dt']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            if df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"Converted {col} to datetime")

    # 3. Fix data types
    for col in df.select_dtypes(include=['object']).columns:
        # Check if it's actually numeric
        try:
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            if numeric_values.notna().sum() / len(df) > 0.8:  # If 80%+ can be converted
                df[col] = numeric_values
                print(f"Converted {col} to numeric")
        except:
            pass

    # 4. Remove invalid rows (common issues)
    if 'Quantity' in df.columns:
        before = len(df)
        df = df[df['Quantity'] > 0]
        print(f"Removed {before - len(df)} rows with negative/zero quantity")

    if 'UnitPrice' in df.columns or 'Price' in df.columns:
        price_col = 'UnitPrice' if 'UnitPrice' in df.columns else 'Price'
        before = len(df)
        df = df[df[price_col] > 0]
        print(f"Removed {before - len(df)} rows with negative/zero price")

    # 5. Handle missing values strategically
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]

    if len(missing_summary) > 0:
        print("\nMissing values to address:")
        print(missing_summary)

    print(f"\nFinal shape: {df.shape}")
    return df

# Example usage
df_clean = clean_marketing_dataset(df)
```

#### Handling Missing Values in Marketing Data

```python
def handle_missing_values(df):
    """
    Context-aware missing value handling for marketing data.
    """
    df = df.copy()

    # Customer IDs: Usually drop if missing (can't analyze without ID)
    id_cols = [col for col in df.columns if 'id' in col.lower() or col == 'CustomerID']
    for col in id_cols:
        if col in df.columns:
            before = len(df)
            df = df.dropna(subset=[col])
            print(f"Dropped {before - len(df)} rows missing {col}")

    # Categorical variables: Fill with 'Unknown' or mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)
            print(f"Filled {col} missing values with 'Unknown'")

    # Numerical variables: Context dependent
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            # Revenue/spend metrics: Fill with 0 (no transaction)
            if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'spend', 'cost', 'price']):
                df[col].fillna(0, inplace=True)
                print(f"Filled {col} with 0 (no transaction)")

            # Count metrics: Fill with 0
            elif any(keyword in col.lower() for keyword in ['count', 'quantity', 'clicks', 'impressions']):
                df[col].fillna(0, inplace=True)
                print(f"Filled {col} with 0 (no activity)")

            # Other metrics: Fill with median
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled {col} with median: {median_val}")

    return df

# Example usage
df_handled = handle_missing_values(df_clean)
```

### Ethical Use of Data

#### Key Principles

1. **Respect Privacy**
   - Even public datasets may contain sensitive information
   - Aggregate data when sharing insights
   - Don't attempt to de-anonymize data

2. **Proper Attribution**
   - Always credit the dataset creator
   - Link to the original Kaggle dataset
   - Mention any modifications you made

3. **License Compliance**
   - Check the dataset license on Kaggle
   - Common licenses: CC0 (public domain), CC BY (attribution required), CC BY-SA (share-alike)
   - Respect commercial use restrictions

4. **Data Integrity**
   - Don't manipulate data to support a predetermined conclusion
   - Document all transformations
   - Be transparent about limitations

#### Attribution Template

```markdown
## Data Attribution

This analysis uses the [Dataset Name] from Kaggle, created by [Author Name].

**Dataset URL**: [Kaggle URL]
**License**: [License Type]
**Last Updated**: [Date]

**Modifications Made**:
- Removed rows with missing CustomerID
- Converted date columns to datetime format
- Created derived metric: TotalPrice = Quantity × UnitPrice
- Filtered to transactions from 2020-2021

**Citation**:
[Author Name]. ([Year]). [Dataset Name] [Data set]. Kaggle. [URL]
```

### Avoiding Data Leakage in ML Projects

Data leakage occurs when information from outside the training dataset is used to create the model.

#### Common Leakage Sources in Marketing Data

1. **Temporal Leakage**
   - Using future information to predict the past
   - **Example**: Using purchase amount to predict purchase likelihood

```python
# WRONG: Using total spend (includes future purchases) to predict first purchase
df['will_purchase'] = (df['TotalSpend'] > 0).astype(int)  # LEAKAGE!

# CORRECT: Use only information available at prediction time
df['first_purchase_date'] = df.groupby('CustomerID')['PurchaseDate'].transform('min')
df['days_since_first'] = (df['PurchaseDate'] - df['first_purchase_date']).dt.days
# Predict second purchase using only first purchase data
```

2. **Target Leakage**
   - Features that are derived from or highly correlated with the target

```python
# WRONG: Using conversion-dependent metrics
features = ['clicks', 'conversions', 'cost_per_conversion']  # Last one is leakage!

# CORRECT: Only use pre-conversion data
features = ['clicks', 'impressions', 'ad_position', 'device_type']
```

3. **Test Set Leakage**
   - Preprocessing using statistics from the entire dataset

```python
# WRONG: Scaling using entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data
X_train, X_test = train_test_split(X_scaled)  # LEAKAGE!

# CORRECT: Fit scaler only on training data
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Apply to test
```

#### Leakage Prevention Checklist

```python
def check_for_leakage(df, target_col, feature_cols):
    """
    Check for potential data leakage issues.
    """
    issues = []

    # 1. Perfect correlation with target
    for col in feature_cols:
        if df[col].dtype in [np.number]:
            corr = df[col].corr(df[target_col])
            if abs(corr) > 0.95:
                issues.append(f"{col} has very high correlation with target: {corr:.3f}")

    # 2. Features that shouldn't be known at prediction time
    future_keywords = ['total', 'lifetime', 'cumulative', 'final', 'outcome']
    for col in feature_cols:
        if any(keyword in col.lower() for keyword in future_keywords):
            issues.append(f"{col} may contain future information")

    # 3. ID columns
    id_keywords = ['id', 'key', 'index']
    for col in feature_cols:
        if any(keyword in col.lower() for keyword in id_keywords):
            if df[col].nunique() == len(df):
                issues.append(f"{col} is a unique identifier (should not be a feature)")

    # Print report
    if issues:
        print("⚠️  POTENTIAL LEAKAGE ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("✓ No obvious leakage issues detected")

    return len(issues) == 0

# Example usage
features = ['Recency', 'Frequency', 'AvgOrderValue']
target = 'WillChurn'
check_for_leakage(df, target, features)
```

---

## 7. Project Ideas

### Project 1: E-Commerce Customer Segmentation

**Dataset**: Online Retail or Olist

**Objective**: Segment customers based on purchasing behavior and develop targeted marketing strategies.

**Step-by-Step Plan**:

1. **Data Preparation** (Week 2 skills)
   - Load and clean transaction data
   - Handle missing CustomerIDs
   - Create TotalPrice column
   - Filter to valid transactions

2. **Feature Engineering** (Week 5 skills)
   - Calculate RFM metrics per customer
   - Add: total orders, avg order value, first/last purchase dates
   - Create time-based features (days since last purchase, customer tenure)

3. **Exploratory Analysis** (Week 5-6 skills)
   - Visualize RFM distributions
   - Identify outliers and patterns
   - Analyze seasonal purchasing trends

4. **Segmentation** (Week 7 skills)
   - Apply K-means clustering on RFM
   - Determine optimal clusters using elbow method
   - Profile each segment

5. **Strategy Development** (Week 11-12 skills)
   - Calculate CLV by segment
   - Recommend marketing tactics per segment
   - Estimate potential revenue impact

**Expected Outcomes**:
- 4-6 distinct customer segments
- Segment profiles with characteristics
- Marketing strategy recommendations
- ROI projections

**Code Starter**:
```python
# Load data
df = pd.read_csv('data.csv', parse_dates=['InvoiceDate'])

# Clean
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Calculate RFM
analysis_date = df['InvoiceDate'].max() + timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Normalize for clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# K-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

# Analyze segments
segment_profile = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
})

print(segment_profile)
```

---

### Project 2: Multi-Channel Attribution Analysis

**Dataset**: Multi-Touch Attribution or Google Analytics

**Objective**: Compare attribution models and optimize marketing budget allocation.

**Step-by-Step Plan**:

1. **Data Understanding** (Week 4 skills)
   - Map customer touchpoint journeys
   - Identify all marketing channels
   - Understand conversion paths

2. **Implement Attribution Models** (Week 9 skills)
   - First touch, last touch
   - Linear attribution
   - Time decay
   - Position-based (if advanced)

3. **Compare Models** (Week 7 skills)
   - Calculate channel value under each model
   - Visualize attribution shifts
   - Test statistical differences

4. **Budget Optimization** (Week 10 skills)
   - Calculate channel ROI
   - Identify over/under-invested channels
   - Simulate budget reallocation scenarios

5. **Recommendations** (Week 10 skills)
   - Propose optimal budget split
   - Estimate revenue impact
   - Create implementation roadmap

**Expected Outcomes**:
- Attribution comparison showing channel value under different models
- Budget reallocation recommendations
- Expected revenue lift from optimization

---

### Project 3: A/B Test Analysis Dashboard

**Dataset**: Digital Advertising or Facebook Ads

**Objective**: Analyze campaign A/B test and build automated decision framework.

**Step-by-Step Plan**:

1. **Test Design** (Week 8 skills)
   - Define test and control groups
   - Select primary and secondary metrics
   - Calculate required sample size

2. **Data Analysis** (Week 7-8 skills)
   - Calculate conversion rates
   - Perform significance testing
   - Check for novelty effects

3. **Visualization** (Week 6 skills)
   - Create performance dashboard
   - Show confidence intervals
   - Visualize statistical power

4. **Decision Framework** (Week 8 skills)
   - Build automated decision logic
   - Consider practical significance
   - Document recommendation

**Expected Outcomes**:
- Clear test winner (if significant)
- Statistical confidence levels
- Implementation recommendation

---

### Project 4: Customer Churn Prediction

**Dataset**: Telco Churn

**Objective**: Predict customer churn and develop retention strategies.

**Step-by-Step Plan**:

1. **Exploratory Analysis** (Week 5 skills)
   - Analyze churn rate overall and by segment
   - Identify churn drivers
   - Visualize patterns

2. **Feature Engineering** (Week 2 skills)
   - Create derived features (tenure groups, service counts)
   - Encode categorical variables
   - Handle missing values

3. **Predictive Modeling** (Week 7 skills)
   - Build logistic regression model
   - Evaluate model performance
   - Identify key predictors

4. **Strategy Development** (Week 11 skills)
   - Segment by churn risk
   - Calculate retention program ROI
   - Prioritize outreach

**Expected Outcomes**:
- Churn probability scores
- Key churn drivers identified
- Targeted retention strategy

---

### Project 5: Marketing Mix Modeling

**Dataset**: Facebook Ads + Create synthetic data for other channels

**Objective**: Quantify impact of each marketing channel on sales.

**Step-by-Step Plan**:

1. **Data Preparation** (Week 3-4 skills)
   - Aggregate spend by channel and week
   - Add sales/revenue data
   - Include seasonality indicators

2. **Model Building** (Week 10 skills)
   - Build linear regression model
   - Test for multicollinearity
   - Validate assumptions

3. **Channel Contribution** (Week 10 skills)
   - Calculate attribution weights
   - Estimate incremental sales
   - Compute ROAS by channel

4. **Optimization** (Week 10 skills)
   - Identify diminishing returns
   - Recommend budget shifts
   - Project revenue impact

**Expected Outcomes**:
- Channel contribution percentages
- ROAS by channel
- Optimized budget allocation

---

### Project 6: Customer Lifetime Value Optimization

**Dataset**: Online Retail or CLV Dataset

**Objective**: Calculate CLV and optimize acquisition spending.

**Step-by-Step Plan**:

1. **Historical CLV** (Week 12 skills)
   - Calculate RFM
   - Aggregate customer revenue
   - Analyze purchase patterns

2. **Predictive CLV** (Week 12 skills)
   - Estimate future purchase probability
   - Project customer lifespan
   - Calculate predicted CLV

3. **Segmentation** (Week 5-6 skills)
   - Segment by CLV quintile
   - Profile each segment
   - Identify growth opportunities

4. **CAC Optimization** (Week 12 skills)
   - Calculate acceptable CAC by segment
   - Compare to actual CAC
   - Recommend acquisition strategy

**Expected Outcomes**:
- CLV distribution
- Segment-specific CLV
- CAC/LTV targets by segment

---

### Project 7: Email Campaign Optimization

**Dataset**: Email Campaign Performance

**Objective**: Optimize email send strategy using A/B testing.

**Step-by-Step Plan**:

1. **Baseline Analysis** (Week 5 skills)
   - Calculate current open/click/conversion rates
   - Segment by customer type
   - Identify trends

2. **A/B Test Design** (Week 8 skills)
   - Test subject lines, send times, content
   - Define success metrics
   - Calculate sample size

3. **Analysis** (Week 8 skills)
   - Test statistical significance
   - Calculate lift
   - Visualize results

4. **Strategy** (Week 11 skills)
   - Estimate incrementality
   - Calculate ROI of optimization
   - Create rollout plan

**Expected Outcomes**:
- Optimized email elements
- Lift in key metrics
- ROI projection

---

### Project 8: Product Recommendation Engine

**Dataset**: Retail Basket Analysis or Online Retail

**Objective**: Build product recommendations based on purchase patterns.

**Step-by-Step Plan**:

1. **Market Basket Analysis** (Week 2-3 skills)
   - Identify frequently bought together items
   - Calculate support, confidence, lift
   - Find strong associations

2. **Collaborative Filtering** (Week 7 skills)
   - Build customer-product matrix
   - Calculate similarity metrics
   - Generate recommendations

3. **Evaluation** (Week 6 skills)
   - Test recommendation quality
   - Calculate potential revenue impact
   - Visualize product networks

4. **Implementation** (Week 10 skills)
   - Create recommendation logic
   - Estimate conversion lift
   - Calculate ROI

**Expected Outcomes**:
- Product association rules
- Recommendation algorithm
- Revenue impact estimate

---

### Project 9: Advertising Campaign Performance Dashboard

**Dataset**: Facebook Ads or Digital Advertising

**Objective**: Create interactive dashboard for campaign monitoring.

**Step-by-Step Plan**:

1. **Metric Definition** (Week 5 skills)
   - Define KPIs (CTR, CPC, CVR, ROAS)
   - Set benchmarks
   - Choose visualization types

2. **Data Pipeline** (Week 3-4 skills)
   - Automate data loading
   - Calculate derived metrics
   - Aggregate by campaign/date

3. **Visualization** (Week 6 skills)
   - Build time series charts
   - Create comparison views
   - Add interactivity (filters)

4. **Insights** (Week 8 skills)
   - Identify anomalies
   - Flag underperforming campaigns
   - Recommend actions

**Expected Outcomes**:
- Automated dashboard
- Key metric trends
- Actionable recommendations

---

### Project 10: Customer Journey Analysis

**Dataset**: Multi-Touch Attribution or Google Analytics

**Objective**: Map and optimize the customer conversion journey.

**Step-by-Step Plan**:

1. **Journey Mapping** (Week 4 skills)
   - Identify touchpoint sequences
   - Calculate path frequencies
   - Find common patterns

2. **Funnel Analysis** (Week 5 skills)
   - Build conversion funnels
   - Calculate drop-off rates
   - Identify bottlenecks

3. **Channel Analysis** (Week 9 skills)
   - Analyze channel sequences
   - Identify assist vs close channels
   - Calculate channel synergies

4. **Optimization** (Week 10-11 skills)
   - Recommend journey improvements
   - Estimate impact of changes
   - Prioritize optimizations

**Expected Outcomes**:
- Journey map visualizations
- Conversion funnel metrics
- Optimization roadmap

---

## Conclusion

This guide provides a comprehensive framework for using Kaggle datasets to enhance your marketing analytics learning. By combining real-world data with structured course modules, you'll develop practical skills that directly translate to professional marketing analytics roles.

**Next Steps**:

1. Set up your Kaggle API credentials
2. Download 2-3 datasets relevant to your current course week
3. Start with the code examples in Section 4
4. Work through the integration examples in Section 5
5. Choose a project from Section 7 to apply your skills

**Additional Resources**:

- Kaggle Learn: https://www.kaggle.com/learn
- Kaggle Notebooks: Browse notebooks on your chosen datasets for inspiration
- Course Slack: Share your Kaggle projects and get feedback

Remember: The best way to learn is by doing. Start exploring datasets today!

---

*Last Updated: November 2024*
*Course: Marketing Measurement Partner Academy*
*Location: `/learning-modules/resources/Kaggle_Datasets_Guide.md`*
