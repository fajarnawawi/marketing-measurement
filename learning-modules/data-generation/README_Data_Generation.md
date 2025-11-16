# Marketing Measurement Synthetic Data Generator

## Overview

This directory contains scripts for generating realistic synthetic marketing data at scale for the **Marketing Measurement Partner Academy**. The generated data simulates real-world marketing campaigns, customer behavior, and attribution patterns.

### What Data is Generated

The generator creates **6 interconnected tables** that mirror a modern marketing data warehouse:

1. **campaigns** - Campaign metadata and configuration
2. **daily_performance** - Daily metrics and KPIs per campaign
3. **customers** - Customer demographics and attributes
4. **conversions** - Transaction-level conversion data
5. **touchpoints** - Multi-touch attribution journey data
6. **cohorts** - Cohort-based retention analysis

### Key Features

- **Realistic Distributions**: Data follows real-world patterns (log-normal spend, gamma-distributed campaign durations)
- **Channel-Specific Performance**: Each channel has unique CTR, CVR, and CPC characteristics
- **Seasonality**: Includes weekly patterns, monthly cycles, and holiday effects
- **Correlated Metrics**: Realistic relationships between spend, conversions, and revenue
- **Customer Journeys**: Multi-touch attribution with 1-8 touchpoints per customer
- **Configurable Scale**: Generate from 1K to 100M+ rows
- **Multiple Export Formats**: CSV, SQLite, and Redshift-compatible formats

---

## Quick Start

### Prerequisites

Install required Python packages:

```bash
pip install numpy pandas tqdm
```

### Basic Usage

```bash
# Generate small dataset (10K campaigns)
python generate_marketing_data.py --size small --output-dir ./data

# Generate medium dataset (100K campaigns) - default
python generate_marketing_data.py

# Generate specific number of campaigns
python generate_marketing_data.py --rows 50000 --output-dir ./my_data

# Generate with all export formats
python generate_marketing_data.py --size large --format csv,sqlite,redshift
```

### Using Pre-configured Scripts

We've included a bash script with common scenarios:

```bash
# Make it executable
chmod +x quick_generate.sh

# Generate small test dataset
./quick_generate.sh small

# Generate medium dataset for learning
./quick_generate.sh medium

# Generate large dataset for Redshift
./quick_generate.sh large

# Generate full production-scale dataset
./quick_generate.sh full
```

---

## Command-Line Arguments Reference

### Size Arguments

Choose between preset sizes or specify custom row counts:

| Argument | Description | Campaign Count |
|----------|-------------|----------------|
| `--size tiny` | Minimal dataset for quick tests | 1,000 |
| `--size small` | Small dataset for development | 10,000 |
| `--size medium` | Medium dataset for learning (default) | 100,000 |
| `--size large` | Large dataset for analysis | 1,000,000 |
| `--size xlarge` | Extra large for Redshift | 10,000,000 |
| `--size xxlarge` | Maximum size for production simulation | 100,000,000 |
| `--rows N` | Custom number of campaigns | N |

### Date Range

| Argument | Description | Default |
|----------|-------------|---------|
| `--days N` | Number of days of historical data | 365 |

**Example:**
```bash
# Generate 2 years of data
python generate_marketing_data.py --days 730
```

### Output Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir PATH` | Output directory path | ./output |
| `--format FORMATS` | Export formats (csv, sqlite, redshift) | csv,sqlite |
| `--compress` | Compress CSV files with gzip | False |

**Example:**
```bash
# Export to all formats with compression
python generate_marketing_data.py --format csv,sqlite,redshift --compress
```

### Generation Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed N` | Random seed for reproducibility | 42 |
| `--customers-multiplier X` | Customer count = campaigns × X | 50.0 |
| `--conversions-multiplier X` | Conversions = campaigns × X | 100.0 |
| `--touchpoints-multiplier X` | Touchpoints = campaigns × X | 200.0 |

**Example:**
```bash
# Generate more touchpoints per campaign
python generate_marketing_data.py --touchpoints-multiplier 500
```

---

## Data Schema Documentation

### Table: campaigns

Campaign metadata and configuration.

| Column | Type | Description |
|--------|------|-------------|
| campaign_id | VARCHAR | Unique campaign identifier (CMP_XXXXXX) |
| campaign_name | VARCHAR | Human-readable campaign name |
| channel | VARCHAR | Marketing channel (Google Ads, Facebook, etc.) |
| campaign_type | VARCHAR | Campaign objective (Brand Awareness, Conversion, etc.) |
| start_date | DATE | Campaign start date |
| end_date | DATE | Campaign end date |
| daily_budget | DECIMAL | Daily budget in USD |
| total_budget | DECIMAL | Total campaign budget |
| status | VARCHAR | Campaign status (Active, Paused, Completed) |
| created_at | TIMESTAMP | Campaign creation timestamp |

**Row Count:** N (specified by user)

**Sample Data:**
```
campaign_id,campaign_name,channel,campaign_type,start_date,end_date,daily_budget,total_budget,status
CMP_000001,Google Ads_Conversion_1,Google Ads,Conversion,2024-03-15,2024-05-14,1250.50,75030.00,Active
CMP_000002,Facebook Ads_Lead Generation_2,Facebook Ads,Lead Generation,2024-01-20,2024-03-05,890.25,40301.25,Completed
```

---

### Table: daily_performance

Daily performance metrics for each campaign.

| Column | Type | Description |
|--------|------|-------------|
| campaign_id | VARCHAR | Foreign key to campaigns table |
| date | DATE | Date of performance |
| impressions | BIGINT | Number of ad impressions |
| clicks | BIGINT | Number of clicks |
| conversions | BIGINT | Number of conversions |
| spend | DECIMAL | Daily spend in USD |
| revenue | DECIMAL | Revenue generated in USD |
| ctr | DECIMAL | Click-through rate (clicks/impressions) |
| cvr | DECIMAL | Conversion rate (conversions/clicks) |
| cpc | DECIMAL | Cost per click |
| cpa | DECIMAL | Cost per acquisition |
| roas | DECIMAL | Return on ad spend |

**Row Count:** N × avg_campaign_duration_days (typically N × 30-45)

**Sample Data:**
```
campaign_id,date,impressions,clicks,conversions,spend,revenue,ctr,cvr,cpc,cpa,roas
CMP_000001,2024-03-15,45230,1584,79,1250.50,6432.15,0.035,0.050,0.79,15.83,5.14
CMP_000001,2024-03-16,38940,1361,68,1189.22,5544.88,0.035,0.050,0.87,17.49,4.66
```

---

### Table: customers

Customer demographic and behavioral attributes.

| Column | Type | Description |
|--------|------|-------------|
| customer_id | VARCHAR | Unique customer identifier (CUST_XXXXXXXX) |
| acquisition_date | DATE | Date customer was acquired |
| country | VARCHAR | Customer country (US, UK, CA, etc.) |
| device_type | VARCHAR | Primary device (Desktop, Mobile, Tablet) |
| segment | VARCHAR | Customer segment (New, Returning, VIP, etc.) |
| age | INT | Customer age |
| gender | VARCHAR | Customer gender (M, F, Other) |
| ltv | DECIMAL | Customer lifetime value in USD |
| num_purchases | INT | Total number of purchases |
| avg_order_value | DECIMAL | Average order value |
| is_active | BOOLEAN | Whether customer is currently active |

**Row Count:** N × customers_multiplier (default: N × 50)

**Sample Data:**
```
customer_id,acquisition_date,country,device_type,segment,age,gender,ltv,num_purchases,avg_order_value,is_active
CUST_00000001,2024-05-12,US,Mobile,New,34,F,487.25,3,162.42,True
CUST_00000002,2024-02-28,UK,Desktop,Returning,45,M,1243.50,8,155.44,True
```

---

### Table: conversions

Transaction-level conversion data.

| Column | Type | Description |
|--------|------|-------------|
| conversion_id | VARCHAR | Unique conversion identifier (CONV_XXXXXXXXXX) |
| campaign_id | VARCHAR | Foreign key to campaigns table |
| customer_id | VARCHAR | Foreign key to customers table |
| conversion_date | DATE | Date of conversion |
| conversion_time | TIME | Time of conversion (HH:MM) |
| order_value | DECIMAL | Order value in USD |
| quantity | INT | Number of items purchased |
| product_category | VARCHAR | Product category |
| is_new_customer | BOOLEAN | Whether this is a new customer conversion |
| payment_method | VARCHAR | Payment method used |

**Row Count:** N × conversions_multiplier (default: N × 100)

**Sample Data:**
```
conversion_id,campaign_id,customer_id,conversion_date,conversion_time,order_value,quantity,product_category,is_new_customer,payment_method
CONV_0000000001,CMP_000015,CUST_00012345,2024-06-15,14:32,89.99,2,Electronics,False,Credit Card
CONV_0000000002,CMP_000087,CUST_00045678,2024-06-15,16:45,124.50,1,Clothing,True,PayPal
```

---

### Table: touchpoints

Multi-touch attribution data (customer journey).

| Column | Type | Description |
|--------|------|-------------|
| touchpoint_id | VARCHAR | Unique touchpoint identifier (TOUCH_XXXXXXXXXXXX) |
| customer_id | VARCHAR | Foreign key to customers table |
| campaign_id | VARCHAR | Foreign key to campaigns table |
| touchpoint_date | DATE | Date of touchpoint |
| touchpoint_time | TIME | Time of touchpoint (HH:MM) |
| position | INT | Position in customer journey (1, 2, 3, ...) |
| is_first_touch | BOOLEAN | Whether this is the first touchpoint |
| is_last_touch | BOOLEAN | Whether this is the last touchpoint |
| device_type | VARCHAR | Device used for this touchpoint |
| interaction_type | VARCHAR | Type of interaction (Click, View, Engagement) |
| time_to_next_touch | INT | Days to next touchpoint (NULL if last) |

**Row Count:** N × touchpoints_multiplier (default: N × 200)

**Sample Data:**
```
touchpoint_id,customer_id,campaign_id,touchpoint_date,touchpoint_time,position,is_first_touch,is_last_touch,device_type,interaction_type,time_to_next_touch
TOUCH_000000000001,CUST_00001234,CMP_000025,2024-05-10,09:15,1,True,False,Mobile,Click,3
TOUCH_000000000002,CUST_00001234,CMP_000031,2024-05-13,14:22,2,False,False,Desktop,View,2
TOUCH_000000000003,CUST_00001234,CMP_000025,2024-05-15,11:45,3,False,True,Mobile,Click,NULL
```

---

### Table: cohorts

Cohort-based retention analysis.

| Column | Type | Description |
|--------|------|-------------|
| cohort_week | VARCHAR | Cohort week (ISO week format: YYYY-Wxx) |
| week_number | INT | Weeks since cohort start (0, 1, 2, ...) |
| cohort_size | INT | Total customers in cohort |
| retained_customers | INT | Customers retained at this week |
| retention_rate | DECIMAL | Retention rate (retained/cohort_size) |
| revenue | DECIMAL | Revenue from retained customers |
| avg_revenue_per_customer | DECIMAL | Average revenue per retained customer |

**Row Count:** num_cohort_weeks × 52 (default: varies by customer acquisition dates)

**Sample Data:**
```
cohort_week,week_number,cohort_size,retained_customers,retention_rate,revenue,avg_revenue_per_customer
2024-W20,0,1234,1234,1.0000,45678.90,37.01
2024-W20,1,1234,1156,0.9368,42345.12,36.63
2024-W20,2,1234,1089,0.8825,38901.45,35.72
```

---

## Entity Relationship Diagram (ERD)

```
┌─────────────────┐
│   campaigns     │
│─────────────────│
│ campaign_id (PK)│◄─────────┐
│ campaign_name   │          │
│ channel         │          │
│ campaign_type   │          │
│ start_date      │          │
│ end_date        │          │
│ daily_budget    │          │
│ total_budget    │          │
└─────────────────┘          │
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        │                    │                    │
┌───────▼──────────┐  ┌──────▼────────┐  ┌───────▼──────────┐
│daily_performance │  │  conversions  │  │   touchpoints    │
│──────────────────│  │───────────────│  │──────────────────│
│ campaign_id (FK) │  │conversion_id  │  │ touchpoint_id    │
│ date             │  │campaign_id(FK)│  │ customer_id (FK) │
│ impressions      │  │customer_id(FK)│  │ campaign_id (FK) │
│ clicks           │  │conversion_date│  │ touchpoint_date  │
│ conversions      │  │order_value    │  │ position         │
│ spend            │  │quantity       │  │ is_first_touch   │
│ revenue          │  │product_cat.   │  │ is_last_touch    │
│ ctr, cvr, cpc... │  └───────────────┘  └──────────────────┘
└──────────────────┘          ▲                    ▲
                              │                    │
                              │                    │
                       ┌──────┴────────┐          │
                       │   customers   │◄─────────┘
                       │───────────────│
                       │customer_id(PK)│
                       │acquisition_dt │
                       │country        │
                       │device_type    │
                       │segment        │
                       │ltv            │
                       │num_purchases  │
                       └───────────────┘
                              ▲
                              │
                       ┌──────┴────────┐
                       │    cohorts    │
                       │───────────────│
                       │ cohort_week   │
                       │ week_number   │
                       │ cohort_size   │
                       │ retained_cust │
                       │ retention_rate│
                       └───────────────┘
```

---

## Data Volume Recommendations

### Development & Testing

**Size:** `tiny` or `small`
- **Campaigns:** 1K - 10K
- **Daily Performance:** 30K - 300K
- **Customers:** 50K - 500K
- **Conversions:** 100K - 1M
- **Total Rows:** ~200K - 2M
- **Disk Space:** ~50 MB - 500 MB
- **Generation Time:** 1-5 minutes

**Use Cases:**
- Local development
- Unit testing
- Quick iterations
- Learning SQL basics

### Learning & Training

**Size:** `medium`
- **Campaigns:** 100K
- **Daily Performance:** 3M
- **Customers:** 5M
- **Conversions:** 10M
- **Total Rows:** ~20M
- **Disk Space:** ~5 GB
- **Generation Time:** 10-30 minutes

**Use Cases:**
- SQL practice
- Dashboard development
- Local analytics
- Partner Academy exercises

### Production Simulation (Redshift)

**Size:** `large` or `xlarge`
- **Campaigns:** 1M - 10M
- **Daily Performance:** 30M - 300M
- **Customers:** 50M - 500M
- **Conversions:** 100M - 1B
- **Total Rows:** ~200M - 2B
- **Disk Space:** ~50 GB - 500 GB
- **Generation Time:** 1-10 hours

**Use Cases:**
- Redshift performance testing
- Query optimization
- Data pipeline development
- Advanced analytics training

### Enterprise Scale

**Size:** `xxlarge`
- **Campaigns:** 100M
- **Daily Performance:** 3B
- **Customers:** 5B
- **Conversions:** 10B
- **Total Rows:** ~20B
- **Disk Space:** ~5 TB
- **Generation Time:** 10-50+ hours

**Use Cases:**
- Extreme performance testing
- Distributed processing
- Big data frameworks (Spark, Presto)

---

## Performance Tips

### Speed Up Generation

1. **Use smaller multipliers** for initial testing:
   ```bash
   python generate_marketing_data.py --customers-multiplier 10 --conversions-multiplier 20
   ```

2. **Generate in batches** for very large datasets:
   ```bash
   # Generate 10 batches of 10M campaigns each
   for i in {1..10}; do
       python generate_marketing_data.py --rows 10000000 --seed $i --output-dir ./batch_$i
   done
   ```

3. **Skip unused tables** by modifying the script (comment out table generation)

4. **Use SSD storage** for faster I/O

### Optimize Memory Usage

For datasets larger than available RAM:

1. **Use chunking** (modify script to write in chunks)
2. **Export only to CSV** (skip SQLite for very large data)
3. **Enable compression** to reduce disk usage:
   ```bash
   python generate_marketing_data.py --compress
   ```

### Redshift Loading Tips

1. **Upload to S3** before loading:
   ```bash
   aws s3 sync ./output/redshift/ s3://your-bucket/marketing-data/
   ```

2. **Use COPY command** for optimal performance (see generated SQL file)

3. **Enable compression** in Redshift table definitions

4. **Set proper distribution and sort keys**:
   ```sql
   -- Add to create_tables.sql
   CREATE TABLE daily_performance (
       ...
   )
   DISTSTYLE KEY
   DISTKEY(campaign_id)
   SORTKEY(date, campaign_id);
   ```

---

## Customization Guide

### Adding New Channels

Edit `CHANNELS` and `CHANNEL_CHARACTERISTICS` in the script:

```python
CHANNELS = [
    'Google Ads',
    'Facebook Ads',
    # Add your channel:
    'Snapchat Ads',
]

CHANNEL_CHARACTERISTICS = {
    # ... existing channels ...
    'Snapchat Ads': {
        'ctr_mean': 0.030,
        'ctr_std': 0.012,
        'cvr_mean': 0.035,
        'cvr_std': 0.014,
        'cpc_mean': 1.40,
        'cpc_std': 0.70
    },
}
```

### Adjusting Seasonality

Modify the `apply_seasonality()` function:

```python
def apply_seasonality(base_value: float, date: datetime) -> float:
    # Your custom seasonality logic
    # Example: Add Black Friday boost
    if date.month == 11 and date.day >= 24:
        return base_value * 2.0  # Double on Black Friday weekend

    # Existing logic...
```

### Changing Distributions

Modify distribution parameters in generation functions:

```python
# Change customer LTV distribution
ltv = np.random.lognormal(mean=6, sigma=1.5)  # Higher LTV

# Change campaign duration
duration = int(np.random.gamma(shape=3, scale=20))  # Longer campaigns
```

### Adding Custom Tables

1. Create a new generation method:
   ```python
   def generate_my_table(self) -> pd.DataFrame:
       data = []
       for i in range(1000):
           data.append({
               'id': i,
               'value': np.random.random()
           })
       return pd.DataFrame(data)
   ```

2. Add to main() function:
   ```python
   tables['my_table'] = generator.generate_my_table()
   ```

---

## Output Structure

After running the generator, your output directory will contain:

```
output/
├── csv/                          # CSV exports
│   ├── campaigns.csv
│   ├── daily_performance.csv
│   ├── customers.csv
│   ├── conversions.csv
│   ├── touchpoints.csv
│   └── cohorts.csv
├── redshift/                     # Redshift-compatible files
│   ├── create_tables.sql        # DDL statements
│   ├── campaigns.csv
│   ├── daily_performance.csv
│   └── ...
├── marketing_data.db            # SQLite database
├── data_summary.txt             # Data summary and statistics
└── data_generation.log          # Generation log file
```

---

## Troubleshooting

### Out of Memory Errors

**Problem:** Script crashes with `MemoryError`

**Solution:**
- Reduce dataset size: `--size medium` instead of `large`
- Lower multipliers: `--customers-multiplier 20`
- Export to CSV only: `--format csv`
- Use chunking (requires script modification)

### Slow Generation

**Problem:** Generation takes too long

**Solution:**
- Start with smaller sizes for testing
- Use SSD storage
- Disable progress bars (modify script: `disable=True` in tqdm)
- Run on a machine with more CPU cores

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
pip install numpy pandas tqdm
```

### Redshift COPY Errors

**Problem:** COPY command fails with delimiter errors

**Solution:**
- Check that pipe delimiter `|` is used
- Verify NULL handling in data
- Ensure proper S3 permissions

---

## Examples

### Example 1: Quick Test Dataset

Generate a small dataset for testing:

```bash
python generate_marketing_data.py \
    --size small \
    --days 90 \
    --output-dir ./test_data \
    --format csv
```

**Output:**
- 10,000 campaigns
- 90 days of data
- ~300K daily performance records
- CSV format only

### Example 2: Full Training Dataset

Generate a comprehensive dataset for Partner Academy training:

```bash
python generate_marketing_data.py \
    --size medium \
    --days 730 \
    --output-dir ./academy_data \
    --format csv,sqlite \
    --seed 12345
```

**Output:**
- 100,000 campaigns
- 2 years of data
- ~20M total records
- CSV + SQLite formats

### Example 3: Production-Scale Redshift Data

Generate large-scale data for Redshift:

```bash
python generate_marketing_data.py \
    --size xlarge \
    --days 365 \
    --output-dir ./redshift_data \
    --format csv,redshift \
    --compress \
    --customers-multiplier 100 \
    --conversions-multiplier 200
```

**Output:**
- 10M campaigns
- 1 year of data
- 2B+ total records
- Compressed CSV + Redshift DDL

### Example 4: Reproducible Dataset

Generate the same dataset across environments:

```bash
python generate_marketing_data.py \
    --rows 50000 \
    --days 365 \
    --seed 42 \
    --output-dir ./reproducible_data
```

Using the same `--seed` value ensures identical data generation.

---

## FAQ

### Q: How realistic is the generated data?

**A:** Very realistic! The data includes:
- Real-world channel performance characteristics
- Realistic distributions (log-normal, gamma)
- Seasonality patterns
- Correlated metrics (higher spend → more conversions)
- Customer journey patterns
- Missing data and outliers at realistic rates

### Q: Can I use this data in production?

**A:** The data is synthetic and designed for training purposes. Use it for:
- Development and testing
- Training and education
- Performance testing
- Demo environments

Do NOT use for:
- Actual business decisions
- Real customer analysis
- Production reporting

### Q: How long does generation take?

**A:** Approximate times on a modern laptop:
- tiny (1K): 30 seconds
- small (10K): 2-5 minutes
- medium (100K): 15-30 minutes
- large (1M): 2-4 hours
- xlarge (10M): 10-20 hours
- xxlarge (100M): 50+ hours

### Q: Can I stop and resume generation?

**A:** No, the current version doesn't support resume. For very large datasets:
- Generate in smaller batches
- Combine results afterward
- Or modify the script to add checkpointing

### Q: How do I query the SQLite database?

**A:**
```bash
sqlite3 output/marketing_data.db

# Example queries
SELECT COUNT(*) FROM campaigns;
SELECT channel, SUM(spend) FROM daily_performance JOIN campaigns USING(campaign_id) GROUP BY channel;
```

### Q: Can I modify the schema?

**A:** Yes! The script is open-source and well-documented. Modify:
- Column names
- Data types
- Distributions
- Add new columns
- Add new tables

---

## Support

For questions or issues:

1. Check this README
2. Review the script docstrings
3. Check the generation log: `data_generation.log`
4. Contact: marketing-measurement-academy@example.com

---

## License

MIT License - Free to use and modify for educational purposes.

## Contributors

Marketing Measurement Partner Academy Team

---

**Last Updated:** 2025-11-16
**Version:** 1.0.0
