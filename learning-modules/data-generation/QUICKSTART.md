# Quick Start Guide - Data Generation

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Make scripts executable:**
   ```bash
   chmod +x generate_marketing_data.py
   chmod +x quick_generate.sh
   chmod +x example_usage.py
   ```

## Generate Your First Dataset

### Option 1: Use Quick Generate Script (Recommended)

```bash
# Generate a small test dataset
./quick_generate.sh test

# Generate a medium dataset for learning
./quick_generate.sh medium
```

### Option 2: Use Python Script Directly

```bash
# Generate small dataset (10K campaigns)
python3 generate_marketing_data.py --size small

# Generate medium dataset with custom output
python3 generate_marketing_data.py --size medium --output-dir ./my_data

# Generate custom size
python3 generate_marketing_data.py --rows 50000 --days 180
```

## Explore the Generated Data

### View Summary

```bash
cat output/data_summary.txt
```

### Query SQLite Database

```bash
sqlite3 output/marketing_data.db

# Example queries:
SELECT COUNT(*) FROM campaigns;
SELECT channel, SUM(spend) as total_spend FROM daily_performance 
  JOIN campaigns USING(campaign_id) GROUP BY channel;
```

### Analyze with Python

```bash
python3 example_usage.py
```

### View CSV Files

```bash
head -n 20 output/csv/campaigns.csv
head -n 20 output/csv/daily_performance.csv
```

## Next Steps

1. Read the full documentation: `README_Data_Generation.md`
2. Load data into your preferred analytics tool
3. Practice SQL queries and data analysis
4. Build dashboards and reports

## Common Use Cases

- **Learning SQL:** Use the SQLite database for practice
- **Testing Dashboards:** Load CSV files into your BI tool
- **Redshift Practice:** Use Redshift export format
- **Data Pipeline Development:** Test ETL processes

## Troubleshooting

- **Missing dependencies:** Run `pip install -r requirements.txt`
- **Out of memory:** Start with smaller datasets (`--size small`)
- **Slow generation:** Use faster storage (SSD) or smaller datasets

For more details, see `README_Data_Generation.md`
