#!/usr/bin/env python3
"""
Example Usage of Marketing Measurement Synthetic Data

This script demonstrates how to load and analyze the generated synthetic data.

Usage:
    python example_usage.py
"""

import sqlite3
from pathlib import Path

import pandas as pd


def load_from_sqlite(db_path: str) -> dict:
    """
    Load all tables from SQLite database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dictionary of table_name -> DataFrame
    """
    print(f"Loading data from {db_path}...")

    tables = {}
    with sqlite3.connect(db_path) as conn:
        # Get list of tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor.fetchall()]

        # Load each table
        for table_name in table_names:
            tables[table_name] = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            print(f"  Loaded {table_name}: {len(tables[table_name]):,} rows")

    return tables


def load_from_csv(csv_dir: str) -> dict:
    """
    Load all tables from CSV directory.

    Args:
        csv_dir: Path to directory containing CSV files

    Returns:
        Dictionary of table_name -> DataFrame
    """
    print(f"Loading data from {csv_dir}...")

    csv_path = Path(csv_dir)
    tables = {}

    for csv_file in csv_path.glob("*.csv*"):
        table_name = csv_file.stem.replace('.csv', '')
        tables[table_name] = pd.read_csv(csv_file)
        print(f"  Loaded {table_name}: {len(tables[table_name]):,} rows")

    return tables


def analyze_campaign_performance(tables: dict):
    """Analyze campaign performance across channels."""
    print("\n" + "=" * 80)
    print("Campaign Performance Analysis")
    print("=" * 80)

    # Join campaigns with daily performance
    campaigns = tables['campaigns']
    daily = tables['daily_performance']

    # Merge tables
    data = daily.merge(campaigns, on='campaign_id', how='left')

    # Analyze by channel
    print("\nPerformance by Channel:")
    print("-" * 80)

    channel_performance = data.groupby('channel').agg({
        'spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum',
        'clicks': 'sum',
        'impressions': 'sum',
    }).round(2)

    channel_performance['ROAS'] = (
        channel_performance['revenue'] / channel_performance['spend']
    ).round(2)

    channel_performance['CTR'] = (
        channel_performance['clicks'] / channel_performance['impressions'] * 100
    ).round(2)

    channel_performance = channel_performance.sort_values('spend', ascending=False)

    print(channel_performance.to_string())

    # Top performing campaigns
    print("\n\nTop 10 Campaigns by ROAS:")
    print("-" * 80)

    campaign_performance = data.groupby(['campaign_id', 'campaign_name', 'channel']).agg({
        'spend': 'sum',
        'revenue': 'sum',
    }).reset_index()

    campaign_performance['ROAS'] = (
        campaign_performance['revenue'] / campaign_performance['spend']
    ).round(2)

    top_campaigns = campaign_performance.nlargest(10, 'ROAS')
    print(top_campaigns.to_string(index=False))


def analyze_customer_segments(tables: dict):
    """Analyze customer segments."""
    print("\n" + "=" * 80)
    print("Customer Segment Analysis")
    print("=" * 80)

    customers = tables['customers']

    # Segment analysis
    print("\nCustomers by Segment:")
    print("-" * 80)

    segment_analysis = customers.groupby('segment').agg({
        'customer_id': 'count',
        'ltv': 'mean',
        'num_purchases': 'mean',
        'avg_order_value': 'mean',
    }).round(2)

    segment_analysis.columns = ['Count', 'Avg LTV', 'Avg Purchases', 'Avg Order Value']
    segment_analysis = segment_analysis.sort_values('Avg LTV', ascending=False)

    print(segment_analysis.to_string())

    # Device analysis
    print("\n\nCustomers by Device Type:")
    print("-" * 80)

    device_analysis = customers.groupby('device_type').agg({
        'customer_id': 'count',
        'ltv': 'mean',
    }).round(2)

    device_analysis.columns = ['Count', 'Avg LTV']
    print(device_analysis.to_string())


def analyze_attribution(tables: dict):
    """Analyze multi-touch attribution."""
    print("\n" + "=" * 80)
    print("Multi-Touch Attribution Analysis")
    print("=" * 80)

    touchpoints = tables['touchpoints']
    campaigns = tables['campaigns']

    # Merge touchpoints with campaigns
    data = touchpoints.merge(campaigns, on='campaign_id', how='left')

    # First-touch attribution
    print("\nFirst-Touch Attribution (by Channel):")
    print("-" * 80)

    first_touch = data[data['is_first_touch'] == 1].groupby('channel').agg({
        'touchpoint_id': 'count',
    }).reset_index()

    first_touch.columns = ['Channel', 'First Touch Count']
    first_touch = first_touch.sort_values('First Touch Count', ascending=False)
    print(first_touch.to_string(index=False))

    # Last-touch attribution
    print("\n\nLast-Touch Attribution (by Channel):")
    print("-" * 80)

    last_touch = data[data['is_last_touch'] == 1].groupby('channel').agg({
        'touchpoint_id': 'count',
    }).reset_index()

    last_touch.columns = ['Channel', 'Last Touch Count']
    last_touch = last_touch.sort_values('Last Touch Count', ascending=False)
    print(last_touch.to_string(index=False))

    # Average touchpoints per customer
    avg_touchpoints = touchpoints.groupby('customer_id').size().mean()
    print(f"\n\nAverage touchpoints per customer: {avg_touchpoints:.2f}")


def analyze_cohort_retention(tables: dict):
    """Analyze cohort retention."""
    print("\n" + "=" * 80)
    print("Cohort Retention Analysis")
    print("=" * 80)

    cohorts = tables['cohorts']

    # Get average retention by week
    print("\nAverage Retention by Week:")
    print("-" * 80)

    retention_by_week = cohorts.groupby('week_number').agg({
        'retention_rate': 'mean',
        'avg_revenue_per_customer': 'mean',
    }).head(12).round(4)

    retention_by_week.columns = ['Avg Retention Rate', 'Avg Revenue per Customer']
    print(retention_by_week.to_string())

    # Best performing cohorts
    print("\n\nTop 5 Cohorts by Week 4 Retention:")
    print("-" * 80)

    week4_cohorts = cohorts[cohorts['week_number'] == 4].nlargest(5, 'retention_rate')
    print(week4_cohorts[['cohort_week', 'cohort_size', 'retention_rate', 'revenue']].to_string(index=False))


def main():
    """Main function to demonstrate data analysis."""
    print("=" * 80)
    print("Marketing Measurement Synthetic Data - Example Usage")
    print("=" * 80)

    # Determine data location
    # Try SQLite first, fall back to CSV
    db_path = Path("./output/marketing_data.db")
    csv_dir = Path("./output/csv")

    if db_path.exists():
        tables = load_from_sqlite(str(db_path))
    elif csv_dir.exists():
        tables = load_from_csv(str(csv_dir))
    else:
        print("\nError: No data found!")
        print("Please generate data first:")
        print("  python generate_marketing_data.py --size small")
        return

    # Run analyses
    analyze_campaign_performance(tables)
    analyze_customer_segments(tables)
    analyze_attribution(tables)
    analyze_cohort_retention(tables)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
