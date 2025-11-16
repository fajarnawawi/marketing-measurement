#!/usr/bin/env python3
"""
Marketing Measurement Synthetic Data Generator

This script generates realistic synthetic marketing data at scale for the
Marketing Measurement Partner Academy. It creates multiple related tables
with realistic distributions, correlations, and seasonality patterns.

Usage:
    python generate_marketing_data.py --size medium --output-dir ./output
    python generate_marketing_data.py --rows 1000000 --format csv,sqlite
    python generate_marketing_data.py --help

Author: Marketing Measurement Partner Academy
License: MIT
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_generation.log')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration and Constants
# ============================================================================

PRESET_SIZES = {
    'tiny': 1_000,
    'small': 10_000,
    'medium': 100_000,
    'large': 1_000_000,
    'xlarge': 10_000_000,
    'xxlarge': 100_000_000,
}

CHANNELS = [
    'Google Ads',
    'Facebook Ads',
    'Instagram Ads',
    'LinkedIn Ads',
    'Twitter Ads',
    'TikTok Ads',
    'Email',
    'Display',
    'Affiliate',
    'Organic Search',
    'Organic Social',
    'Direct',
    'Referral',
]

CAMPAIGN_TYPES = [
    'Brand Awareness',
    'Lead Generation',
    'Conversion',
    'Retargeting',
    'Engagement',
    'Video Views',
    'App Install',
    'Store Visits',
]

DEVICE_TYPES = ['Desktop', 'Mobile', 'Tablet']
COUNTRIES = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR', 'IN', 'MX']
SEGMENTS = ['New', 'Returning', 'VIP', 'At-Risk', 'Churned']

# Channel-specific performance characteristics
CHANNEL_CHARACTERISTICS = {
    'Google Ads': {'ctr_mean': 0.035, 'ctr_std': 0.015, 'cvr_mean': 0.05, 'cvr_std': 0.02, 'cpc_mean': 2.50, 'cpc_std': 1.20},
    'Facebook Ads': {'ctr_mean': 0.025, 'ctr_std': 0.012, 'cvr_mean': 0.04, 'cvr_std': 0.015, 'cpc_mean': 1.80, 'cpc_std': 0.90},
    'Instagram Ads': {'ctr_mean': 0.028, 'ctr_std': 0.013, 'cvr_mean': 0.038, 'cvr_std': 0.014, 'cpc_mean': 1.90, 'cpc_std': 0.95},
    'LinkedIn Ads': {'ctr_mean': 0.022, 'ctr_std': 0.010, 'cvr_mean': 0.06, 'cvr_std': 0.025, 'cpc_mean': 5.50, 'cpc_std': 2.50},
    'Twitter Ads': {'ctr_mean': 0.020, 'ctr_std': 0.009, 'cvr_mean': 0.03, 'cvr_std': 0.012, 'cpc_mean': 1.50, 'cpc_std': 0.70},
    'TikTok Ads': {'ctr_mean': 0.032, 'ctr_std': 0.014, 'cvr_mean': 0.035, 'cvr_std': 0.013, 'cpc_mean': 1.20, 'cpc_std': 0.60},
    'Email': {'ctr_mean': 0.015, 'ctr_std': 0.008, 'cvr_mean': 0.08, 'cvr_std': 0.03, 'cpc_mean': 0.10, 'cpc_std': 0.05},
    'Display': {'ctr_mean': 0.008, 'ctr_std': 0.004, 'cvr_mean': 0.02, 'cvr_std': 0.008, 'cpc_mean': 0.80, 'cpc_std': 0.40},
    'Affiliate': {'ctr_mean': 0.012, 'ctr_std': 0.006, 'cvr_mean': 0.045, 'cvr_std': 0.018, 'cpc_mean': 1.00, 'cpc_std': 0.50},
    'Organic Search': {'ctr_mean': 0.040, 'ctr_std': 0.018, 'cvr_mean': 0.055, 'cvr_std': 0.022, 'cpc_mean': 0.00, 'cpc_std': 0.00},
    'Organic Social': {'ctr_mean': 0.018, 'ctr_std': 0.008, 'cvr_mean': 0.03, 'cvr_std': 0.012, 'cpc_mean': 0.00, 'cpc_std': 0.00},
    'Direct': {'ctr_mean': 0.050, 'ctr_std': 0.020, 'cvr_mean': 0.07, 'cvr_std': 0.028, 'cpc_mean': 0.00, 'cpc_std': 0.00},
    'Referral': {'ctr_mean': 0.030, 'ctr_std': 0.014, 'cvr_mean': 0.05, 'cvr_std': 0.02, 'cpc_mean': 0.00, 'cpc_std': 0.00},
}


# ============================================================================
# Utility Functions
# ============================================================================

def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def get_date_range(days: int = 365) -> Tuple[datetime, datetime]:
    """Get date range for data generation."""
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


def apply_seasonality(base_value: float, date: datetime) -> float:
    """
    Apply realistic seasonality patterns to a base value.

    Includes:
    - Weekly patterns (weekends vs weekdays)
    - Monthly patterns (month-end spikes)
    - Holiday seasons (Q4 boost, summer dip)
    """
    # Day of week effect (weekends are 70% of weekdays)
    dow_multiplier = 0.7 if date.weekday() >= 5 else 1.0

    # Month effect (Q4 boost, summer dip)
    month_multipliers = {
        1: 0.9, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.05, 6: 0.95,
        7: 0.85, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.25, 12: 1.4
    }
    month_multiplier = month_multipliers.get(date.month, 1.0)

    # End of month spike (last 3 days)
    eom_multiplier = 1.15 if date.day >= 28 else 1.0

    return base_value * dow_multiplier * month_multiplier * eom_multiplier


def add_realistic_noise(value: float, noise_level: float = 0.1) -> float:
    """Add realistic noise to a value."""
    noise = np.random.normal(0, noise_level * value)
    return max(0, value + noise)


def generate_correlated_metrics(
    n: int,
    base_metric: np.ndarray,
    correlation: float = 0.7,
    noise_level: float = 0.2
) -> np.ndarray:
    """Generate a metric correlated with a base metric."""
    correlated = correlation * base_metric + (1 - correlation) * np.random.randn(n)
    noise = np.random.normal(0, noise_level, n)
    return np.maximum(0, correlated + noise)


# ============================================================================
# Data Generation Classes
# ============================================================================

class MarketingDataGenerator:
    """Main class for generating synthetic marketing data."""

    def __init__(
        self,
        num_campaigns: int = 100,
        days: int = 365,
        seed: int = 42
    ):
        """
        Initialize the data generator.

        Args:
            num_campaigns: Number of campaigns to generate
            days: Number of days of historical data
            seed: Random seed for reproducibility
        """
        self.num_campaigns = num_campaigns
        self.days = days
        set_random_seed(seed)
        self.start_date, self.end_date = get_date_range(days)

        logger.info(f"Initialized generator: {num_campaigns} campaigns, {days} days")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")

    def generate_campaigns(self) -> pd.DataFrame:
        """
        Generate campaign metadata table.

        Returns:
            DataFrame with campaign information
        """
        logger.info(f"Generating {self.num_campaigns} campaigns...")

        campaigns = []
        for i in tqdm(range(self.num_campaigns), desc="Campaigns", disable=False):
            channel = np.random.choice(CHANNELS)
            campaign_type = np.random.choice(CAMPAIGN_TYPES)

            # Campaign start date (random within date range)
            campaign_start = self.start_date + timedelta(
                days=np.random.randint(0, max(1, self.days - 30))
            )

            # Campaign duration (7-90 days, with most being 30 days)
            duration = int(np.random.gamma(shape=2, scale=15))
            duration = min(max(duration, 7), 90)
            campaign_end = campaign_start + timedelta(days=duration)

            # Budget (realistic distribution)
            daily_budget = np.random.lognormal(mean=7, sigma=1.5)
            total_budget = daily_budget * duration

            campaigns.append({
                'campaign_id': f'CMP_{i+1:06d}',
                'campaign_name': f'{channel}_{campaign_type}_{i+1}',
                'channel': channel,
                'campaign_type': campaign_type,
                'start_date': campaign_start.date(),
                'end_date': campaign_end.date(),
                'daily_budget': round(daily_budget, 2),
                'total_budget': round(total_budget, 2),
                'status': np.random.choice(['Active', 'Paused', 'Completed'], p=[0.6, 0.1, 0.3]),
                'created_at': campaign_start - timedelta(days=np.random.randint(1, 7)),
            })

        df = pd.DataFrame(campaigns)
        logger.info(f"Generated {len(df)} campaigns")
        return df

    def generate_daily_performance(self, campaigns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate daily performance metrics for each campaign.

        Args:
            campaigns_df: DataFrame with campaign metadata

        Returns:
            DataFrame with daily performance metrics
        """
        logger.info("Generating daily performance data...")

        performance_data = []

        for _, campaign in tqdm(campaigns_df.iterrows(), total=len(campaigns_df), desc="Daily Performance"):
            campaign_id = campaign['campaign_id']
            channel = campaign['channel']
            daily_budget = campaign['daily_budget']

            # Get channel characteristics
            char = CHANNEL_CHARACTERISTICS[channel]

            # Generate daily data for campaign duration
            current_date = campaign['start_date']
            end_date = min(campaign['end_date'], self.end_date.date())

            while current_date <= end_date:
                current_datetime = datetime.combine(current_date, datetime.min.time())

                # Apply seasonality to budget
                adjusted_budget = apply_seasonality(daily_budget, current_datetime)
                spend = add_realistic_noise(adjusted_budget, noise_level=0.15)

                # Generate impressions based on spend and CPC
                base_cpc = np.random.normal(char['cpc_mean'], char['cpc_std'])
                base_cpc = max(0.01, base_cpc)

                if base_cpc > 0:
                    impressions = int(spend / base_cpc * 1000)  # Rough conversion
                else:
                    impressions = int(spend * 10000)  # For organic channels

                impressions = max(0, int(add_realistic_noise(impressions, 0.2)))

                # Generate clicks with realistic CTR
                base_ctr = np.random.normal(char['ctr_mean'], char['ctr_std'])
                base_ctr = max(0.001, min(0.2, base_ctr))
                seasonal_ctr = apply_seasonality(base_ctr, current_datetime)
                clicks = int(impressions * seasonal_ctr)
                clicks = max(0, clicks)

                # Generate conversions with realistic CVR
                base_cvr = np.random.normal(char['cvr_mean'], char['cvr_std'])
                base_cvr = max(0.001, min(0.3, base_cvr))
                seasonal_cvr = apply_seasonality(base_cvr, current_datetime)
                conversions = int(clicks * seasonal_cvr)
                conversions = max(0, conversions)

                # Calculate revenue (with correlation to conversions)
                if conversions > 0:
                    avg_order_value = np.random.lognormal(mean=4, sigma=0.8)
                    revenue = conversions * avg_order_value
                else:
                    revenue = 0

                # Calculate derived metrics
                ctr = clicks / impressions if impressions > 0 else 0
                cvr = conversions / clicks if clicks > 0 else 0
                cpc = spend / clicks if clicks > 0 else 0
                cpa = spend / conversions if conversions > 0 else 0
                roas = revenue / spend if spend > 0 else 0

                performance_data.append({
                    'campaign_id': campaign_id,
                    'date': current_date,
                    'impressions': impressions,
                    'clicks': clicks,
                    'conversions': conversions,
                    'spend': round(spend, 2),
                    'revenue': round(revenue, 2),
                    'ctr': round(ctr, 6),
                    'cvr': round(cvr, 6),
                    'cpc': round(cpc, 4),
                    'cpa': round(cpa, 2),
                    'roas': round(roas, 4),
                })

                current_date += timedelta(days=1)

        df = pd.DataFrame(performance_data)
        logger.info(f"Generated {len(df):,} daily performance records")
        return df

    def generate_customers(self, num_customers: int) -> pd.DataFrame:
        """
        Generate customer demographic and attribute data.

        Args:
            num_customers: Number of customers to generate

        Returns:
            DataFrame with customer data
        """
        logger.info(f"Generating {num_customers:,} customers...")

        customers = []

        for i in tqdm(range(num_customers), desc="Customers"):
            # Customer acquisition date
            acquisition_date = self.start_date + timedelta(
                days=np.random.randint(0, self.days)
            )

            # Customer lifetime value (log-normal distribution)
            ltv = np.random.lognormal(mean=5, sigma=1.2)

            # Number of purchases (correlated with LTV)
            num_purchases = max(1, int(ltv / 100) + np.random.poisson(2))

            # Age and demographics
            age = int(np.random.normal(40, 15))
            age = max(18, min(85, age))

            customers.append({
                'customer_id': f'CUST_{i+1:08d}',
                'acquisition_date': acquisition_date.date(),
                'country': np.random.choice(COUNTRIES, p=[0.5, 0.1, 0.08, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04]),
                'device_type': np.random.choice(DEVICE_TYPES, p=[0.4, 0.5, 0.1]),
                'segment': np.random.choice(SEGMENTS, p=[0.25, 0.35, 0.15, 0.15, 0.10]),
                'age': age,
                'gender': np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04]),
                'ltv': round(ltv, 2),
                'num_purchases': num_purchases,
                'avg_order_value': round(ltv / num_purchases, 2),
                'is_active': np.random.choice([True, False], p=[0.7, 0.3]),
            })

        df = pd.DataFrame(customers)
        logger.info(f"Generated {len(df):,} customers")
        return df

    def generate_conversions(
        self,
        campaigns_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        num_conversions: int
    ) -> pd.DataFrame:
        """
        Generate transaction-level conversion data.

        Args:
            campaigns_df: DataFrame with campaign metadata
            customers_df: DataFrame with customer data
            num_conversions: Number of conversions to generate

        Returns:
            DataFrame with conversion data
        """
        logger.info(f"Generating {num_conversions:,} conversions...")

        conversions = []

        campaign_ids = campaigns_df['campaign_id'].values
        customer_ids = customers_df['customer_id'].values

        for i in tqdm(range(num_conversions), desc="Conversions"):
            # Random conversion date within date range
            conversion_date = self.start_date + timedelta(
                days=np.random.randint(0, self.days)
            )

            # Order value (log-normal distribution)
            order_value = np.random.lognormal(mean=4, sigma=0.8)

            # Product quantity (usually 1-3 items)
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.07, 0.03])

            conversions.append({
                'conversion_id': f'CONV_{i+1:010d}',
                'campaign_id': np.random.choice(campaign_ids),
                'customer_id': np.random.choice(customer_ids),
                'conversion_date': conversion_date.date(),
                'conversion_time': f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                'order_value': round(order_value, 2),
                'quantity': quantity,
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books']),
                'is_new_customer': np.random.choice([True, False], p=[0.3, 0.7]),
                'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card', 'Other'], p=[0.5, 0.3, 0.15, 0.05]),
            })

        df = pd.DataFrame(conversions)
        logger.info(f"Generated {len(df):,} conversions")
        return df

    def generate_touchpoints(
        self,
        campaigns_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        num_touchpoints: int
    ) -> pd.DataFrame:
        """
        Generate multi-touch attribution data (customer journey).

        Args:
            campaigns_df: DataFrame with campaign metadata
            customers_df: DataFrame with customer data
            num_touchpoints: Number of touchpoints to generate

        Returns:
            DataFrame with touchpoint data
        """
        logger.info(f"Generating {num_touchpoints:,} touchpoints...")

        touchpoints = []

        campaign_ids = campaigns_df['campaign_id'].values
        customer_ids = customers_df['customer_id'].values

        # Generate customer journeys (multiple touchpoints per customer)
        journeys_per_customer = num_touchpoints // len(customer_ids)

        for customer_id in tqdm(customer_ids, desc="Touchpoints"):
            # Number of touchpoints for this customer (typically 2-8)
            num_customer_touchpoints = np.random.choice(
                range(1, 9),
                p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02]
            )

            # First touchpoint date
            first_touch_date = self.start_date + timedelta(
                days=np.random.randint(0, max(1, self.days - 30))
            )

            for touch_num in range(num_customer_touchpoints):
                # Subsequent touchpoints are spaced out
                days_offset = sum(np.random.poisson(3) for _ in range(touch_num))
                touchpoint_date = first_touch_date + timedelta(days=days_offset)

                if touchpoint_date > self.end_date:
                    break

                # Touch position in journey
                position = touch_num + 1
                is_first_touch = (position == 1)
                is_last_touch = (position == num_customer_touchpoints)

                touchpoints.append({
                    'touchpoint_id': f'TOUCH_{len(touchpoints)+1:012d}',
                    'customer_id': customer_id,
                    'campaign_id': np.random.choice(campaign_ids),
                    'touchpoint_date': touchpoint_date.date(),
                    'touchpoint_time': f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                    'position': position,
                    'is_first_touch': is_first_touch,
                    'is_last_touch': is_last_touch,
                    'device_type': np.random.choice(DEVICE_TYPES, p=[0.4, 0.5, 0.1]),
                    'interaction_type': np.random.choice(['Click', 'View', 'Engagement'], p=[0.5, 0.4, 0.1]),
                    'time_to_next_touch': np.random.poisson(3) if not is_last_touch else None,
                })

                if len(touchpoints) >= num_touchpoints:
                    break

            if len(touchpoints) >= num_touchpoints:
                break

        df = pd.DataFrame(touchpoints)
        logger.info(f"Generated {len(df):,} touchpoints")
        return df

    def generate_cohorts(
        self,
        customers_df: pd.DataFrame,
        num_weeks: int = 52
    ) -> pd.DataFrame:
        """
        Generate cohort-based retention data.

        Args:
            customers_df: DataFrame with customer data
            num_weeks: Number of weeks to track retention

        Returns:
            DataFrame with cohort retention data
        """
        logger.info(f"Generating cohort data for {num_weeks} weeks...")

        cohorts = []

        # Group customers by acquisition week
        customers_df['acquisition_week'] = pd.to_datetime(
            customers_df['acquisition_date']
        ).dt.to_period('W').astype(str)

        cohort_groups = customers_df.groupby('acquisition_week')

        for cohort_week, cohort_customers in tqdm(cohort_groups, desc="Cohorts"):
            cohort_size = len(cohort_customers)

            # Generate retention for each week
            for week_num in range(num_weeks):
                # Retention drops exponentially
                retention_rate = 1.0 * np.exp(-0.05 * week_num)
                retention_rate += np.random.normal(0, 0.02)  # Add noise
                retention_rate = max(0, min(1, retention_rate))

                retained_customers = int(cohort_size * retention_rate)

                # Revenue from retained customers
                avg_revenue_per_customer = np.random.lognormal(mean=3, sigma=0.5)
                total_revenue = retained_customers * avg_revenue_per_customer

                cohorts.append({
                    'cohort_week': cohort_week,
                    'week_number': week_num,
                    'cohort_size': cohort_size,
                    'retained_customers': retained_customers,
                    'retention_rate': round(retention_rate, 4),
                    'revenue': round(total_revenue, 2),
                    'avg_revenue_per_customer': round(avg_revenue_per_customer, 2),
                })

        df = pd.DataFrame(cohorts)
        logger.info(f"Generated {len(df):,} cohort records")
        return df


# ============================================================================
# Export Functions
# ============================================================================

def export_to_csv(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    compress: bool = False
) -> None:
    """
    Export tables to CSV files.

    Args:
        tables: Dictionary of table_name -> DataFrame
        output_dir: Output directory path
        compress: Whether to compress CSV files
    """
    logger.info(f"Exporting {len(tables)} tables to CSV...")

    csv_dir = output_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    for table_name, df in tqdm(tables.items(), desc="CSV Export"):
        if compress:
            filename = csv_dir / f'{table_name}.csv.gz'
            df.to_csv(filename, index=False, compression='gzip')
        else:
            filename = csv_dir / f'{table_name}.csv'
            df.to_csv(filename, index=False)

        logger.info(f"  Exported {table_name}: {len(df):,} rows -> {filename}")


def export_to_sqlite(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    db_name: str = 'marketing_data.db'
) -> None:
    """
    Export tables to SQLite database.

    Args:
        tables: Dictionary of table_name -> DataFrame
        output_dir: Output directory path
        db_name: Database filename
    """
    logger.info(f"Exporting {len(tables)} tables to SQLite...")

    db_path = output_dir / db_name

    with sqlite3.connect(db_path) as conn:
        for table_name, df in tqdm(tables.items(), desc="SQLite Export"):
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"  Exported {table_name}: {len(df):,} rows")

    logger.info(f"SQLite database created: {db_path}")


def export_to_redshift(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path
) -> None:
    """
    Export tables in Redshift-compatible format (CSV + DDL).

    Args:
        tables: Dictionary of table_name -> DataFrame
        output_dir: Output directory path
    """
    logger.info(f"Exporting {len(tables)} tables for Redshift...")

    redshift_dir = output_dir / 'redshift'
    redshift_dir.mkdir(parents=True, exist_ok=True)

    # Export CSVs
    for table_name, df in tqdm(tables.items(), desc="Redshift CSV"):
        filename = redshift_dir / f'{table_name}.csv'
        df.to_csv(filename, index=False, sep='|', na_rep='NULL')

    # Generate DDL statements
    ddl_file = redshift_dir / 'create_tables.sql'

    with open(ddl_file, 'w') as f:
        f.write("-- Redshift DDL for Marketing Measurement Data\n")
        f.write("-- Generated: " + datetime.now().isoformat() + "\n\n")

        for table_name, df in tables.items():
            f.write(f"DROP TABLE IF EXISTS {table_name};\n")
            f.write(f"CREATE TABLE {table_name} (\n")

            columns = []
            for col, dtype in df.dtypes.items():
                if dtype == 'object':
                    col_type = 'VARCHAR(255)'
                elif dtype == 'int64':
                    col_type = 'BIGINT'
                elif dtype == 'float64':
                    col_type = 'DECIMAL(18,6)'
                elif dtype == 'bool':
                    col_type = 'BOOLEAN'
                else:
                    col_type = 'VARCHAR(255)'

                columns.append(f"    {col} {col_type}")

            f.write(',\n'.join(columns))
            f.write("\n);\n\n")

        # Add COPY commands
        f.write("\n-- COPY commands (update bucket path)\n")
        for table_name in tables.keys():
            f.write(f"""
COPY {table_name}
FROM 's3://your-bucket/marketing-data/{table_name}.csv'
CREDENTIALS 'aws_access_key_id=YOUR_KEY;aws_secret_access_key=YOUR_SECRET'
DELIMITER '|'
IGNOREHEADER 1
NULL AS 'NULL';
\n""")

    logger.info(f"Redshift DDL created: {ddl_file}")


def export_summary(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path
) -> None:
    """
    Export data summary and statistics.

    Args:
        tables: Dictionary of table_name -> DataFrame
        output_dir: Output directory path
    """
    summary_file = output_dir / 'data_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Marketing Measurement Synthetic Data - Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        for table_name, df in tables.items():
            f.write(f"\n{table_name.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Rows: {len(df):,}\n")
            f.write(f"Columns: {len(df.columns)}\n")
            f.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
            f.write(f"\nColumns:\n")
            for col in df.columns:
                f.write(f"  - {col} ({df[col].dtype})\n")

            f.write(f"\nSample (first 5 rows):\n")
            f.write(df.head().to_string())
            f.write("\n\n")

    logger.info(f"Summary file created: {summary_file}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point for the data generator."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic marketing data for the Partner Academy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate small dataset for testing
  python generate_marketing_data.py --size small --output-dir ./data

  # Generate 1M rows with all export formats
  python generate_marketing_data.py --rows 1000000 --format csv,sqlite,redshift

  # Generate large dataset with specific seed
  python generate_marketing_data.py --size large --seed 12345 --days 730
        """
    )

    # Size arguments (mutually exclusive)
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        '--size',
        choices=list(PRESET_SIZES.keys()),
        default='medium',
        help='Preset size (default: medium)'
    )
    size_group.add_argument(
        '--rows',
        type=int,
        help='Custom number of campaigns (overrides --size)'
    )

    # Date range
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data (default: 365)'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='csv,sqlite',
        help='Export formats: csv,sqlite,redshift (default: csv,sqlite)'
    )

    parser.add_argument(
        '--compress',
        action='store_true',
        help='Compress CSV files with gzip'
    )

    # Generation options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--customers-multiplier',
        type=float,
        default=50.0,
        help='Customer count multiplier (default: 50x campaigns)'
    )

    parser.add_argument(
        '--conversions-multiplier',
        type=float,
        default=100.0,
        help='Conversions count multiplier (default: 100x campaigns)'
    )

    parser.add_argument(
        '--touchpoints-multiplier',
        type=float,
        default=200.0,
        help='Touchpoints count multiplier (default: 200x campaigns)'
    )

    args = parser.parse_args()

    # Determine number of campaigns
    if args.rows:
        num_campaigns = args.rows
    else:
        num_campaigns = PRESET_SIZES[args.size]

    # Calculate derived counts
    num_customers = int(num_campaigns * args.customers_multiplier)
    num_conversions = int(num_campaigns * args.conversions_multiplier)
    num_touchpoints = int(num_campaigns * args.touchpoints_multiplier)

    # Parse export formats
    export_formats = [fmt.strip() for fmt in args.format.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    logger.info("=" * 80)
    logger.info("Marketing Measurement Data Generator")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Campaigns: {num_campaigns:,}")
    logger.info(f"  Customers: {num_customers:,}")
    logger.info(f"  Conversions: {num_conversions:,}")
    logger.info(f"  Touchpoints: {num_touchpoints:,}")
    logger.info(f"  Days: {args.days}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Formats: {', '.join(export_formats)}")
    logger.info("=" * 80)

    # Initialize generator
    generator = MarketingDataGenerator(
        num_campaigns=num_campaigns,
        days=args.days,
        seed=args.seed
    )

    # Generate tables
    tables = {}

    try:
        # 1. Campaigns
        tables['campaigns'] = generator.generate_campaigns()

        # 2. Daily Performance
        tables['daily_performance'] = generator.generate_daily_performance(
            tables['campaigns']
        )

        # 3. Customers
        tables['customers'] = generator.generate_customers(num_customers)

        # 4. Conversions
        tables['conversions'] = generator.generate_conversions(
            tables['campaigns'],
            tables['customers'],
            num_conversions
        )

        # 5. Touchpoints
        tables['touchpoints'] = generator.generate_touchpoints(
            tables['campaigns'],
            tables['customers'],
            num_touchpoints
        )

        # 6. Cohorts
        tables['cohorts'] = generator.generate_cohorts(
            tables['customers'],
            num_weeks=52
        )

        # Export data
        logger.info("\n" + "=" * 80)
        logger.info("Exporting data...")
        logger.info("=" * 80)

        if 'csv' in export_formats:
            export_to_csv(tables, output_dir, compress=args.compress)

        if 'sqlite' in export_formats:
            export_to_sqlite(tables, output_dir)

        if 'redshift' in export_formats:
            export_to_redshift(tables, output_dir)

        # Export summary
        export_summary(tables, output_dir)

        # Final statistics
        logger.info("\n" + "=" * 80)
        logger.info("Generation Complete!")
        logger.info("=" * 80)
        total_rows = sum(len(df) for df in tables.values())
        total_size = sum(df.memory_usage(deep=True).sum() for df in tables.values()) / 1024 / 1024

        logger.info(f"Total rows: {total_rows:,}")
        logger.info(f"Total memory: {total_size:.2f} MB")
        logger.info(f"Output directory: {output_dir.absolute()}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
