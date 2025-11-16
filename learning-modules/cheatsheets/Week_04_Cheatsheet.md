# Week 4: SQL Advanced - Quick Reference Cheatsheet

## 📋 Window Functions

### ROW_NUMBER
```sql
-- Assign unique row number within partition
SELECT
    campaign_name,
    channel,
    revenue,
    ROW_NUMBER() OVER (ORDER BY revenue DESC) AS overall_rank,
    ROW_NUMBER() OVER (PARTITION BY channel ORDER BY revenue DESC) AS channel_rank
FROM campaigns;

-- Get top 3 campaigns per channel
WITH ranked_campaigns AS (
    SELECT
        campaign_name,
        channel,
        revenue,
        ROW_NUMBER() OVER (PARTITION BY channel ORDER BY revenue DESC) AS rank
    FROM campaigns
)
SELECT *
FROM ranked_campaigns
WHERE rank <= 3;
```

### RANK and DENSE_RANK
```sql
-- RANK: Gaps in ranking when there are ties
-- DENSE_RANK: No gaps in ranking
SELECT
    campaign_name,
    revenue,
    RANK() OVER (ORDER BY revenue DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY revenue DESC) AS dense_rank,
    ROW_NUMBER() OVER (ORDER BY revenue DESC) AS row_num
FROM campaigns;

-- Example output:
-- campaign_name | revenue | rank | dense_rank | row_num
-- Campaign_A    | 10000   | 1    | 1          | 1
-- Campaign_B    | 10000   | 1    | 1          | 2  (tie)
-- Campaign_C    | 9000    | 3    | 2          | 3  (rank skips 2)
-- Campaign_D    | 8000    | 4    | 3          | 4
```

### NTILE
```sql
-- Divide rows into N groups (quartiles, deciles, etc.)
SELECT
    campaign_name,
    revenue,
    NTILE(4) OVER (ORDER BY revenue) AS quartile,
    NTILE(10) OVER (ORDER BY revenue) AS decile
FROM campaigns;

-- Performance quartiles by channel
SELECT
    campaign_name,
    channel,
    revenue,
    NTILE(4) OVER (PARTITION BY channel ORDER BY revenue) AS performance_quartile
FROM campaigns;
```

### LAG and LEAD
```sql
-- Compare with previous/next row
SELECT
    date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY date) AS prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY date) AS next_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY date) AS day_over_day_change
FROM daily_performance;

-- Calculate day-over-day percentage change
SELECT
    date,
    revenue,
    LAG(revenue) OVER (ORDER BY date) AS prev_revenue,
    (revenue - LAG(revenue) OVER (ORDER BY date)) * 100.0 /
        NULLIF(LAG(revenue) OVER (ORDER BY date), 0) AS pct_change
FROM daily_performance;

-- Default value for first/last row
SELECT
    date,
    revenue,
    LAG(revenue, 1, 0) OVER (ORDER BY date) AS prev_revenue  -- 0 as default
FROM daily_performance;
```

### Running Totals & Moving Averages
```sql
-- Running total
SELECT
    date,
    cost,
    SUM(cost) OVER (ORDER BY date) AS running_total
FROM daily_performance;

-- Running total by channel
SELECT
    date,
    channel,
    cost,
    SUM(cost) OVER (PARTITION BY channel ORDER BY date) AS channel_running_total
FROM daily_performance;

-- Moving average (7-day)
SELECT
    date,
    revenue,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7day
FROM daily_performance;

-- Moving sum (last 30 days)
SELECT
    date,
    conversions,
    SUM(conversions) OVER (
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS conversions_30d
FROM daily_performance;
```

### FIRST_VALUE and LAST_VALUE
```sql
-- Get first and last values in window
SELECT
    date,
    channel,
    revenue,
    FIRST_VALUE(revenue) OVER (
        PARTITION BY channel
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_day_revenue,
    LAST_VALUE(revenue) OVER (
        PARTITION BY channel
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_day_revenue
FROM daily_performance;
```

### Window Frame Specifications
```sql
-- ROWS vs RANGE
SELECT
    date,
    revenue,
    -- Physical rows
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) AS avg_rows,
    -- Logical range (based on values)
    AVG(revenue) OVER (
        ORDER BY date
        RANGE BETWEEN INTERVAL '2 days' PRECEDING AND INTERVAL '2 days' FOLLOWING
    ) AS avg_range
FROM daily_performance;

-- Frame options
-- ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- All rows up to current
-- ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING  -- Current row to end
-- ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING          -- 7-row window
-- ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING          -- Previous 5 rows only
```

---

## 🔄 Common Table Expressions (CTEs)

### Basic CTE
```sql
-- Single CTE
WITH high_performers AS (
    SELECT
        campaign_id,
        campaign_name,
        revenue / NULLIF(cost, 0) AS roas
    FROM campaigns
    WHERE cost > 0
)
SELECT *
FROM high_performers
WHERE roas > 3.0
ORDER BY roas DESC;
```

### Multiple CTEs
```sql
-- Multiple CTEs (comma-separated)
WITH campaign_metrics AS (
    SELECT
        campaign_id,
        SUM(cost) AS total_cost,
        SUM(revenue) AS total_revenue,
        SUM(conversions) AS total_conversions
    FROM performance
    GROUP BY campaign_id
),
campaign_details AS (
    SELECT
        c.campaign_id,
        c.campaign_name,
        c.channel,
        m.total_cost,
        m.total_revenue,
        m.total_conversions
    FROM campaigns c
    INNER JOIN campaign_metrics m ON c.campaign_id = m.campaign_id
)
SELECT
    campaign_name,
    channel,
    total_cost,
    total_revenue,
    total_revenue / NULLIF(total_cost, 0) AS roas,
    total_cost / NULLIF(total_conversions, 0) AS cpa
FROM campaign_details
WHERE total_cost > 1000
ORDER BY roas DESC;
```

### Recursive CTEs
```sql
-- Generate date series
WITH RECURSIVE date_series AS (
    -- Base case
    SELECT DATE '2024-01-01' AS date
    UNION ALL
    -- Recursive case
    SELECT date + INTERVAL '1 day'
    FROM date_series
    WHERE date < DATE '2024-12-31'
)
SELECT date FROM date_series;

-- Campaign hierarchy (if campaigns have parent campaigns)
WITH RECURSIVE campaign_hierarchy AS (
    -- Top-level campaigns
    SELECT
        campaign_id,
        campaign_name,
        parent_campaign_id,
        1 AS level
    FROM campaigns
    WHERE parent_campaign_id IS NULL

    UNION ALL

    -- Child campaigns
    SELECT
        c.campaign_id,
        c.campaign_name,
        c.parent_campaign_id,
        ch.level + 1
    FROM campaigns c
    INNER JOIN campaign_hierarchy ch ON c.parent_campaign_id = ch.campaign_id
)
SELECT * FROM campaign_hierarchy
ORDER BY level, campaign_id;
```

### CTEs for Complex Aggregations
```sql
-- Calculate percentiles and compare
WITH campaign_stats AS (
    SELECT
        campaign_id,
        campaign_name,
        channel,
        revenue / NULLIF(cost, 0) AS roas
    FROM campaigns
    WHERE cost > 0
),
channel_benchmarks AS (
    SELECT
        channel,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY roas) AS p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY roas) AS median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY roas) AS p75,
        AVG(roas) AS avg_roas
    FROM campaign_stats
    GROUP BY channel
)
SELECT
    cs.campaign_name,
    cs.channel,
    cs.roas,
    cb.median AS channel_median,
    cs.roas - cb.median AS diff_from_median,
    CASE
        WHEN cs.roas >= cb.p75 THEN 'Top 25%'
        WHEN cs.roas >= cb.median THEN 'Above Median'
        WHEN cs.roas >= cb.p25 THEN 'Below Median'
        ELSE 'Bottom 25%'
    END AS performance_tier
FROM campaign_stats cs
INNER JOIN channel_benchmarks cb ON cs.channel = cb.channel
ORDER BY cs.channel, cs.roas DESC;
```

---

## 💾 Temporary Tables

### Creating Temporary Tables
```sql
-- Create temporary table
CREATE TEMP TABLE campaign_summary AS
SELECT
    campaign_id,
    SUM(cost) AS total_cost,
    SUM(revenue) AS total_revenue,
    SUM(conversions) AS total_conversions
FROM performance
GROUP BY campaign_id;

-- Use temporary table
SELECT
    c.campaign_name,
    cs.total_cost,
    cs.total_revenue,
    cs.total_revenue / NULLIF(cs.total_cost, 0) AS roas
FROM campaigns c
INNER JOIN campaign_summary cs ON c.campaign_id = cs.campaign_id
WHERE cs.total_cost > 1000;

-- Temporary table with explicit columns
CREATE TEMP TABLE top_channels (
    channel VARCHAR(50),
    total_revenue DECIMAL(10,2),
    campaign_count INT
);

INSERT INTO top_channels
SELECT
    channel,
    SUM(revenue) AS total_revenue,
    COUNT(*) AS campaign_count
FROM campaigns
GROUP BY channel
HAVING SUM(revenue) > 50000;
```

### When to Use Temp Tables vs CTEs
```sql
-- ✅ Use CTE: One-time use, simple query
WITH active_campaigns AS (
    SELECT * FROM campaigns WHERE is_active = true
)
SELECT * FROM active_campaigns;

-- ✅ Use Temp Table: Multiple uses, complex operations
CREATE TEMP TABLE active_campaign_metrics AS
SELECT
    campaign_id,
    SUM(cost) AS cost,
    SUM(revenue) AS revenue
FROM performance
GROUP BY campaign_id;

-- Use multiple times
SELECT * FROM active_campaign_metrics WHERE cost > 1000;
SELECT AVG(revenue) FROM active_campaign_metrics;
SELECT * FROM active_campaign_metrics ORDER BY revenue DESC LIMIT 10;

-- Temp tables are dropped at end of session
DROP TABLE IF EXISTS active_campaign_metrics;
```

---

## 🔗 Advanced Joins & Unions

### CROSS JOIN
```sql
-- Cartesian product (all combinations)
SELECT
    c.channel,
    d.date
FROM (SELECT DISTINCT channel FROM campaigns) c
CROSS JOIN (
    SELECT GENERATE_SERIES(
        DATE '2024-01-01',
        DATE '2024-01-31',
        INTERVAL '1 day'
    )::DATE AS date
) d;

-- Useful for filling gaps in data
WITH all_dates AS (
    SELECT GENERATE_SERIES(
        DATE '2024-01-01',
        DATE '2024-01-31',
        INTERVAL '1 day'
    )::DATE AS date
),
all_channels AS (
    SELECT DISTINCT channel FROM campaigns
),
date_channel_grid AS (
    SELECT * FROM all_dates CROSS JOIN all_channels
)
SELECT
    dcg.date,
    dcg.channel,
    COALESCE(p.revenue, 0) AS revenue
FROM date_channel_grid dcg
LEFT JOIN performance p
    ON dcg.date = p.date
    AND dcg.channel = p.channel;
```

### UNION, UNION ALL, INTERSECT, EXCEPT
```sql
-- UNION (removes duplicates)
SELECT campaign_id, campaign_name FROM search_campaigns
UNION
SELECT campaign_id, campaign_name FROM social_campaigns;

-- UNION ALL (keeps duplicates, faster)
SELECT campaign_id, cost FROM jan_performance
UNION ALL
SELECT campaign_id, cost FROM feb_performance;

-- INTERSECT (common rows)
SELECT campaign_id FROM high_cost_campaigns
INTERSECT
SELECT campaign_id FROM high_revenue_campaigns;

-- EXCEPT (in first but not second)
SELECT campaign_id FROM all_campaigns
EXCEPT
SELECT campaign_id FROM paused_campaigns;
```

### Lateral Joins
```sql
-- Get top 3 performing campaigns per channel
SELECT
    c.channel,
    top_campaigns.*
FROM (SELECT DISTINCT channel FROM campaigns) c
CROSS JOIN LATERAL (
    SELECT campaign_name, revenue
    FROM campaigns
    WHERE channel = c.channel
    ORDER BY revenue DESC
    LIMIT 3
) top_campaigns;

-- Latest performance for each campaign
SELECT
    c.campaign_id,
    c.campaign_name,
    latest.*
FROM campaigns c
CROSS JOIN LATERAL (
    SELECT date, cost, revenue
    FROM performance p
    WHERE p.campaign_id = c.campaign_id
    ORDER BY date DESC
    LIMIT 1
) latest;
```

---

## 📅 Advanced Date/Time Functions

### Date Arithmetic
```sql
-- Date differences
SELECT
    campaign_name,
    start_date,
    end_date,
    end_date - start_date AS duration_days,
    AGE(end_date, start_date) AS duration_interval
FROM campaigns;

-- Add/subtract intervals
SELECT
    date,
    date + INTERVAL '1 day' AS tomorrow,
    date - INTERVAL '1 week' AS last_week,
    date + INTERVAL '1 month' AS next_month,
    date - INTERVAL '1 year' AS last_year
FROM performance;
```

### Date Truncation & Rounding
```sql
-- Truncate to period start
SELECT
    date,
    DATE_TRUNC('day', date) AS day_start,
    DATE_TRUNC('week', date) AS week_start,
    DATE_TRUNC('month', date) AS month_start,
    DATE_TRUNC('quarter', date) AS quarter_start,
    DATE_TRUNC('year', date) AS year_start
FROM performance;

-- Weekly aggregation
SELECT
    DATE_TRUNC('week', date) AS week_start,
    SUM(cost) AS weekly_cost,
    SUM(revenue) AS weekly_revenue
FROM performance
GROUP BY DATE_TRUNC('week', date)
ORDER BY week_start;
```

### Date Generation
```sql
-- Generate date series
SELECT GENERATE_SERIES(
    DATE '2024-01-01',
    DATE '2024-12-31',
    INTERVAL '1 day'
)::DATE AS date;

-- Business days only (Monday-Friday)
WITH all_dates AS (
    SELECT GENERATE_SERIES(
        DATE '2024-01-01',
        DATE '2024-12-31',
        INTERVAL '1 day'
    )::DATE AS date
)
SELECT date
FROM all_dates
WHERE EXTRACT(DOW FROM date) BETWEEN 1 AND 5;  -- 1=Mon, 5=Fri
```

### Fiscal Periods
```sql
-- Fiscal year (e.g., starts April 1)
SELECT
    date,
    CASE
        WHEN EXTRACT(MONTH FROM date) >= 4 THEN EXTRACT(YEAR FROM date)
        ELSE EXTRACT(YEAR FROM date) - 1
    END AS fiscal_year,
    CASE
        WHEN EXTRACT(MONTH FROM date) >= 4
        THEN EXTRACT(MONTH FROM date) - 3
        ELSE EXTRACT(MONTH FROM date) + 9
    END AS fiscal_month
FROM performance;
```

---

## 🔤 String Manipulation

### String Functions
```sql
-- Concatenation
SELECT
    first_name || ' ' || last_name AS full_name,
    CONCAT(first_name, ' ', last_name) AS full_name_alt,
    CONCAT_WS(' - ', channel, campaign_type, region) AS campaign_key
FROM campaigns;

-- Case conversion
SELECT
    UPPER(campaign_name) AS upper_name,
    LOWER(campaign_name) AS lower_name,
    INITCAP(campaign_name) AS title_name  -- Capitalize Each Word
FROM campaigns;

-- Substring
SELECT
    campaign_name,
    SUBSTRING(campaign_name, 1, 10) AS short_name,
    SUBSTRING(campaign_name FROM 'Search.*') AS extract_pattern,
    LEFT(campaign_name, 5) AS first_5,
    RIGHT(campaign_name, 5) AS last_5
FROM campaigns;

-- Trimming
SELECT
    TRIM(campaign_name) AS trimmed,
    LTRIM(campaign_name) AS left_trim,
    RTRIM(campaign_name) AS right_trim,
    TRIM(BOTH ' ' FROM campaign_name) AS trim_spaces
FROM campaigns;

-- Replace
SELECT
    REPLACE(campaign_name, 'Google', 'Search') AS updated_name,
    REGEXP_REPLACE(campaign_name, '[0-9]', '', 'g') AS remove_numbers
FROM campaigns;
```

### Pattern Matching
```sql
-- LIKE patterns
SELECT * FROM campaigns WHERE campaign_name LIKE 'Search%';
SELECT * FROM campaigns WHERE campaign_name LIKE '%2024%';
SELECT * FROM campaigns WHERE campaign_name LIKE 'Brand_____';  -- Exactly 5 chars after Brand

-- ILIKE (case-insensitive)
SELECT * FROM campaigns WHERE campaign_name ILIKE '%search%';

-- Regular expressions
SELECT * FROM campaigns
WHERE campaign_name ~ '^Search';  -- Starts with Search

SELECT * FROM campaigns
WHERE campaign_name ~* 'search';  -- Case-insensitive regex

-- Extract with regex
SELECT
    campaign_name,
    SUBSTRING(campaign_name FROM '[0-9]+') AS extracted_numbers,
    REGEXP_MATCHES(campaign_name, '([A-Z][a-z]+)', 'g') AS words
FROM campaigns;
```

### String Aggregation
```sql
-- Combine multiple rows into single string
SELECT
    channel,
    STRING_AGG(campaign_name, ', ' ORDER BY campaign_name) AS all_campaigns
FROM campaigns
GROUP BY channel;

-- Array operations
SELECT
    channel,
    ARRAY_AGG(campaign_name ORDER BY campaign_name) AS campaign_array
FROM campaigns
GROUP BY channel;
```

---

## 🎯 Advanced Marketing Analytics Queries

### Cohort Analysis
```sql
-- User cohort analysis by signup month
WITH user_cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', signup_date) AS cohort_month,
        DATE_TRUNC('month', purchase_date) AS purchase_month
    FROM user_purchases
),
cohort_metrics AS (
    SELECT
        cohort_month,
        purchase_month,
        COUNT(DISTINCT user_id) AS users,
        SUM(revenue) AS revenue
    FROM user_cohorts
    GROUP BY cohort_month, purchase_month
)
SELECT
    cohort_month,
    purchase_month,
    users,
    revenue,
    EXTRACT(MONTH FROM AGE(purchase_month, cohort_month)) AS months_since_signup
FROM cohort_metrics
ORDER BY cohort_month, purchase_month;
```

### Attribution Modeling
```sql
-- First-touch attribution
WITH ranked_touches AS (
    SELECT
        user_id,
        touchpoint_id,
        channel,
        conversion_id,
        ROW_NUMBER() OVER (PARTITION BY user_id, conversion_id ORDER BY timestamp) AS touch_rank
    FROM customer_journey
    WHERE conversion_id IS NOT NULL
)
SELECT
    channel,
    COUNT(*) AS attributed_conversions,
    SUM(conversion_value) AS attributed_revenue
FROM ranked_touches
WHERE touch_rank = 1
GROUP BY channel;

-- Last-touch attribution
WITH ranked_touches AS (
    SELECT
        user_id,
        touchpoint_id,
        channel,
        conversion_id,
        ROW_NUMBER() OVER (PARTITION BY user_id, conversion_id ORDER BY timestamp DESC) AS touch_rank
    FROM customer_journey
    WHERE conversion_id IS NOT NULL
)
SELECT
    channel,
    COUNT(*) AS attributed_conversions,
    SUM(conversion_value) AS attributed_revenue
FROM ranked_touches
WHERE touch_rank = 1
GROUP BY channel;

-- Linear attribution (equal credit to all touchpoints)
WITH conversion_touchpoints AS (
    SELECT
        conversion_id,
        channel,
        conversion_value,
        COUNT(*) OVER (PARTITION BY conversion_id) AS touchpoint_count
    FROM customer_journey
    WHERE conversion_id IS NOT NULL
)
SELECT
    channel,
    COUNT(*) AS touchpoints,
    SUM(conversion_value / touchpoint_count) AS attributed_revenue
FROM conversion_touchpoints
GROUP BY channel;
```

### Customer Lifetime Value (LTV)
```sql
-- Calculate LTV by cohort
WITH customer_revenue AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', first_purchase_date) AS cohort_month,
        SUM(order_value) AS total_revenue,
        COUNT(*) AS order_count,
        MAX(order_date) - MIN(order_date) AS customer_lifetime_days
    FROM orders
    GROUP BY customer_id, DATE_TRUNC('month', first_purchase_date)
)
SELECT
    cohort_month,
    COUNT(*) AS customers,
    AVG(total_revenue) AS avg_ltv,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_revenue) AS median_ltv,
    AVG(order_count) AS avg_orders,
    AVG(customer_lifetime_days) AS avg_lifetime_days
FROM customer_revenue
GROUP BY cohort_month
ORDER BY cohort_month;
```

### Marketing Mix Modeling (Contribution Analysis)
```sql
-- Channel contribution to overall performance
WITH channel_performance AS (
    SELECT
        channel,
        SUM(cost) AS channel_cost,
        SUM(revenue) AS channel_revenue,
        SUM(conversions) AS channel_conversions
    FROM campaigns
    GROUP BY channel
),
total_performance AS (
    SELECT
        SUM(cost) AS total_cost,
        SUM(revenue) AS total_revenue,
        SUM(conversions) AS total_conversions
    FROM campaigns
)
SELECT
    cp.channel,
    cp.channel_cost,
    cp.channel_revenue,
    cp.channel_conversions,
    cp.channel_cost * 100.0 / tp.total_cost AS cost_share_pct,
    cp.channel_revenue * 100.0 / tp.total_revenue AS revenue_share_pct,
    cp.channel_conversions * 100.0 / tp.total_conversions AS conversion_share_pct,
    cp.channel_revenue / NULLIF(cp.channel_cost, 0) AS channel_roas,
    tp.total_revenue / NULLIF(tp.total_cost, 0) AS overall_roas,
    (cp.channel_revenue / NULLIF(cp.channel_cost, 0)) /
        NULLIF((tp.total_revenue / NULLIF(tp.total_cost, 0)), 0) AS roas_vs_avg
FROM channel_performance cp
CROSS JOIN total_performance tp
ORDER BY revenue_share_pct DESC;
```

### Time-Based Performance Analysis
```sql
-- Compare performance by day of week
SELECT
    TO_CHAR(date, 'Day') AS day_of_week,
    EXTRACT(DOW FROM date) AS dow_num,
    COUNT(*) AS days,
    AVG(cost) AS avg_daily_cost,
    AVG(revenue) AS avg_daily_revenue,
    AVG(revenue / NULLIF(cost, 0)) AS avg_roas,
    SUM(conversions) AS total_conversions
FROM performance
GROUP BY TO_CHAR(date, 'Day'), EXTRACT(DOW FROM date)
ORDER BY dow_num;

-- Hour of day analysis (for intraday data)
SELECT
    EXTRACT(HOUR FROM timestamp) AS hour_of_day,
    COUNT(*) AS events,
    SUM(conversions) AS conversions,
    AVG(conversion_value) AS avg_value
FROM events
GROUP BY EXTRACT(HOUR FROM timestamp)
ORDER BY hour_of_day;
```

---

## 💡 Query Optimization Tips

### Use EXPLAIN to Analyze Queries
```sql
-- See query execution plan
EXPLAIN
SELECT * FROM campaigns
WHERE channel = 'Google' AND cost > 1000;

-- See actual execution statistics
EXPLAIN ANALYZE
SELECT * FROM campaigns
WHERE channel = 'Google' AND cost > 1000;
```

### Indexing Strategies
```sql
-- Create index on frequently filtered columns
CREATE INDEX idx_campaigns_channel ON campaigns(channel);
CREATE INDEX idx_performance_date ON performance(date);
CREATE INDEX idx_performance_campaign_id ON performance(campaign_id);

-- Composite index for multiple columns
CREATE INDEX idx_campaigns_channel_cost ON campaigns(channel, cost);

-- Partial index (filtered)
CREATE INDEX idx_active_campaigns ON campaigns(channel)
WHERE is_active = true;

-- Drop unused indexes
DROP INDEX idx_campaigns_channel;
```

### Query Optimization Techniques
```sql
-- ✅ Good: Filter early
SELECT c.campaign_name, p.revenue
FROM campaigns c
INNER JOIN (
    SELECT campaign_id, SUM(revenue) AS revenue
    FROM performance
    WHERE date >= '2024-01-01'  -- Filter before join
    GROUP BY campaign_id
) p ON c.campaign_id = p.campaign_id;

-- ❌ Bad: Filter late
SELECT c.campaign_name, p.revenue
FROM campaigns c
INNER JOIN (
    SELECT campaign_id, SUM(revenue) AS revenue
    FROM performance
    GROUP BY campaign_id
) p ON c.campaign_id = p.campaign_id
WHERE p.revenue > 10000;  -- Could filter earlier

-- ✅ Good: Avoid SELECT *
SELECT campaign_id, campaign_name, cost
FROM campaigns;

-- ❌ Bad: SELECT *
SELECT * FROM campaigns;  -- Returns unnecessary columns

-- ✅ Good: Use EXISTS for existence checks
SELECT campaign_name
FROM campaigns c
WHERE EXISTS (
    SELECT 1 FROM performance p WHERE p.campaign_id = c.campaign_id
);

-- ❌ Slower: Use IN with subquery
SELECT campaign_name
FROM campaigns c
WHERE campaign_id IN (
    SELECT campaign_id FROM performance
);
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Window Functions
```sql
-- Problem: Calculate running total and moving average of daily revenue
SELECT
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date) AS running_total,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma_7day,
    revenue - LAG(revenue) OVER (ORDER BY date) AS day_over_day_change,
    (revenue - LAG(revenue) OVER (ORDER BY date)) * 100.0 /
        NULLIF(LAG(revenue) OVER (ORDER BY date), 0) AS pct_change
FROM daily_performance
ORDER BY date;
```

### Exercise 2: CTEs for Complex Analysis
```sql
-- Problem: Find campaigns that outperform their channel average
WITH campaign_metrics AS (
    SELECT
        c.campaign_id,
        c.campaign_name,
        c.channel,
        SUM(p.revenue) / NULLIF(SUM(p.cost), 0) AS roas
    FROM campaigns c
    INNER JOIN performance p ON c.campaign_id = p.campaign_id
    GROUP BY c.campaign_id, c.campaign_name, c.channel
),
channel_averages AS (
    SELECT
        channel,
        AVG(roas) AS avg_roas,
        STDDEV(roas) AS stddev_roas
    FROM campaign_metrics
    GROUP BY channel
)
SELECT
    cm.campaign_name,
    cm.channel,
    cm.roas,
    ca.avg_roas AS channel_avg,
    cm.roas - ca.avg_roas AS diff_from_avg,
    (cm.roas - ca.avg_roas) / NULLIF(ca.stddev_roas, 0) AS z_score
FROM campaign_metrics cm
INNER JOIN channel_averages ca ON cm.channel = ca.channel
WHERE cm.roas > ca.avg_roas
ORDER BY z_score DESC;
```

### Exercise 3: Advanced Attribution
```sql
-- Problem: Time-decay attribution (more recent touches get more credit)
WITH journey_with_weights AS (
    SELECT
        conversion_id,
        channel,
        touchpoint_order,
        total_touchpoints,
        conversion_value,
        -- Exponential decay: more recent = higher weight
        POWER(2, touchpoint_order - 1) AS touch_weight
    FROM (
        SELECT
            conversion_id,
            channel,
            conversion_value,
            ROW_NUMBER() OVER (
                PARTITION BY conversion_id
                ORDER BY timestamp DESC
            ) AS touchpoint_order,
            COUNT(*) OVER (PARTITION BY conversion_id) AS total_touchpoints
        FROM customer_journey
        WHERE conversion_id IS NOT NULL
    ) ranked_touches
),
weighted_attribution AS (
    SELECT
        conversion_id,
        channel,
        conversion_value,
        touch_weight,
        SUM(touch_weight) OVER (PARTITION BY conversion_id) AS total_weight,
        touch_weight / SUM(touch_weight) OVER (PARTITION BY conversion_id) AS attribution_share
    FROM journey_with_weights
)
SELECT
    channel,
    COUNT(DISTINCT conversion_id) AS conversions_touched,
    SUM(conversion_value * attribution_share) AS attributed_revenue,
    AVG(attribution_share) AS avg_attribution_share
FROM weighted_attribution
GROUP BY channel
ORDER BY attributed_revenue DESC;
```

### Exercise 4: Cohort Retention
```sql
-- Problem: Calculate monthly retention rates by signup cohort
WITH user_months AS (
    SELECT
        user_id,
        DATE_TRUNC('month', signup_date) AS cohort_month,
        DATE_TRUNC('month', activity_date) AS activity_month,
        EXTRACT(MONTH FROM AGE(activity_date, signup_date)) AS months_since_signup
    FROM user_activity
),
cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT user_id) AS cohort_size
    FROM user_months
    WHERE months_since_signup = 0
    GROUP BY cohort_month
),
retention_data AS (
    SELECT
        um.cohort_month,
        um.months_since_signup,
        COUNT(DISTINCT um.user_id) AS active_users,
        cs.cohort_size
    FROM user_months um
    INNER JOIN cohort_sizes cs ON um.cohort_month = cs.cohort_month
    GROUP BY um.cohort_month, um.months_since_signup, cs.cohort_size
)
SELECT
    cohort_month,
    months_since_signup,
    active_users,
    cohort_size,
    active_users * 100.0 / cohort_size AS retention_rate
FROM retention_data
ORDER BY cohort_month, months_since_signup;
```

### Exercise 5: Performance Segmentation
```sql
-- Problem: Segment campaigns into performance tiers with detailed stats
WITH campaign_performance AS (
    SELECT
        c.campaign_id,
        c.campaign_name,
        c.channel,
        SUM(p.cost) AS total_cost,
        SUM(p.revenue) AS total_revenue,
        SUM(p.conversions) AS total_conversions,
        SUM(p.revenue) / NULLIF(SUM(p.cost), 0) AS roas
    FROM campaigns c
    INNER JOIN performance p ON c.campaign_id = p.campaign_id
    GROUP BY c.campaign_id, c.campaign_name, c.channel
),
performance_percentiles AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY roas) AS p25,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY roas) AS p50,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY roas) AS p75
    FROM campaign_performance
)
SELECT
    channel,
    COUNT(*) AS campaigns,
    SUM(CASE WHEN roas >= pp.p75 THEN 1 ELSE 0 END) AS top_tier,
    SUM(CASE WHEN roas >= pp.p50 AND roas < pp.p75 THEN 1 ELSE 0 END) AS mid_tier,
    SUM(CASE WHEN roas >= pp.p25 AND roas < pp.p50 THEN 1 ELSE 0 END) AS low_tier,
    SUM(CASE WHEN roas < pp.p25 THEN 1 ELSE 0 END) AS bottom_tier,
    AVG(roas) AS avg_roas,
    SUM(total_revenue) AS total_revenue,
    SUM(total_cost) AS total_cost
FROM campaign_performance cp
CROSS JOIN performance_percentiles pp
GROUP BY channel, pp.p75, pp.p50, pp.p25
ORDER BY avg_roas DESC;
```

---

## 🔍 Quick Reference Table

| Function | Purpose | Example |
|----------|---------|---------|
| ROW_NUMBER() | Unique sequential number | `ROW_NUMBER() OVER (ORDER BY revenue DESC)` |
| RANK() | Rank with gaps for ties | `RANK() OVER (ORDER BY revenue DESC)` |
| DENSE_RANK() | Rank without gaps | `DENSE_RANK() OVER (ORDER BY revenue DESC)` |
| LAG() | Previous row value | `LAG(revenue) OVER (ORDER BY date)` |
| LEAD() | Next row value | `LEAD(revenue) OVER (ORDER BY date)` |
| SUM() OVER | Running total | `SUM(cost) OVER (ORDER BY date)` |
| AVG() OVER | Moving average | `AVG(revenue) OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` |
| WITH | CTE definition | `WITH cte AS (SELECT ...) SELECT * FROM cte` |
| CROSS JOIN | Cartesian product | `SELECT * FROM t1 CROSS JOIN t2` |
| LATERAL | Correlated subquery | `CROSS JOIN LATERAL (SELECT ... WHERE ...)` |

---

**Quick Navigation:**
- [← Week 3 Cheatsheet](Week_03_Cheatsheet.md)
- [Week 5 Cheatsheet →](Week_05_Cheatsheet.md)
- [Back to Main README](../README.md)
