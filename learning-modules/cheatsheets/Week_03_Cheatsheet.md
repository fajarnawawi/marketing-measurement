# Week 3: SQL Basics - Quick Reference Cheatsheet

## 📋 SQL Fundamentals

### SELECT Statements
```sql
-- Select all columns
SELECT * FROM campaigns;

-- Select specific columns
SELECT campaign_name, cost, revenue
FROM campaigns;

-- Select with alias
SELECT
    campaign_name AS campaign,
    cost AS spend,
    revenue AS rev
FROM campaigns;

-- Select distinct values
SELECT DISTINCT channel FROM campaigns;
SELECT DISTINCT channel, campaign_type FROM campaigns;

-- Select with calculation
SELECT
    campaign_name,
    cost,
    revenue,
    revenue - cost AS profit,
    revenue / cost AS roas
FROM campaigns;

-- Limit results
SELECT * FROM campaigns LIMIT 10;
SELECT * FROM campaigns LIMIT 10 OFFSET 20;  -- Skip first 20
```

### WHERE Clause & Filtering
```sql
-- Basic comparisons
SELECT * FROM campaigns WHERE cost > 5000;
SELECT * FROM campaigns WHERE channel = 'Google';
SELECT * FROM campaigns WHERE roas >= 3.0;
SELECT * FROM campaigns WHERE conversions <> 0;  -- Not equal

-- AND / OR / NOT
SELECT * FROM campaigns
WHERE cost > 5000 AND revenue > 15000;

SELECT * FROM campaigns
WHERE channel = 'Google' OR channel = 'Facebook';

SELECT * FROM campaigns
WHERE NOT channel = 'Display';

-- BETWEEN (inclusive)
SELECT * FROM campaigns
WHERE cost BETWEEN 1000 AND 5000;

-- IN (multiple values)
SELECT * FROM campaigns
WHERE channel IN ('Google', 'Facebook', 'Instagram');

-- LIKE (pattern matching)
SELECT * FROM campaigns
WHERE campaign_name LIKE 'Search%';      -- Starts with 'Search'

SELECT * FROM campaigns
WHERE campaign_name LIKE '%Sale%';       -- Contains 'Sale'

SELECT * FROM campaigns
WHERE campaign_name LIKE '_earch%';      -- Single character wildcard

-- IS NULL / IS NOT NULL
SELECT * FROM campaigns WHERE revenue IS NULL;
SELECT * FROM campaigns WHERE revenue IS NOT NULL;

-- Complex conditions
SELECT * FROM campaigns
WHERE (channel = 'Google' OR channel = 'Facebook')
  AND cost > 1000
  AND conversions IS NOT NULL;
```

### ORDER BY (Sorting)
```sql
-- Sort ascending (default)
SELECT * FROM campaigns ORDER BY cost;
SELECT * FROM campaigns ORDER BY cost ASC;

-- Sort descending
SELECT * FROM campaigns ORDER BY revenue DESC;

-- Sort by multiple columns
SELECT * FROM campaigns
ORDER BY channel ASC, cost DESC;

-- Sort by calculated column
SELECT
    campaign_name,
    revenue / cost AS roas
FROM campaigns
ORDER BY roas DESC;

-- Sort with alias
SELECT
    campaign_name,
    revenue / cost AS roas
FROM campaigns
ORDER BY roas DESC
LIMIT 10;
```

---

## 🔗 JOIN Operations

### Sample Schema
```sql
-- campaigns table
CREATE TABLE campaigns (
    campaign_id INT PRIMARY KEY,
    campaign_name VARCHAR(100),
    channel VARCHAR(50),
    start_date DATE,
    budget DECIMAL(10, 2)
);

-- performance table
CREATE TABLE performance (
    performance_id INT PRIMARY KEY,
    campaign_id INT,
    date DATE,
    impressions INT,
    clicks INT,
    cost DECIMAL(10, 2),
    conversions INT,
    revenue DECIMAL(10, 2)
);

-- channels table
CREATE TABLE channels (
    channel_id INT PRIMARY KEY,
    channel_name VARCHAR(50),
    category VARCHAR(50)
);
```

### INNER JOIN
```sql
-- Basic inner join
SELECT
    c.campaign_name,
    p.cost,
    p.revenue
FROM campaigns c
INNER JOIN performance p ON c.campaign_id = p.campaign_id;

-- Join multiple tables
SELECT
    c.campaign_name,
    ch.channel_name,
    ch.category,
    p.cost,
    p.revenue
FROM campaigns c
INNER JOIN performance p ON c.campaign_id = p.campaign_id
INNER JOIN channels ch ON c.channel = ch.channel_name;

-- Join with WHERE
SELECT
    c.campaign_name,
    p.cost,
    p.revenue
FROM campaigns c
INNER JOIN performance p ON c.campaign_id = p.campaign_id
WHERE p.cost > 1000;
```

### LEFT JOIN (LEFT OUTER JOIN)
```sql
-- Keep all campaigns, even without performance data
SELECT
    c.campaign_name,
    c.budget,
    COALESCE(p.cost, 0) AS cost,
    COALESCE(p.revenue, 0) AS revenue
FROM campaigns c
LEFT JOIN performance p ON c.campaign_id = p.campaign_id;

-- Find campaigns with no performance data
SELECT
    c.campaign_name,
    c.budget
FROM campaigns c
LEFT JOIN performance p ON c.campaign_id = p.campaign_id
WHERE p.campaign_id IS NULL;
```

### RIGHT JOIN (RIGHT OUTER JOIN)
```sql
-- Keep all performance records, even without campaign data
SELECT
    p.date,
    p.cost,
    p.revenue,
    c.campaign_name
FROM campaigns c
RIGHT JOIN performance p ON c.campaign_id = p.campaign_id;
```

### FULL OUTER JOIN
```sql
-- Keep all records from both tables
SELECT
    c.campaign_name,
    p.cost,
    p.revenue
FROM campaigns c
FULL OUTER JOIN performance p ON c.campaign_id = p.campaign_id;
```

### SELF JOIN
```sql
-- Compare campaigns from same channel
SELECT
    c1.campaign_name AS campaign_1,
    c2.campaign_name AS campaign_2,
    c1.budget
FROM campaigns c1
INNER JOIN campaigns c2
    ON c1.channel = c2.channel
    AND c1.campaign_id < c2.campaign_id;
```

---

## 📊 GROUP BY & Aggregations

### Basic GROUP BY
```sql
-- Group by single column
SELECT
    channel,
    COUNT(*) AS campaign_count
FROM campaigns
GROUP BY channel;

-- Group by multiple columns
SELECT
    channel,
    campaign_type,
    COUNT(*) AS campaign_count,
    SUM(cost) AS total_cost
FROM campaigns
GROUP BY channel, campaign_type;
```

### Aggregate Functions
```sql
-- Common aggregations
SELECT
    channel,
    COUNT(*) AS campaign_count,
    SUM(cost) AS total_cost,
    AVG(cost) AS avg_cost,
    MIN(cost) AS min_cost,
    MAX(cost) AS max_cost,
    SUM(revenue) AS total_revenue
FROM performance
GROUP BY channel;

-- COUNT variations
SELECT
    COUNT(*) AS total_rows,              -- Count all rows
    COUNT(revenue) AS revenue_count,      -- Count non-null revenues
    COUNT(DISTINCT channel) AS unique_channels
FROM campaigns;
```

### HAVING Clause
```sql
-- Filter aggregated results (use HAVING, not WHERE)
SELECT
    channel,
    SUM(cost) AS total_cost,
    AVG(revenue / cost) AS avg_roas
FROM campaigns
GROUP BY channel
HAVING SUM(cost) > 10000;

-- Multiple HAVING conditions
SELECT
    channel,
    COUNT(*) AS campaign_count,
    SUM(cost) AS total_cost
FROM campaigns
GROUP BY channel
HAVING COUNT(*) >= 5
   AND SUM(cost) > 20000;

-- WHERE vs HAVING
SELECT
    channel,
    SUM(cost) AS total_cost
FROM campaigns
WHERE cost > 1000              -- Filter before grouping
GROUP BY channel
HAVING SUM(cost) > 10000;      -- Filter after grouping
```

### ORDER BY with GROUP BY
```sql
-- Order aggregated results
SELECT
    channel,
    SUM(revenue) / SUM(cost) AS roas
FROM campaigns
GROUP BY channel
ORDER BY roas DESC;

-- LIMIT with GROUP BY
SELECT
    channel,
    SUM(cost) AS total_cost
FROM campaigns
GROUP BY channel
ORDER BY total_cost DESC
LIMIT 5;
```

---

## 🔍 Subqueries

### Subquery in WHERE
```sql
-- Find campaigns with above-average cost
SELECT campaign_name, cost
FROM campaigns
WHERE cost > (SELECT AVG(cost) FROM campaigns);

-- Find campaigns from top-performing channel
SELECT campaign_name, channel, revenue
FROM campaigns
WHERE channel = (
    SELECT channel
    FROM campaigns
    GROUP BY channel
    ORDER BY SUM(revenue) DESC
    LIMIT 1
);
```

### Subquery in FROM (Derived Table)
```sql
-- Calculate metrics from aggregated data
SELECT
    channel,
    total_cost,
    total_revenue,
    total_revenue / total_cost AS roas
FROM (
    SELECT
        channel,
        SUM(cost) AS total_cost,
        SUM(revenue) AS total_revenue
    FROM campaigns
    GROUP BY channel
) AS channel_summary
WHERE total_cost > 5000;
```

### Subquery in SELECT
```sql
-- Compare each campaign to channel average
SELECT
    campaign_name,
    channel,
    cost,
    (SELECT AVG(cost)
     FROM campaigns c2
     WHERE c2.channel = c1.channel) AS channel_avg_cost
FROM campaigns c1;
```

### IN / NOT IN with Subquery
```sql
-- Find campaigns in top 3 channels by revenue
SELECT campaign_name, channel, revenue
FROM campaigns
WHERE channel IN (
    SELECT channel
    FROM campaigns
    GROUP BY channel
    ORDER BY SUM(revenue) DESC
    LIMIT 3
);

-- Find campaigns NOT in underperforming channels
SELECT campaign_name, channel
FROM campaigns
WHERE channel NOT IN (
    SELECT channel
    FROM campaigns
    GROUP BY channel
    HAVING AVG(revenue / cost) < 2.0
);
```

### EXISTS / NOT EXISTS
```sql
-- Find campaigns with performance data
SELECT c.campaign_name
FROM campaigns c
WHERE EXISTS (
    SELECT 1
    FROM performance p
    WHERE p.campaign_id = c.campaign_id
);

-- Find campaigns without performance data
SELECT c.campaign_name
FROM campaigns c
WHERE NOT EXISTS (
    SELECT 1
    FROM performance p
    WHERE p.campaign_id = c.campaign_id
);
```

---

## 💼 CASE Statements

### Basic CASE
```sql
-- Simple categorization
SELECT
    campaign_name,
    cost,
    CASE
        WHEN cost < 1000 THEN 'Low'
        WHEN cost < 5000 THEN 'Medium'
        ELSE 'High'
    END AS cost_category
FROM campaigns;

-- Multiple conditions
SELECT
    campaign_name,
    revenue / cost AS roas,
    CASE
        WHEN revenue / cost >= 4.0 THEN 'Excellent'
        WHEN revenue / cost >= 2.5 THEN 'Good'
        WHEN revenue / cost >= 1.5 THEN 'Fair'
        ELSE 'Poor'
    END AS performance
FROM campaigns;
```

### CASE in Aggregations
```sql
-- Count by category
SELECT
    channel,
    COUNT(*) AS total_campaigns,
    SUM(CASE WHEN revenue / cost >= 3.0 THEN 1 ELSE 0 END) AS high_roas_count,
    SUM(CASE WHEN revenue / cost < 2.0 THEN 1 ELSE 0 END) AS low_roas_count
FROM campaigns
GROUP BY channel;

-- Conditional aggregation
SELECT
    channel,
    SUM(CASE WHEN campaign_type = 'Search' THEN cost ELSE 0 END) AS search_cost,
    SUM(CASE WHEN campaign_type = 'Display' THEN cost ELSE 0 END) AS display_cost,
    SUM(CASE WHEN campaign_type = 'Video' THEN cost ELSE 0 END) AS video_cost
FROM campaigns
GROUP BY channel;
```

### CASE for Data Cleaning
```sql
-- Standardize values
SELECT
    campaign_id,
    CASE
        WHEN channel IN ('google', 'GOOGLE', 'Google Ads') THEN 'Google'
        WHEN channel IN ('facebook', 'FACEBOOK', 'FB') THEN 'Facebook'
        WHEN channel IN ('instagram', 'IG') THEN 'Instagram'
        ELSE channel
    END AS channel_clean
FROM campaigns;
```

---

## 🎯 Common Marketing SQL Patterns

### Campaign Performance Report
```sql
-- Comprehensive campaign metrics
SELECT
    c.campaign_name,
    c.channel,
    c.budget,
    SUM(p.impressions) AS total_impressions,
    SUM(p.clicks) AS total_clicks,
    SUM(p.conversions) AS total_conversions,
    SUM(p.cost) AS total_cost,
    SUM(p.revenue) AS total_revenue,
    SUM(p.clicks) * 1.0 / NULLIF(SUM(p.impressions), 0) AS ctr,
    SUM(p.conversions) * 1.0 / NULLIF(SUM(p.clicks), 0) AS cvr,
    SUM(p.cost) / NULLIF(SUM(p.conversions), 0) AS cpa,
    SUM(p.revenue) / NULLIF(SUM(p.cost), 0) AS roas
FROM campaigns c
LEFT JOIN performance p ON c.campaign_id = p.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.channel, c.budget
ORDER BY roas DESC;
```

### Top Performing Channels
```sql
-- Channel comparison
SELECT
    channel,
    COUNT(DISTINCT campaign_id) AS campaign_count,
    SUM(cost) AS total_cost,
    SUM(revenue) AS total_revenue,
    SUM(conversions) AS total_conversions,
    SUM(revenue) / SUM(cost) AS roas,
    SUM(cost) / SUM(conversions) AS cpa,
    SUM(revenue - cost) AS total_profit
FROM (
    SELECT
        c.campaign_id,
        c.channel,
        SUM(p.cost) AS cost,
        SUM(p.revenue) AS revenue,
        SUM(p.conversions) AS conversions
    FROM campaigns c
    INNER JOIN performance p ON c.campaign_id = p.campaign_id
    GROUP BY c.campaign_id, c.channel
) AS campaign_totals
GROUP BY channel
ORDER BY roas DESC;
```

### Daily Trend Analysis
```sql
-- Daily performance metrics
SELECT
    date,
    SUM(impressions) AS total_impressions,
    SUM(clicks) AS total_clicks,
    SUM(conversions) AS total_conversions,
    SUM(cost) AS total_cost,
    SUM(revenue) AS total_revenue,
    SUM(revenue) / NULLIF(SUM(cost), 0) AS roas,
    SUM(clicks) * 100.0 / NULLIF(SUM(impressions), 0) AS ctr_pct,
    SUM(conversions) * 100.0 / NULLIF(SUM(clicks), 0) AS cvr_pct
FROM performance
WHERE date >= '2024-01-01' AND date <= '2024-12-31'
GROUP BY date
ORDER BY date;
```

### Budget vs Actual Spending
```sql
-- Compare budget to actual spending
SELECT
    c.campaign_name,
    c.budget,
    COALESCE(SUM(p.cost), 0) AS actual_spend,
    c.budget - COALESCE(SUM(p.cost), 0) AS remaining_budget,
    COALESCE(SUM(p.cost), 0) * 100.0 / NULLIF(c.budget, 0) AS pct_spent,
    CASE
        WHEN COALESCE(SUM(p.cost), 0) > c.budget THEN 'Over Budget'
        WHEN COALESCE(SUM(p.cost), 0) > c.budget * 0.9 THEN 'Near Budget'
        ELSE 'Under Budget'
    END AS budget_status
FROM campaigns c
LEFT JOIN performance p ON c.campaign_id = p.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.budget
ORDER BY pct_spent DESC;
```

### Conversion Funnel
```sql
-- Full funnel metrics
SELECT
    channel,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks,
    SUM(conversions) AS conversions,
    SUM(clicks) * 100.0 / NULLIF(SUM(impressions), 0) AS impression_to_click_pct,
    SUM(conversions) * 100.0 / NULLIF(SUM(clicks), 0) AS click_to_conversion_pct,
    SUM(conversions) * 100.0 / NULLIF(SUM(impressions), 0) AS impression_to_conversion_pct
FROM performance
GROUP BY channel
ORDER BY impression_to_conversion_pct DESC;
```

---

## 📅 Working with Dates

### Date Functions (PostgreSQL/MySQL)
```sql
-- Extract date parts
SELECT
    date,
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(MONTH FROM date) AS month,
    EXTRACT(DAY FROM date) AS day,
    EXTRACT(DOW FROM date) AS day_of_week  -- 0=Sunday
FROM performance;

-- Date formatting
SELECT
    date,
    TO_CHAR(date, 'YYYY-MM-DD') AS formatted_date,
    TO_CHAR(date, 'Month') AS month_name,
    TO_CHAR(date, 'Day') AS day_name
FROM performance;

-- Date arithmetic
SELECT
    date,
    date + INTERVAL '7 days' AS week_later,
    date - INTERVAL '1 month' AS month_ago
FROM performance;

-- Current date/time
SELECT
    CURRENT_DATE AS today,
    CURRENT_TIMESTAMP AS now;
```

### Date Filtering
```sql
-- Filter by date range
SELECT * FROM performance
WHERE date BETWEEN '2024-01-01' AND '2024-12-31';

SELECT * FROM performance
WHERE date >= '2024-01-01' AND date < '2025-01-01';

-- Last 30 days
SELECT * FROM performance
WHERE date >= CURRENT_DATE - INTERVAL '30 days';

-- This month
SELECT * FROM performance
WHERE EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM CURRENT_DATE)
  AND EXTRACT(MONTH FROM date) = EXTRACT(MONTH FROM CURRENT_DATE);
```

### Date Aggregations
```sql
-- Group by month
SELECT
    DATE_TRUNC('month', date) AS month,
    SUM(cost) AS monthly_cost,
    SUM(revenue) AS monthly_revenue
FROM performance
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;

-- Group by week
SELECT
    DATE_TRUNC('week', date) AS week_start,
    SUM(cost) AS weekly_cost,
    SUM(revenue) AS weekly_revenue
FROM performance
GROUP BY DATE_TRUNC('week', date)
ORDER BY week_start;

-- Year-over-year comparison
SELECT
    EXTRACT(YEAR FROM date) AS year,
    EXTRACT(MONTH FROM date) AS month,
    SUM(revenue) AS revenue
FROM performance
GROUP BY EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date)
ORDER BY year, month;
```

---

## 💡 Best Practices & Tips

### Query Optimization
```sql
-- ✅ Good: Filter before joining
SELECT c.campaign_name, p.revenue
FROM campaigns c
INNER JOIN (
    SELECT campaign_id, SUM(revenue) AS revenue
    FROM performance
    WHERE date >= '2024-01-01'
    GROUP BY campaign_id
) p ON c.campaign_id = p.campaign_id;

-- ✅ Good: Use specific columns instead of SELECT *
SELECT campaign_id, campaign_name, cost
FROM campaigns;

-- ✅ Good: Use LIMIT for testing
SELECT * FROM large_table LIMIT 100;
```

### Avoiding NULL Issues
```sql
-- Use COALESCE for default values
SELECT
    campaign_name,
    COALESCE(revenue, 0) AS revenue,
    COALESCE(cost, 0) AS cost
FROM campaigns;

-- Use NULLIF to avoid division by zero
SELECT
    campaign_name,
    revenue / NULLIF(cost, 0) AS roas
FROM campaigns;

-- Combine both for safety
SELECT
    campaign_name,
    COALESCE(revenue, 0) / NULLIF(COALESCE(cost, 0), 0) AS roas
FROM campaigns;
```

### Formatting & Readability
```sql
-- ✅ Good: Well-formatted query
SELECT
    c.campaign_name,
    ch.channel_name,
    SUM(p.cost) AS total_cost,
    SUM(p.revenue) AS total_revenue,
    SUM(p.revenue) / NULLIF(SUM(p.cost), 0) AS roas
FROM campaigns c
INNER JOIN channels ch ON c.channel_id = ch.channel_id
INNER JOIN performance p ON c.campaign_id = p.campaign_id
WHERE p.date >= '2024-01-01'
GROUP BY c.campaign_name, ch.channel_name
HAVING SUM(p.cost) > 1000
ORDER BY roas DESC
LIMIT 10;
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Basic Queries
```sql
-- Problem: Find all campaigns with cost > 5000
SELECT campaign_name, cost, revenue
FROM campaigns
WHERE cost > 5000
ORDER BY cost DESC;

-- Problem: Count campaigns per channel
SELECT
    channel,
    COUNT(*) AS campaign_count
FROM campaigns
GROUP BY channel
ORDER BY campaign_count DESC;

-- Problem: Find top 5 campaigns by ROAS
SELECT
    campaign_name,
    revenue / cost AS roas
FROM campaigns
WHERE cost > 0
ORDER BY roas DESC
LIMIT 5;
```

### Exercise 2: Joins
```sql
-- Problem: Get all campaigns with their total performance metrics
SELECT
    c.campaign_name,
    c.channel,
    c.budget,
    COUNT(p.performance_id) AS days_active,
    SUM(p.cost) AS total_cost,
    SUM(p.revenue) AS total_revenue,
    SUM(p.conversions) AS total_conversions
FROM campaigns c
LEFT JOIN performance p ON c.campaign_id = p.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.channel, c.budget;

-- Problem: Find campaigns with no performance data in last 7 days
SELECT c.campaign_name
FROM campaigns c
LEFT JOIN performance p
    ON c.campaign_id = p.campaign_id
    AND p.date >= CURRENT_DATE - INTERVAL '7 days'
WHERE p.performance_id IS NULL;
```

### Exercise 3: Aggregations
```sql
-- Problem: Channel performance summary
SELECT
    channel,
    COUNT(DISTINCT campaign_id) AS num_campaigns,
    SUM(cost) AS total_cost,
    SUM(revenue) AS total_revenue,
    SUM(conversions) AS total_conversions,
    AVG(revenue / NULLIF(cost, 0)) AS avg_roas,
    SUM(revenue - cost) AS total_profit
FROM (
    SELECT
        c.channel,
        c.campaign_id,
        SUM(p.cost) AS cost,
        SUM(p.revenue) AS revenue,
        SUM(p.conversions) AS conversions
    FROM campaigns c
    INNER JOIN performance p ON c.campaign_id = p.campaign_id
    GROUP BY c.channel, c.campaign_id
) campaign_totals
GROUP BY channel
HAVING SUM(cost) > 10000
ORDER BY total_profit DESC;
```

### Exercise 4: Subqueries
```sql
-- Problem: Find campaigns performing better than channel average
SELECT
    c.campaign_name,
    c.channel,
    p.roas,
    ca.channel_avg_roas
FROM campaigns c
INNER JOIN (
    SELECT
        campaign_id,
        SUM(revenue) / NULLIF(SUM(cost), 0) AS roas
    FROM performance
    GROUP BY campaign_id
) p ON c.campaign_id = p.campaign_id
INNER JOIN (
    SELECT
        c2.channel,
        AVG(p2.roas) AS channel_avg_roas
    FROM campaigns c2
    INNER JOIN (
        SELECT
            campaign_id,
            SUM(revenue) / NULLIF(SUM(cost), 0) AS roas
        FROM performance
        GROUP BY campaign_id
    ) p2 ON c2.campaign_id = p2.campaign_id
    GROUP BY c2.channel
) ca ON c.channel = ca.channel
WHERE p.roas > ca.channel_avg_roas;
```

### Exercise 5: Case Statements
```sql
-- Problem: Categorize campaigns and count by category
SELECT
    channel,
    SUM(CASE WHEN roas >= 4.0 THEN 1 ELSE 0 END) AS excellent,
    SUM(CASE WHEN roas >= 2.5 AND roas < 4.0 THEN 1 ELSE 0 END) AS good,
    SUM(CASE WHEN roas >= 1.5 AND roas < 2.5 THEN 1 ELSE 0 END) AS fair,
    SUM(CASE WHEN roas < 1.5 THEN 1 ELSE 0 END) AS poor,
    AVG(roas) AS avg_roas
FROM (
    SELECT
        c.channel,
        c.campaign_id,
        SUM(p.revenue) / NULLIF(SUM(p.cost), 0) AS roas
    FROM campaigns c
    INNER JOIN performance p ON c.campaign_id = p.campaign_id
    GROUP BY c.channel, c.campaign_id
) campaign_roas
GROUP BY channel
ORDER BY avg_roas DESC;
```

---

## 🔍 Quick Reference Table

| Operation | SQL Syntax |
|-----------|------------|
| Select all | `SELECT * FROM table` |
| Select specific columns | `SELECT col1, col2 FROM table` |
| Filter rows | `WHERE condition` |
| Sort results | `ORDER BY col DESC` |
| Limit results | `LIMIT n` |
| Inner join | `INNER JOIN table2 ON table1.id = table2.id` |
| Left join | `LEFT JOIN table2 ON table1.id = table2.id` |
| Group by | `GROUP BY col` |
| Filter groups | `HAVING condition` |
| Count rows | `COUNT(*)` |
| Sum values | `SUM(col)` |
| Average | `AVG(col)` |
| Min/Max | `MIN(col)`, `MAX(col)` |
| Distinct values | `SELECT DISTINCT col FROM table` |
| Case statement | `CASE WHEN condition THEN value END` |

---

**Quick Navigation:**
- [← Week 2 Cheatsheet](Week_02_Cheatsheet.md)
- [Week 4 Cheatsheet →](Week_04_Cheatsheet.md)
- [Back to Main README](../README.md)
