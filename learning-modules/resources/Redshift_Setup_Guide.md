# Amazon Redshift Setup Guide for Marketing Analytics
## Marketing Measurement Partner Academy - Advanced Track

---

## 1. Introduction

### What is Amazon Redshift?

Amazon Redshift is a fully managed, petabyte-scale cloud data warehouse service designed for analytics workloads. It uses columnar storage and massively parallel processing (MPP) architecture to deliver fast query performance on large datasets.

**Key Features for Marketing Analytics:**
- **Columnar Storage**: Optimized for analytical queries that aggregate large datasets
- **Massively Parallel Processing**: Distributes query execution across multiple nodes
- **Scalability**: Handle billions of marketing events and impressions
- **SQL Compatibility**: Uses PostgreSQL-compatible SQL syntax
- **AWS Integration**: Native integration with S3, Glue, QuickSight, and other AWS services

### Comparison: SQLite vs PostgreSQL vs Redshift

| Feature | SQLite | PostgreSQL | Amazon Redshift |
|---------|--------|------------|-----------------|
| **Architecture** | Embedded, file-based | Client-server | Cloud data warehouse (MPP) |
| **Scalability** | Up to ~280 TB (theoretical) | Multi-TB | Petabyte-scale |
| **Concurrency** | Limited (write locks) | Excellent (MVCC) | Excellent (MPP) |
| **Use Case** | Local dev, prototypes | OLTP + Analytics | Large-scale analytics (OLAP) |
| **Cost** | Free | Free (hosting costs) | Pay-per-use |
| **Query Speed (1M rows)** | Seconds | Seconds | Sub-second |
| **Query Speed (1B rows)** | Not practical | Minutes | Seconds |
| **Best For** | Development, testing | Production apps | Data warehousing, BI |

### When to Use Redshift

**Use Redshift When:**
- Analyzing **10M+ marketing events** regularly
- Running complex multi-table joins across **large datasets**
- Building **enterprise BI dashboards** (QuickSight, Tableau, Looker)
- Need **sub-second queries** on billions of rows
- Integrating with AWS ecosystem (S3, Athena, EMR)
- Scaling analytics for **multiple brands/regions**
- Performing **attribution modeling** across millions of touchpoints

**Stick with SQLite/PostgreSQL When:**
- Dataset < 10M rows
- Simple queries with limited joins
- Development and prototyping
- Budget constraints
- Single-user analysis

### Cost Considerations

**Pricing Overview (as of 2024):**
- **DC2 Cluster**: ~$0.25/hour for dc2.large (2 nodes) = ~$180/month
- **RA3 Cluster**: ~$1.08/hour for ra3.xlplus (2 nodes) = ~$777/month
- **Redshift Serverless**: $0.36/RPU-hour (pay for query runtime only)
- **Storage**: $0.024/GB-month for RA3 managed storage

**Cost Optimization:**
- Pause clusters when not in use (no compute charges)
- Use Redshift Serverless for intermittent workloads
- Reserved instances for 24/7 production (up to 75% discount)
- Compress data (reduce storage costs by 70-90%)

**Monthly Cost Estimate for Learning:**
```
Scenario: 2-node DC2 cluster, 2 hours/day usage
- Compute: 2 hrs/day × 30 days × $0.25 = $15/month
- Storage: 100GB × $0.024 = $2.40/month
Total: ~$17-20/month
```

---

## 2. AWS Account Setup

### Creating an AWS Account

1. **Navigate to AWS Homepage**
   - Visit: https://aws.amazon.com
   - Click "Create an AWS Account"

2. **Provide Account Information**
   - Email address
   - Password
   - AWS account name (e.g., "Marketing Analytics Lab")

3. **Contact Information**
   - Choose "Personal" for learning purposes
   - Provide name, phone, address

4. **Payment Information**
   - Enter credit card (required even for free tier)
   - You won't be charged if you stay within free tier limits

5. **Identity Verification**
   - Phone verification via SMS or call

6. **Select Support Plan**
   - Choose "Basic Support - Free" for learning

### Setting up IAM Users and Permissions

**IMPORTANT**: Never use your root account for daily operations!

#### Step 1: Create IAM User for Redshift Administration

```bash
# Login to AWS Console -> IAM Dashboard
# Navigate to: https://console.aws.amazon.com/iam/
```

1. Click **"Users"** → **"Add users"**
2. Username: `redshift-admin`
3. Select: **"Provide user access to AWS Management Console"**
4. Click **"Next: Permissions"**

#### Step 2: Attach Permissions Policies

Attach these managed policies:
- `AmazonRedshiftFullAccess` - Full Redshift management
- `AmazonS3ReadOnlyAccess` - Load data from S3 (optional)
- `IAMReadOnlyAccess` - View IAM roles (optional)

**For custom policy (principle of least privilege):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "redshift:CreateCluster",
        "redshift:DeleteCluster",
        "redshift:ModifyCluster",
        "redshift:DescribeClusters",
        "redshift:RebootCluster",
        "redshift:PauseCluster",
        "redshift:ResumeCluster",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeInternetGateways",
        "ec2:AuthorizeSecurityGroupIngress"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Step 3: Create Access Keys for Programmatic Access

1. Click on user → **"Security credentials"** tab
2. Scroll to **"Access keys"** → **"Create access key"**
3. Choose **"Command Line Interface (CLI)"**
4. **Download .csv file** - store securely!

```bash
# Save credentials to ~/.aws/credentials
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### Security Best Practices

**1. Enable Multi-Factor Authentication (MFA)**
```
IAM Dashboard → Users → Select user → Security credentials →
Assigned MFA device → Manage → Virtual MFA device
```
Use Google Authenticator or Authy.

**2. Rotate Access Keys Regularly**
- Set calendar reminder every 90 days
- Create new key before deleting old one (zero downtime)

**3. Use IAM Roles for EC2/Lambda**
- Never hardcode credentials in code
- Attach IAM roles to compute resources

**4. Enable CloudTrail**
- Audit all API calls to Redshift
- Detect unauthorized access attempts

**5. Use Secrets Manager for Database Passwords**
```python
import boto3
import json

def get_redshift_credentials():
    client = boto3.client('secretsmanager', region_name='us-east-1')
    secret = client.get_secret_value(SecretId='redshift/marketing/credentials')
    return json.loads(secret['SecretString'])
```

### Free Tier Considerations

**AWS Free Tier for Redshift:**
- **2-month trial**: 750 hours/month of DC2.Large node (single-node cluster)
- **Availability**: Must be new AWS customer (within 12 months)
- **Limitations**: Only available in select regions

**After Free Tier Expires:**
- Switch to Redshift Serverless (pay-per-query)
- Pause cluster when not in use
- Use reserved instances for long-term savings

**Monitoring Free Tier Usage:**
```
AWS Console → Billing Dashboard → Free Tier Usage
```

**Setting Up Billing Alerts:**
```
1. AWS Console → CloudWatch → Alarms → Create Alarm
2. Select metric: EstimatedCharges
3. Set threshold: $10 (or your budget)
4. Configure SNS notification to your email
```

---

## 3. Redshift Cluster Setup

### Creating a Redshift Cluster (Step-by-Step)

#### Step 1: Navigate to Redshift Console

1. Login to AWS Console: https://console.aws.amazon.com
2. Search for "Redshift" in the services search bar
3. Click **"Amazon Redshift"**
4. Click **"Create cluster"**

#### Step 2: Cluster Configuration

**Cluster Identifier:**
```
marketing-analytics-cluster
```
(Must be unique, lowercase, alphanumeric with hyphens)

**What do you want to do?**
- Select: **"I'll choose"** (for custom configuration)

#### Step 3: Node Configuration

**Node type options:**

| Node Type | vCPUs | Memory | Storage | Price/Hour | Use Case |
|-----------|-------|--------|---------|------------|----------|
| dc2.large | 2 | 15 GB | 0.16 TB SSD | $0.25 | Dev/test, small datasets |
| dc2.8xlarge | 32 | 244 GB | 2.56 TB SSD | $4.80 | Production, compute-heavy |
| ra3.xlplus | 4 | 32 GB | Managed | $1.08 | Production, large storage |
| ra3.4xlarge | 12 | 96 GB | Managed | $3.26 | Enterprise, petabyte-scale |

**For Learning: Choose DC2.Large**
- **Node type**: dc2.large
- **Number of nodes**: 1 (single-node cluster)
  - Note: Single-node clusters cannot be paused (limitation)
  - For pause capability, choose 2 nodes (costs 2x)

**For Production: Choose RA3**
- Scales storage independently
- Better cost for large datasets (> 1TB)

#### Step 4: Database Configuration

**Database name:**
```
marketing_db
```

**Master username:**
```
admin
```
(or `marketing_admin`)

**Master password:**
- **Requirements**: 8-64 characters, at least one uppercase, lowercase, and number
- **Example**: `Marketing2024!Analytics`
- **⚠️ IMPORTANT**: Save this password securely! You cannot recover it.

**Recommended**: Use AWS Secrets Manager
```bash
# Store password in Secrets Manager
aws secretsmanager create-secret \
    --name redshift/marketing/admin-password \
    --secret-string '{"username":"admin","password":"Marketing2024!Analytics"}'
```

#### Step 5: Cluster Permissions (IAM Role)

**Purpose**: Allow Redshift to access S3 for data loading

1. Click **"Manage IAM roles"**
2. Click **"Create IAM role"**
3. Role name: `RedshiftS3AccessRole`
4. Attach policy: `AmazonS3ReadOnlyAccess`
5. Click **"Associate IAM role"**

**Or create custom role:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::marketing-data-bucket/*",
        "arn:aws:s3:::marketing-data-bucket"
      ]
    }
  ]
}
```

#### Step 6: Additional Configurations

**Expand "Additional configurations"**

**Network and security:**
- **Virtual private cloud (VPC)**: Default VPC
- **VPC security groups**: default
- **Cluster subnet group**: default
- **Publicly accessible**: ✅ **Yes** (for learning purposes)
  - ⚠️ **Security Warning**: Only for development! Use VPN/bastion for production.

**Database configurations:**
- **Port**: 5439 (default Redshift port)
- **Enhanced VPC routing**: No (not needed for learning)

**Maintenance:**
- **Maintenance window**: Select convenient time (e.g., Sunday 3:00 AM UTC)

**Monitoring:**
- **Use CloudWatch alarms**: No (to avoid extra charges)

**Backup:**
- **Automated snapshot retention**: 1 day (minimum for learning)
- **Manual snapshot retention**: Not needed

**Encryption:**
- **Encrypt database**: Yes (default, no extra cost)
- **Use AWS KMS**: aws/redshift (default key)

#### Step 7: Review and Create

1. Review all settings
2. **Estimated monthly cost**: Check the estimate
3. Click **"Create cluster"**

**Cluster Creation Time**: 5-10 minutes

**Monitor status:**
```bash
# Using AWS CLI
aws redshift describe-clusters \
    --cluster-identifier marketing-analytics-cluster \
    --query 'Clusters[0].ClusterStatus' \
    --output text
```

Status progression:
```
creating → modifying → available
```

### VPC and Security Group Configuration

#### Making Cluster Publicly Accessible (For Learning)

**⚠️ SECURITY WARNING**: Only for development/learning environments!

**Step 1: Modify Cluster to be Publicly Accessible**

1. Redshift Console → Clusters → Select your cluster
2. Click **"Actions"** → **"Modify"**
3. Scroll to **"Network and security"**
4. Check **"Publicly accessible"** → **Yes**
5. Click **"Modify cluster"**

**Step 2: Configure Security Group**

1. Redshift Console → Select cluster → **"Properties"** tab
2. Under **"Network and security"**, click on the VPC security group link
3. Click **"Edit inbound rules"**
4. Click **"Add rule"**

**Rule configuration:**
```
Type: Custom TCP
Protocol: TCP
Port range: 5439
Source: My IP (or Custom: YOUR_IP_ADDRESS/32)
Description: Redshift access from my IP
```

**⚠️ NEVER use 0.0.0.0/0 (allows access from anywhere)**

**For multiple IPs (e.g., home + office):**
```
Rule 1: YOUR_HOME_IP/32
Rule 2: YOUR_OFFICE_IP/32
```

**Find your IP:**
```bash
curl https://checkip.amazonaws.com
# Or visit: https://whatismyipaddress.com
```

**Step 3: Verify Endpoint**

After cluster becomes available:
```
Cluster → Properties → Endpoint

Example:
marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com:5439
```

**Step 4: Test Connection**

```bash
# Using psql (if installed)
psql -h marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com \
     -U admin \
     -d marketing_db \
     -p 5439

# You should see:
Password for user admin:
marketing_db=#
```

### Production VPC Configuration (Reference)

For production environments, use private subnets with controlled access:

```
Architecture:
Internet → Application Load Balancer → Web App (Public Subnet)
                                         ↓
                                    Redshift (Private Subnet)
                                         ↑
Corporate Network → VPN → VPC → Private Subnet
```

**Key components:**
1. **Private Subnet**: Redshift cluster not publicly accessible
2. **VPN/Direct Connect**: Secure connection from corporate network
3. **Bastion Host**: Jump server for SSH access
4. **NAT Gateway**: Outbound internet access for updates
5. **VPC Endpoints**: Private connection to S3

---

## 4. Connection Methods

### Using psycopg2 from Python

**Installation:**
```bash
pip install psycopg2-binary
```

**Basic Connection:**
```python
import psycopg2

# Connection parameters
conn = psycopg2.connect(
    host='marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com',
    port=5439,
    dbname='marketing_db',
    user='admin',
    password='Marketing2024!Analytics'
)

# Create cursor
cur = conn.cursor()

# Execute query
cur.execute("SELECT version();")
print(cur.fetchone())

# Close connection
cur.close()
conn.close()
```

**Expected output:**
```
('PostgreSQL 8.0.2 on i686-pc-linux-gnu, compiled by GCC gcc (GCC) 3.4.2 20041017 (Red Hat 3.4.2-6.fc3), Redshift 1.0.50629',)
```

**Production-Ready Connection with Error Handling:**
```python
import psycopg2
from psycopg2 import OperationalError
import sys

def create_connection():
    """Create database connection with error handling."""
    try:
        conn = psycopg2.connect(
            host='your-cluster.region.redshift.amazonaws.com',
            port=5439,
            dbname='marketing_db',
            user='admin',
            password='your-password',
            connect_timeout=10,  # 10 second timeout
            sslmode='require'     # Require SSL connection
        )
        print("✓ Connected to Redshift successfully")
        return conn
    except OperationalError as e:
        print(f"✗ Connection failed: {e}")
        sys.exit(1)

# Usage
conn = create_connection()
```

### Using SQL Workbench/J

**SQL Workbench/J** is a free, DBMS-independent SQL query tool.

**Step 1: Download and Install**
1. Download from: https://www.sql-workbench.eu/downloads.html
2. Requires Java 11+ (download from: https://adoptium.net/)
3. Extract ZIP and run `sqlworkbench.jar`

**Step 2: Download Redshift JDBC Driver**
1. Visit: https://docs.aws.amazon.com/redshift/latest/mgmt/jdbc20-install.html
2. Download: `redshift-jdbc42-2.1.0.x.jar`
3. Save to a known location

**Step 3: Configure Driver in SQL Workbench**
1. File → Manage Drivers
2. Click "New" (blank page icon)
3. Fill in:
   - **Name**: Amazon Redshift
   - **Library**: Click folder icon → select downloaded JAR
   - **Classname**: `com.amazon.redshift.jdbc.Driver` (auto-fills)
   - **Sample URL**: `jdbc:redshift://endpoint:5439/database`

**Step 4: Create Connection Profile**
1. File → Connect window
2. Click "New profile" icon
3. Fill in:
   - **Name**: Marketing Analytics Cluster
   - **Driver**: Amazon Redshift
   - **URL**: `jdbc:redshift://your-cluster.region.redshift.amazonaws.com:5439/marketing_db`
   - **Username**: `admin`
   - **Password**: (your password)
   - **Autocommit**: ✓ (check)

**Step 5: Test Connection**
- Click "Test Connection"
- Should see: "Connection to 'Marketing Analytics Cluster' successful"

**Step 6: Execute Queries**
```sql
-- Test query
SELECT current_database(), current_user, version();

-- Create table
CREATE TABLE test_campaigns (
    campaign_id INT,
    campaign_name VARCHAR(255),
    impressions BIGINT
);

-- Insert data
INSERT INTO test_campaigns VALUES
(1, 'Summer Sale', 1000000),
(2, 'Black Friday', 5000000);

-- Query
SELECT * FROM test_campaigns;
```

### Using AWS Query Editor

**AWS Query Editor v2** - Browser-based SQL editor (no installation required)

**Step 1: Access Query Editor**
1. AWS Console → Redshift → Query Editor v2
2. Or direct link: https://console.aws.amazon.com/sqlworkbench/home

**Step 2: Connect to Cluster**
1. Click **"Add connection"**
2. Choose: **"Redshift cluster"**
3. **Authentication**: Temporary credentials using your AWS login
   - ⚠️ Requires IAM permission: `redshift:GetClusterCredentials`
4. Or choose: **"Database username and password"**
   - Cluster: marketing-analytics-cluster
   - Database: marketing_db
   - User: admin
   - Password: (your password)
5. Click **"Connect"**

**Step 3: Run Queries**
```sql
-- Browser-based editor with autocomplete
SELECT schemaname, tablename,
       size_in_mb
FROM svv_table_info
ORDER BY size_in_mb DESC
LIMIT 10;
```

**Features:**
- **SQL autocomplete** (table names, columns)
- **Query history** (view past queries)
- **Export results** (CSV, JSON)
- **Share queries** with team members
- **Saved queries** (organize frequently used SQL)

**Limitations:**
- 100MB result set limit
- 5-minute query timeout
- No local file import (must use S3)

### Connection String Format

**Standard PostgreSQL format:**
```
postgresql://[user]:[password]@[host]:[port]/[database]
```

**Example:**
```
postgresql://admin:Marketing2024!Analytics@marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com:5439/marketing_db
```

**SQLAlchemy format:**
```python
from sqlalchemy import create_engine

# Redshift-specific dialect
connection_string = (
    "redshift+psycopg2://admin:Marketing2024!Analytics@"
    "marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com:5439/"
    "marketing_db"
)

engine = create_engine(connection_string)
```

**JDBC format (for Java/Spark):**
```
jdbc:redshift://marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com:5439/marketing_db?user=admin&password=Marketing2024!Analytics
```

### Credentials Management

**⚠️ NEVER hardcode credentials in code or commit to Git!**

#### Method 1: Environment Variables

```bash
# ~/.bashrc or ~/.zshrc
export REDSHIFT_HOST="marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com"
export REDSHIFT_PORT="5439"
export REDSHIFT_DB="marketing_db"
export REDSHIFT_USER="admin"
export REDSHIFT_PASSWORD="Marketing2024!Analytics"
```

```python
import os
import psycopg2

conn = psycopg2.connect(
    host=os.environ['REDSHIFT_HOST'],
    port=os.environ['REDSHIFT_PORT'],
    dbname=os.environ['REDSHIFT_DB'],
    user=os.environ['REDSHIFT_USER'],
    password=os.environ['REDSHIFT_PASSWORD']
)
```

#### Method 2: Config File (gitignored)

**config/database.ini:**
```ini
[redshift]
host = marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com
port = 5439
database = marketing_db
user = admin
password = Marketing2024!Analytics
```

**Python code:**
```python
import configparser
import psycopg2

def get_db_config(filename='config/database.ini', section='redshift'):
    parser = configparser.ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in {filename}')

    return db

# Usage
config = get_db_config()
conn = psycopg2.connect(**config)
```

**Add to .gitignore:**
```
config/database.ini
.env
credentials.json
```

#### Method 3: AWS Secrets Manager (Production)

```python
import boto3
import json
import psycopg2

def get_redshift_credentials(secret_name='redshift/marketing/credentials'):
    """Retrieve credentials from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name='us-east-1')

    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        raise

# Usage
creds = get_redshift_credentials()

conn = psycopg2.connect(
    host=creds['host'],
    port=creds['port'],
    dbname=creds['dbname'],
    user=creds['username'],
    password=creds['password']
)
```

**Create secret via CLI:**
```bash
aws secretsmanager create-secret \
    --name redshift/marketing/credentials \
    --description "Redshift credentials for marketing analytics" \
    --secret-string '{
        "host": "marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com",
        "port": "5439",
        "dbname": "marketing_db",
        "username": "admin",
        "password": "Marketing2024!Analytics"
    }'
```

#### Method 4: IAM Authentication (Most Secure)

No password needed - uses temporary credentials from AWS IAM.

```python
import boto3
import psycopg2

def get_temporary_credentials(cluster_id, db_user, db_name, region='us-east-1'):
    """Get temporary Redshift credentials using IAM."""
    client = boto3.client('redshift', region_name=region)

    response = client.get_cluster_credentials(
        ClusterIdentifier=cluster_id,
        DbUser=db_user,
        DbName=db_name,
        DurationSeconds=3600  # 1 hour
    )

    return response

# Usage
creds = get_temporary_credentials(
    cluster_id='marketing-analytics-cluster',
    db_user='admin',
    db_name='marketing_db'
)

conn = psycopg2.connect(
    host='marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com',
    port=5439,
    dbname='marketing_db',
    user=creds['DbUser'],
    password=creds['DbPassword']
)
```

**Required IAM permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "redshift:GetClusterCredentials",
      "Resource": [
        "arn:aws:redshift:us-east-1:123456789012:dbuser:marketing-analytics-cluster/admin",
        "arn:aws:redshift:us-east-1:123456789012:dbname:marketing-analytics-cluster/marketing_db"
      ]
    }
  ]
}
```

---

## 5. Python Integration

### Installing Required Libraries

```bash
# Core libraries
pip install psycopg2-binary sqlalchemy pandas

# Redshift-specific SQLAlchemy dialect
pip install sqlalchemy-redshift

# Optional: AWS SDK for Python
pip install boto3

# Optional: For parallel data loading
pip install redshift-connector
```

**Or use requirements.txt:**
```
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
sqlalchemy-redshift==0.8.14
pandas==2.1.4
boto3==1.34.10
numpy==1.26.2
```

```bash
pip install -r requirements.txt
```

### Connection Code Examples

#### Basic Connection with psycopg2

```python
import psycopg2
from psycopg2 import sql
import os

class RedshiftConnection:
    """Context manager for Redshift connections."""

    def __init__(self):
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = psycopg2.connect(
            host=os.environ['REDSHIFT_HOST'],
            port=os.environ['REDSHIFT_PORT'],
            dbname=os.environ['REDSHIFT_DB'],
            user=os.environ['REDSHIFT_USER'],
            password=os.environ['REDSHIFT_PASSWORD'],
            sslmode='require',
            connect_timeout=10
        )
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.cursor.close()
            self.conn.close()

# Usage
with RedshiftConnection() as cursor:
    cursor.execute("SELECT COUNT(*) FROM marketing_events")
    count = cursor.fetchone()[0]
    print(f"Total events: {count:,}")
```

#### SQLAlchemy Engine

```python
from sqlalchemy import create_engine, text
import os

def get_redshift_engine():
    """Create SQLAlchemy engine for Redshift."""
    connection_string = (
        f"redshift+psycopg2://{os.environ['REDSHIFT_USER']}:"
        f"{os.environ['REDSHIFT_PASSWORD']}@"
        f"{os.environ['REDSHIFT_HOST']}:{os.environ['REDSHIFT_PORT']}/"
        f"{os.environ['REDSHIFT_DB']}"
    )

    engine = create_engine(
        connection_string,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections after 1 hour
        echo=False           # Set True for SQL logging
    )

    return engine

# Usage
engine = get_redshift_engine()

# Execute query
with engine.connect() as conn:
    result = conn.execute(text("SELECT version()"))
    print(result.fetchone())
```

### Pandas Integration

#### Reading Data from Redshift

```python
import pandas as pd
from sqlalchemy import create_engine
import os

# Create engine
engine = get_redshift_engine()

# Method 1: Simple read
query = """
SELECT
    campaign_id,
    campaign_name,
    SUM(impressions) as total_impressions,
    SUM(clicks) as total_clicks,
    SUM(conversions) as total_conversions
FROM marketing_events
WHERE event_date >= '2024-01-01'
GROUP BY campaign_id, campaign_name
ORDER BY total_conversions DESC
"""

df = pd.read_sql(query, engine)
print(df.head())
```

**Output:**
```
   campaign_id       campaign_name  total_impressions  total_clicks  total_conversions
0            1         Summer Sale          1000000          25000                500
1            2       Black Friday          5000000         125000               3000
```

#### Writing Data to Redshift

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
campaigns = pd.DataFrame({
    'campaign_id': range(1, 101),
    'campaign_name': [f'Campaign_{i}' for i in range(1, 101)],
    'channel': np.random.choice(['Google', 'Facebook', 'Email', 'Display'], 100),
    'budget': np.random.randint(1000, 50000, 100),
    'start_date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'status': np.random.choice(['Active', 'Paused', 'Completed'], 100)
})

# Write to Redshift
campaigns.to_sql(
    name='campaigns',
    con=engine,
    schema='public',
    if_exists='replace',  # 'append', 'fail', or 'replace'
    index=False,
    method='multi',  # Batch inserts (faster)
    chunksize=1000   # Insert 1000 rows at a time
)

print("✓ Data loaded successfully")
```

### Best Practices for Large Datasets

#### 1. Use Chunked Reading for Large Tables

```python
def read_large_table(table_name, chunksize=100000):
    """Read large table in chunks to avoid memory issues."""
    query = f"SELECT * FROM {table_name}"

    chunks = []
    for chunk in pd.read_sql(query, engine, chunksize=chunksize):
        # Process each chunk
        processed_chunk = chunk[chunk['revenue'] > 0]  # Example filter
        chunks.append(processed_chunk)

    # Combine all chunks
    df = pd.concat(chunks, ignore_index=True)
    return df

# Usage
df = read_large_table('marketing_events', chunksize=50000)
```

#### 2. Use LIMIT for Prototyping

```python
# Don't load entire table during development
query_dev = "SELECT * FROM marketing_events LIMIT 10000"
df_sample = pd.read_sql(query_dev, engine)

# Develop your analysis on sample
# ...

# Then run on full dataset
query_prod = "SELECT * FROM marketing_events"
df_full = pd.read_sql(query_prod, engine)
```

#### 3. Push Computation to Redshift (Not Pandas)

**❌ BAD - Pull all data, then filter in Pandas:**
```python
# Loads 1 billion rows into memory!
df = pd.read_sql("SELECT * FROM events", engine)
df_filtered = df[df['event_date'] >= '2024-01-01']
```

**✅ GOOD - Filter in Redshift:**
```python
# Only loads filtered results
query = "SELECT * FROM events WHERE event_date >= '2024-01-01'"
df = pd.read_sql(query, engine)
```

#### 4. Use COPY Command for Bulk Loads (10x-100x Faster)

**Instead of `to_sql()` for large datasets, use COPY from S3:**

```python
import boto3
import pandas as pd

def load_to_redshift_via_s3(df, table_name, s3_bucket, s3_key):
    """Load large DataFrame to Redshift via S3 COPY command."""

    # Step 1: Save DataFrame to CSV
    csv_buffer = df.to_csv(index=False, sep='|', header=False)

    # Step 2: Upload to S3
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=s3_key,
        Body=csv_buffer
    )

    # Step 3: Execute COPY command in Redshift
    copy_query = f"""
    COPY {table_name}
    FROM 's3://{s3_bucket}/{s3_key}'
    IAM_ROLE 'arn:aws:iam::123456789012:role/RedshiftS3AccessRole'
    DELIMITER '|'
    IGNOREHEADER 1
    DATEFORMAT 'auto'
    TIMEFORMAT 'auto'
    COMPUPDATE ON
    STATUPDATE ON;
    """

    with engine.connect() as conn:
        conn.execute(text(copy_query))
        conn.commit()

    print(f"✓ Loaded {len(df):,} rows to {table_name}")

# Usage
large_df = pd.read_csv('large_marketing_data.csv')
load_to_redshift_via_s3(
    df=large_df,
    table_name='marketing_events',
    s3_bucket='my-marketing-data',
    s3_key='events/2024-11-16.csv'
)
```

**Performance comparison:**
```
Method             | 1M rows | 10M rows | 100M rows
-------------------|---------|----------|----------
to_sql()           | 5 min   | 50 min   | 8 hours
COPY from S3       | 30 sec  | 3 min    | 30 min
```

#### 5. Use Redshift Data Types Efficiently

```python
# Optimize DataFrame dtypes before loading
df['campaign_id'] = df['campaign_id'].astype('int32')  # Not int64
df['impressions'] = df['impressions'].astype('int64')
df['ctr'] = df['ctr'].astype('float32')  # Not float64
df['event_date'] = pd.to_datetime(df['event_date'])

# Specify dtypes explicitly
dtype_mapping = {
    'campaign_id': 'INTEGER',
    'campaign_name': 'VARCHAR(255)',
    'impressions': 'BIGINT',
    'clicks': 'INTEGER',
    'ctr': 'REAL',
    'event_date': 'DATE'
}

df.to_sql('campaigns', engine, dtype=dtype_mapping, if_exists='replace')
```

### Query Optimization Tips

#### 1. Use EXPLAIN to Analyze Query Plans

```python
def explain_query(query):
    """Show query execution plan."""
    explain_query = f"EXPLAIN {query}"

    with engine.connect() as conn:
        result = conn.execute(text(explain_query))
        for row in result:
            print(row[0])

# Usage
query = """
SELECT campaign_name, SUM(revenue)
FROM marketing_events
WHERE event_date >= '2024-01-01'
GROUP BY campaign_name
"""

explain_query(query)
```

**Look for:**
- **Seq Scan** (slow) vs **Index Scan** (fast)
- **Hash Join** (fast) vs **Nested Loop** (slow for large tables)
- **Sort** operations (use sort keys to avoid)

#### 2. Use Distribution Keys for JOIN Performance

```python
# Create tables with distribution keys
create_table_query = """
CREATE TABLE campaigns (
    campaign_id INTEGER DISTKEY SORTKEY,
    campaign_name VARCHAR(255),
    channel VARCHAR(50)
);

CREATE TABLE events (
    event_id BIGINT IDENTITY(1,1),
    campaign_id INTEGER DISTKEY,  -- Same as campaigns.campaign_id
    event_date DATE SORTKEY,
    impressions INTEGER,
    clicks INTEGER
);
"""

with engine.connect() as conn:
    conn.execute(text(create_table_query))
    conn.commit()
```

**Why this matters:**
- Rows with same `campaign_id` are stored on same node
- JOINs don't require data movement across nodes
- 10x-100x faster JOINs

#### 3. Use WHERE Clauses with Sort Keys

```python
# ✅ GOOD - Uses sort key (event_date) in WHERE
query_fast = """
SELECT * FROM events
WHERE event_date BETWEEN '2024-01-01' AND '2024-01-31'
"""

# ❌ SLOW - No sort key in WHERE
query_slow = """
SELECT * FROM events
WHERE campaign_id = 123
"""
# Even though there's a DISTKEY, without sort key filter, it scans all blocks
```

#### 4. Use Columnar Compression

```python
# Analyze table to apply compression
analyze_query = "ANALYZE marketing_events;"

with engine.connect() as conn:
    conn.execute(text(analyze_query))
    conn.commit()

# Check compression
compression_query = """
SELECT
    schemaname,
    tablename,
    column_name,
    type,
    encoding
FROM pg_table_def
WHERE tablename = 'marketing_events'
ORDER BY column_name;
"""

df_compression = pd.read_sql(compression_query, engine)
print(df_compression)
```

**Expected encodings:**
- **LZO**: General purpose
- **ZSTD**: High compression (default for new tables)
- **DELTA**: Good for timestamps, sequential IDs
- **RUNLENGTH**: Good for low-cardinality columns (status, category)

#### 5. Use VACUUM and ANALYZE Regularly

```python
def optimize_table(table_name):
    """Optimize table performance."""
    with engine.connect() as conn:
        # Reclaim space and sort rows
        conn.execute(text(f"VACUUM {table_name};"))

        # Update statistics for query planner
        conn.execute(text(f"ANALYZE {table_name};"))

        conn.commit()

    print(f"✓ Optimized {table_name}")

# Run after large data loads or deletes
optimize_table('marketing_events')
```

---

## 6. Sample Marketing Schema

### Multi-Table Schema Design for Marketing Analytics

**Entity-Relationship Diagram (ERD):**
```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│   CAMPAIGNS     │       │     EVENTS       │       │   CONVERSIONS   │
├─────────────────┤       ├──────────────────┤       ├─────────────────┤
│ campaign_id (PK)│◄──────┤ campaign_id (FK) │       │ conversion_id   │
│ campaign_name   │       │ event_id (PK)    │       │ event_id (FK)   │───┐
│ channel         │       │ event_date       │       │ conversion_date │   │
│ budget          │       │ user_id          │       │ revenue         │   │
│ start_date      │       │ impressions      │       │ product_id      │   │
│ end_date        │       │ clicks           │       └─────────────────┘   │
│ status          │       │ cost             │                             │
└─────────────────┘       └──────────────────┘                             │
                                  │                                         │
                                  └─────────────────────────────────────────┘

┌─────────────────┐       ┌──────────────────┐
│     CHANNELS    │       │      USERS       │
├─────────────────┤       ├──────────────────┤
│ channel_id (PK) │       │ user_id (PK)     │
│ channel_name    │       │ first_seen       │
│ cost_per_click  │       │ country          │
│ category        │       │ device_type      │
└─────────────────┘       │ user_segment     │
                          └──────────────────┘
```

### Creating Tables with Distribution Keys

```sql
-- Drop existing tables (if any)
DROP TABLE IF EXISTS conversions CASCADE;
DROP TABLE IF EXISTS events CASCADE;
DROP TABLE IF EXISTS campaigns CASCADE;
DROP TABLE IF EXISTS channels CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1. CAMPAIGNS TABLE
-- Distribution: KEY (campaign_id) - for JOINs with events
-- Sort: campaign_id for range queries
CREATE TABLE campaigns (
    campaign_id INTEGER PRIMARY KEY DISTKEY SORTKEY,
    campaign_name VARCHAR(255) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    budget DECIMAL(12,2),
    start_date DATE NOT NULL,
    end_date DATE,
    target_audience VARCHAR(100),
    status VARCHAR(20) DEFAULT 'Active',
    created_at TIMESTAMP DEFAULT GETDATE()
)
DISTSTYLE KEY;

-- 2. EVENTS TABLE (Largest table - fact table)
-- Distribution: KEY (campaign_id) - co-located with campaigns
-- Sort: event_date for time-series queries
CREATE TABLE events (
    event_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    campaign_id INTEGER NOT NULL DISTKEY,
    event_date DATE NOT NULL SORTKEY,
    event_timestamp TIMESTAMP NOT NULL,
    user_id BIGINT,
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    cost DECIMAL(10,2),
    device_type VARCHAR(20),
    geo_country VARCHAR(2),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
)
DISTSTYLE KEY;

-- 3. CONVERSIONS TABLE
-- Distribution: KEY (campaign_id) - co-located with events
-- Sort: conversion_date for time-series analysis
CREATE TABLE conversions (
    conversion_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    event_id BIGINT,
    campaign_id INTEGER NOT NULL DISTKEY,
    conversion_date DATE NOT NULL SORTKEY,
    conversion_timestamp TIMESTAMP NOT NULL,
    user_id BIGINT NOT NULL,
    revenue DECIMAL(10,2) NOT NULL,
    product_id INTEGER,
    product_category VARCHAR(100),
    quantity INTEGER DEFAULT 1,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
)
DISTSTYLE KEY;

-- 4. CHANNELS TABLE (Dimension table - small)
-- Distribution: ALL - replicate on all nodes for faster JOINs
CREATE TABLE channels (
    channel_id INTEGER PRIMARY KEY,
    channel_name VARCHAR(50) UNIQUE NOT NULL,
    channel_category VARCHAR(50),
    avg_cpc DECIMAL(6,2),
    avg_cpm DECIMAL(6,2),
    platform VARCHAR(50)
)
DISTSTYLE ALL;

-- 5. USERS TABLE
-- Distribution: KEY (user_id) - for user-level analysis
-- Sort: first_seen for cohort analysis
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY DISTKEY,
    first_seen TIMESTAMP NOT NULL SORTKEY,
    last_seen TIMESTAMP,
    country VARCHAR(2),
    device_type VARCHAR(20),
    user_segment VARCHAR(50),
    ltv DECIMAL(10,2) DEFAULT 0,
    total_conversions INTEGER DEFAULT 0
)
DISTSTYLE KEY;
```

### Setting Sort Keys for Performance

**Why sort keys matter:**
- Redshift stores data in 1MB blocks
- Sort keys minimize blocks scanned
- Critical for WHERE clauses and JOINs

**Compound Sort Key (multiple columns):**
```sql
-- For queries filtering by campaign AND date
CREATE TABLE events_v2 (
    event_id BIGINT IDENTITY(1,1),
    campaign_id INTEGER DISTKEY,
    event_date DATE,
    impressions INTEGER,
    COMPOUND SORTKEY (campaign_id, event_date)  -- Order matters!
);

-- Optimized for:
-- WHERE campaign_id = 123 AND event_date >= '2024-01-01'
```

**Interleaved Sort Key (equal weight to all columns):**
```sql
-- For queries filtering by ANY combination of columns
CREATE TABLE events_v3 (
    event_id BIGINT IDENTITY(1,1),
    campaign_id INTEGER,
    event_date DATE,
    user_id BIGINT,
    INTERLEAVED SORTKEY (campaign_id, event_date, user_id)
);

-- Optimized for:
-- WHERE campaign_id = 123
-- WHERE event_date >= '2024-01-01'
-- WHERE user_id = 456
-- WHERE campaign_id = 123 AND user_id = 456
```

**⚠️ Trade-off**: Interleaved sort keys are slower to load/VACUUM.

### Sample DDL Statements

**Complete schema creation script:**

```sql
-- =============================================================================
-- MARKETING ANALYTICS SCHEMA - AMAZON REDSHIFT
-- =============================================================================

-- Create schema (optional - organize tables)
CREATE SCHEMA IF NOT EXISTS marketing;
SET search_path TO marketing, public;

-- -----------------------------------------------------------------------------
-- 1. DIMENSION TABLES (small, lookup tables)
-- -----------------------------------------------------------------------------

-- Channels (replicated on all nodes)
CREATE TABLE marketing.dim_channels (
    channel_id INTEGER PRIMARY KEY,
    channel_name VARCHAR(50) NOT NULL,
    channel_category VARCHAR(50),
    avg_cpc DECIMAL(6,2),
    avg_cpm DECIMAL(6,2),
    platform VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE
)
DISTSTYLE ALL
SORTKEY (channel_name);

-- Geographies (replicated on all nodes)
CREATE TABLE marketing.dim_geo (
    geo_id INTEGER PRIMARY KEY,
    country_code VARCHAR(2) NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    continent VARCHAR(50)
)
DISTSTYLE ALL
SORTKEY (country_code);

-- Products (replicated on all nodes)
CREATE TABLE marketing.dim_products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    product_category VARCHAR(100),
    price DECIMAL(10,2),
    margin_pct DECIMAL(5,2)
)
DISTSTYLE ALL
SORTKEY (product_category, product_name);

-- -----------------------------------------------------------------------------
-- 2. FACT TABLES (large, transaction tables)
-- -----------------------------------------------------------------------------

-- Campaigns
CREATE TABLE marketing.fact_campaigns (
    campaign_id INTEGER PRIMARY KEY DISTKEY SORTKEY,
    campaign_name VARCHAR(255) NOT NULL ENCODE LZO,
    channel_id INTEGER NOT NULL,
    budget DECIMAL(12,2) ENCODE ZSTD,
    start_date DATE NOT NULL ENCODE DELTA,
    end_date DATE ENCODE DELTA,
    target_audience VARCHAR(100) ENCODE LZO,
    status VARCHAR(20) DEFAULT 'Active' ENCODE RUNLENGTH,
    created_at TIMESTAMP DEFAULT GETDATE() ENCODE ZSTD,
    FOREIGN KEY (channel_id) REFERENCES marketing.dim_channels(channel_id)
)
DISTSTYLE KEY;

-- Marketing Events (largest table)
CREATE TABLE marketing.fact_events (
    event_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    campaign_id INTEGER NOT NULL DISTKEY,
    event_date DATE NOT NULL ENCODE DELTA,
    event_timestamp TIMESTAMP NOT NULL ENCODE ZSTD,
    user_id BIGINT ENCODE ZSTD,
    impressions INTEGER DEFAULT 0 ENCODE ZSTD,
    clicks INTEGER DEFAULT 0 ENCODE ZSTD,
    cost DECIMAL(10,2) ENCODE ZSTD,
    device_type VARCHAR(20) ENCODE RUNLENGTH,
    geo_id INTEGER ENCODE ZSTD,
    FOREIGN KEY (campaign_id) REFERENCES marketing.fact_campaigns(campaign_id),
    FOREIGN KEY (geo_id) REFERENCES marketing.dim_geo(geo_id),
    COMPOUND SORTKEY (campaign_id, event_date)
)
DISTSTYLE KEY;

-- Conversions
CREATE TABLE marketing.fact_conversions (
    conversion_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    event_id BIGINT ENCODE ZSTD,
    campaign_id INTEGER NOT NULL DISTKEY,
    conversion_date DATE NOT NULL ENCODE DELTA,
    conversion_timestamp TIMESTAMP NOT NULL ENCODE ZSTD,
    user_id BIGINT NOT NULL ENCODE ZSTD,
    revenue DECIMAL(10,2) NOT NULL ENCODE ZSTD,
    product_id INTEGER ENCODE ZSTD,
    quantity INTEGER DEFAULT 1 ENCODE ZSTD,
    FOREIGN KEY (campaign_id) REFERENCES marketing.fact_campaigns(campaign_id),
    FOREIGN KEY (product_id) REFERENCES marketing.dim_products(product_id),
    COMPOUND SORTKEY (campaign_id, conversion_date)
)
DISTSTYLE KEY;

-- User Activity
CREATE TABLE marketing.fact_users (
    user_id BIGINT PRIMARY KEY DISTKEY,
    first_seen TIMESTAMP NOT NULL SORTKEY ENCODE ZSTD,
    last_seen TIMESTAMP ENCODE ZSTD,
    geo_id INTEGER ENCODE ZSTD,
    device_type VARCHAR(20) ENCODE RUNLENGTH,
    user_segment VARCHAR(50) ENCODE LZO,
    ltv DECIMAL(10,2) DEFAULT 0 ENCODE ZSTD,
    total_conversions INTEGER DEFAULT 0 ENCODE ZSTD,
    FOREIGN KEY (geo_id) REFERENCES marketing.dim_geo(geo_id)
)
DISTSTYLE KEY;

-- -----------------------------------------------------------------------------
-- 3. GRANT PERMISSIONS (if using multiple users)
-- -----------------------------------------------------------------------------

GRANT ALL ON SCHEMA marketing TO admin;
GRANT SELECT ON ALL TABLES IN SCHEMA marketing TO analyst_role;
```

### Loading Sample Data

#### Method 1: Direct INSERT (Small datasets)

```sql
-- Insert sample channels
INSERT INTO marketing.dim_channels VALUES
(1, 'Google Ads', 'Paid Search', 1.50, 8.00, 'Google'),
(2, 'Facebook Ads', 'Paid Social', 0.75, 12.00, 'Meta'),
(3, 'Email', 'Owned', 0.00, 0.00, 'Internal'),
(4, 'Display', 'Programmatic', 0.50, 5.00, 'Various'),
(5, 'TikTok Ads', 'Paid Social', 0.65, 10.00, 'TikTok');

-- Insert sample geographies
INSERT INTO marketing.dim_geo VALUES
(1, 'US', 'United States', 'North America', 'Americas'),
(2, 'CA', 'Canada', 'North America', 'Americas'),
(3, 'GB', 'United Kingdom', 'Western Europe', 'Europe'),
(4, 'DE', 'Germany', 'Western Europe', 'Europe'),
(5, 'AU', 'Australia', 'Oceania', 'Oceania');

-- Insert sample products
INSERT INTO marketing.dim_products VALUES
(1, 'Premium Subscription', 'Subscription', 99.99, 85.00),
(2, 'Basic Subscription', 'Subscription', 49.99, 80.00),
(3, 'Pro Tools Add-on', 'Add-on', 29.99, 90.00),
(4, 'Enterprise License', 'License', 999.99, 70.00);

-- Insert sample campaigns
INSERT INTO marketing.fact_campaigns
(campaign_id, campaign_name, channel_id, budget, start_date, end_date, target_audience, status)
VALUES
(1, 'Summer Sale 2024', 1, 50000.00, '2024-06-01', '2024-08-31', 'All Users', 'Completed'),
(2, 'Black Friday 2024', 2, 100000.00, '2024-11-01', '2024-11-30', 'High Intent', 'Active'),
(3, 'Email Nurture Q4', 3, 5000.00, '2024-10-01', '2024-12-31', 'Existing Users', 'Active'),
(4, 'Brand Awareness Display', 4, 75000.00, '2024-01-01', '2024-12-31', 'New Users', 'Active');
```

#### Method 2: COPY from S3 (Large datasets)

```python
# Python script to generate and load sample data

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import boto3
from sqlalchemy import create_engine, text

# Generate sample events (1 million rows)
def generate_sample_events(n_rows=1000000):
    np.random.seed(42)

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(300)]

    events = pd.DataFrame({
        'campaign_id': np.random.choice([1, 2, 3, 4], n_rows, p=[0.3, 0.4, 0.2, 0.1]),
        'event_date': np.random.choice(dates, n_rows),
        'event_timestamp': pd.date_range('2024-01-01', periods=n_rows, freq='10S'),
        'user_id': np.random.randint(1, 100000, n_rows),
        'impressions': np.random.randint(1, 1000, n_rows),
        'clicks': np.random.randint(0, 50, n_rows),
        'cost': np.round(np.random.uniform(0.1, 100, n_rows), 2),
        'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], n_rows, p=[0.4, 0.5, 0.1]),
        'geo_id': np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    })

    return events

# Generate sample conversions (50k rows, ~5% conversion rate)
def generate_sample_conversions(events_df, conversion_rate=0.05):
    n_conversions = int(len(events_df) * conversion_rate)

    conversions = pd.DataFrame({
        'event_id': np.random.choice(range(1, len(events_df)), n_conversions, replace=False),
        'campaign_id': np.random.choice([1, 2, 3, 4], n_conversions),
        'conversion_date': pd.date_range('2024-01-01', periods=n_conversions, freq='1H'),
        'conversion_timestamp': pd.date_range('2024-01-01', periods=n_conversions, freq='1H'),
        'user_id': np.random.randint(1, 100000, n_conversions),
        'revenue': np.round(np.random.choice([49.99, 99.99, 29.99, 999.99], n_conversions), 2),
        'product_id': np.random.choice([1, 2, 3, 4], n_conversions),
        'quantity': np.random.choice([1, 1, 1, 2, 3], n_conversions, p=[0.7, 0.15, 0.1, 0.04, 0.01])
    })

    return conversions

# Upload to S3 and load to Redshift
def load_to_redshift_via_s3(df, table_name, s3_bucket, s3_key, iam_role_arn):
    """Load DataFrame to Redshift via S3 COPY."""

    # Save to CSV
    csv_file = f'/tmp/{table_name}.csv'
    df.to_csv(csv_file, index=False, sep='|', header=False)

    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(csv_file, s3_bucket, s3_key)
    print(f"✓ Uploaded {s3_key} to S3")

    # Execute COPY command
    copy_query = f"""
    COPY marketing.{table_name}
    FROM 's3://{s3_bucket}/{s3_key}'
    IAM_ROLE '{iam_role_arn}'
    DELIMITER '|'
    DATEFORMAT 'auto'
    TIMEFORMAT 'auto'
    COMPUPDATE ON
    STATUPDATE ON
    MAXERROR 10;
    """

    engine = get_redshift_engine()
    with engine.connect() as conn:
        result = conn.execute(text(copy_query))
        conn.commit()

    print(f"✓ Loaded {len(df):,} rows to {table_name}")

# Main execution
if __name__ == "__main__":
    # Generate data
    print("Generating sample events...")
    events = generate_sample_events(n_rows=1000000)

    print("Generating sample conversions...")
    conversions = generate_sample_conversions(events)

    # Load to Redshift
    S3_BUCKET = 'my-marketing-data'
    IAM_ROLE = 'arn:aws:iam::123456789012:role/RedshiftS3AccessRole'

    load_to_redshift_via_s3(
        df=events,
        table_name='fact_events',
        s3_bucket=S3_BUCKET,
        s3_key='sample-data/events.csv',
        iam_role_arn=IAM_ROLE
    )

    load_to_redshift_via_s3(
        df=conversions,
        table_name='fact_conversions',
        s3_bucket=S3_BUCKET,
        s3_key='sample-data/conversions.csv',
        iam_role_arn=IAM_ROLE
    )

    print("✓ Sample data loaded successfully!")
```

---

## 7. Cost Management

### Understanding Redshift Pricing

**Pricing Components:**

1. **Compute (Nodes)**
   - **DC2 (Dense Compute)**: SSD storage, compute-optimized
   - **RA3 (Managed Storage)**: Scales compute and storage independently
   - **Charged by**: Node-hour (billed per second, 60-second minimum)

2. **Storage (RA3 only)**
   - **Managed Storage**: $0.024/GB-month
   - Scales independently from compute
   - Automatically moved to S3 when not accessed

3. **Redshift Spectrum**
   - Query data in S3 without loading
   - $5.00 per TB scanned

4. **Data Transfer**
   - **Within same AZ**: Free
   - **Between AZs**: $0.01/GB
   - **To internet**: $0.09/GB (first 1 GB free)

**Pricing Examples (us-east-1, as of 2024):**

| Configuration | Nodes | vCPU | Memory | Storage | $/Hour | $/Month (24/7) |
|---------------|-------|------|--------|---------|--------|----------------|
| dc2.large (1 node) | 1 | 2 | 15 GB | 160 GB | $0.25 | $180 |
| dc2.large (2 nodes) | 2 | 4 | 30 GB | 320 GB | $0.50 | $360 |
| dc2.8xlarge (2 nodes) | 2 | 64 | 488 GB | 5.12 TB | $9.60 | $6,912 |
| ra3.xlplus (2 nodes) | 2 | 8 | 64 GB | Managed | $2.16 | $1,555 |
| ra3.4xlarge (2 nodes) | 2 | 24 | 192 GB | Managed | $6.52 | $4,694 |

**+ Storage (RA3):**
```
1 TB × $0.024/GB = $24/month
10 TB × $0.024/GB = $240/month
```

### Pausing Clusters When Not in Use

**⚠️ Important**: Only **multi-node DC2/RA3 clusters** can be paused (not single-node).

#### Pause via AWS Console

1. Redshift Console → Clusters → Select cluster
2. Click **"Actions"** → **"Pause"**
3. Confirm pause

**During pause:**
- ✅ No compute charges
- ✅ Storage charges continue (RA3 only)
- ❌ Cannot query data
- ❌ Snapshots continue (charged)

**Resume:**
- Takes 1-3 minutes
- Click **"Actions"** → **"Resume"**

#### Pause via AWS CLI

```bash
# Pause cluster
aws redshift pause-cluster \
    --cluster-identifier marketing-analytics-cluster

# Resume cluster
aws redshift resume-cluster \
    --cluster-identifier marketing-analytics-cluster

# Check status
aws redshift describe-clusters \
    --cluster-identifier marketing-analytics-cluster \
    --query 'Clusters[0].ClusterStatus' \
    --output text
```

#### Automated Pause/Resume with Lambda

**Schedule pause at night, resume in morning:**

```python
# lambda_function.py
import boto3
import os

redshift = boto3.client('redshift')
CLUSTER_ID = os.environ['CLUSTER_ID']

def lambda_handler(event, context):
    action = event.get('action')  # 'pause' or 'resume'

    try:
        if action == 'pause':
            redshift.pause_cluster(ClusterIdentifier=CLUSTER_ID)
            return {'statusCode': 200, 'body': f'Paused {CLUSTER_ID}'}

        elif action == 'resume':
            redshift.resume_cluster(ClusterIdentifier=CLUSTER_ID)
            return {'statusCode': 200, 'body': f'Resumed {CLUSTER_ID}'}

        else:
            return {'statusCode': 400, 'body': 'Invalid action'}

    except Exception as e:
        return {'statusCode': 500, 'body': str(e)}
```

**EventBridge Schedule:**
```bash
# Pause at 6 PM (18:00 UTC)
aws events put-rule \
    --name pause-redshift-evening \
    --schedule-expression "cron(0 18 * * ? *)"

# Resume at 8 AM (08:00 UTC)
aws events put-rule \
    --name resume-redshift-morning \
    --schedule-expression "cron(0 8 * * MON-FRI *)"
```

**Cost savings:**
```
Scenario: 2-node DC2 cluster, paused 16 hours/day
- Without pause: 24 hrs × 30 days × $0.50/hr = $360/month
- With pause: 8 hrs × 30 days × $0.50/hr = $120/month
Savings: $240/month (67%)
```

### Monitoring Usage and Costs

#### CloudWatch Metrics

**Key metrics to monitor:**

1. **CPUUtilization**: % of CPU used
2. **DatabaseConnections**: Number of active connections
3. **PercentageDiskSpaceUsed**: Storage utilization
4. **ReadIOPS / WriteIOPS**: I/O operations
5. **QueryDuration**: Average query execution time

**View in AWS Console:**
```
Redshift → Clusters → Select cluster → Monitoring tab
```

**Query CloudWatch via CLI:**
```bash
aws cloudwatch get-metric-statistics \
    --namespace AWS/Redshift \
    --metric-name CPUUtilization \
    --dimensions Name=ClusterIdentifier,Value=marketing-analytics-cluster \
    --start-time 2024-11-01T00:00:00Z \
    --end-time 2024-11-16T00:00:00Z \
    --period 3600 \
    --statistics Average \
    --output table
```

#### Cost Explorer

**View Redshift costs:**
1. AWS Console → Cost Explorer
2. Filters:
   - **Service**: Amazon Redshift
   - **Time range**: Last 30 days
3. Group by: **Usage Type** (to see compute vs storage breakdown)

#### Query System Tables

**Find most expensive queries:**
```sql
SELECT
    userid,
    query,
    service_class,
    elapsed / 1000000.0 AS elapsed_seconds,
    query_cpu_time / 1000000.0 AS cpu_seconds,
    TRIM(querytxt) AS query_text
FROM stl_query
WHERE userid > 1  -- Exclude system queries
  AND starttime >= DATEADD(day, -7, GETDATE())
ORDER BY elapsed DESC
LIMIT 20;
```

**Monitor storage usage:**
```sql
SELECT
    schemaname,
    tablename,
    ROUND(size_in_mb, 2) AS size_mb,
    ROUND(size_in_mb / 1024.0, 2) AS size_gb,
    rows
FROM (
    SELECT
        schemaname,
        tablename,
        SUM(mbytes) AS size_in_mb,
        SUM(num_rows) AS rows
    FROM (
        SELECT
            n.nspname AS schemaname,
            c.relname AS tablename,
            (c.reltuples)::BIGINT AS num_rows,
            ((c.relpages * 8.0) / 1024.0) AS mbytes
        FROM pg_class c
        LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'r'
    ) sub
    GROUP BY schemaname, tablename
) sizes
ORDER BY size_in_mb DESC
LIMIT 20;
```

**Active connections:**
```sql
SELECT
    usename,
    COUNT(*) AS connection_count,
    MAX(query_start) AS last_query
FROM pg_stat_activity
WHERE usename != 'rdsdb'
GROUP BY usename
ORDER BY connection_count DESC;
```

### Setting Up Billing Alerts

#### Step 1: Enable Billing Alerts

1. AWS Console → Account (top right) → **Billing and Cost Management**
2. Left sidebar → **Billing preferences**
3. Check **"Receive Billing Alerts"**
4. Save preferences

#### Step 2: Create CloudWatch Alarm

```bash
# Create SNS topic for billing alerts
aws sns create-topic --name billing-alerts

# Subscribe email to topic
aws sns subscribe \
    --topic-arn arn:aws:sns:us-east-1:123456789012:billing-alerts \
    --protocol email \
    --notification-endpoint your-email@example.com

# Confirm subscription (check email)

# Create CloudWatch alarm
aws cloudwatch put-metric-alarm \
    --alarm-name redshift-monthly-cost-alert \
    --alarm-description "Alert when Redshift monthly cost exceeds $50" \
    --namespace AWS/Billing \
    --metric-name EstimatedCharges \
    --dimensions Name=ServiceName,Value=AmazonRedshift \
    --statistic Maximum \
    --period 21600 \
    --evaluation-periods 1 \
    --threshold 50 \
    --comparison-operator GreaterThanThreshold \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts
```

**Multi-tier alerts:**
```bash
# Warning at $30
aws cloudwatch put-metric-alarm \
    --alarm-name redshift-cost-warning \
    --threshold 30 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts

# Critical at $50
aws cloudwatch put-metric-alarm \
    --alarm-name redshift-cost-critical \
    --threshold 50 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts
```

### Cleanup Instructions

**When you're done with learning/testing:**

#### 1. Delete Cluster

**⚠️ WARNING**: This permanently deletes all data! Create final snapshot if needed.

**Via Console:**
1. Redshift → Clusters → Select cluster
2. **Actions** → **Delete**
3. Options:
   - **Create final snapshot**: Yes (recommended) or No
   - **Snapshot identifier**: `marketing-cluster-final-snapshot`
4. Type cluster name to confirm
5. Click **Delete cluster**

**Via CLI:**
```bash
# With final snapshot (recommended)
aws redshift delete-cluster \
    --cluster-identifier marketing-analytics-cluster \
    --final-cluster-snapshot-identifier marketing-cluster-final-20241116

# Without snapshot (⚠️ data will be lost!)
aws redshift delete-cluster \
    --cluster-identifier marketing-analytics-cluster \
    --skip-final-cluster-snapshot
```

#### 2. Delete Snapshots

**Automated snapshots** are deleted automatically after retention period. **Manual snapshots** must be deleted manually.

```bash
# List snapshots
aws redshift describe-cluster-snapshots \
    --cluster-identifier marketing-analytics-cluster

# Delete specific snapshot
aws redshift delete-cluster-snapshot \
    --snapshot-identifier marketing-cluster-final-20241116
```

#### 3. Delete S3 Data

```bash
# Delete sample data from S3
aws s3 rm s3://my-marketing-data/sample-data/ --recursive
```

#### 4. Delete IAM Roles

```bash
# Detach policy
aws iam detach-role-policy \
    --role-name RedshiftS3AccessRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Delete role
aws iam delete-role --role-name RedshiftS3AccessRole
```

#### 5. Verify No Charges

Wait 24-48 hours, then check:
```
AWS Console → Billing Dashboard → Bills

Ensure Redshift charges = $0.00
```

---

## 8. Alternative: Redshift Serverless

### What is Redshift Serverless?

**Redshift Serverless** automatically provisions and scales data warehouse capacity. No need to choose node types or cluster sizes.

**Key Differences:**

| Feature | Provisioned Cluster | Serverless |
|---------|---------------------|------------|
| **Setup** | Choose node type/count | Automatic |
| **Scaling** | Manual resize | Auto-scales |
| **Pricing** | Per-hour (even when idle) | Per-second (only when running queries) |
| **Pause** | Manual | Automatic (after inactivity) |
| **Best For** | 24/7 workloads, predictable usage | Intermittent, variable workloads |

### When to Use Redshift Serverless

**✅ Use Serverless When:**
- Running queries **intermittently** (not 24/7)
- Unpredictable workload patterns
- Learning/development (pay only for query time)
- Variable user concurrency
- Don't want to manage infrastructure

**❌ Stick with Provisioned When:**
- 24/7 production workloads
- Predictable, steady usage
- Need maximum control over performance
- Reserved Instance discounts make sense

### Setup Instructions

#### Step 1: Create Namespace

**Namespace** = logical container for databases, users, permissions

1. Redshift Console → **Redshift Serverless**
2. Click **"Create workgroup"**
3. **Namespace settings:**
   - **Namespace name**: `marketing-serverless`
   - **Database name**: `marketing_db`
   - **Admin username**: `admin`
   - **Admin password**: (create secure password)
4. **Encryption**: Default (AWS-managed key)
5. **Permissions**: Attach IAM role for S3 access (optional)
6. Click **"Next"**

#### Step 2: Create Workgroup

**Workgroup** = compute resources for running queries

1. **Workgroup name**: `marketing-workgroup`
2. **Base capacity**: 8 RPU (Redshift Processing Units)
   - Minimum: 8 RPU (approximately 2 nodes)
   - Each RPU = ~2 vCPUs + 16 GB memory
   - Auto-scales up to 512 RPU based on workload
3. **Network and security:**
   - **VPC**: Default VPC
   - **Subnets**: Select available subnets
   - **Security groups**: Default (modify later for access)
   - **Publicly accessible**: Yes (for learning)
4. **Limits:**
   - **Maximum RPU**: 512 (default, can adjust)
5. Click **"Create workgroup"**

**Creation time**: 2-3 minutes

#### Step 3: Configure Security Group

Same as provisioned cluster:
```
1. Serverless dashboard → Click workgroup
2. Data access → Security groups → Edit
3. Add inbound rule:
   Type: Custom TCP
   Port: 5439
   Source: My IP
```

#### Step 4: Get Endpoint

```
Serverless dashboard → Workgroups → marketing-workgroup → General information

Endpoint example:
marketing-workgroup.123456789012.us-east-1.redshift-serverless.amazonaws.com:5439
```

### Connection from Python

**Identical to provisioned cluster:**

```python
import psycopg2

conn = psycopg2.connect(
    host='marketing-workgroup.123456789012.us-east-1.redshift-serverless.amazonaws.com',
    port=5439,
    dbname='marketing_db',
    user='admin',
    password='your-password'
)

cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())
cur.close()
conn.close()
```

### Cost Comparison

**Pricing: $0.36 per RPU-hour** (billed per second)

**Example scenarios:**

**Scenario 1: Learning (2 hours/day)**
```
Usage: 2 hours/day × 30 days = 60 hours/month
Base capacity: 8 RPU
Cost: 60 hours × 8 RPU × $0.36 = $172.80/month

Provisioned equivalent (dc2.large, paused 22 hrs/day):
2 hours × 30 days × $0.25 = $15/month

Winner: Provisioned (with pause)
```

**Scenario 2: Intermittent queries (10 queries/day, 5 min each)**
```
Usage: 10 × 5 min = 50 min/day = 25 hours/month
Base capacity: 8 RPU
Cost: 25 hours × 8 RPU × $0.36 = $72/month

Provisioned (cannot pause between queries):
24/7: 720 hours × $0.25 = $180/month

Winner: Serverless
```

**Scenario 3: Business hours only (8 AM - 6 PM, Mon-Fri)**
```
Usage: 10 hours/day × 22 days = 220 hours/month
Base capacity: 8 RPU (scales to ~16 during peak)
Average: 12 RPU
Cost: 220 hours × 12 RPU × $0.36 = $950/month

Provisioned (paused nights/weekends):
220 hours × $0.50 (2 nodes) = $110/month

Winner: Provisioned (with pause)
```

**Key Insight**: Serverless is cost-effective for **truly intermittent** workloads, but provisioned clusters with pause are cheaper for regular schedules.

### Managing Serverless Costs

**1. Set RPU Limits**
```
Workgroup settings → Edit → Limits
Set maximum RPU: 32 (instead of 512)
```

**2. Auto-pause After Inactivity**

Serverless automatically pauses after **no queries for configured time** (default: none).

**Enable via CLI:**
```bash
aws redshift-serverless update-workgroup \
    --workgroup-name marketing-workgroup \
    --max-capacity 32 \
    --base-capacity 8
```

**3. Monitor RPU Usage**

```sql
-- Query Redshift system tables
SELECT
    DATE_TRUNC('hour', start_time) AS hour,
    AVG(capacity_used) AS avg_rpu,
    MAX(capacity_used) AS peak_rpu,
    COUNT(*) AS query_count
FROM sys_serverless_usage
WHERE start_time >= DATEADD(day, -7, GETDATE())
GROUP BY 1
ORDER BY 1 DESC;
```

**4. Use CloudWatch Alarms**

```bash
aws cloudwatch put-metric-alarm \
    --alarm-name serverless-rpu-high \
    --namespace AWS/Redshift-Serverless \
    --metric-name ComputeCapacity \
    --dimensions Name=WorkgroupName,Value=marketing-workgroup \
    --statistic Average \
    --period 3600 \
    --evaluation-periods 2 \
    --threshold 64 \
    --comparison-operator GreaterThanThreshold \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts
```

---

## 9. Troubleshooting

### Common Connection Issues

#### Issue 1: Connection Timeout

**Error:**
```
psycopg2.OperationalError: could not connect to server: Connection timed out
```

**Causes & Solutions:**

**1. Security Group Not Configured**
```bash
# Check security group inbound rules
aws ec2 describe-security-groups \
    --group-ids sg-0123456789abcdef0 \
    --query 'SecurityGroups[0].IpPermissions'
```

**Solution**: Add inbound rule for port 5439 from your IP.

**2. Cluster Not Publicly Accessible**
```bash
# Check if publicly accessible
aws redshift describe-clusters \
    --cluster-identifier marketing-analytics-cluster \
    --query 'Clusters[0].PubliclyAccessible'
```

**Solution**: Modify cluster to enable public access (learning only!).

**3. Wrong Endpoint or Port**
```python
# Verify endpoint
import psycopg2

conn = psycopg2.connect(
    host='marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com',  # ✓ Full endpoint
    port=5439,  # ✓ Default Redshift port, not 5432 (PostgreSQL)
    dbname='marketing_db',
    user='admin',
    password='password'
)
```

**4. Firewall Blocking Outbound Port 5439**
```bash
# Test connection from command line
telnet marketing-analytics-cluster.c4xyz123.us-east-1.redshift.amazonaws.com 5439

# If timeout, firewall is blocking
# Solution: Configure firewall to allow outbound 5439
```

#### Issue 2: Authentication Failed

**Error:**
```
psycopg2.OperationalError: FATAL: password authentication failed for user "admin"
```

**Causes & Solutions:**

**1. Wrong Password**
- ⚠️ Passwords are case-sensitive
- No way to recover lost password (must reset)

**Reset password:**
```bash
aws redshift modify-cluster \
    --cluster-identifier marketing-analytics-cluster \
    --master-user-password 'NewPassword123!'
```

**2. Wrong Username**
```python
# Double-check username
conn = psycopg2.connect(
    user='admin',  # Default, not 'root' or 'postgres'
    ...
)
```

**3. Wrong Database Name**
```python
# Check database exists
conn = psycopg2.connect(
    dbname='marketing_db',  # Must match created database
    ...
)
```

**List databases:**
```sql
SELECT datname FROM pg_database;
```

#### Issue 3: SSL/TLS Errors

**Error:**
```
psycopg2.OperationalError: SSL SYSCALL error: EOF detected
```

**Solution**: Explicitly set SSL mode:

```python
import psycopg2

conn = psycopg2.connect(
    host='...',
    port=5439,
    dbname='marketing_db',
    user='admin',
    password='password',
    sslmode='require'  # Force SSL connection
)
```

**SSL modes:**
- `disable`: No SSL (⚠️ not recommended)
- `require`: Require SSL (recommended for production)
- `prefer`: Try SSL, fall back to non-SSL

### Permission Errors

#### Issue 1: Insufficient Privileges

**Error:**
```
ERROR: permission denied for schema marketing
```

**Solution**: Grant permissions to user

```sql
-- As admin user, grant permissions
GRANT USAGE ON SCHEMA marketing TO analyst_user;
GRANT SELECT ON ALL TABLES IN SCHEMA marketing TO analyst_user;

-- Grant future tables too
ALTER DEFAULT PRIVILEGES IN SCHEMA marketing
GRANT SELECT ON TABLES TO analyst_user;
```

#### Issue 2: Cannot Create Tables

**Error:**
```
ERROR: permission denied for schema public
```

**Solution**: Grant CREATE privilege

```sql
GRANT CREATE ON SCHEMA public TO username;

-- Or create in own schema
CREATE SCHEMA username;
GRANT ALL ON SCHEMA username TO username;
```

#### Issue 3: IAM Role Issues (COPY from S3)

**Error:**
```
ERROR: S3ServiceException: Access Denied
```

**Causes:**

**1. IAM Role Not Attached to Cluster**
```bash
# Check attached roles
aws redshift describe-clusters \
    --cluster-identifier marketing-analytics-cluster \
    --query 'Clusters[0].IamRoles'
```

**Solution**: Attach IAM role in Redshift console.

**2. IAM Role Missing S3 Permissions**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket/*",
        "arn:aws:s3:::my-bucket"
      ]
    }
  ]
}
```

**3. Wrong IAM Role ARN in COPY Command**
```sql
-- ✓ Correct - Full ARN
COPY table_name
FROM 's3://bucket/path'
IAM_ROLE 'arn:aws:iam::123456789012:role/RedshiftS3AccessRole';

-- ✗ Wrong - Partial ARN
COPY table_name
FROM 's3://bucket/path'
IAM_ROLE 'RedshiftS3AccessRole';  -- Missing full ARN
```

### Query Performance Problems

#### Issue 1: Slow Queries

**Diagnostic queries:**

```sql
-- Find long-running queries
SELECT
    pid,
    user_name,
    starttime,
    DATEDIFF(seconds, starttime, GETDATE()) AS runtime_seconds,
    TRIM(query_text) AS query
FROM stv_recents
WHERE status = 'Running'
ORDER BY runtime_seconds DESC;

-- Kill long-running query (if needed)
SELECT pg_terminate_backend(pid);
```

**Common causes:**

**1. Missing Distribution Key**
```sql
-- Check distribution style
SELECT
    tablename,
    diststyle
FROM pg_table_def
WHERE schemaname = 'marketing'
  AND tablename = 'events';
```

**Solution**: Recreate table with DISTKEY on JOIN columns.

**2. Missing Sort Key**
```sql
-- Check sort keys
SELECT
    tablename,
    column_name,
    sortkey
FROM pg_table_def
WHERE schemaname = 'marketing'
  AND tablename = 'events'
  AND sortkey > 0
ORDER BY sortkey;
```

**Solution**: Add SORTKEY on frequently filtered columns.

**3. Table Not Vacuumed**
```sql
-- Check table stats
SELECT
    schemaname || '.' || tablename AS table,
    unsorted,
    vacuum_sort_benefit,
    stats_off
FROM svv_table_info
WHERE schemaname = 'marketing'
ORDER BY vacuum_sort_benefit DESC;
```

**Solution**: Run VACUUM and ANALYZE
```sql
VACUUM marketing.events;
ANALYZE marketing.events;
```

**4. Outdated Statistics**
```sql
-- Query planner uses wrong estimates
-- Solution: Run ANALYZE
ANALYZE marketing.events;
```

#### Issue 2: Out of Memory Errors

**Error:**
```
ERROR: Cannot allocate memory
```

**Causes:**

**1. Query Too Complex**
- Reduce JOIN complexity
- Break into multiple steps with temp tables

```sql
-- Create temp table for intermediate result
CREATE TEMP TABLE temp_campaign_stats AS
SELECT
    campaign_id,
    SUM(impressions) AS total_impressions
FROM events
WHERE event_date >= '2024-01-01'
GROUP BY campaign_id;

-- Join with smaller dataset
SELECT
    c.campaign_name,
    t.total_impressions
FROM temp_campaign_stats t
JOIN campaigns c ON t.campaign_id = c.campaign_id;
```

**2. Too Many Concurrent Queries**
```sql
-- Check workload management (WLM) queues
SELECT * FROM stv_wlm_query_state;
```

**Solution**: Adjust WLM configuration to limit concurrency.

### Timeout Issues

#### Issue 1: Query Timeout

**Error:**
```
psycopg2.errors.QueryCanceled: Query execution was interrupted
```

**Causes:**

**1. Statement Timeout**
```sql
-- Check timeout setting
SHOW statement_timeout;

-- Set higher timeout (in milliseconds)
SET statement_timeout = 600000;  -- 10 minutes
```

**2. WLM Query Timeout**
```sql
-- Check WLM configuration
SELECT * FROM stv_wlm_service_class_config;
```

**Solution**: Adjust WLM queue timeout in Redshift console.

#### Issue 2: Connection Timeout

```python
# Increase connection timeout
conn = psycopg2.connect(
    host='...',
    port=5439,
    dbname='marketing_db',
    user='admin',
    password='password',
    connect_timeout=30  # 30 seconds (default is 10)
)
```

#### Issue 3: Idle Session Timeout

**Error:**
```
psycopg2.OperationalError: server closed the connection unexpectedly
```

**Solution**: Use connection pooling or re-establish connection:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    connection_string,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # ✓ Test connections before using
    pool_recycle=3600    # ✓ Recycle after 1 hour
)
```

---

## 10. Migration from SQLite

### Exporting Data from SQLite

#### Method 1: Export to CSV

```python
import sqlite3
import pandas as pd

# Connect to SQLite database
sqlite_conn = sqlite3.connect('marketing_analytics.db')

# Get list of tables
tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    sqlite_conn
)

# Export each table to CSV
for table_name in tables['name']:
    print(f"Exporting {table_name}...")

    df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_conn)
    df.to_csv(f"export/{table_name}.csv", index=False)

    print(f"  ✓ Exported {len(df):,} rows")

sqlite_conn.close()
```

#### Method 2: Direct DataFrame Transfer

```python
import sqlite3
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Source: SQLite
sqlite_conn = sqlite3.connect('marketing_analytics.db')

# Destination: Redshift
redshift_engine = create_engine(
    'redshift+psycopg2://admin:password@endpoint:5439/marketing_db'
)

# Migrate table
table_name = 'campaigns'

# Read from SQLite
df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_conn)

# Write to Redshift
df.to_sql(
    table_name,
    redshift_engine,
    schema='public',
    if_exists='replace',
    index=False,
    method='multi',
    chunksize=1000
)

print(f"✓ Migrated {len(df):,} rows from SQLite to Redshift")
```

### Importing to Redshift

#### Small Datasets (< 1M rows): Direct Insert

```python
import sqlite3
import psycopg2
import pandas as pd

# Export from SQLite
sqlite_conn = sqlite3.connect('marketing_analytics.db')
df = pd.read_sql("SELECT * FROM campaigns", sqlite_conn)

# Connect to Redshift
redshift_conn = psycopg2.connect(
    host='endpoint',
    port=5439,
    dbname='marketing_db',
    user='admin',
    password='password'
)

# Insert data
cursor = redshift_conn.cursor()

for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO campaigns
        (campaign_id, campaign_name, channel, budget, start_date)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (row['campaign_id'], row['campaign_name'], row['channel'],
         row['budget'], row['start_date'])
    )

redshift_conn.commit()
cursor.close()
redshift_conn.close()
```

#### Large Datasets (> 1M rows): Via S3

```python
import sqlite3
import pandas as pd
import boto3
from sqlalchemy import create_engine, text

def migrate_table_via_s3(
    table_name,
    sqlite_db,
    s3_bucket,
    s3_key,
    redshift_engine,
    iam_role_arn
):
    """Migrate large table from SQLite to Redshift via S3."""

    # Step 1: Export from SQLite to CSV
    print(f"Exporting {table_name} from SQLite...")
    sqlite_conn = sqlite3.connect(sqlite_db)
    df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_conn)
    sqlite_conn.close()

    csv_file = f'/tmp/{table_name}.csv'
    df.to_csv(csv_file, index=False, sep='|', header=False)
    print(f"  ✓ Exported {len(df):,} rows")

    # Step 2: Upload to S3
    print(f"Uploading to S3...")
    s3_client = boto3.client('s3')
    s3_client.upload_file(csv_file, s3_bucket, s3_key)
    print(f"  ✓ Uploaded to s3://{s3_bucket}/{s3_key}")

    # Step 3: COPY to Redshift
    print(f"Loading to Redshift...")
    copy_query = f"""
    COPY {table_name}
    FROM 's3://{s3_bucket}/{s3_key}'
    IAM_ROLE '{iam_role_arn}'
    DELIMITER '|'
    IGNOREHEADER 0
    DATEFORMAT 'auto'
    TIMEFORMAT 'auto'
    COMPUPDATE ON
    STATUPDATE ON;
    """

    with redshift_engine.connect() as conn:
        conn.execute(text(copy_query))
        conn.commit()

    print(f"  ✓ Loaded {len(df):,} rows to Redshift")

# Usage
redshift_engine = create_engine(
    'redshift+psycopg2://admin:password@endpoint:5439/marketing_db'
)

migrate_table_via_s3(
    table_name='marketing_events',
    sqlite_db='marketing_analytics.db',
    s3_bucket='my-marketing-data',
    s3_key='migration/events.csv',
    redshift_engine=redshift_engine,
    iam_role_arn='arn:aws:iam::123456789012:role/RedshiftS3AccessRole'
)
```

### Schema Translation

**SQLite → Redshift equivalents:**

#### Data Type Mapping

| SQLite | Redshift | Notes |
|--------|----------|-------|
| `INTEGER` | `INTEGER` or `BIGINT` | Use BIGINT for > 2B values |
| `REAL` | `REAL` or `DOUBLE PRECISION` | REAL = 4 bytes, DOUBLE = 8 bytes |
| `TEXT` | `VARCHAR(n)` | Specify max length (n) |
| `BLOB` | `VARCHAR(MAX)` | Base64 encode binary data |
| `NUMERIC` | `DECIMAL(p,s)` | Specify precision (p) and scale (s) |
| `DATE` | `DATE` | Same |
| `DATETIME` | `TIMESTAMP` | Redshift uses TIMESTAMP |

#### Auto-Increment Translation

**SQLite:**
```sql
CREATE TABLE campaigns (
    campaign_id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_name TEXT
);
```

**Redshift:**
```sql
CREATE TABLE campaigns (
    campaign_id INTEGER IDENTITY(1,1) PRIMARY KEY,
    campaign_name VARCHAR(255)
);
```

Or use `BIGINT IDENTITY(1,1)` for larger sequences.

#### Schema Extraction & Translation Script

```python
import sqlite3
import re

def sqlite_to_redshift_type(sqlite_type):
    """Map SQLite type to Redshift type."""
    type_map = {
        'INTEGER': 'INTEGER',
        'BIGINT': 'BIGINT',
        'REAL': 'REAL',
        'DOUBLE': 'DOUBLE PRECISION',
        'TEXT': 'VARCHAR(500)',  # Default to 500 chars
        'BLOB': 'VARCHAR(MAX)',
        'NUMERIC': 'DECIMAL(18,2)',
        'DATE': 'DATE',
        'DATETIME': 'TIMESTAMP',
        'TIMESTAMP': 'TIMESTAMP'
    }

    sqlite_type_upper = sqlite_type.upper()

    # Handle VARCHAR(n)
    if 'VARCHAR' in sqlite_type_upper:
        return sqlite_type_upper

    # Handle DECIMAL(p,s)
    if 'DECIMAL' in sqlite_type_upper or 'NUMERIC' in sqlite_type_upper:
        return sqlite_type_upper if '(' in sqlite_type else 'DECIMAL(18,2)'

    return type_map.get(sqlite_type_upper, 'VARCHAR(500)')

def extract_sqlite_schema(sqlite_db, table_name):
    """Extract CREATE TABLE statement from SQLite."""
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()

    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    create_sql = cursor.fetchone()[0]

    conn.close()
    return create_sql

def convert_to_redshift_ddl(sqlite_ddl, table_name, distkey=None, sortkey=None):
    """Convert SQLite DDL to Redshift DDL."""

    # Parse columns from SQLite DDL
    # This is simplified - production code should use proper SQL parser
    lines = sqlite_ddl.split('\n')
    redshift_columns = []

    for line in lines[1:-1]:  # Skip CREATE TABLE and closing paren
        line = line.strip().rstrip(',')
        if not line or line.startswith('CONSTRAINT') or line.startswith('FOREIGN'):
            continue

        # Parse column definition
        parts = line.split()
        if len(parts) >= 2:
            col_name = parts[0].strip('`"')
            sqlite_type = parts[1]

            # Convert type
            redshift_type = sqlite_to_redshift_type(sqlite_type)

            # Handle AUTOINCREMENT
            if 'AUTOINCREMENT' in line.upper():
                redshift_type = redshift_type.replace('INTEGER', 'INTEGER IDENTITY(1,1)')

            # Handle NOT NULL, PRIMARY KEY
            constraints = ''
            if 'NOT NULL' in line.upper():
                constraints += ' NOT NULL'
            if 'PRIMARY KEY' in line.upper() and 'AUTOINCREMENT' not in line.upper():
                constraints += ' PRIMARY KEY'

            redshift_columns.append(f"    {col_name} {redshift_type}{constraints}")

    # Build Redshift DDL
    redshift_ddl = f"CREATE TABLE {table_name} (\n"
    redshift_ddl += ',\n'.join(redshift_columns)
    redshift_ddl += "\n)"

    # Add distribution and sort keys
    if distkey:
        redshift_ddl += f"\nDISTKEY({distkey})"
    if sortkey:
        redshift_ddl += f"\nSORTKEY({sortkey})"

    redshift_ddl += ";"

    return redshift_ddl

# Usage
sqlite_ddl = extract_sqlite_schema('marketing_analytics.db', 'campaigns')
print("SQLite DDL:")
print(sqlite_ddl)
print("\n" + "="*80 + "\n")

redshift_ddl = convert_to_redshift_ddl(
    sqlite_ddl,
    table_name='campaigns',
    distkey='campaign_id',
    sortkey='start_date'
)
print("Redshift DDL:")
print(redshift_ddl)
```

**Example output:**

**SQLite:**
```sql
CREATE TABLE campaigns (
    campaign_id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_name TEXT NOT NULL,
    channel TEXT,
    budget REAL,
    start_date DATE
);
```

**Redshift:**
```sql
CREATE TABLE campaigns (
    campaign_id INTEGER IDENTITY(1,1) PRIMARY KEY,
    campaign_name VARCHAR(500) NOT NULL,
    channel VARCHAR(500),
    budget REAL,
    start_date DATE
)
DISTKEY(campaign_id)
SORTKEY(start_date);
```

### Complete Migration Script

```python
"""
Complete SQLite to Redshift Migration Script
"""

import sqlite3
import pandas as pd
import boto3
from sqlalchemy import create_engine, text
import os

class SQLiteToRedshiftMigration:
    def __init__(self, sqlite_db, redshift_engine, s3_bucket, iam_role_arn):
        self.sqlite_db = sqlite_db
        self.redshift_engine = redshift_engine
        self.s3_bucket = s3_bucket
        self.iam_role_arn = iam_role_arn
        self.s3_client = boto3.client('s3')

    def get_tables(self):
        """Get list of tables from SQLite."""
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

    def migrate_table(self, table_name, distkey=None, sortkey=None):
        """Migrate single table from SQLite to Redshift."""
        print(f"\n{'='*80}")
        print(f"Migrating: {table_name}")
        print(f"{'='*80}")

        # Step 1: Extract schema (you'd need to implement convert_to_redshift_ddl)
        print("1. Creating table in Redshift...")
        # Simplified - in production, extract and convert schema properly

        # Step 2: Export data
        print("2. Exporting data from SQLite...")
        sqlite_conn = sqlite3.connect(self.sqlite_db)
        df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_conn)
        sqlite_conn.close()
        print(f"   ✓ Exported {len(df):,} rows")

        if len(df) == 0:
            print("   ⚠ Table is empty, skipping...")
            return

        # Step 3: Upload to S3
        print("3. Uploading to S3...")
        s3_key = f"migration/{table_name}.csv"
        csv_buffer = df.to_csv(index=False, sep='|', header=False)
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=csv_buffer
        )
        print(f"   ✓ Uploaded to s3://{self.s3_bucket}/{s3_key}")

        # Step 4: COPY to Redshift
        print("4. Loading to Redshift...")

        # Get column names for COPY command
        columns = ', '.join(df.columns)

        copy_query = f"""
        COPY {table_name} ({columns})
        FROM 's3://{self.s3_bucket}/{s3_key}'
        IAM_ROLE '{self.iam_role_arn}'
        DELIMITER '|'
        IGNOREHEADER 0
        DATEFORMAT 'auto'
        TIMEFORMAT 'auto'
        COMPUPDATE ON
        STATUPDATE ON
        MAXERROR 10;
        """

        with self.redshift_engine.connect() as conn:
            result = conn.execute(text(copy_query))
            conn.commit()

        print(f"   ✓ Loaded {len(df):,} rows")

        # Step 5: Verify
        print("5. Verifying...")
        with self.redshift_engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.fetchone()[0]

        if count == len(df):
            print(f"   ✓ Verification successful: {count:,} rows")
        else:
            print(f"   ✗ Row count mismatch! SQLite: {len(df):,}, Redshift: {count:,}")

    def migrate_all(self, table_config=None):
        """Migrate all tables."""
        tables = self.get_tables()

        print(f"\nFound {len(tables)} tables: {', '.join(tables)}")

        for table in tables:
            config = table_config.get(table, {}) if table_config else {}
            self.migrate_table(
                table,
                distkey=config.get('distkey'),
                sortkey=config.get('sortkey')
            )

        print("\n" + "="*80)
        print("Migration Complete!")
        print("="*80)

# Usage
if __name__ == "__main__":
    # Configuration
    SQLITE_DB = 'marketing_analytics.db'
    REDSHIFT_ENGINE = create_engine(
        'redshift+psycopg2://admin:password@endpoint:5439/marketing_db'
    )
    S3_BUCKET = 'my-marketing-data'
    IAM_ROLE = 'arn:aws:iam::123456789012:role/RedshiftS3AccessRole'

    # Table-specific configuration
    TABLE_CONFIG = {
        'campaigns': {'distkey': 'campaign_id', 'sortkey': 'start_date'},
        'events': {'distkey': 'campaign_id', 'sortkey': 'event_date'},
        'conversions': {'distkey': 'campaign_id', 'sortkey': 'conversion_date'}
    }

    # Run migration
    migrator = SQLiteToRedshiftMigration(
        sqlite_db=SQLITE_DB,
        redshift_engine=REDSHIFT_ENGINE,
        s3_bucket=S3_BUCKET,
        iam_role_arn=IAM_ROLE
    )

    migrator.migrate_all(table_config=TABLE_CONFIG)
```

---

## Conclusion

This guide covered everything you need to set up and use Amazon Redshift for marketing analytics at scale. Key takeaways:

1. **Start Small**: Use free tier or Serverless for learning
2. **Optimize Early**: Design schema with DIST/SORT keys from the start
3. **Monitor Costs**: Set billing alerts, pause clusters when idle
4. **Use S3 for Loading**: COPY command is 10-100x faster than INSERT
5. **Think Columnar**: Redshift excels at aggregations, not row-by-row operations

**Next Steps:**
- Complete Module 9 (Advanced SQL Techniques)
- Experiment with sample marketing data
- Build attribution models with window functions
- Integrate with BI tools (QuickSight, Tableau)

**Resources:**
- AWS Redshift Documentation: https://docs.aws.amazon.com/redshift/
- Redshift Best Practices: https://docs.aws.amazon.com/redshift/latest/dg/best-practices.html
- AWS Pricing Calculator: https://calculator.aws/
- Redshift Spectrum Guide: https://docs.aws.amazon.com/redshift/latest/dg/c-using-spectrum.html

Happy analyzing! 🚀
