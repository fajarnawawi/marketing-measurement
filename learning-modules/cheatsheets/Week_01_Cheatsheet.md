# Week 1: Python Foundations - Quick Reference Cheatsheet

## 📋 Core Concepts

### Variables & Data Types
```python
# Integer (whole numbers)
impressions = 50000
clicks = 2500

# Float (decimals)
cost = 1250.50
cpa = 10.25

# String (text)
campaign_name = "Summer Sale 2024"
channel = 'Google Ads'

# Boolean (True/False)
is_active = True
is_converting = False

# Type checking
type(impressions)  # <class 'int'>
type(cost)        # <class 'float'>
```

### String Formatting
```python
# F-strings (recommended)
print(f"Campaign: {campaign_name}")
print(f"CPA: ${cpa:.2f}")           # 2 decimal places
print(f"CTR: {ctr:.2%}")            # Percentage format

# Format specifications
f"{value:.2f}"   # 2 decimal places: 12.35
f"{value:.2%}"   # Percentage: 12.35%
f"{value:,}"     # Thousands separator: 1,234,567
f"{value:>10}"   # Right-align in 10 spaces
f"{value:<10}"   # Left-align in 10 spaces
```

### Operators
```python
# Arithmetic
cost + revenue      # Addition
revenue - cost      # Subtraction
cost * 2           # Multiplication
revenue / cost     # Division (float result)
revenue // cost    # Floor division (integer result)
revenue % cost     # Modulo (remainder)
revenue ** 2       # Exponentiation

# Comparison
cpa < 20           # Less than
cpa <= 20          # Less than or equal
cpa > 15           # Greater than
cpa >= 15          # Greater than or equal
cpa == 20          # Equal to
cpa != 20          # Not equal to

# Logical
(roas > 3) and (cpa < 20)      # Both must be True
(roas > 3) or (cpa < 20)       # At least one must be True
not (roas > 3)                  # Negation
```

### Conditionals
```python
# If/elif/else
if roas >= 4.0:
    status = "Excellent"
elif roas >= 2.5:
    status = "Good"
elif roas >= 1.5:
    status = "Fair"
else:
    status = "Poor"

# Ternary operator (one-line if/else)
status = "Good" if roas >= 3.0 else "Poor"

# Multiple conditions
if roas >= 3.0 and conversions >= 100:
    print("Strong performance")
```

### Loops
```python
# For loop with range
for i in range(5):              # 0, 1, 2, 3, 4
    print(i)

for i in range(1, 6):           # 1, 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2):       # 0, 2, 4, 6, 8 (step by 2)
    print(i)

# For loop with list
channels = ["Google", "Facebook", "Instagram"]
for channel in channels:
    print(channel)

# For loop with enumerate (index + value)
for i, channel in enumerate(channels):
    print(f"{i}: {channel}")

# While loop
spent = 0
budget = 1000
while spent < budget:
    spent += 10
    print(f"Spent: ${spent}")
```

### Lists
```python
# Creating lists
channels = ["Google", "Facebook", "Instagram", "TikTok"]
costs = [1000, 1500, 2000, 1200]
empty_list = []

# Accessing elements
channels[0]        # "Google" (first element)
channels[-1]       # "TikTok" (last element)
channels[1:3]      # ["Facebook", "Instagram"] (slice)
channels[:2]       # ["Google", "Facebook"] (first 2)
channels[2:]       # ["Instagram", "TikTok"] (from index 2)

# List methods
channels.append("LinkedIn")           # Add to end
channels.insert(1, "Twitter")        # Insert at index 1
channels.remove("TikTok")            # Remove by value
channels.pop()                       # Remove and return last item
channels.pop(0)                      # Remove and return item at index 0

# List operations
len(channels)                        # Length: 4
sum(costs)                           # Sum: 5700
min(costs)                           # Minimum: 1000
max(costs)                           # Maximum: 2000
sorted(costs)                        # Sorted list (ascending)
sorted(costs, reverse=True)          # Sorted descending

# List comprehension
cpas = [costs[i] / conversions[i] for i in range(len(costs))]
high_cost = [c for c in costs if c > 1500]
```

### Dictionaries
```python
# Creating dictionaries
campaign = {
    "name": "Summer Sale",
    "channel": "Google",
    "cost": 1250.50,
    "conversions": 125,
    "revenue": 6250.00
}

# Accessing values
campaign["name"]              # "Summer Sale"
campaign.get("name")          # "Summer Sale"
campaign.get("budget", 0)     # 0 (default if key doesn't exist)

# Adding/updating
campaign["status"] = "active"
campaign["cost"] = 1300.00

# Dictionary methods
campaign.keys()               # dict_keys(['name', 'channel', ...])
campaign.values()             # dict_values(['Summer Sale', 'Google', ...])
campaign.items()              # dict_items([('name', 'Summer Sale'), ...])

# Checking keys
"name" in campaign           # True
"budget" in campaign         # False

# Looping through dictionaries
for key in campaign:
    print(f"{key}: {campaign[key]}")

for key, value in campaign.items():
    print(f"{key}: {value}")
```

### Functions
```python
# Basic function
def calculate_cpa(cost, conversions):
    return cost / conversions

# Function with default parameter
def calculate_cpa(cost, conversions, currency="USD"):
    cpa = cost / conversions
    symbol = "$" if currency == "USD" else "€"
    return f"{symbol}{cpa:.2f}"

# Multiple return values
def calculate_metrics(impressions, clicks, conversions, cost):
    ctr = clicks / impressions
    cvr = conversions / clicks
    cpa = cost / conversions
    return ctr, cvr, cpa

# Using the function
ctr, cvr, cpa = calculate_metrics(100000, 5000, 250, 5000)

# Docstring (documentation)
def calculate_roas(revenue, cost):
    """
    Calculate Return on Ad Spend.

    Parameters:
    - revenue (float): Total revenue generated
    - cost (float): Total ad spend

    Returns:
    - float: ROAS value
    """
    return revenue / cost
```

---

## 🎯 Marketing Metrics Quick Reference

### Essential Formulas
```python
# Funnel Metrics
CTR = clicks / impressions                          # Click-Through Rate
CVR = conversions / clicks                          # Conversion Rate
CPC = cost / clicks                                 # Cost Per Click
CPA = cost / conversions                            # Cost Per Acquisition
CPM = (cost / impressions) * 1000                   # Cost Per Mille (thousand)

# Performance Metrics
ROAS = revenue / cost                               # Return on Ad Spend
ROI = (revenue - cost) / cost                       # Return on Investment
Profit = revenue - cost                             # Net profit

# Efficiency Metrics
AOV = revenue / conversions                         # Average Order Value
Revenue_Per_Click = revenue / clicks
```

### Common Patterns
```python
# Calculate all metrics for a campaign
def analyze_campaign(impressions, clicks, conversions, cost, revenue):
    metrics = {
        'ctr': clicks / impressions if impressions > 0 else 0,
        'cvr': conversions / clicks if clicks > 0 else 0,
        'cpc': cost / clicks if clicks > 0 else 0,
        'cpa': cost / conversions if conversions > 0 else 0,
        'cpm': (cost / impressions) * 1000 if impressions > 0 else 0,
        'roas': revenue / cost if cost > 0 else 0
    }
    return metrics

# Performance evaluation
def evaluate_performance(roas, target_roas=3.0):
    if roas >= target_roas * 1.2:
        return "Excellent - Scale up"
    elif roas >= target_roas:
        return "Good - Maintain"
    elif roas >= target_roas * 0.8:
        return "Fair - Optimize"
    else:
        return "Poor - Pause"

# Multi-campaign comparison
campaigns = [
    {"name": "Search", "cost": 2000, "revenue": 12000},
    {"name": "Social", "cost": 3500, "revenue": 10500},
    {"name": "Display", "cost": 1500, "revenue": 6000}
]

for campaign in campaigns:
    roas = campaign['revenue'] / campaign['cost']
    campaign['roas'] = roas
    campaign['performance'] = evaluate_performance(roas)

# Sort by performance
sorted_campaigns = sorted(campaigns, key=lambda x: x['roas'], reverse=True)
```

---

## 💡 Common Patterns & Best Practices

### Error Handling
```python
# Avoid division by zero
if conversions > 0:
    cpa = cost / conversions
else:
    cpa = 0

# Using try/except
try:
    cpa = cost / conversions
except ZeroDivisionError:
    cpa = 0

# Ternary operator for safe division
cpa = cost / conversions if conversions > 0 else 0
```

### Working with Multiple Campaigns
```python
# Lists of values
campaign_names = ["Search", "Social", "Display"]
costs = [2000, 3500, 1500]
revenues = [12000, 10500, 6000]

# Calculate ROAS for all
roas_values = []
for i in range(len(costs)):
    roas = revenues[i] / costs[i]
    roas_values.append(roas)
    print(f"{campaign_names[i]}: {roas:.2f}x")

# List comprehension version
roas_values = [revenues[i] / costs[i] for i in range(len(costs))]

# Using zip
for name, cost, revenue in zip(campaign_names, costs, revenues):
    roas = revenue / cost
    print(f"{name}: ROAS = {roas:.2f}x")
```

### Data Validation
```python
def validate_campaign_data(impressions, clicks, conversions):
    """Validate that campaign data makes sense."""
    errors = []

    # Clicks can't exceed impressions
    if clicks > impressions:
        errors.append("Clicks exceed impressions")

    # Conversions can't exceed clicks
    if conversions > clicks:
        errors.append("Conversions exceed clicks")

    # All values should be non-negative
    if any(x < 0 for x in [impressions, clicks, conversions]):
        errors.append("Negative values detected")

    if errors:
        return False, errors
    return True, []

# Usage
is_valid, errors = validate_campaign_data(100000, 5000, 250)
if not is_valid:
    print(f"Data validation failed: {errors}")
```

---

## 🚀 Quick Tips

### Performance
- Use list comprehensions for better performance
- Avoid unnecessary loops
- Use built-in functions (sum, min, max) when possible

### Readability
- Use meaningful variable names (`campaign_cost` not `cc`)
- Add comments for complex logic
- Keep functions focused on one task
- Use f-strings for formatting

### Common Mistakes
```python
# ❌ Wrong: Using = instead of ==
if roas = 3.0:  # SyntaxError

# ✅ Correct
if roas == 3.0:

# ❌ Wrong: Forgetting to convert types
clicks = "2500"
ctr = clicks / impressions  # TypeError

# ✅ Correct
clicks = int("2500")
ctr = clicks / impressions

# ❌ Wrong: Indentation errors
if roas > 3.0:
print("Good")  # IndentationError

# ✅ Correct
if roas > 3.0:
    print("Good")
```

---

## 📚 Practice Exercises Solutions

### Exercise: Calculate Campaign Metrics
```python
# Given data
impressions = 100000
clicks = 5000
conversions = 250
cost = 5000
revenue = 25000

# Solution
ctr = clicks / impressions
cvr = conversions / clicks
cpa = cost / conversions
roas = revenue / cost

print(f"CTR: {ctr:.2%}")
print(f"CVR: {cvr:.2%}")
print(f"CPA: ${cpa:.2f}")
print(f"ROAS: {roas:.2f}x")
```

### Exercise: Find Best Performing Channel
```python
campaigns = [
    {"name": "Google", "cost": 5000, "revenue": 20000},
    {"name": "Facebook", "cost": 3500, "revenue": 14000},
    {"name": "Instagram", "cost": 2000, "revenue": 10000}
]

# Calculate ROAS for each
for campaign in campaigns:
    campaign['roas'] = campaign['revenue'] / campaign['cost']

# Find best performer
best = max(campaigns, key=lambda x: x['roas'])
print(f"Best performer: {best['name']} with ROAS of {best['roas']:.2f}x")
```

### Exercise: Budget Allocation
```python
total_budget = 10000
channels = {
    "Google": {"weight": 0.4, "min_budget": 2000},
    "Facebook": {"weight": 0.35, "min_budget": 1500},
    "Instagram": {"weight": 0.25, "min_budget": 1000}
}

# Allocate budget
for channel, config in channels.items():
    allocated = total_budget * config['weight']
    if allocated < config['min_budget']:
        allocated = config['min_budget']
    print(f"{channel}: ${allocated:,.2f}")
```

---

## 🔍 Debugging Tips

```python
# Print debugging
print(f"Cost: {cost}, Conversions: {conversions}")
print(f"Type of cost: {type(cost)}")

# Check variable values
print(f"Debug - ROAS calculation: {revenue} / {cost} = {revenue/cost}")

# Use repr() for exact representation
print(repr(campaign_name))

# Check if variable exists
try:
    print(campaign_budget)
except NameError:
    print("campaign_budget is not defined")
```

---

**Quick Navigation:**
- [Week 2 Cheatsheet →](Week_02_Cheatsheet.md)
- [Back to Main README](../README.md)
