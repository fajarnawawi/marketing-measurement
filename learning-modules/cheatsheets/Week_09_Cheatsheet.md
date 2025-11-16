# Week 9: Attribution Modeling - Quick Reference Cheatsheet

## 📋 Core Concepts

### Attribution Model Types

```python
"""
Common Attribution Models:

1. Last-Touch (Last-Click): 100% credit to last touchpoint
2. First-Touch (First-Click): 100% credit to first touchpoint
3. Linear: Equal credit to all touchpoints
4. Time Decay: More credit to recent touchpoints
5. Position-Based (U-Shaped): 40% first, 40% last, 20% middle
6. Data-Driven: Uses statistical modeling (Markov, Shapley)
"""

# Example customer journey
journey = [
    {'channel': 'Organic Search', 'timestamp': '2024-01-01'},
    {'channel': 'Facebook', 'timestamp': '2024-01-05'},
    {'channel': 'Email', 'timestamp': '2024-01-08'},
    {'channel': 'Google Ads', 'timestamp': '2024-01-10'},  # Converted
]

conversion_value = 100  # dollars
```

### Last-Touch Attribution
```python
def last_touch_attribution(journey, conversion_value):
    """
    Assign 100% credit to the last touchpoint before conversion.
    """
    if not journey:
        return {}

    last_channel = journey[-1]['channel']

    attribution = {channel['channel']: 0 for channel in journey}
    attribution[last_channel] = conversion_value

    return attribution

# Example
journey = [
    {'channel': 'Organic Search'},
    {'channel': 'Facebook'},
    {'channel': 'Email'},
    {'channel': 'Google Ads'},  # Gets 100% credit
]

attribution = last_touch_attribution(journey, 100)
print("Last-Touch Attribution:", attribution)
# Output: {'Organic Search': 0, 'Facebook': 0, 'Email': 0, 'Google Ads': 100}
```

### First-Touch Attribution
```python
def first_touch_attribution(journey, conversion_value):
    """
    Assign 100% credit to the first touchpoint.
    """
    if not journey:
        return {}

    first_channel = journey[0]['channel']

    attribution = {channel['channel']: 0 for channel in journey}
    attribution[first_channel] = conversion_value

    return attribution

# Example
attribution = first_touch_attribution(journey, 100)
print("First-Touch Attribution:", attribution)
# Output: {'Organic Search': 100, 'Facebook': 0, 'Email': 0, 'Google Ads': 0}
```

### Linear Attribution
```python
def linear_attribution(journey, conversion_value):
    """
    Distribute credit equally across all touchpoints.
    """
    if not journey:
        return {}

    n_touchpoints = len(journey)
    credit_per_touchpoint = conversion_value / n_touchpoints

    # Get unique channels
    channels = list(set(channel['channel'] for channel in journey))
    attribution = {ch: 0 for ch in channels}

    # Distribute credit
    for touchpoint in journey:
        attribution[touchpoint['channel']] += credit_per_touchpoint

    return attribution

# Example
attribution = linear_attribution(journey, 100)
print("Linear Attribution:", attribution)
# Each of 4 touchpoints gets $25
```

### Time Decay Attribution
```python
import numpy as np
from datetime import datetime

def time_decay_attribution(journey, conversion_value, half_life_days=7):
    """
    Assign more credit to recent touchpoints using exponential decay.

    half_life_days: number of days for credit to decay to 50%
    """
    if not journey:
        return {}

    # Parse timestamps
    conversion_date = datetime.strptime(journey[-1]['timestamp'], '%Y-%m-%d')

    # Calculate weights
    weights = []
    for touchpoint in journey:
        touch_date = datetime.strptime(touchpoint['timestamp'], '%Y-%m-%d')
        days_before_conversion = (conversion_date - touch_date).days

        # Exponential decay: weight = 2^(-days/half_life)
        weight = 2 ** (-days_before_conversion / half_life_days)
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Assign credit
    channels = list(set(tp['channel'] for tp in journey))
    attribution = {ch: 0 for ch in channels}

    for touchpoint, weight in zip(journey, normalized_weights):
        attribution[touchpoint['channel']] += weight * conversion_value

    return attribution

# Example
journey_with_dates = [
    {'channel': 'Organic Search', 'timestamp': '2024-01-01'},  # 9 days ago
    {'channel': 'Facebook', 'timestamp': '2024-01-05'},         # 5 days ago
    {'channel': 'Email', 'timestamp': '2024-01-08'},            # 2 days ago
    {'channel': 'Google Ads', 'timestamp': '2024-01-10'},       # Today
]

attribution = time_decay_attribution(journey_with_dates, 100, half_life_days=7)
print("Time Decay Attribution:")
for channel, value in attribution.items():
    print(f"  {channel}: ${value:.2f}")
```

### Position-Based (U-Shaped) Attribution
```python
def position_based_attribution(journey, conversion_value,
                                first_pct=0.4, last_pct=0.4):
    """
    U-shaped attribution:
    - First touchpoint: 40% (default)
    - Last touchpoint: 40% (default)
    - Middle touchpoints: remaining 20% divided equally
    """
    if not journey:
        return {}

    n_touchpoints = len(journey)

    # Get unique channels
    channels = list(set(tp['channel'] for tp in journey))
    attribution = {ch: 0 for ch in channels}

    if n_touchpoints == 1:
        # Only one touchpoint gets everything
        attribution[journey[0]['channel']] = conversion_value
    elif n_touchpoints == 2:
        # Split between first and last
        attribution[journey[0]['channel']] += conversion_value * first_pct
        attribution[journey[-1]['channel']] += conversion_value * last_pct
    else:
        # First touchpoint
        attribution[journey[0]['channel']] += conversion_value * first_pct

        # Last touchpoint
        attribution[journey[-1]['channel']] += conversion_value * last_pct

        # Middle touchpoints
        middle_credit = conversion_value * (1 - first_pct - last_pct)
        middle_touchpoints = journey[1:-1]
        credit_per_middle = middle_credit / len(middle_touchpoints)

        for touchpoint in middle_touchpoints:
            attribution[touchpoint['channel']] += credit_per_middle

    return attribution

# Example
attribution = position_based_attribution(journey_with_dates, 100)
print("Position-Based Attribution:")
for channel, value in attribution.items():
    print(f"  {channel}: ${value:.2f}")
```

---

## 🎯 Customer Journey Analysis

### Journey Representation
```python
import pandas as pd

# Customer journeys as DataFrame
journeys = pd.DataFrame([
    {
        'user_id': 'user_001',
        'journey': ['Organic', 'Facebook', 'Email', 'Google Ads'],
        'conversion_value': 100,
        'converted': True
    },
    {
        'user_id': 'user_002',
        'journey': ['Facebook', 'Google Ads'],
        'conversion_value': 150,
        'converted': True
    },
    {
        'user_id': 'user_003',
        'journey': ['Organic', 'Facebook'],
        'conversion_value': 0,
        'converted': False
    },
    {
        'user_id': 'user_004',
        'journey': ['Email', 'Google Ads', 'Email'],
        'conversion_value': 200,
        'converted': True
    },
])

print(journeys)
```

### Path Analysis
```python
def analyze_conversion_paths(journeys_df):
    """
    Analyze which paths lead to conversion.
    """
    # Convert paths to strings for grouping
    journeys_df['path'] = journeys_df['journey'].apply(lambda x: ' → '.join(x))

    # Group by path
    path_analysis = journeys_df.groupby('path').agg({
        'converted': ['sum', 'count'],
        'conversion_value': 'sum'
    }).reset_index()

    path_analysis.columns = ['path', 'conversions', 'total_journeys', 'total_value']
    path_analysis['conversion_rate'] = (
        path_analysis['conversions'] / path_analysis['total_journeys']
    )

    # Sort by conversions
    path_analysis = path_analysis.sort_values('conversions', ascending=False)

    return path_analysis

# Example
path_stats = analyze_conversion_paths(journeys)
print("\nConversion Path Analysis:")
print(path_stats)
```

### Touchpoint Frequency
```python
from collections import Counter

def touchpoint_frequency(journeys_df, converted_only=True):
    """
    Count how often each channel appears in customer journeys.
    """
    if converted_only:
        journeys_df = journeys_df[journeys_df['converted']]

    # Flatten all touchpoints
    all_touchpoints = []
    for journey in journeys_df['journey']:
        all_touchpoints.extend(journey)

    # Count frequency
    frequency = Counter(all_touchpoints)

    # Convert to DataFrame
    freq_df = pd.DataFrame([
        {'channel': ch, 'frequency': count}
        for ch, count in frequency.most_common()
    ])

    freq_df['percentage'] = freq_df['frequency'] / freq_df['frequency'].sum()

    return freq_df

# Example
freq_stats = touchpoint_frequency(journeys)
print("\nTouchpoint Frequency (Converted Journeys):")
print(freq_stats)
```

### Average Path Length
```python
def path_length_analysis(journeys_df):
    """
    Analyze journey length statistics.
    """
    journeys_df['path_length'] = journeys_df['journey'].apply(len)

    stats = journeys_df.groupby('converted').agg({
        'path_length': ['mean', 'median', 'min', 'max']
    }).round(2)

    print("\nPath Length Analysis:")
    print(stats)

    return stats

# Example
path_length_analysis(journeys)
```

---

## 💡 Multi-Touch Attribution Patterns

### Complete Attribution Analyzer
```python
import pandas as pd
import numpy as np

class AttributionAnalyzer:
    """
    Apply multiple attribution models to customer journeys.
    """

    def __init__(self, journeys_df):
        """
        journeys_df: DataFrame with columns ['user_id', 'journey', 'conversion_value', 'converted']
        """
        self.journeys = journeys_df[journeys_df['converted']].copy()

    def last_touch(self):
        """Last-touch attribution."""
        attribution = {}

        for _, row in self.journeys.iterrows():
            last_channel = row['journey'][-1]
            attribution[last_channel] = attribution.get(last_channel, 0) + row['conversion_value']

        return pd.Series(attribution).sort_values(ascending=False)

    def first_touch(self):
        """First-touch attribution."""
        attribution = {}

        for _, row in self.journeys.iterrows():
            first_channel = row['journey'][0]
            attribution[first_channel] = attribution.get(first_channel, 0) + row['conversion_value']

        return pd.Series(attribution).sort_values(ascending=False)

    def linear(self):
        """Linear attribution."""
        attribution = {}

        for _, row in self.journeys.iterrows():
            journey = row['journey']
            value_per_touch = row['conversion_value'] / len(journey)

            for channel in journey:
                attribution[channel] = attribution.get(channel, 0) + value_per_touch

        return pd.Series(attribution).sort_values(ascending=False)

    def position_based(self, first_pct=0.4, last_pct=0.4):
        """Position-based (U-shaped) attribution."""
        attribution = {}

        for _, row in self.journeys.iterrows():
            journey = row['journey']
            value = row['conversion_value']
            n = len(journey)

            if n == 1:
                attribution[journey[0]] = attribution.get(journey[0], 0) + value
            elif n == 2:
                attribution[journey[0]] = attribution.get(journey[0], 0) + value * first_pct
                attribution[journey[-1]] = attribution.get(journey[-1], 0) + value * last_pct
            else:
                # First
                attribution[journey[0]] = attribution.get(journey[0], 0) + value * first_pct
                # Last
                attribution[journey[-1]] = attribution.get(journey[-1], 0) + value * last_pct
                # Middle
                middle_value = value * (1 - first_pct - last_pct) / (n - 2)
                for channel in journey[1:-1]:
                    attribution[channel] = attribution.get(channel, 0) + middle_value

        return pd.Series(attribution).sort_values(ascending=False)

    def compare_models(self):
        """Compare all attribution models."""
        results = pd.DataFrame({
            'Last-Touch': self.last_touch(),
            'First-Touch': self.first_touch(),
            'Linear': self.linear(),
            'Position-Based': self.position_based()
        }).fillna(0)

        # Add totals
        results.loc['TOTAL'] = results.sum()

        return results

# Example usage
journeys = pd.DataFrame([
    {'user_id': 'u1', 'journey': ['Organic', 'Facebook', 'Google Ads'],
     'conversion_value': 100, 'converted': True},
    {'user_id': 'u2', 'journey': ['Facebook', 'Email', 'Google Ads'],
     'conversion_value': 150, 'converted': True},
    {'user_id': 'u3', 'journey': ['Organic', 'Facebook'],
     'conversion_value': 0, 'converted': False},
    {'user_id': 'u4', 'journey': ['Email', 'Google Ads'],
     'conversion_value': 200, 'converted': True},
    {'user_id': 'u5', 'journey': ['Organic', 'Email', 'Facebook', 'Google Ads'],
     'conversion_value': 120, 'converted': True},
])

analyzer = AttributionAnalyzer(journeys)
comparison = analyzer.compare_models()

print("Attribution Model Comparison:")
print(comparison.round(2))
```

### Calculate ROI by Attribution Model
```python
def calculate_channel_roi(attribution_results, channel_costs):
    """
    Calculate ROI for each channel under different attribution models.

    attribution_results: DataFrame from compare_models()
    channel_costs: dict of {channel: cost}
    """
    roi_results = {}

    for model in attribution_results.columns:
        roi = {}
        for channel in channel_costs.keys():
            if channel in attribution_results.index:
                revenue = attribution_results.loc[channel, model]
                cost = channel_costs[channel]
                roi[channel] = (revenue - cost) / cost if cost > 0 else 0
            else:
                roi[channel] = -1  # No revenue attributed

        roi_results[model] = roi

    roi_df = pd.DataFrame(roi_results)

    return roi_df

# Example
channel_costs = {
    'Google Ads': 200,
    'Facebook': 150,
    'Email': 50,
    'Organic': 0  # No direct cost
}

roi = calculate_channel_roi(comparison, channel_costs)
print("\nROI by Attribution Model:")
print(roi.round(2))
```

---

## 🚀 Markov Chain Attribution (Basics)

### Transition Matrix
```python
import numpy as np
import pandas as pd
from collections import defaultdict

def build_transition_matrix(journeys_df):
    """
    Build transition probability matrix from customer journeys.
    """
    # Count transitions
    transitions = defaultdict(lambda: defaultdict(int))

    for _, row in journeys_df.iterrows():
        journey = ['START'] + row['journey'] + ['CONVERSION' if row['converted'] else 'NULL']

        # Count each transition
        for i in range(len(journey) - 1):
            from_state = journey[i]
            to_state = journey[i + 1]
            transitions[from_state][to_state] += 1

    # Convert to probabilities
    transition_probs = {}
    for from_state, to_states in transitions.items():
        total = sum(to_states.values())
        transition_probs[from_state] = {
            to_state: count / total
            for to_state, count in to_states.items()
        }

    return transition_probs

# Example
journeys = pd.DataFrame([
    {'journey': ['Organic', 'Facebook', 'Google Ads'], 'converted': True},
    {'journey': ['Facebook', 'Google Ads'], 'converted': True},
    {'journey': ['Organic', 'Facebook'], 'converted': False},
    {'journey': ['Email', 'Google Ads'], 'converted': True},
])

trans_matrix = build_transition_matrix(journeys)

print("Transition Probabilities:")
for from_state, to_states in trans_matrix.items():
    print(f"\n{from_state}:")
    for to_state, prob in to_states.items():
        print(f"  → {to_state}: {prob:.2%}")
```

### Removal Effect
```python
def markov_removal_effect(journeys_df, channel_to_remove):
    """
    Calculate conversion probability with and without a specific channel.
    This measures the channel's contribution.
    """
    # Baseline: conversion rate with all channels
    total_conversions = journeys_df['converted'].sum()
    total_journeys = len(journeys_df)
    baseline_cvr = total_conversions / total_journeys

    # Remove channel from journeys
    modified_journeys = journeys_df.copy()
    modified_journeys['journey_filtered'] = modified_journeys['journey'].apply(
        lambda j: [ch for ch in j if ch != channel_to_remove]
    )

    # Journeys that had the channel but now empty → no conversion
    modified_journeys['converted_filtered'] = modified_journeys.apply(
        lambda row: row['converted'] if len(row['journey_filtered']) > 0 else False,
        axis=1
    )

    # New conversion rate
    modified_conversions = modified_journeys['converted_filtered'].sum()
    modified_cvr = modified_conversions / total_journeys

    # Removal effect
    removal_effect = baseline_cvr - modified_cvr

    print(f"Channel: {channel_to_remove}")
    print(f"Baseline CVR: {baseline_cvr:.2%}")
    print(f"CVR without {channel_to_remove}: {modified_cvr:.2%}")
    print(f"Removal Effect: {removal_effect:.2%}")
    print(f"→ {channel_to_remove} contributes {removal_effect:.2%} to conversions")

    return removal_effect

# Example
journeys = pd.DataFrame([
    {'journey': ['Organic', 'Facebook', 'Google Ads'], 'converted': True},
    {'journey': ['Facebook', 'Google Ads'], 'converted': True},
    {'journey': ['Organic', 'Facebook'], 'converted': False},
    {'journey': ['Email', 'Google Ads'], 'converted': True},
    {'journey': ['Organic', 'Email'], 'converted': False},
])

# Test removal of Google Ads
removal_effect = markov_removal_effect(journeys, 'Google Ads')
```

### Markov Attribution
```python
def markov_attribution(journeys_df, total_conversions_value):
    """
    Simplified Markov chain attribution based on removal effect.
    """
    # Get unique channels
    all_channels = set()
    for journey in journeys_df['journey']:
        all_channels.update(journey)

    # Calculate removal effect for each channel
    removal_effects = {}
    for channel in all_channels:
        effect = markov_removal_effect(journeys_df, channel)
        removal_effects[channel] = effect

    # Normalize removal effects to sum to 1
    total_effect = sum(removal_effects.values())

    # Attribute value proportionally to removal effects
    attribution = {}
    for channel, effect in removal_effects.items():
        if total_effect > 0:
            attribution[channel] = (effect / total_effect) * total_conversions_value
        else:
            attribution[channel] = 0

    return pd.Series(attribution).sort_values(ascending=False)

# Example (would show detailed output)
# attribution = markov_attribution(journeys, total_conversions_value=1000)
```

---

## 🎯 Shapley Value Attribution (Introduction)

### Shapley Value Concept
```python
"""
Shapley Value Attribution:
- From cooperative game theory
- Distributes "payout" fairly based on marginal contribution
- Considers all possible orderings of channels
- Computationally expensive: O(2^n) for n channels

Formula:
Shapley_i = Σ [|S|!(n-|S|-1)! / n!] × [v(S ∪ {i}) - v(S)]

Where:
- S is a subset of channels not including i
- v(S) is the value/conversion probability with channels in S
- n is total number of channels
"""

from itertools import combinations, permutations
import numpy as np

def shapley_value_simple(journeys_df, total_value):
    """
    Simplified Shapley value calculation for small datasets.
    Warning: Computationally expensive for >5 channels.
    """
    # Get all channels
    all_channels = set()
    for journey in journeys_df['journey']:
        all_channels.update(journey)

    channels = list(all_channels)
    n_channels = len(channels)

    print(f"Calculating Shapley values for {n_channels} channels...")
    print(f"This requires evaluating {2**n_channels} coalitions.")

    if n_channels > 6:
        print("⚠ Warning: Too many channels for exact calculation")
        return None

    # Function to calculate conversion value for a coalition
    def coalition_value(coalition):
        """
        Calculate total conversion value when only coalition channels are present.
        """
        if not coalition:
            return 0

        value = 0
        for _, row in journeys_df[journeys_df['converted']].iterrows():
            # Check if journey uses only coalition channels
            journey_channels = set(row['journey'])
            if journey_channels.issubset(set(coalition)):
                value += row['conversion_value']

        return value

    # Calculate Shapley value for each channel
    shapley_values = {}

    for channel in channels:
        value = 0

        # Consider all possible coalitions not containing this channel
        other_channels = [ch for ch in channels if ch != channel]

        for r in range(len(other_channels) + 1):
            for coalition in combinations(other_channels, r):
                coalition_list = list(coalition)

                # Marginal contribution
                without_channel = coalition_value(coalition_list)
                with_channel = coalition_value(coalition_list + [channel])
                marginal = with_channel - without_channel

                # Weight
                n = n_channels
                s = len(coalition)
                weight = np.math.factorial(s) * np.math.factorial(n - s - 1) / np.math.factorial(n)

                value += weight * marginal

        shapley_values[channel] = value

    return pd.Series(shapley_values).sort_values(ascending=False)

# Note: This is illustrative - real Shapley calculation requires more sophisticated implementation
```

---

## 💡 Marketing Attribution Patterns

### Pattern 1: Multi-Model Comparison Dashboard
```python
import pandas as pd
import numpy as np

def attribution_dashboard(journeys_df, channel_costs):
    """
    Complete attribution analysis with multiple models.
    """
    # Initialize analyzer
    analyzer = AttributionAnalyzer(journeys_df)

    # Get attribution for each model
    attribution = analyzer.compare_models()

    # Calculate metrics for each model
    results = {}

    for model in attribution.columns:
        model_attribution = attribution[model].drop('TOTAL')

        # ROI by channel
        roi = {}
        for channel in model_attribution.index:
            cost = channel_costs.get(channel, 0)
            revenue = model_attribution[channel]
            roi[channel] = (revenue - cost) / cost if cost > 0 else np.inf

        # Overall metrics
        total_revenue = model_attribution.sum()
        total_cost = sum(channel_costs.values())
        overall_roi = (total_revenue - total_cost) / total_cost

        results[model] = {
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'overall_roi': overall_roi,
            'channel_roi': roi
        }

    # Print dashboard
    print("="*80)
    print("ATTRIBUTION MODEL COMPARISON DASHBOARD")
    print("="*80)

    # Revenue attribution
    print("\n1. REVENUE ATTRIBUTION")
    print("-"*80)
    print(attribution.round(2))

    # Overall ROI
    print("\n2. OVERALL ROI BY MODEL")
    print("-"*80)
    for model, metrics in results.items():
        print(f"{model:20s}: {metrics['overall_roi']:>8.2%}")

    # Channel ROI comparison
    print("\n3. CHANNEL ROI BY MODEL")
    print("-"*80)
    roi_df = pd.DataFrame({
        model: metrics['channel_roi']
        for model, metrics in results.items()
    })
    print(roi_df.round(2))

    # Model agreement
    print("\n4. MODEL AGREEMENT")
    print("-"*80)
    # Coefficient of variation for each channel
    cv = attribution.drop('TOTAL').std(axis=1) / attribution.drop('TOTAL').mean(axis=1)
    cv_df = pd.DataFrame({'Coefficient_of_Variation': cv}).sort_values('Coefficient_of_Variation')
    print(cv_df.round(2))
    print("\nLow CV = models agree, High CV = models disagree")

    return attribution, results

# Example usage would go here
```

### Pattern 2: Time-Based Attribution Windows
```python
from datetime import datetime, timedelta

def attribution_with_windows(journeys_df, conversion_value, window_days=30):
    """
    Apply attribution with time-based lookback windows.

    Only touchpoints within window_days of conversion are counted.
    """
    attributed_journeys = []

    for _, row in journeys_df[journeys_df['converted']].iterrows():
        journey = row['journey']
        timestamps = row['timestamps']  # List of datetime objects
        conversion_date = timestamps[-1]

        # Filter touchpoints within window
        windowed_journey = []
        for i, (channel, timestamp) in enumerate(zip(journey, timestamps)):
            days_before = (conversion_date - timestamp).days
            if days_before <= window_days:
                windowed_journey.append(channel)

        attributed_journeys.append({
            'user_id': row['user_id'],
            'journey': windowed_journey,
            'conversion_value': row['conversion_value'],
            'converted': True
        })

    return pd.DataFrame(attributed_journeys)

# Example: 7-day vs 30-day attribution window
# journeys_7day = attribution_with_windows(journeys, 100, window_days=7)
# journeys_30day = attribution_with_windows(journeys, 100, window_days=30)
```

### Pattern 3: Custom Attribution Rules
```python
def custom_attribution(journey, conversion_value, rules):
    """
    Apply custom business rules for attribution.

    rules: dict mapping channel to weight/priority
    """
    attribution = {channel: 0 for channel in set(journey)}

    # Example rules:
    # - Branded search gets minimal credit (user already knew brand)
    # - Last non-direct click gets priority
    # - Assisted conversions get partial credit

    # Identify channel types
    branded_channels = ['Branded Search', 'Direct']
    last_touch = journey[-1]

    if last_touch in branded_channels and len(journey) > 1:
        # Give credit to last non-branded
        for channel in reversed(journey[:-1]):
            if channel not in branded_channels:
                attribution[channel] = conversion_value * 0.6
                break
        # Remaining to last touch
        attribution[last_touch] = conversion_value * 0.4
    else:
        # Standard position-based
        if len(journey) == 1:
            attribution[journey[0]] = conversion_value
        else:
            attribution[journey[0]] += conversion_value * 0.4
            attribution[journey[-1]] += conversion_value * 0.4

            if len(journey) > 2:
                middle_credit = conversion_value * 0.2 / (len(journey) - 2)
                for channel in journey[1:-1]:
                    attribution[channel] += middle_credit

    return attribution

# Example
journey = ['Facebook', 'Email', 'Google Ads', 'Branded Search']
rules = {'Branded Search': 'low_priority'}
attr = custom_attribution(journey, 100, rules)
print(attr)
```

---

## 🚀 Quick Tips

### Choosing Attribution Models
1. **Last-Touch**: Good for direct response campaigns, simple reporting
2. **First-Touch**: Good for understanding acquisition channels
3. **Linear**: Fair but may over-credit "passive" touchpoints
4. **Position-Based**: Good balance, emphasizes acquisition and conversion
5. **Data-Driven (Markov/Shapley)**: Most accurate but complex, needs data

### Data Requirements
- **Minimum**: 100-200 conversions for basic models
- **Recommended**: 1000+ conversions for data-driven models
- **Path diversity**: Need variety in customer journeys
- **Tracking**: Accurate cross-device/cross-session tracking

### Common Mistakes
```python
# ❌ Wrong: Ignoring non-converting journeys
converting_journeys_only = df[df['converted'] == True]
# This biases analysis!

# ✅ Correct: Include all journeys for proper analysis
all_journeys = df  # Both converted and non-converted

# ❌ Wrong: Not de-duplicating consecutive same-channel touches
journey = ['Facebook', 'Facebook', 'Facebook', 'Email']
# Triple-counts Facebook!

# ✅ Correct: De-duplicate consecutive touches
def dedupe_journey(journey):
    return [journey[i] for i in range(len(journey))
            if i == 0 or journey[i] != journey[i-1]]

journey_clean = dedupe_journey(journey)  # ['Facebook', 'Email']

# ❌ Wrong: Treating all touchpoints equally regardless of time
# 90 days ago vs yesterday should be different

# ✅ Correct: Apply time decay or use lookback windows
# (see time_decay_attribution function above)
```

---

## 📚 Practice Exercises Solutions

### Exercise 1: Compare Attribution Models
```python
import pandas as pd

# Customer journey data
journeys = pd.DataFrame([
    {'user_id': 'u1', 'journey': ['Organic', 'Facebook', 'Email', 'Google Ads'],
     'conversion_value': 150, 'converted': True},
    {'user_id': 'u2', 'journey': ['Facebook', 'Google Ads'],
     'conversion_value': 200, 'converted': True},
    {'user_id': 'u3', 'journey': ['Organic', 'Email', 'Google Ads'],
     'conversion_value': 100, 'converted': True},
    {'user_id': 'u4', 'journey': ['Email', 'Facebook', 'Google Ads'],
     'conversion_value': 175, 'converted': True},
    {'user_id': 'u5', 'journey': ['Organic', 'Facebook'],
     'conversion_value': 0, 'converted': False},
])

# Apply all attribution models
analyzer = AttributionAnalyzer(journeys)
comparison = analyzer.compare_models()

print("Attribution Model Comparison:")
print(comparison.round(2))

# Calculate channel costs
channel_costs = {
    'Google Ads': 250,
    'Facebook': 200,
    'Email': 50,
    'Organic': 0
}

# Calculate ROI
roi = calculate_channel_roi(comparison, channel_costs)
print("\nROI by Model and Channel:")
print(roi.round(2))

# Which channel has most stable attribution?
attribution_no_total = comparison.drop('TOTAL')
cv = attribution_no_total.std(axis=1) / attribution_no_total.mean(axis=1)

print("\nAttribution Stability (Coefficient of Variation):")
print(cv.sort_values())
print("\nLower = more stable across models")
```

### Exercise 2: Path Analysis
```python
# Analyze most common conversion paths
journeys = pd.DataFrame([
    {'journey': ['Organic', 'Facebook', 'Google Ads'], 'converted': True, 'value': 100},
    {'journey': ['Organic', 'Facebook', 'Google Ads'], 'converted': True, 'value': 120},
    {'journey': ['Facebook', 'Google Ads'], 'converted': True, 'value': 150},
    {'journey': ['Facebook', 'Google Ads'], 'converted': True, 'value': 130},
    {'journey': ['Email', 'Google Ads'], 'converted': True, 'value': 90},
    {'journey': ['Organic', 'Email', 'Facebook'], 'converted': False, 'value': 0},
    {'journey': ['Facebook'], 'converted': False, 'value': 0},
])

# Convert to path strings
journeys['path'] = journeys['journey'].apply(lambda x: ' → '.join(x))

# Analyze converting paths
converting = journeys[journeys['converted']].groupby('path').agg({
    'converted': 'count',
    'value': 'sum'
}).rename(columns={'converted': 'count'})

converting['avg_value'] = converting['value'] / converting['count']
converting = converting.sort_values('count', ascending=False)

print("Top Converting Paths:")
print(converting)

# Path length analysis
journeys['path_length'] = journeys['journey'].apply(len)
path_length_stats = journeys.groupby('converted')['path_length'].agg(['mean', 'median', 'min', 'max'])

print("\nPath Length by Conversion:")
print(path_length_stats)
```

### Exercise 3: Channel Effectiveness
```python
from collections import Counter

# Analyze which channels appear most in converting vs non-converting journeys
converting_journeys = journeys[journeys['converted']]
non_converting_journeys = journeys[~journeys['converted']]

# Touchpoint frequency
converting_touchpoints = []
for journey in converting_journeys['journey']:
    converting_touchpoints.extend(journey)

non_converting_touchpoints = []
for journey in non_converting_journeys['journey']:
    non_converting_touchpoints.extend(journey)

converting_freq = Counter(converting_touchpoints)
non_converting_freq = Counter(non_converting_touchpoints)

# Compare
all_channels = set(list(converting_freq.keys()) + list(non_converting_freq.keys()))

comparison = []
for channel in all_channels:
    conv_count = converting_freq.get(channel, 0)
    non_conv_count = non_converting_freq.get(channel, 0)
    total_count = conv_count + non_conv_count

    conversion_rate = conv_count / total_count if total_count > 0 else 0

    comparison.append({
        'channel': channel,
        'converting_appearances': conv_count,
        'non_converting_appearances': non_conv_count,
        'total_appearances': total_count,
        'conversion_rate': conversion_rate
    })

comparison_df = pd.DataFrame(comparison).sort_values('conversion_rate', ascending=False)

print("Channel Effectiveness Analysis:")
print(comparison_df.round(3))

print("\nInsights:")
best_channel = comparison_df.iloc[0]
print(f"Most effective channel: {best_channel['channel']}")
print(f"  Conversion rate: {best_channel['conversion_rate']:.1%}")
```

### Exercise 4: Build Custom Attribution Model
```python
# Build custom model: 50% last-touch, 30% first-touch, 20% evenly distributed to middle
def custom_50_30_20_attribution(journey, conversion_value):
    """
    Custom attribution:
    - 50% to last touchpoint
    - 30% to first touchpoint
    - 20% evenly to middle touchpoints (if any)
    """
    n = len(journey)
    attribution = {ch: 0 for ch in set(journey)}

    if n == 1:
        attribution[journey[0]] = conversion_value
    elif n == 2:
        attribution[journey[0]] = conversion_value * 0.30
        attribution[journey[-1]] = conversion_value * 0.50
        # Remaining 20% lost (no middle)
    else:
        # First
        attribution[journey[0]] = conversion_value * 0.30
        # Last
        attribution[journey[-1]] = conversion_value * 0.50
        # Middle
        middle_touchpoints = journey[1:-1]
        middle_credit = (conversion_value * 0.20) / len(middle_touchpoints)
        for channel in middle_touchpoints:
            attribution[channel] += middle_credit

    return attribution

# Apply to all journeys
custom_attribution_results = {}

for _, row in journeys[journeys['converted']].iterrows():
    attr = custom_50_30_20_attribution(row['journey'], row['value'])
    for channel, value in attr.items():
        custom_attribution_results[channel] = custom_attribution_results.get(channel, 0) + value

print("Custom 50-30-20 Attribution Model Results:")
for channel, value in sorted(custom_attribution_results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {channel}: ${value:.2f}")
```

---

## 🔍 Attribution Model Selection Guide

### Decision Tree
```
Start
  ├─ Need simple, explainable model?
  │   ├─ Yes → Last-Touch or First-Touch
  │   └─ No → Continue
  │
  ├─ Have <200 conversions?
  │   ├─ Yes → Position-Based (U-shaped)
  │   └─ No → Continue
  │
  ├─ Have >1000 conversions with diverse paths?
  │   ├─ Yes → Data-Driven (Markov/Shapley)
  │   └─ No → Linear or Position-Based
  │
  └─ Specific business requirements?
      └─ Yes → Custom rule-based model
```

### Model Comparison Table

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **Last-Touch** | Simple, matches platform reporting | Ignores customer journey | Direct response, last-click |
| **First-Touch** | Credits acquisition | Ignores nurturing | Top-of-funnel focus |
| **Linear** | Fair, considers all touches | May over-credit passive | Equal-value touchpoints |
| **Position-Based** | Balances acquisition + conversion | Arbitrary weights | Most businesses |
| **Time Decay** | Recent touches matter more | Decay rate arbitrary | Long sales cycles |
| **Markov** | Data-driven, marginal contribution | Complex, needs data | Mature attribution |
| **Shapley** | Theoretically optimal | Very complex | Advanced analytics |

---

**Quick Navigation:**
- [← Week 8 Cheatsheet](Week_08_Cheatsheet.md)
- [Week 10 Cheatsheet →](Week_10_Cheatsheet.md)
- [Back to Main README](../README.md)
