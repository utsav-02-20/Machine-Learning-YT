# Insurance Dataset — EDA, Preprocessing & Feature Selection Revision Notes

> **Source:** `pro(2).ipynb`  
> **Dataset:** `insurance.csv`  
> **Main topics:** EDA, visualization, categorical encoding, feature engineering, standardization, Pearson correlation, p-values, Chi-square feature selection.

---

# 1. Libraries Used
      
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')
```

| Library | Purpose |
|---|---|
| NumPy | Numerical operations |
| Pandas | Data loading/manipulation |
| Seaborn | Statistical visualization |
| Matplotlib | Plotting |
| `StandardScaler` | Feature standardization |
| `pearsonr` | Pearson correlation + p-value |
| `chi2_contingency` | Chi-square test |
| `warnings` | Suppress warning messages |

---

# 2. Dataset Loading

```python
df = pd.read_csv('insurance.csv')
df
```

`pd.read_csv()` loads the CSV dataset into a Pandas DataFrame.

The notebook works with insurance-related features such as:

- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`
- `charges`

Here, `charges` is the numerical target that the notebook later uses for feature-selection analysis.

---

# 3. Exploratory Data Analysis (EDA)

The notebook starts with basic EDA before preprocessing.

## EDA Workflow

```text
Load Dataset
     ↓
Inspect Shape
     ↓
View Head
     ↓
Check Info / Data Types
     ↓
Statistical Summary
     ↓
Missing Values
     ↓
Visualize Distributions
     ↓
Check Correlations
```

---

# 4. Basic Dataset Inspection

## 4.1 Shape

```python
df.shape
```

Returns:

```text
(number_of_rows, number_of_columns)
```

### Remember

```python
df.shape[0]   # rows
df.shape[1]   # columns
```

---

## 4.2 First Rows

```python
df.head()
```

Displays the first 5 rows by default.

```python
df.head(10)
```

displays the first 10 rows.

---

## 4.3 Dataset Information

```python
df.info()
```

Provides:

- Column names
- Number of entries
- Non-null counts
- Data types
- Memory usage

---

## 4.4 Statistical Summary

```python
df.describe()
```

For numerical columns, gives:

| Statistic | Meaning |
|---|---|
| count | Number of non-null observations |
| mean | Average |
| std | Standard deviation |
| min | Minimum |
| 25% | First quartile |
| 50% | Median |
| 75% | Third quartile |
| max | Maximum |

---

## 4.5 Missing Values

```python
df.isnull().sum()
```

Counts missing values in every column.

### Important

```text
isnull()       → identifies missing values
isnull().sum() → counts missing values
```

The notebook checks for missing values before proceeding with preprocessing.

---

## 4.6 Column Names

```python
print(df.columns)
```

Displays all column names.

The notebook also executes:

```python
print(df.info())
```

Note that `df.info()` itself prints information and normally returns `None`, so wrapping it inside `print()` is unnecessary.

---

# 5. Numerical Columns

The notebook explicitly defines:

```python
numeric_columns = [
    'age',
    'bmi',
    'children',
    'charges'
]
```

These are used for numerical visualization.

---

# 6. Histograms

The notebook creates histograms for every numerical feature:

```python
for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(
        df[col],
        kde=True,
        bins=20
    )
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('frequency')
    plt.show()
```

## `sns.histplot()`

A histogram shows the distribution of numerical data.

### Parameters

```python
sns.histplot(
    data,
    kde=True,
    bins=20
)
```

- `kde=True` → adds density curve
- `bins=20` → divides values into 20 intervals

### Uses

Histograms help identify:

- Distribution shape
- Concentration of values
- Skewness
- Possible unusual values

---

# 7. Count Plots

Count plots are used for categorical/discrete variables.

## 7.1 Children

```python
sns.countplot(x=df['children'])
plt.show()
```

Shows how frequently each number of children occurs.

---

## 7.2 Sex

```python
sns.countplot(y=df['sex'])
plt.show()
```

Here the categorical variable is placed on the y-axis.

---

## 7.3 Smoker

```python
sns.countplot(x=df['smoker'])
plt.show()
```

Shows the frequency of:

```text
yes
no
```

---

# 8. Box Plots

```python
for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
```

A box plot summarizes a numerical distribution.

## It helps identify

- Median
- Quartiles
- Interquartile Range
- Spread
- Potential outliers

### Important

Box plot is particularly useful for detecting possible outliers.

---

# 9. Correlation Heatmap

```python
plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
```

This calculates correlations between numerical variables and displays them as a heatmap.

---

## Correlation coefficient

Pearson correlation generally ranges from:

\[
-1 \leq r \leq 1
\]

| `r` | Interpretation |
|---|---|
| +1 | Perfect positive linear relationship |
| Close to +1 | Strong positive relationship |
| 0 | Little/no linear relationship |
| Close to -1 | Strong negative relationship |
| -1 | Perfect negative linear relationship |

### `annot=True`

Displays numerical correlation values inside heatmap cells.

### `numeric_only=True`

Uses only numerical columns.

---

# 10. Data Cleaning and Processing

The notebook creates a copy before modifying the data:

```python
df_cleaned = df.copy()
```

## Why copy?

It preserves the original DataFrame:

```text
df          → original data
df_cleaned  → preprocessing version
```

This is a good practice because changes to `df_cleaned` do not intentionally overwrite the original dataset.

---

# 11. Checking the Cleaned Dataset

The notebook checks:

```python
df_cleaned.shape
df_cleaned.head()
df_cleaned.isnull().sum()
df_cleaned.dtypes
```

These verify:

- Dimensions
- Sample records
- Missing values
- Data types

---

# 12. Frequency of Categorical Values

## Sex

```python
df_cleaned['sex'].value_counts()
```

Counts occurrences of each category.

Example conceptually:

```text
male    → count
female  → count
```

---

## Smoker

```python
df_cleaned['smoker'].value_counts()
```

Counts:

```text
yes
no
```

---

## Unique values

```python
df_cleaned['smoker'].unique()
```

Returns the unique categories present in the column.

### Difference

```text
value_counts() → values + frequency
unique()       → unique values only
```

---

# 13. Encoding `sex`

The notebook converts:

```text
male   → 0
female → 1
```

using:

```python
df_cleaned['sex'] = df_cleaned['sex'].map({
    'male': 0,
    'female': 1
})
```

## `map()`

Used to replace values according to a mapping dictionary.

### General pattern

```python
df['column'] = df['column'].map({
    'category1': 0,
    'category2': 1
})
```

---

# 14. Encoding `smoker`

The notebook first checks the categories:

```python
df_cleaned['smoker'].value_counts()

print(df_cleaned['smoker'].unique())
```

Then:

```python
df_cleaned['smoker'] = df_cleaned['smoker'].replace({
    'yes': 1,
    'no': 0
})
```

### Result

```text
yes → 1
no  → 0
```

---

## `map()` vs `replace()`

### `map()`

```python
df['sex'].map({
    'male': 0,
    'female': 1
})
```

Maps values according to a dictionary.

### `replace()`

```python
df['smoker'].replace({
    'yes': 1,
    'no': 0
})
```

Replaces specified values.

Both are useful for categorical conversion.

---

# 15. Renaming Columns

The notebook renames:

```text
sex    → is_female
smoker → is_smoker
```

using:

```python
df_cleaned.rename(
    columns={
        'sex': 'is_female',
        'smoker': 'is_smoker'
    },
    inplace=True
)
```

## Why?

The new names communicate the meaning of the encoded binary variables more clearly.

After encoding:

```text
is_female = 0 → male
is_female = 1 → female

is_smoker = 0 → non-smoker
is_smoker = 1 → smoker
```

---

# 16. Region Encoding

First:

```python
df['region'].value_counts()
```

checks the distribution of regions.

Then:

```python
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=['region'],
    dtype=int
)
```

converts the categorical `region` column into separate binary columns.

Expected categories in the dataset:

```text
northeast
northwest
southeast
southwest
```

The resulting columns are:

```text
region_northeast
region_northwest
region_southeast
region_southwest
```

Each contains `0` or `1`.

---

# 17. One-Hot Encoding

`pd.get_dummies()` performs one-hot encoding.

### Example

Original:

| region |
|---|
| northeast |
| southwest |

After encoding:

| region_northeast | region_southwest |
|---:|---:|
| 1 | 0 |
| 0 | 1 |

### Why encode?

Most machine-learning algorithms require numerical feature representations.

---

# 18. Feature Engineering

The notebook starts feature engineering with:

```python
# feature engineering and extraction
```

Feature engineering means creating or transforming features to make them more useful for analysis/modeling.

---

# 19. BMI Distribution

```python
sns.histplot(df['bmi'])
```

This visualizes the BMI distribution.

---

# 20. Creating BMI Categories

The notebook creates:

```python
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[
        0,
        18.5,
        24.9,
        29.9,
        float('inf')
    ],
    labels=[
        'underweight',
        'normal',
        'overweight',
        'obese'
    ]
)
```

This converts continuous BMI into categorical ranges.

---

# 21. BMI Categories Used in Notebook

| BMI Range | Category |
|---|---|
| 0–18.5 | Underweight |
| 18.5–24.9 | Normal |
| 24.9–29.9 | Overweight |
| 29.9+ | Obese |

### Important

The notebook uses the exact bin boundaries:

```python
[0, 18.5, 24.9, 29.9, inf]
```

and labels:

```text
underweight
normal
overweight
obese
```

---

# 22. `pd.cut()`

`pd.cut()` converts continuous numerical values into discrete intervals/categories.

### General syntax

```python
pd.cut(
    column,
    bins=[...],
    labels=[...]
)
```

### Example

```python
df['category'] = pd.cut(
    df['age'],
    bins=[0, 18, 60, 100],
    labels=['young', 'adult', 'senior']
)
```

---

# 23. Checking BMI Category Frequencies

```python
df_cleaned['bmi_category'].value_counts()
```

Counts the number of records in each BMI category.

---

# 24. Encoding BMI Categories

The notebook converts BMI categories into numerical dummy variables:

```python
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=['bmi_category'],
    dtype=int
)
```

This produces:

```text
bmi_category_underweight
bmi_category_normal
bmi_category_overweight
bmi_category_obese
```

Each contains binary values.

---

# 25. Standard Scaling

The notebook scales:

```python
cols = [
    'age',
    'bmi',
    'children'
]

scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(
    df_cleaned[cols]
)
```

---

## Standardization Formula

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- `x` = original value
- `μ` = mean
- `σ` = standard deviation
- `z` = standardized value

After standardization:

```text
mean ≈ 0
standard deviation ≈ 1
```

---

# 26. Why Standardize?

The features may have different ranges:

```text
age       → tens
bmi       → tens
children  → small integers
```

Standardization puts selected numerical features on a comparable scale.

This is particularly useful for algorithms sensitive to feature magnitude.

---

# 27. `fit_transform()`

```python
scaler.fit_transform(data)
```

performs:

### Step 1 — Fit

Learns:

- Mean
- Standard deviation

### Step 2 — Transform

Applies:

\[
z=\frac{x-\mu}{\sigma}
\]

### Memory trick

```text
fit       → learn
transform → apply
fit_transform → learn + apply
```

---

# 28. Selected Features

The notebook defines:

```python
selected_features = [
    'age',
    'bmi',
    'children',
    'is_smoker',
    'is_female',
    'region_northeast',
    'region_northwest',
    'region_southeast',
    'region_southwest',
    'bmi_category_underweight',
    'bmi_category_normal',
    'bmi_category_overweight',
    'bmi_category_obese'
]
```

These features are later tested against:

```text
charges
```

---

# 29. Ensuring Integer Type

The notebook explicitly converts:

```python
df_cleaned['is_smoker'] = df_cleaned['is_smoker'].astype('int64')
```

### Why?

To ensure that the feature is represented as an integer.

```text
0 → non-smoker
1 → smoker
```

---

# 30. Pearson Correlation

The notebook imports:

```python
from scipy.stats import pearsonr
```

Then calculates correlation between each selected feature and `charges`.

```python
results_ori = []

for feature in selected_features:
    corr, p_value = pearsonr(
        df_cleaned[feature],
        df_cleaned['charges']
    )

    results_ori.append([
        feature,
        corr,
        p_value
    ])
```

---

# 31. Pearson Correlation Formula

Pearson correlation measures the strength and direction of a **linear relationship** between two variables.

Conceptually:

\[
r =
\frac{
\operatorname{Cov}(X,Y)
}{
\sigma_X\sigma_Y
}
\]

Range:

\[
-1 \le r \le 1
\]

---

# 32. Interpreting Pearson Correlation

### Positive correlation

```text
X increases → Y tends to increase
```

### Negative correlation

```text
X increases → Y tends to decrease
```

### Near zero

```text
Weak/no linear relationship
```

### Important

Correlation does **not** prove causation.

---

# 33. P-Value

`pearsonr()` returns:

```python
corr, p_value
```

The p-value helps evaluate whether the observed correlation provides statistical evidence against the null hypothesis of no linear association.

The notebook later uses:

```python
alpha = 0.05
```

as its significance threshold for the Chi-square test.

A common interpretation in this notebook's testing framework is:

```text
p < 0.05 → statistically significant
p >= 0.05 → not statistically significant
```

---

# 34. Storing Correlation Results

The notebook stores:

```python
results_ori.append([
    feature,
    corr,
    p_value
])
```

So each result contains:

```text
Feature
Correlation
P-value
```

---

# 35. Printing Correlation Results

```python
print(f'feature   correlation   p_value')

for feature, corr, p_value in results_ori:
    print(
        f'{feature:25} '
        f'{corr:.4f} '
        f'{p_value:.6e}'
    )
```

### Formatting

```text
:.4f       → 4 digits after decimal
:.6e       → scientific notation with 6 decimal places
```

---

# 36. Creating a Results DataFrame

```python
results_ori_table = pd.DataFrame(
    results_ori,
    columns=[
        'Features',
        'Correlation',
        'P_Value'
    ]
)
```

This converts the list of results into a structured DataFrame.

---

# 37. Sorting Correlation Results

## Ascending

```python
results_ori_table_sorted_ascending = (
    results_ori_table.sort_values(
        by='Correlation'
    )
)
```

Sorts from smallest to largest correlation.

---

## Descending

```python
results_ori_table_sorted_not_ascending = (
    results_ori_table.sort_values(
        by='Correlation',
        ascending=False
    )
)
```

Sorts from largest to smallest correlation.

---

## Print without index

```python
print(
    results_ori_table_sorted_not_ascending
    .to_string(index=False)
)
```

### `index=False`

Prevents DataFrame index numbers from being printed.

---

# 38. Chi-Square Feature Selection

The notebook then applies a Chi-square test.

Import:

```python
from scipy.stats import chi2_contingency
```

The code sets:

```python
alpha = 0.05
```

---

# 39. Why Bin `charges`?

The target:

```text
charges
```

is continuous.

For the Chi-square test, the notebook creates categorical bins:

```python
df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)
```

---

# 40. `pd.qcut()`

`qcut()` divides data into quantile-based groups.

Here:

```python
q=4
```

creates four groups.

Conceptually:

```text
charges
   ↓
Quartile 1
Quartile 2
Quartile 3
Quartile 4
```

`labels=False` assigns integer category labels.

---

# 41. Chi-Square Test

For every selected feature:

```python
for col in selected_features:
    contingency = pd.crosstab(
        df_cleaned[col],
        df_cleaned['charges_bin']
    )

    chi2_stats, p_val, _, _ = chi2_contingency(
        contingency
    )
```

---

# 42. Contingency Table

```python
pd.crosstab(
    df_cleaned[col],
    df_cleaned['charges_bin']
)
```

creates a frequency table showing combinations of:

```text
Feature category/value
        ×
Charges category
```

Example conceptual structure:

| Feature | Low charges | Q2 | Q3 | High charges |
|---|---:|---:|---:|---:|
| Category A | ... | ... | ... | ... |
| Category B | ... | ... | ... | ... |

---

# 43. Chi-Square Test Concept

The Chi-square test checks whether two categorical variables are statistically associated.

### Null hypothesis (H₀)

> The variables are independent.

### Alternative hypothesis (H₁)

> The variables are associated.

---

# 44. Decision Rule Used in Notebook

The notebook uses:

```python
decision = (
    'Reject Null (Keep Feature)'
    if p_val < alpha
    else
    'Accept Null (Drop Feature)'
)
```

With:

```python
alpha = 0.05
```

Therefore:

### If:

\[
p < 0.05
\]

→ Reject null hypothesis  
→ Keep feature

### If:

\[
p \geq 0.05
\]

→ Do not reject null hypothesis  
→ Notebook labels it as drop feature

> **Terminology note:** In formal statistical writing, "fail to reject the null" is generally preferred over "accept the null." The notebook uses `"Accept Null (Drop Feature)"`, so that exact notebook terminology is preserved here.

---

# 45. Storing Chi-Square Results

The notebook stores:

```python
chi2_results[col] = {
    'chi2_statistic': chi2_stats,
    'p_value': p_val,
    'Decision': decision
}
```

Each feature gets:

- Chi-square statistic
- P-value
- Decision

---

# 46. Converting Results to DataFrame

```python
chi2_results_table = pd.DataFrame(
    chi2_results
)

chi2_results_table
```

This makes the Chi-square results easier to inspect.

---

# 47. Features Marked for Dropping

The notebook finally defines:

```python
drop_features = [
    'region_northeast',
    'region_northwest',
    'region_southwest',
    'bmi_category_underweight',
    'bmi_category_normal',
    'bmi_category_overweight',
    'bmi_category_obese'
]
```

These are the features selected by the notebook's final `drop_features` list.

### Important

The notebook itself does not show an actual:

```python
df_cleaned.drop(...)
```

operation after defining the list. It only creates the list of features to drop.

---

# 48. Complete Pipeline

The complete notebook workflow is:

```text
insurance.csv
      ↓
Load with Pandas
      ↓
EDA
      ↓
Shape / Head / Info / Describe
      ↓
Missing-value check
      ↓
Histograms
      ↓
Count plots
      ↓
Box plots
      ↓
Correlation heatmap
      ↓
Copy original DataFrame
      ↓
Encode sex
      ↓
Encode smoker
      ↓
Rename encoded columns
      ↓
One-hot encode region
      ↓
Feature engineering
      ↓
Create BMI categories
      ↓
One-hot encode BMI categories
      ↓
Standardize age, BMI, children
      ↓
Pearson correlation with charges
      ↓
Create correlation results table
      ↓
Chi-square feature selection
      ↓
Create drop_features list
```

---

# 49. Important Code Cheat Sheet

## Load

```python
df = pd.read_csv('insurance.csv')
```

## Inspect

```python
df.shape
df.head()
df.info()
df.describe()
df.isnull().sum()
df.columns
```

## Frequency

```python
df['column'].value_counts()
df['column'].unique()
```

## Copy

```python
df_cleaned = df.copy()
```

## Binary mapping

```python
df['sex'] = df['sex'].map({
    'male': 0,
    'female': 1
})
```

## Binary replacement

```python
df['smoker'] = df['smoker'].replace({
    'yes': 1,
    'no': 0
})
```

## Rename

```python
df.rename(
    columns={'old': 'new'},
    inplace=True
)
```

## One-hot encoding

```python
df = pd.get_dummies(
    df,
    columns=['region'],
    dtype=int
)
```

## Binning

```python
df['category'] = pd.cut(
    df['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=[
        'underweight',
        'normal',
        'overweight',
        'obese'
    ]
)
```

## Standardization

```python
scaler = StandardScaler()

df[cols] = scaler.fit_transform(
    df[cols]
)
```

## Pearson correlation

```python
corr, p_value = pearsonr(x, y)
```

## Chi-square

```python
chi2, p, dof, expected = chi2_contingency(
    contingency_table
)
```

---

# 50. Key Differences to Remember

| Concept | Purpose |
|---|---|
| `value_counts()` | Frequency of values |
| `unique()` | Unique values |
| `map()` | Map values using a mapping |
| `replace()` | Replace specified values |
| `get_dummies()` | One-hot encode categories |
| `cut()` | Create fixed interval categories |
| `qcut()` | Create quantile-based categories |
| `corr()` | Correlation matrix |
| `pearsonr()` | Correlation + p-value for two variables |
| `chi2_contingency()` | Chi-square test for categorical association |
| `StandardScaler()` | Standardize numerical features |

---

# 51. Pearson vs Chi-Square

| Property | Pearson Correlation | Chi-Square |
|---|---|---|
| Main use | Linear relationship | Association between categorical variables |
| Variables | Numerical | Categorical |
| Output | Correlation + p-value | Chi-square statistic + p-value |
| Correlation range | `-1` to `+1` | No `-1` to `+1` range |
| Notebook target | `charges` directly | Binned `charges` |
| Main interpretation | Strength/direction of linear association | Independence/association |

---

# 52. `cut()` vs `qcut()`

| `pd.cut()` | `pd.qcut()` |
|---|---|
| Uses specified numerical boundaries | Uses quantiles |
| Creates fixed intervals | Creates groups based on distribution |
| Example: BMI ranges | Example: charge quartiles |
| Used for `bmi_category` | Used for `charges_bin` |

### Memory trick

```text
cut  → fixed CUT points
qcut → quantile CUT points
```

---

# 53. Statistical Testing Decision Rule

The notebook uses:

```python
alpha = 0.05
```

Remember:

```text
p < 0.05
    ↓
statistically significant
    ↓
reject H₀
```

and:

```text
p >= 0.05
    ↓
not statistically significant
    ↓
do not reject H₀
```

For this notebook's feature-selection logic:

```text
p < 0.05  → Keep Feature
p >= 0.05 → Drop Feature
```

---

# 54. Viva / Exam Questions

## Q1. What is EDA?

Exploratory Data Analysis is the process of inspecting, summarizing, visualizing, and understanding a dataset before further modeling.

---

## Q2. Why use `df.copy()`?

To create a separate DataFrame for preprocessing while preserving the original dataset.

---

## Q3. What does `value_counts()` do?

It counts the frequency of each unique value.

---

## Q4. Why encode `sex` and `smoker`?

They are categorical values, so the notebook converts them into numerical binary representations.

---

## Q5. What is one-hot encoding?

It converts categorical classes into separate binary indicator columns.

---

## Q6. What does `pd.cut()` do?

Converts a continuous variable into categories using specified bin boundaries.

---

## Q7. What does `pd.qcut()` do?

Divides data into quantile-based groups.

---

## Q8. Why create `bmi_category`?

It transforms continuous BMI into meaningful categories that can be treated as categorical features.

---

## Q9. Why standardize features?

To put selected numerical features on a comparable scale.

---

## Q10. What is the formula for standardization?

\[
z=\frac{x-\mu}{\sigma}
\]

---

## Q11. What does Pearson correlation measure?

The strength and direction of a linear relationship between two numerical variables.

---

## Q12. What is the range of Pearson correlation?

\[
[-1,+1]
\]

---

## Q13. What does a p-value indicate?

It provides evidence for assessing whether an observed relationship is statistically significant under the chosen statistical test.

---

## Q14. What is the null hypothesis for the Chi-square independence test?

The two categorical variables are independent.

---

## Q15. What is a contingency table?

A frequency table showing combinations of categories from two categorical variables.

---

## Q16. Why was `charges` converted into `charges_bin`?

Because the notebook applies a Chi-square test, which is designed for categorical data. `qcut()` converts continuous charges into four quantile-based categories.

---

## Q17. What does `alpha = 0.05` mean?

It is the significance threshold used by the notebook for its statistical decision rule.

---

## Q18. What happens when `p < 0.05`?

The notebook rejects the null hypothesis and marks the feature to be kept.

---

## Q19. What does `annot=True` do?

Displays numerical values on the heatmap.

---

## Q20. What does `ascending=False` do?

Sorts values from largest to smallest.

---

# 55. Last-Minute Revision

```text
EDA
│
├── shape()             → dimensions
├── head()              → first rows
├── info()              → types + non-null
├── describe()          → statistics
├── isnull().sum()      → missing count
└── columns             → column names

Visualization
│
├── histplot()          → numerical distribution
├── countplot()         → category frequency
├── boxplot()           → spread/outliers
└── heatmap()           → correlation matrix

Preprocessing
│
├── copy()              → preserve original
├── map()               → mapping
├── replace()           → replacement
├── rename()            → rename columns
└── get_dummies()       → one-hot encoding

Feature Engineering
│
├── pd.cut()            → fixed bins
├── bmi_category        → BMI groups
└── get_dummies()       → encode BMI groups

Scaling
│
└── StandardScaler()
       ↓
   z = (x - μ) / σ

Feature Selection
│
├── pearsonr()
│      ├── correlation
│      └── p-value
│
└── chi2_contingency()
       ├── chi-square statistic
       └── p-value
```

---

# 56. Super-Short Memory Formula Sheet

### Standardization

\[
\boxed{z=\frac{x-\mu}{\sigma}}
\]

### Pearson correlation

\[
\boxed{-1\leq r\leq1}
\]

### Significance rule used in notebook

\[
\boxed{p<0.05\Rightarrow\text{Reject }H_0}
\]

### Chi-square workflow

```text
Categorical Feature
       ↓
charges → qcut()
       ↓
Contingency Table
       ↓
chi2_contingency()
       ↓
p-value
       ↓
p < 0.05 → Keep
p ≥ 0.05 → Drop
```

---

# 57. Final Exam Checklist

Before an exam, make sure you can explain:

- [ ] EDA
- [ ] `shape`, `head`, `info`, `describe`
- [ ] Missing-value checking
- [ ] `value_counts()` vs `unique()`
- [ ] Histogram
- [ ] Count plot
- [ ] Box plot
- [ ] Correlation heatmap
- [ ] `df.copy()`
- [ ] `map()` and `replace()`
- [ ] Binary encoding
- [ ] One-hot encoding
- [ ] `pd.cut()`
- [ ] `pd.qcut()`
- [ ] Feature engineering
- [ ] StandardScaler
- [ ] Standardization formula
- [ ] Pearson correlation
- [ ] P-value
- [ ] Chi-square test
- [ ] Contingency table
- [ ] Alpha/significance level
- [ ] Feature-selection decision rule
- [ ] Difference between Pearson and Chi-square

---

## One-Line Memory Trick

> **Inspect → Visualize → Clean → Encode → Engineer → Scale → Correlate → Test → Select**
