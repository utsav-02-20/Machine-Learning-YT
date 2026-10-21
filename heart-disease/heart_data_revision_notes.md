# Data Analysis & Preprocessing — Revision Notes

> **Source:** `pro(1).ipynb`  
> **Dataset used:** `heart.csv`  
> **Main focus:** Pandas data inspection, missing/invalid-value handling, visualization with Seaborn/Matplotlib, categorical encoding, correlation, and standard scaling.

---

## 1. Libraries Used
  
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### Purpose

| Library | Main use in notebook |
|---|---|
| `NumPy (np)` | Numerical operations |
| `Pandas (pd)` | Data loading, inspection, cleaning, preprocessing |
| `Seaborn (sns)` | Statistical/data visualization |
| `Matplotlib (plt)` | Plot creation and figure management |

---

# 2. Loading a Dataset

```python
df = pd.read_csv('heart.csv')
df
```

### `pd.read_csv()`

Loads a CSV file into a Pandas **DataFrame**.

### Important

```python
df = pd.read_csv("file.csv")
```

After loading, `df` contains rows and columns of the dataset.

---

# 3. Basic Dataset Inspection

## 3.1 `df.shape`

```python
df.shape
```

Returns:

```text
(number_of_rows, number_of_columns)
```

### Example

```text
(918, 12)
```

means 918 rows and 12 columns.

### Remember

- `shape` → dimensions
- `shape[0]` → number of rows
- `shape[1]` → number of columns

---

## 3.2 `df.head()`

```python
df.head()
```

Displays the **first 5 rows** by default.

You can also specify the number:

```python
df.head(10)
```

→ first 10 rows.

### Exam point

> `head()` is used for quickly viewing the beginning of a DataFrame.

---

## 3.3 `df.info()`

```python
df.info()
```

Provides information about:

- Number of rows
- Column names
- Number of non-null values
- Data types
- Memory usage

### Typical data types

- `int64`
- `float64`
- `object`
- `bool`
- `category`
- `string`

### Exam point

> `info()` is useful for checking dataset structure and identifying possible missing values/data-type issues.

---

## 3.4 `df.describe()`

```python
df.describe()
```

Generates descriptive statistics for numerical columns.

Important statistics include:

- `count`
- `mean`
- `std`
- `min`
- `25%`
- `50%`
- `75%`
- `max`

### Interpretation

| Statistic | Meaning |
|---|---|
| Count | Number of non-null observations |
| Mean | Arithmetic average |
| Std | Standard deviation |
| Min | Minimum value |
| 25% | First quartile |
| 50% | Median |
| 75% | Third quartile |
| Max | Maximum value |

### Important

`50%` represents the **median**.

---

# 4. Checking Missing Values

## 4.1 `df.isnull()`

```python
df.isnull()
```

Returns a DataFrame of Boolean values:

- `True` → value is missing
- `False` → value is not missing

---

## 4.2 `df.isnull().sum()`

```python
df.isnull().sum()
```

Counts missing values in every column.

### Why?

Boolean values behave numerically in this operation:

```text
True  → 1
False → 0
```

Therefore, summing gives the number of missing values.

### Exam point

```python
df.isnull().sum()
```

is one of the most common commands for detecting missing values.

---

# 5. Viewing Column Names

```python
print(df.columns)
```

Returns the column labels of the DataFrame.

### Example columns in the notebook

The notebook works with columns such as:

- `Age`
- `Sex`
- `ChestPainType`
- `RestingBP`
- `Cholesterol`
- `FastingBS`
- `MaxHR`
- `Oldpeak`
- `HeartDisease`

---

# 6. Detecting Duplicate Rows

## 6.1 `df.duplicated()`

```python
df.duplicated()
```

Returns `True` for rows that are duplicates of an earlier row.

---

## 6.2 Count duplicates

```python
df.duplicated().sum()
```

Returns the total number of duplicate rows.

### Quick revision

```text
duplicated()      → identifies duplicate rows
duplicated().sum() → counts duplicate rows
```

---

# 7. Selecting Numerical Columns for Visualization

The notebook defines:

```python
numeric_columns = [
    'age',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'MaxHR',
    'Oldpeak',
    'HeartDisease'
]
```

These are numerical variables selected for analysis.

### Note

The notebook later uses `Age` with a capital `A` in plotting code. Pandas column names are **case-sensitive**, so column names must exactly match the DataFrame.

---

# 8. Bar Plot of Heart Disease

```python
df['HeartDisease'].value_counts().plot(kind='bar')
```

### What happens?

`value_counts()` counts how many times each unique value occurs.

For a binary target such as:

```text
0 → No heart disease
1 → Heart disease
```

the bar chart shows the number of samples in each class.

### `kind='bar'`

Creates a bar chart.

---

# 9. Histograms

A histogram shows the **distribution of numerical data**.

The notebook defines:

```python
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)
```

Then:

```python
plt.figure(figsize=(10, 8))
plt.tight_layout

plotting("Age", 1)
plotting("Cholesterol", 2)
plotting("RestingBP", 3)
plotting("MaxHR", 4)
```

---

## 9.1 `sns.histplot()`

```python
sns.histplot(df[var], kde=True)
```

Creates a histogram.

### `kde=True`

Adds a **Kernel Density Estimate (KDE)** curve to visualize the approximate probability density.

---

## 9.2 `plt.subplot()`

```python
plt.subplot(2, 2, num)
```

Creates a grid of:

```text
2 rows × 2 columns
```

The third argument determines the subplot position.

```text
1  2
3  4
```

---

## 9.3 `plt.figure(figsize=(10,8))`

Sets the figure size.

---

## 9.4 `plt.tight_layout()`

Adjusts spacing between plots to reduce overlap.

### Notebook note

The notebook contains:

```python
plt.tight_layout
```

without parentheses. The intended function call is:

```python
plt.tight_layout()
```

---

# 10. Data Cleaning — Invalid Zero Values

The notebook treats some zero values as impossible/invalid values for variables such as:

- `Cholesterol`
- `RestingBP`

Instead of leaving these zero values, it calculates a mean from the non-zero observations.

---

# 11. Replacing Zero Cholesterol Values

First calculate the mean using only non-zero values:

```python
ch_mean = df.loc[
    df['Cholesterol'] != 0,
    'Cholesterol'
].mean()
```

### Explanation

```python
df['Cholesterol'] != 0
```

creates a Boolean condition selecting rows where cholesterol is not zero.

```python
df.loc[condition, 'Cholesterol']
```

selects the corresponding cholesterol values.

```python
.mean()
```

calculates their mean.

---

## Replace zero values

```python
df['Cholesterol'] = df['Cholesterol'].replace(
    0,
    ch_mean
)
```

Every `0` in the `Cholesterol` column is replaced by the calculated mean.

Then:

```python
df['Cholesterol'] = df['Cholesterol'].round(2)
```

rounds values to 2 decimal places.

### General pattern

```python
column_mean = df.loc[df['column'] != 0, 'column'].mean()

df['column'] = df['column'].replace(0, column_mean)
df['column'] = df['column'].round(2)
```

---

# 12. Replacing Zero Resting Blood Pressure Values

The notebook calculates:

```python
RestinBP_mean = df.loc[
    df['RestingBP'] != 0,
    'RestingBP'
].mean()
```

Then it contains:

```python
df['RestingBP'] = df['RestingBP'].replace(0, ch_mean)
df['RestingBP'] = df['RestingBP'].round(2)
```

### Important notebook observation

The calculated variable is:

```python
RestinBP_mean
```

but the replacement line uses:

```python
ch_mean
```

So the notebook's code uses the cholesterol mean for the RestingBP replacement. This appears to be an inconsistency/likely coding mistake in the notebook.

If the intention is to replace zero `RestingBP` values with its own non-zero mean, the corresponding operation would be:

```python
df['RestingBP'] = df['RestingBP'].replace(
    0,
    RestinBP_mean
)
```

This note is included because it is important for revision/debugging; the original notebook code should not be confused with the intended preprocessing logic.

---

# 13. Numerical vs Categorical Columns

A very important preprocessing step is identifying numerical and categorical variables.

---

## 13.1 Numerical columns

```python
num_colms = df.select_dtypes(
    include=['number']
).columns

num_colms
```

### Meaning

`select_dtypes()` selects columns according to their data type.

```python
include=['number']
```

selects numerical columns.

Examples:

- Age
- RestingBP
- Cholesterol
- MaxHR
- Oldpeak

---

## 13.2 Categorical columns

```python
cate_colms = df.select_dtypes(
    include=['object', 'category', 'string']
).columns

cate_colms
```

Selects categorical/text-like columns.

Examples include variables such as:

- `Sex`
- `ChestPainType`

### Quick comparison

| Type | Code |
|---|---|
| Numerical | `include=['number']` |
| Categorical | `include=['object','category','string']` |

---

# 14. Count Plot

## Gender distribution

```python
sns.countplot(x=df['Sex'])
```

A count plot displays the number of observations in each category.

### Use

Useful for categorical variables.

---

# 15. Count Plot with Hue

The notebook uses:

```python
sns.countplot(
    x=df['ChestPainType'],
    hue=df['HeartDisease']
)
```

### What does `hue` do?

`hue` adds another categorical variable to split/color the bars.

Here:

- X-axis → Chest pain type
- Hue → Heart disease status

This allows comparison of heart disease cases across chest pain categories.

The notebook notes that the distribution suggests higher heart disease occurrence for certain chest-pain categories, particularly `ASY`, compared with categories such as `ATA`.

### Important

A plot can show **association/distribution**, but it should not automatically be interpreted as proof of causation.

---

# 16. Gender vs Heart Disease

```python
sns.countplot(
    x=df['Sex'],
    hue=df['HeartDisease']
)
```

### Interpretation in notebook

The notebook observes that heart disease occurrence is higher among male patients than female patients in this dataset.

### Exam point

This is a **dataset-level observation**, not a universal medical conclusion.

---

# 17. Box Plot

```python
sns.boxplot(
    x='HeartDisease',
    y='Cholesterol',
    data=df
)
```

A box plot compares the distribution of cholesterol for different heart disease classes.

### Box plot shows

- Median
- Quartiles
- Spread
- Potential outliers

### Important components

```text
        upper whisker
             |
         ┌───────┐
         │       │
         │ median│
         │       │
         └───────┘
             |
        lower whisker
```

The box represents the interquartile range (IQR).

---

# 18. Violin Plot

The notebook uses:

```python
sns.violinplot(
    x='HeartDisease',
    y='Cholesterol',
    data=df
)
```

and:

```python
sns.violinplot(
    x='HeartDisease',
    y='Age',
    data=df
)
```

## What is a violin plot?

A violin plot combines ideas from:

- Box plots
- Distribution/density plots

It shows how the values are distributed across different categories.

### Interpretation

Wider regions indicate greater density of observations around those values.

---

# 19. Correlation Heatmap

```python
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
```

This is one of the most important exploratory-data-analysis operations in the notebook.

---

## 19.1 Correlation

Correlation measures the strength and direction of a linear relationship between two numerical variables.

The correlation coefficient generally ranges from:

```text
-1 to +1
```

### Interpretation

| Correlation | Meaning |
|---|---|
| +1 | Perfect positive linear relationship |
| 0 | No linear relationship |
| -1 | Perfect negative linear relationship |

### Positive correlation

When one variable increases, the other tends to increase.

### Negative correlation

When one variable increases, the other tends to decrease.

---

## 19.2 `df.corr(numeric_only=True)`

Calculates correlations for numerical columns.

`numeric_only=True` prevents non-numerical columns from being included.

---

## 19.3 `sns.heatmap()`

```python
sns.heatmap(correlation_matrix, annot=True)
```

Displays correlation values visually as a color-coded matrix.

### `annot=True`

Displays the numerical correlation values inside the cells.

---

# 20. Data Preprocessing

The notebook converts categorical variables into numerical form using:

```python
df_encode = pd.get_dummies(
    df,
    drop_first=True
)
```

---

# 21. One-Hot Encoding with `pd.get_dummies()`

Machine-learning algorithms generally require numerical input.

Categorical data such as:

```text
Sex = M / F
```

cannot always be directly used as numerical model input.

`pd.get_dummies()` performs **one-hot encoding**.

### Example

Original:

| Sex |
|---|
| M |
| F |
| M |

One-hot encoding can produce:

| Sex_M |
|---:|
| 1 |
| 0 |
| 1 |

---

# 22. Why `drop_first=True`?

```python
pd.get_dummies(df, drop_first=True)
```

drops one category from each categorical variable.

### Reason

For a binary variable:

```text
Sex_M = 1 → Male
Sex_M = 0 → Female
```

A separate `Sex_F` column is unnecessary because it can be inferred.

This can reduce redundant information and avoid perfect linear dependency between dummy variables.

---

# 23. Boolean to Integer Conversion

After encoding, dummy columns can appear as Boolean values:

```text
True
False
```

The notebook converts them to integers:

```python
df_encode = df_encode.astype(int)
```

### Conversion

```text
True  → 1
False → 0
```

### Why?

Many machine-learning workflows expect numerical values such as `0` and `1`.

---

# 24. Standard Scaling

The notebook imports:

```python
from sklearn.preprocessing import StandardScaler
```

Then selects numerical columns:

```python
numerical_col = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]
```

Creates a scaler:

```python
scaler = StandardScaler()
```

and applies it:

```python
df_encode[numerical_col] = scaler.fit_transform(
    df_encode[numerical_col]
)
```

---

# 25. Standardization Formula

Standard scaling transforms a value using:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- `x` = original value
- `μ` = mean of the feature
- `σ` = standard deviation
- `z` = standardized value

After standardization, the feature generally has:

```text
Mean ≈ 0
Standard deviation ≈ 1
```

---

# 26. `StandardScaler()`

```python
scaler = StandardScaler()
```

Creates a standardization object.

### `fit_transform()`

```python
scaler.fit_transform(data)
```

does two operations:

1. **fit** → calculates required statistics such as mean and standard deviation
2. **transform** → applies the standardization formula

### Quick memory trick

```text
fit       → learn parameters
transform → apply parameters
fit_transform → learn + apply
```

---

# 27. Why Scale Numerical Features?

The numerical features in the dataset can have different ranges.

For example:

```text
Age          → tens
Cholesterol  → hundreds
MaxHR        → hundreds
Oldpeak      → smaller values
```

Without scaling, variables with larger numerical magnitudes can have an undesirable influence in algorithms that depend on distances or feature magnitudes.

Standardization puts selected numerical features on a comparable scale.

---

# 28. Complete Notebook Workflow

The notebook follows this general data-analysis pipeline:

```text
Load Libraries
      ↓
Load Dataset
      ↓
Inspect Dataset
      ↓
Check Shape / Head / Info / Describe
      ↓
Check Missing Values
      ↓
Check Duplicate Rows
      ↓
Explore Numerical & Categorical Variables
      ↓
Visualize Data
      ↓
Detect Invalid Zero Values
      ↓
Clean Invalid Values
      ↓
Analyze Relationships
      ↓
Encode Categorical Variables
      ↓
Convert Boolean Values to 0/1
      ↓
Standard Scale Numerical Features
      ↓
Prepared Dataset
```

---

# 29. Important Pandas Commands — Quick Revision Table

| Command | Purpose |
|---|---|
| `pd.read_csv()` | Load CSV |
| `df.shape` | Rows × columns |
| `df.head()` | First 5 rows |
| `df.info()` | Structure and data types |
| `df.describe()` | Statistical summary |
| `df.isnull()` | Detect missing values |
| `df.isnull().sum()` | Count missing values |
| `df.columns` | Column names |
| `df.duplicated()` | Detect duplicate rows |
| `df.duplicated().sum()` | Count duplicates |
| `df.select_dtypes()` | Select columns by data type |
| `df.loc[]` | Label/condition based selection |
| `df.replace()` | Replace values |
| `df.round()` | Round numerical values |
| `df.corr()` | Correlation matrix |
| `pd.get_dummies()` | One-hot encoding |
| `.astype(int)` | Convert to integers |

---

# 30. Important Seaborn/Matplotlib Commands

| Command | Purpose |
|---|---|
| `sns.histplot()` | Histogram/distribution |
| `sns.countplot()` | Count of categorical values |
| `sns.boxplot()` | Distribution + quartiles/outliers |
| `sns.violinplot()` | Distribution/density by category |
| `sns.heatmap()` | Matrix visualization |
| `plt.figure()` | Create figure |
| `plt.subplot()` | Create subplot |
| `plt.tight_layout()` | Improve plot spacing |

---

# 31. Important Concepts for Exam

## DataFrame

A Pandas DataFrame is a two-dimensional tabular data structure containing rows and columns.

---

## Missing Value

A missing value represents unavailable/empty data.

Detection:

```python
df.isnull().sum()
```

---

## Duplicate

A row that repeats an earlier row.

Detection:

```python
df.duplicated()
```

Count:

```python
df.duplicated().sum()
```

---

## Numerical Data

Data represented by numerical values.

Examples:

```text
Age
Cholesterol
RestingBP
MaxHR
```

---

## Categorical Data

Data representing categories/classes.

Examples:

```text
Sex
ChestPainType
```

---

## One-Hot Encoding

Converts categorical values into separate binary indicator columns.

```python
pd.get_dummies(df, drop_first=True)
```

---

## Standardization

Converts numerical features to approximately:

```text
mean = 0
standard deviation = 1
```

Formula:

\[
z = \frac{x-\mu}{\sigma}
\]

---

## Correlation

Measures the strength and direction of a linear relationship between numerical variables.

Range:

\[
-1 \leq r \leq 1
\]

---

# 32. Frequently Asked Viva/Exam Questions

### Q1. What does `df.shape` return?

It returns the number of rows and columns in the DataFrame.

---

### Q2. What is the default output of `df.head()`?

The first 5 rows.

---

### Q3. What is the purpose of `df.info()`?

To inspect the DataFrame structure, column names, non-null counts, and data types.

---

### Q4. How do you count missing values?

```python
df.isnull().sum()
```

---

### Q5. How do you count duplicate rows?

```python
df.duplicated().sum()
```

---

### Q6. What does `value_counts()` do?

It counts occurrences of each unique value.

Example:

```python
df['HeartDisease'].value_counts()
```

---

### Q7. What does `hue` do in Seaborn?

It divides a plot according to another categorical variable, allowing comparison between groups.

---

### Q8. What is a histogram used for?

To visualize the distribution of numerical data.

---

### Q9. What is a box plot used for?

To summarize a numerical distribution using quartiles, median, spread, and possible outliers.

---

### Q10. What is a violin plot?

A visualization showing the distribution/density of numerical values across categories.

---

### Q11. What does `annot=True` do in a heatmap?

It displays the numerical values inside the heatmap cells.

---

### Q12. Why use `numeric_only=True` in `corr()`?

To calculate correlations only for numerical columns.

---

### Q13. Why is one-hot encoding needed?

To convert categorical variables into numerical representations suitable for machine-learning workflows.

---

### Q14. Why use `drop_first=True`?

To remove one dummy category and avoid redundant dummy variables.

---

### Q15. What does `astype(int)` do?

Converts values to integer type. In this notebook it converts Boolean dummy values:

```text
True → 1
False → 0
```

---

### Q16. What is standard scaling?

A transformation that standardizes a numerical feature using:

\[
z=\frac{x-\mu}{\sigma}
\]

---

### Q17. What does `fit_transform()` do?

It learns the scaling parameters from the data and then transforms the data.

---

# 33. Code Patterns to Memorize

## Load and inspect

```python
import pandas as pd

df = pd.read_csv("heart.csv")

df.shape
df.head()
df.info()
df.describe()
```

---

## Missing values

```python
df.isnull().sum()
```

---

## Duplicate rows

```python
df.duplicated().sum()
```

---

## Numerical columns

```python
num_cols = df.select_dtypes(
    include=['number']
).columns
```

---

## Categorical columns

```python
cat_cols = df.select_dtypes(
    include=['object', 'category', 'string']
).columns
```

---

## One-hot encoding

```python
df_encode = pd.get_dummies(
    df,
    drop_first=True
)

df_encode = df_encode.astype(int)
```

---

## Standard scaling

```python
from sklearn.preprocessing import StandardScaler

numerical_col = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'MaxHR',
    'Oldpeak'
]

scaler = StandardScaler()

df_encode[numerical_col] = scaler.fit_transform(
    df_encode[numerical_col]
)
```

---

# 34. Last-Minute Revision Sheet

```text
Pandas
│
├── read_csv()          → load data
├── shape               → dimensions
├── head()              → first rows
├── info()              → structure + types
├── describe()          → statistics
├── isnull()            → missing values
├── isnull().sum()      → missing-value count
├── duplicated()        → duplicate detection
├── duplicated().sum()  → duplicate count
├── columns             → column names
├── loc[]               → conditional selection
├── replace()           → replace values
└── select_dtypes()     → select data types

Visualization
│
├── histplot()          → numerical distribution
├── countplot()         → category counts
├── boxplot()           → quartiles/outliers
├── violinplot()        → density/distribution
└── heatmap()           → matrix/correlation visualization

Preprocessing
│
├── get_dummies()       → categorical encoding
├── drop_first=True     → remove redundant dummy
├── astype(int)         → Boolean → 0/1
└── StandardScaler()    → standardization

Standardization
│
└── z = (x - μ) / σ

Correlation
│
└── -1 ≤ r ≤ +1
```

---

# 35. Most Important Things to Remember

1. **`shape`** → dimensions of dataset.
2. **`head()`** → first 5 rows.
3. **`info()`** → data types + non-null information.
4. **`describe()`** → statistical summary.
5. **`isnull().sum()`** → count missing values.
6. **`duplicated().sum()`** → count duplicate rows.
7. **`value_counts()`** → frequency of categories/values.
8. **Histogram** → numerical distribution.
9. **Count plot** → categorical frequency.
10. **Box plot** → median, quartiles, spread, outliers.
11. **Violin plot** → distribution/density.
12. **Heatmap + correlation** → visualize relationships between numerical variables.
13. **`get_dummies()`** → one-hot encoding.
14. **`drop_first=True`** → remove one category per encoded feature.
15. **`astype(int)`** → convert Boolean values to `0/1`.
16. **`StandardScaler`** → standardize numerical features.
17. **Standardization formula:**  
   \[
   z = \frac{x-\mu}{\sigma}
   \]
18. Always distinguish **missing values** from **invalid values** such as an impossible `0`.
19. Always verify column names because Pandas is **case-sensitive**.
20. Check preprocessing code carefully for accidental variable reuse, such as the `RestingBP` replacement line in the source notebook.

---

## One-Line Memory Trick

> **Inspect → Clean → Visualize → Encode → Scale → Prepare for ML**
