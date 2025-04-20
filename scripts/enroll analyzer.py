import pandas as pd
from tqdm import tqdm
import os

"""
Enrollment Analyzer:
Reads enrollment CSV, computes age and gender stats per degree code,
and displays summary tables.
"""

# Read the CSV file
print("Reading CSV file...")
file_path = os.path.join(os.path.dirname(__file__), 'demoenrollFASTER.csv')
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Convert date columns to datetime
print("Converting dates...")
df['DATE_OF_BIRTH'] = pd.to_datetime(df['DATE_OF_BIRTH'])

# Extract year and semester from TERM_CODE_ADMIT
df['ADMIT_YEAR'] = df['TERM_CODE_ADMIT'].astype(str).str[:4].astype(int)
df['ADMIT_SEMESTER'] = df['TERM_CODE_ADMIT'].astype(str).str[4:].astype(int)

# Calculate enrollment dates
print("Calculating enrollment dates...")
semester_month = {10: 9, 20: 1, 30: 5}
df['ENROLL_MONTH'] = df['ADMIT_SEMESTER'].map(semester_month)
df['ENROLL_DATE'] = pd.to_datetime(dict(
    year=df['ADMIT_YEAR'],
    month=df['ENROLL_MONTH'],
    day=1
))
df.drop(columns=['ENROLL_MONTH'], inplace=True)

# Calculate age at enrollment
print("Calculating ages...")
df['AGE_AT_ENROLLMENT'] = (df['ENROLL_DATE'] - df['DATE_OF_BIRTH']).dt.total_seconds() / (365.25 * 24 * 60 * 60)

"""
Compute detailed per-degree statistics with progress bar
"""
degree_codes = df['STU_DEGREE_CODE'].unique()
total_students_all = df['STUDENT_ID'].nunique()
stats = []
for code in tqdm(degree_codes, desc='Aggregating stats per degree'):
    sub = df[df['STU_DEGREE_CODE'] == code]
    total_students = sub['STUDENT_ID'].nunique()
    ages = sub['AGE_AT_ENROLLMENT']
    # basic stats
    avg_age = round(ages.mean(), 1)
    median_age = round(ages.median(), 1)
    std_age = round(ages.std(), 1)
    # additional stats
    min_age = round(ages.min(), 1)
    max_age = round(ages.max(), 1)
    q1 = round(ages.quantile(0.25), 1)
    q3 = round(ages.quantile(0.75), 1)
    # gender counts
    gender_counts = sub.drop_duplicates('STUDENT_ID')['GENDER'].value_counts()
    female = int(gender_counts.get('F', 0))
    male = int(gender_counts.get('M', 0))
    # percent of total
    pct_total = round(total_students / total_students_all * 100, 1)
    stats.append({
        'Degree_Code': code,
        'Total_Students': total_students,
        'Pct_of_Total': pct_total,
        'Male': male,
        'Female': female,
        'Avg_Age': avg_age,
        'Median_Age': median_age,
        'Std_Age': std_age,
        'Min_Age': min_age,
        'Max_Age': max_age,
        'Q1_Age': q1,
        'Q3_Age': q3,
    })
result = pd.DataFrame(stats).sort_values('Total_Students', ascending=False)

# Display results
print("\nDetailed Enrollment Stats by Degree Code:")
print(result.to_string(index=False))
print(f"\nTotal Unique Students: {total_students_all}")

# Additional breakdown by admission year
year_counts = df['ADMIT_YEAR'].value_counts().sort_index()
print("\nEnrollment Counts by Admit Year:")
print(year_counts.to_string())