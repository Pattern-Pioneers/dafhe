import pandas as pd
import datetime

# Read the CSV file
df = pd.read_csv('25years enrollment sample.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Convert date columns to datetime
df['DATE_OF_BIRTH'] = pd.to_datetime(df['DATE_OF_BIRTH'])

# Extract year and semester from TERM_CODE_ADMIT
df['ADMIT_YEAR'] = df['TERM_CODE_ADMIT'].astype(str).str[:4].astype(int)
df['ADMIT_SEMESTER'] = df['TERM_CODE_ADMIT'].astype(str).str[4:].astype(int)

# Calculate enrollment date (approximate to middle of semester)
semester_month = {10: 1, 20: 5, 30: 9}  # Map semester codes to months
df['ENROLL_DATE'] = df.apply(lambda x: pd.Timestamp(year=x['ADMIT_YEAR'], 
                                                   month=semester_month[x['ADMIT_SEMESTER']], 
                                                   day=1), axis=1)

# Calculate age at enrollment
df['AGE_AT_ENROLLMENT'] = (df['ENROLL_DATE'] - df['DATE_OF_BIRTH']).dt.total_seconds() / (365.25 * 24 * 60 * 60)

# Group by degree code and calculate statistics
enrollment_stats = df.groupby('STU_DEGREE_CODE').agg({
    'STUDENT_ID': 'nunique',  # Count unique students
    'AGE_AT_ENROLLMENT': 'mean',
}).reset_index()

# Calculate gender counts for unique students
gender_counts = df.drop_duplicates('STUDENT_ID').groupby('STU_DEGREE_CODE')['GENDER'].value_counts().unstack(fill_value=0)

# Merge all statistics
result = pd.merge(enrollment_stats, gender_counts, on='STU_DEGREE_CODE')
result.columns = ['Degree_Code', 'Total_Students', 'Avg_Age', 'Female', 'Male']
result['Avg_Age'] = result['Avg_Age'].round(1)

# Reorder columns
result = result[['Degree_Code', 'Male', 'Female', 'Total_Students', 'Avg_Age']]

# Sort by total students in descending order
result = result.sort_values('Total_Students', ascending=False)

# Add percentage of total
total_students = result['Total_Students'].sum()
result['Percentage'] = (result['Total_Students'] / total_students * 100).round(1)

# Display results
print("\nEnrollment Distribution by Degree Code:")
print("==============================================================")
print(result.to_string(index=False))
print(f"\nTotal Unique Students: {total_students}")