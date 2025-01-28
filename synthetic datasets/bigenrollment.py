import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Seed for reproducibility
Faker.seed(0)
random.seed(0)

# Define the degree programs, cities, and qualifications
degree_programs = [
    "BSc. Computer Science",
    "BSc. Computer Science (Special)",
    "BSc. Computer Science with Management",
    "BSc. Information Technology",
    "BSc. Information Technology (Special)"
]

cities = [
    "Port of Spain", "San Fernando", "Arima", "Chaguanas", "Couva",
    "Point Fortin", "Sangre Grande", "Tunapuna", "Siparia", "Penal"
]

qualifications = ["CAPE", "CSEC", "Other"]

# Gender distribution based on real-world trends
gender_distribution = ["Male"] * 60 + ["Female"] * 40

# Generate synthetic data
data = set()
years = list(range(2000, 2025))

# Realistic number of records for each year
for year in years:
    if year <= 2012:
        num_records = random.randint(100, 175)
    else:
        num_records = random.randint(175, 410)
    for _ in range(num_records):
        # Ensure realistic age at the time of enrollment
        dob_year = year - random.randint(17, 22)
        dob_month = random.randint(1, 12)
        # Generate a valid date of birth
        dob = fake.date_of_birth(minimum_age=17, maximum_age=22).replace(year=dob_year, month=dob_month).strftime("%m-%Y")
        gender = random.choice(gender_distribution)
        nationality = "Trinidad and Tobago"
        address = random.choice(cities)
        qualification = random.choice(qualifications)
        degree_enrolled = random.choice(degree_programs)
        
        record = (
            dob,
            gender,
            nationality,
            address,
            qualification,
            degree_enrolled,
            year
        )
        
        data.add(record)

# Convert to DataFrame and save to CSV
columns = [
    "Date of Birth",
    "Gender",
    "Nationality",
    "Address",
    "Qualification",
    "Degree Enrolled in",
    "Year Enrolled"
]
df = pd.DataFrame(data, columns=columns)
df.to_csv("bigenrollment_data.csv", index=False)

print("bigenrollment_data.csv successfully generated!")