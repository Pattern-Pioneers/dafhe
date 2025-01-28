import pandas as pd
import random
from faker import Faker

fake = Faker()

# Seed for reproducibility
Faker.seed(0)

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
gender_distribution = ["Male"] * 55 + ["Female"] * 45

# Generate synthetic data
data = []
for _ in range(100):
    dob = fake.date_of_birth(minimum_age=42, maximum_age=47).strftime("%m-%Y")
    gender = random.choice(gender_distribution)
    nationality = "Trinidad and Tobago"
    address = random.choice(cities)
    qualification = random.choice(qualifications)
    degree_enrolled = random.choice(degree_programs)
    year_enrolled = random.choice([2000, 2001, 2002, 2003, 2004])
    
    record = {
        "Date of Birth": dob,
        "Gender": gender,
        "Nationality": nationality,
        "Address": address,
        "Qualification": qualification,
        "Degree Enrolled in": degree_enrolled,
        "Year Enrolled": year_enrolled
    }
    
    data.append(record)

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)
df.to_csv("enrollment_data.csv", index=False)

print("enrollment_data.csv successfully generated!")