"""
- Generates synthetic student IDs and admission records
- Uses real world city/country data for addresses
- Simulates realistic enrollment patterns across semesters
- Maintains demographic distributions (e.g. 60% female students)
- Outputs data in CSV format suitable for database import
"""

import random
from datetime import datetime
import csv
import pandas as pd
import faker

def generate_random_date(start_year, end_year):
    """
    Generate a random date between two years.
    
    Args:
        start_year (int): The earliest possible year
        end_year (int): The latest possible year
    
    Returns:
        str: A date string in DD/MM/YYYY format
    """
    fake = faker.Faker()
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    random_date = fake.date_between(start_date=start_date, end_date=end_date)
    return random_date.strftime("%d/%m/%Y")

def generate_student_data(num_students):
    """
    Generate synthetic student registration records.
    
    Args:
        num_students (int): Number of unique students to generate
    
    Returns:
        list: List of dictionaries containing student records with multiple semesters per student
    
    Note:
        - Student IDs are unique 9-digit numbers starting with 318
        - 92% of students are from Trinidad and Tobago (TTO)
        - Gender distribution is skewed 60% female
        - Most students complete in 6 semesters (some take 7-8)
    """
    student_data = []
    student_ids = random.sample(range(318000000, 318999999), num_students)

    # Predefined data
    admit_terms = []
    for _ in range(num_students):
        year = random.randint(2000, 2004)
        semester = random.choices(['10', '20', '30'], weights=[8, 1, 1])[0]  # Bias towards Semester 1
        admit_terms.append(int(f"{year}1{semester}"))
    
    # Load the world cities data
    worldcities_df = pd.read_csv(r"synthetic datasets/worldcities.csv")
    
    # Extract unique cities, states, and nations (using iso3)
    cities = worldcities_df["city"].unique().tolist()
    states = worldcities_df["admin_name"].unique().tolist()
    nations = worldcities_df["iso3"].unique().tolist()
    countries = worldcities_df["country"].unique().tolist()  # new extraction of country names
    
    #TODO review skew code for contributing countries
    # Ensure at least 92% of the students are from TTO
    weighted_nations = ["TTO"] * 92 + nations
    tto_ratio = 92
    total_remaining = 100 - tto_ratio

    # Create dictionary of country ratios from 2022 data (excluding TTO)
    country_stats = pd.read_csv("synthetic datasets/country_stats.csv")
    country_stats = country_stats[country_stats['country_iso3'] != 'TTO']
    country_stats = country_stats[country_stats['country_iso3'] != 'UG_TOTAL']

    # Calculate proportions based on 2022 values
    total_2022 = country_stats['2022'].sum()
    country_weights = []

    for country in nations:
        if country == 'TTO':
            continue
        matches = country_stats[country_stats['country_iso3'] == country]
        if not matches.empty and not pd.isna(matches['2022'].iloc[0]):
            weight = int((matches['2022'].iloc[0] / total_2022) * total_remaining)
            country_weights.extend([country] * max(weight, 1))

    weighted_nations = ["TTO"] * tto_ratio + country_weights

    genders = ["M", "F"]
    religions = ["N/A", "Other", "None", "Christian", "Seventh-Day Adventist", "Spiritual Baptist", "Pentecostal", "Roman Catholic", 
                 "Muslim", "Hindu", "Rastafarian", "Buddhist", "Wesleyan", "Jehovah's Witness", "Ethiopian Orthodox",
                 "Islam", "Presbyterian", "Methodist", "Anglican", "Brethren", "Baha'i", 
                 "Church of Christ", "Baptist", "Evangelical Church", "Nazarene", "Moravian", "Church of God"]
    
    		
    
    marital_statuses = ["Single", "Married", "Common-Law", "Separated", "Divorced"]
    degree_codes = ["CS-MAJO", "CS-SPEC", "CS-MANA", "CS-MASP", "IT-MAJO", "IT-SPEC"]

    term_codes = []
    for year in range(2000, 2005):
        term_codes.extend([f"{year}10", f"{year}20", f"{year}30"]) #TODO this is hardcoded for 3 semesters across 5 years

    for i in range(num_students):
        student_id = student_ids[i]
        admit_term = admit_terms[i]
        birth_date = generate_random_date(1975, 1990)

        # Choose a nation first
        nation = random.choice(weighted_nations)

        # Filter worldcities_df for the chosen nation so cities are accurate
        nation_cities = worldcities_df[worldcities_df["iso3"] == nation]
        if not nation_cities.empty:
            chosen_city = nation_cities.sample(1).iloc[0]
            city = chosen_city["city"]
            state = chosen_city["admin_name"] 
            country = chosen_city["country"]  # new: get country name from record
        else:
            # Fallback if no cities are found for nation
            city = random.choice(cities)
            state = random.choice(states)
            country = random.choice(countries)  # new fallback for country

        gender = random.choices(genders, weights=[4, 6])[0]  # Skewed to 60% female
        religion = random.choice(religions)
        marital_status = random.choice(marital_statuses)
        faculty_code = "FST"  #TODO note we are focusing on FST students alone for now at least
        degree_code = random.choice(degree_codes)

        # Generate multiple records for each student
        num_semesters = random.choices([6, 7, 8], weights=[5, 3, 2])[0]  # Bias towards 6 semesters
        start_year = int(str(admit_term)[:4])

        for j in range(num_semesters):
            term_code_eff = int(term_codes[(start_year - 2000) * 3 + j % 3])
            student_record = {
                "STUDENT_ID": student_id,
                "TERM_CODE_EFF": term_code_eff,
                "TERM_CODE_ADMIT": admit_term,
                "DATE_OF_BIRTH": birth_date,
                "CITY": city,
                "STATE": state,
                "NATION_ISO3": nation,
                "NATION_NAME": country,
                "GENDER": gender,
                "RELIGION": religion,
                "MARITAL_STATUS": marital_status,
                "FACULTY_CODE": faculty_code,
                "STU_DEGREE_CODE": degree_code
            }
            student_data.append(student_record)

    return student_data

def save_to_csv(student_data, filename):
    fieldnames = [
        "STUDENT_ID", "TERM_CODE_EFF", "TERM_CODE_ADMIT", "DATE_OF_BIRTH",
        "CITY", "STATE", "NATION_ISO3", "NATION_NAME",
        "GENDER", "RELIGION", "MARITAL_STATUS", "FACULTY_CODE", "STU_DEGREE_CODE"
    ]

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(student_data)

# Configuration constants
STUDENT_ID_PREFIX = 318  # Prefix for all student IDs
TRINIDAD_BIAS = 92      # Percentage of students from Trinidad
FEMALE_RATIO = 0.6      # Ratio of female students

# Generate data for 500 students
students = generate_student_data(500)

# Save the generated data to a CSV file
save_to_csv(students, 'university_registration_data.csv')

print("Data has been generated and saved to university_registration_data.csv")