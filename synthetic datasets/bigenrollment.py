import random
from datetime import datetime, timedelta
import csv

def generate_random_date(start_year, end_year):
    start_date = datetime(year=start_year, month=1, day=1)
    end_date = datetime(year=end_year, month=12, day=31)
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date.strftime("%d/%m/%Y")

def generate_student_data(num_students):
    student_data = []
    student_ids = random.sample(range(318000000, 318999999), num_students)

    # Predefined data
    admit_terms = []
    for _ in range(num_students):
        year = random.randint(2000, 2004)
        semester = random.choices(['10', '20', '30'], weights=[8, 1, 1])[0]  # Bias towards Semester 1
        admit_terms.append(int(f"{year}1{semester}"))

    cities = ["Port of Spain", "San Fernando", "Scarborough", "Bridgetown", "Kingston", "Nassau", "Havana"]
    states = ["Trinidad", "Tobago", "Barbados", "Jamaica", "Bahamas", "Cuba"]
    nations = ["Trinidad and Tobago", "Barbados", "Jamaica", "Bahamas", "Cuba", "Guyana", "Suriname"]
    genders = ["M", "F"]
    religions = ["Christian", "Seventh-Day Adventist", "Spiritual Baptist", "Pentecostal", "Roman Catholic", 
                 "Muslim", "Hindu", "Rastafarian", "Buddhist"]
    marital_statuses = ["Single", "Married", "Common-Law", "Separated", "Divorced"]
    degree_codes = ["CS-MAJO", "CS-SPEC", "CS-MANA", "CS-MASP", "IT-MAJO", "IT-SPEC"]

    term_codes = []
    for year in range(2000, 2005):
        term_codes.extend([f"{year}10", f"{year}20", f"{year}30"]) #keep

    for i in range(num_students):
        student_id = student_ids[i]
        admit_term = admit_terms[i]
        birth_date = generate_random_date(1975, 1990)
        city = random.choice(cities)
        state = random.choice(states)
        nation = random.choice(nations)
        gender = random.choice(genders)
        religion = random.choice(religions)
        marital_status = random.choice(marital_statuses)
        faculty_code = "FST"
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
                "NATION": nation,
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
        "CITY", "STATE", "NATION", "GENDER", "RELIGION",
        "MARITAL_STATUS", "FACULTY_CODE", "STU_DEGREE_CODE"
    ]

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(student_data)

# Generate data for 100 students
students = generate_student_data(100)

# Save the generated data to a CSV file
save_to_csv(students, 'university_registration_data.csv')

print("Data has been generated and saved to university_registration_data.csv")