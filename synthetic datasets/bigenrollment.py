import pandas as pd
import random
from faker import Faker
import calendar

# Initialize Faker and seed for reproducibility
fake = Faker()
Faker.seed(-1)
random.seed(-1)

# Define the degree programs, cities, and qualifications
degree_programs = [
    "CS-MAJ", # Computer Science Major
    "CS-SPE", # Computer Science (Special)
    "CS-MAN", # Computer Science with Management
    "CS-MSP", # Computer Science with Management (Special)
    "IT-MAJ", # Information Technology Major
    "IT-SPE", # Information Technology (Special)
    "IT-MAN", # Information Technology with Management
]

cities = [
    "Port of Spain",
    "San Fernando",
    "Arima",
    "Chaguanas",
    "Couva",
    "Point Fortin",
    "Sangre Grande",
    "Tunapuna",
    "Siparia",
    "Penal",
]

# Gender distribution based on real-world trends
gender_distribution = ["Male"] * 40 + ["Female"] * 60

# Remove year_enrollment_data declaration since we're now using degree_enrollment.csv
# New: Load degree enrollment statistics from CSV to ground the dataset generation realistically
csv_path = "synthetic datasets\degree_enrollment.csv"
df_degrees = pd.read_csv(csv_path)
csv_to_program = {
    "CS-MAJO": "CS-MAJO",
    "CS-SPEC": "CS-SPEC",
    "CS-MANA": "CS-MANA",
    "CS-MASP": "CS-MASP",
    "IT-MAJO": "IT-MAJO",
    "IT-SPEC": "IT-SPEC",
}

degree_yearly_enrollment = {}

for _, row in df_degrees.iterrows():
    year_val = int(row["year"])
    degree_yearly_enrollment[year_val] = {}
    for csv_col, prog in csv_to_program.items():
        val = row[csv_col]
        count = 0 if (val == '-' or pd.isna(val)) else int(val)
        degree_yearly_enrollment[year_val][prog] = count
    # For missing degree code in CSV, assign 0 (e.g. IT-MAN)
    degree_yearly_enrollment[year_val]["IT-MAN"] = 0

# List of enrollment years
years = list(range(2000, 2024))

records = []

def generate_student_id():
    return int("308" + ''.join(str(random.randint(0, 9)) for _ in range(6)))
    
student_id_set = set()  # Keep track of used IDs to avoid duplicates

for year in years:
    if year in degree_yearly_enrollment:
        # For years with degree-specific counts, generate records per degree
        for degree_code, count in degree_yearly_enrollment[year].items():
            for _ in range(count):
                # Generate unique student ID
                while True:
                    student_id = generate_student_id()
                    if student_id not in student_id_set:
                        student_id_set.add(student_id)
                        break

                # Generate date of birth ensuring age is between 17 and 22 at enrollment
                age_at_enrollment = random.randint(17, 24)
                dob_year = year - age_at_enrollment
                dob_month = random.randint(1, 12)
                dob_date = fake.date_of_birth(minimum_age=17, maximum_age=22)
                valid_day = min(dob_date.day, calendar.monthrange(dob_year, dob_month)[1])
                dob_formatted = dob_date.replace(year=dob_year, month=dob_month, day=valid_day).strftime("%m-%Y")
                
                gender = random.choice(gender_distribution)
                
                address = random.choice(cities)
                
                # Use the specific degree_code from the dictionary
                
                # Nationality distribution remains the same
                nationalities = (
                    ["Trinidad and Tobago"] * 923
                    + [
                        "Anguilla",
                        "Antigua and Barbuda",
                        "Barbados",
                        "Belize",
                        "British Virgin Islands",
                        "Dominica",
                        "Grenada",
                        "Guyana",
                        "Jamaica",
                        "Monsterrat",
                        "St. Kitts and Nevis",
                        "St. Lucia",
                        "St. Vincent and the Grenadines",
                        "The Bahamas",
                        "Turks and Caicos Islands",
                    ] * 8
                    + [
                        "United States",
                        "Canada",
                        "Nigeria",
                        "India",
                    ] * 2
                )
                nationality = random.choice(nationalities)
                
                record = (
                    student_id,
                    dob_formatted,
                    gender,
                    nationality,
                    address,
                    degree_code,
                    year,
                )
                records.append(record)
    continue

# Define the column names for the CSV
columns = [
    "STUDENT_ID",
    "DATE_OF_BIRTH",
    "GENDER",
    "NATIONALITY",
    "ADDRESS",
    "STU_DEGREE_CODE",
    "TERM_CODE_ADMIT",
]

# Create a DataFrame and write it to CSV
df = pd.DataFrame(records, columns=columns)
df.to_csv("bigen_fancy.csv", index=False)

print("bigen_fancy.csv successfully generated!")