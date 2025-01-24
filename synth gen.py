import pandas as pd
import numpy as np

# Example approximate distributions
gender_dist = ["Male", "Female"]
gender_probs = [0.55, 0.45]  # real-world approximation

program_list = [
    "BSc. Computer Science",
    "BSc. Computer Science (Special)",
    "BSc. Computer Science with Management",
    "BSc. Information Technology",
    "BSc. Information Technology (Special)",
]
# Example probabilities (must sum to 1)
program_probs = [0.30, 0.25, 0.10, 0.25, 0.10]  # Adjusted to six probabilities

cities = ["Kingston", "Point Fortin", "Port of Spain", "San Fernando", "Bridgetown", "Nassau"]
city_probs = [0.15, 0.15, 0.30, 0.20, 0.10, 0.10]

nationalities = ["Jamaica", "Trinidad", "Bahamas", "Barbados", "Guyana"]
nationality_probs = [0.10, 0.50, 0.15, 0.15, 0.10]

num_students = 200
mean_age = 21
std_age = 2.5
start_year = 2000
end_year = 2004

# Generate features with probabilities
gender = np.random.choice(gender_dist, size=num_students, p=gender_probs)
program = np.random.choice(program_list, size=num_students, p=program_probs)
city = np.random.choice(cities, size=num_students, p=city_probs)
nationality = np.random.choice(nationalities, size=num_students, p=nationality_probs)

# Skew the age slightly (e.g., more younger students)
age = np.random.normal(loc=mean_age, scale=std_age, size=num_students).astype(int)
age = np.clip(age, 17, 40)

enrollment_year = np.random.randint(start_year, end_year + 1, size=num_students)

df = pd.DataFrame({
    "Gender": gender,
    "Age": age,
    "EnrollmentYear": enrollment_year,
    "Program": program,
    "City": city,
    "Nationality": nationality  # Included nationality
})

# Save to CSV
df.to_csv("synth_data.csv", index=False)
