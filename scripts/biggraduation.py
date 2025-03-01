import csv
import random

def main():
    """
    This script generates a CSV file named 'university_graduation.csv' with fake data
    for university graduation records between 2000 and 2024.
    Fields:
    - Term_Code_Grad: format YYYY(Semester)0, for example 201610, 201620, 201630.
    - Term_Code_Admit: format YYYY(Semester)0, for example 201610, 201620, 201630, 
      ensuring all students are admitted in Semester 1.
    - Gender: 'M', 'F'.
    - Programme_Major: One of 
        ["BSC Computer Science (Major)", "BSC Computer Science (Spec)", 
         "BSC Computer Science and Management", "BSC Information Technology (Major)", 
         "BSC Information Technology (Spec)"].
    - Faculty_Code: 'SA' for grad years <= 2012, otherwise 'ST'.
    - Class_Of_Degree_Code: 'SU', 'SL', 'P', 'F'.
    """

    random.seed(42)  # For reproducible results
    num_records = 8500  # Adjust as needed

    # Define possible programme majors
    programme_majors = [
        "BSC Computer Science (Major)",
        "BSC Computer Science (Spec)",
        "BSC Computer Science and Management",
        "BSC Information Technology (Major)",
        "BSC Information Technology (Spec)"
    ]

    # Possible classes of degree
    classes_of_degree = ["SU"] * 44 + ["SL"] * 25 + ["P"] * 10 + ["F"] * 21

    # gender skew
    gender_balance = random.uniform(0.55, 0.65)  # Adjust female percentage between 55% and 65%
    num_female = int(num_records * gender_balance)
    num_male = num_records - num_female
    genders = ["F"] * num_female + ["M"] * num_male
    random.shuffle(genders) # shuffle so that the first records are not all female

    with open("25grad.csv", mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow([
            "Term_Code_Grad", 
            "Term_Code_Admit", 
            "Gender",
            "Programme_Major", 
            "Faculty_Code", 
            "Class_Of_Degree_Code"
        ])

        for _ in range(num_records):
            # Randomly generate an admission year within 2000–2024 
            admit_year = random.randint(2000, 2024)
            # All students admitted in Semester 1
            term_code_admit = f"{admit_year}10"

            # Graduation typically happens 2 to 5 years after Y, but capped at 2024
            graduation_delay = random.randint(2, 5)
            grad_year = admit_year + graduation_delay
            if grad_year > 2024:
                grad_year = 2024

            # Pick a random semester (1, 2, or 3) for graduation
            grad_sem = random.choice([1, 2, 3])
            term_code_grad = f"{grad_year}{grad_sem}0"

            # Determine faculty code based on graduation year
            faculty_code = "SA" if grad_year <= 2012 else "ST"

            # Pick random values for other fields
            gender = random.choice(genders)
            programme_major = random.choice(programme_majors)
            class_of_degree = random.choice(classes_of_degree)

            # Write row to CSV
            writer.writerow([
                term_code_grad,
                term_code_admit,
                gender,
                programme_major,
                faculty_code,
                class_of_degree
            ])

if __name__ == "__main__":
    main()