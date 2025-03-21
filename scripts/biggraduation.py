import csv
import random

def main():

    # Function to generate GPA based on class of degree
    def generate_gpa(class_of_degree):
        if (class_of_degree == "SU"):  # First Class
            return round(random.uniform(3.60, 4.30), 2)
        elif (class_of_degree == "SL"):  # Upper Class
            return round(random.uniform(3.00, 3.59), 2)
        elif (class_of_degree == "P"):  # Lower Class
            return round(random.uniform(2.50, 2.99), 2)
        elif (class_of_degree == "F"):  # Pass
            return round(random.uniform(2.00, 2.49), 2)
        else:
            return 0.0  # Default case

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

    # Program-specific class of degree distributions
    program_class_distributions = {
        "BSC Computer Science (Major)": {
            "SU": 17.5,  # First Class: ~15%-20%
            "SL": 37.5,  # Second Upper Class: ~35%-40%
            "P": 27.5,   # Second Lower Class: ~25%-30%
            "F": 12.5    # Pass: ~10%-15%
        },
        "BSC Computer Science (Spec)": {
            "SU": 12.5,  # First Class: ~10%-15%
            "SL": 32.5,  # Second Upper Class: ~30%-35%
            "P": 32.5,   # Second Lower Class: ~30%-35%
            "F": 17.5    # Pass: ~15%-20%
        },
        "BSC Computer Science and Management": {
            "SU": 17.5,  # First Class: ~15%-20%
            "SL": 42.5,  # Second Upper Class: ~40%-45%
            "P": 27.5,   # Second Lower Class: ~25%-30%
            "F": 7.5     # Pass: ~5%-10%
        },
        "BSC Information Technology (Major)": {
            "SU": 12.5,  # First Class: ~10%-15%
            "SL": 35.0,  # Second Upper Class: ~30%-40%
            "P": 32.5,   # Second Lower Class: ~30%-35%
            "F": 12.5    # Pass: ~10%-15%
        },
        "BSC Information Technology (Spec)": {
            "SU": 10.0,  # First Class: ~8%-12%
            "SL": 27.5,  # Second Upper Class: ~25%-30%
            "P": 32.5,   # Second Lower Class: ~30%-35%
            "F": 22.5    # Pass: ~20%-25%
        }
    }

    # Function to select class of degree based on program major
    def select_class_of_degree(program_major):
        distribution = program_class_distributions[program_major]
        classes = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(classes, weights=weights, k=1)[0]

    # gender skew
    gender_balance = random.uniform(0.55, 0.65)  # Adjust female percentage between 55% and 65%
    num_female = int(num_records * gender_balance)
    num_male = num_records - num_female
    genders = ["F"] * num_female + ["M"] * num_male
    random.shuffle(genders) # shuffle so that the first records are not all female

    with open("grads_gpa.csv", mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write header
        writer.writerow([
            "Term_Code_Grad", 
            "Term_Code_Admit", 
            "Gender",
            "Programme_Major", 
            "Faculty_Code", 
            "Class_Of_Degree_Code",
            "GPA"
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

            # Pick random value for program major
            programme_major = random.choice(programme_majors)
            
            # Select class of degree based on program major
            class_of_degree = select_class_of_degree(programme_major)
            
            # Generate GPA based on class of degree
            gpa = generate_gpa(class_of_degree)

            # Write row to CSV
            writer.writerow([
                term_code_grad,
                term_code_admit,
                genders.pop(),  # Use pop() to get a gender from the list
                programme_major,
                faculty_code,
                class_of_degree,
                gpa
            ])

if __name__ == "__main__":
    main()