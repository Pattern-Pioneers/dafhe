import pandas as pd
import random

# Define the years, degree programs, and other categories
years = [2000, 2001, 2002, 2003, 2004]
degree_programs = [
    "BSc. Computer Science",
    "BSc. Computer Science (Special)",
    "BSc. Computer Science with Management",
    "BSc. Information Technology",
    "BSc. Information Technology (Special)"
]

# Initialize lists to store the generated data
data = []

# Generate synthetic graduation data
for year in years:
    total_enrolled = random.randint(150, 250)
    
    # Generate program-specific enrollment such that they add up to total_enrolled
    remaining_students = total_enrolled
    program_enrollment = {}
    
    for i, program in enumerate(degree_programs):
        if i == len(degree_programs) - 1:
            # Assign remaining students to the last program to ensure the total adds up
            program_enrollment[program] = remaining_students
        else:
            # Randomly assign students to each program
            enrolled = random.randint(10, remaining_students - 10 * (len(degree_programs) - i - 1))
            program_enrollment[program] = enrolled
            remaining_students -= enrolled

    total_dropped_out = random.randint(10, 30)
    total_transferred = random.randint(5, 20)
    full_time = random.randint(80, 150)
    part_time = total_enrolled - full_time
    sponsored = random.randint(50, 100)
    self_funded = total_enrolled - sponsored
    
    record = {
        "Year": year,
        "Total Enrolled": total_enrolled,
        "Dropped Out": total_dropped_out,
        "Transferred": total_transferred,
        "Full-time": full_time,
        "Part-time": part_time,
        "Sponsored": sponsored,
        "Self-funded": self_funded
    }
    
    # Add program-specific enrollment to the record
    record.update(program_enrollment)
    
    data.append(record)

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("graduation_datav1.csv", index=False)

print("graduation_datav1.csv successfully generated!")