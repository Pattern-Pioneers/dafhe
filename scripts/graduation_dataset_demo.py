#!/usr/bin/env python3
"""
Demo script to generate synthetic graduation records,
using OpenRouter LLM for realistic program, GPA, and gender distributions.
"""
import os
import csv
import random
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODEL_LIST = [
        "google/learnlm-1.5-pro-experimental:free",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]

# Load environment
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={"X-Title": "DAFHE Data Generation"},
)

def clean_json_response(text: str) -> str:
    """Strip Markdown fences and extract JSON content from a response."""
    if text.startswith("```"):
        start = text.find("{")
        end = text.rfind("}")
        text = text[start:end+1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()

def create_gpa_prompt(program: str) -> str:
    """Build prompt for GPA distribution for a given program."""
    return f"""Generate realistic degree classification distribution for {program} at a university in Trinidad and Tobago.

Provide percentages for each class of degree:
- F (First Class): 3.60-4.30
- SU (Second Upper Class): 3.00-3.59
- SL (Second Lower Class): 2.50-2.99
- P (Pass): 2.00-2.49

The percentages must sum EXACTLY to 100%. Respond with only JSON."""

def create_gpa_schema() -> dict:
    """Return JSON schema for validating GPA distribution responses."""
    return {
        "type": "object",
        "properties": {"F": {"type": "number"}, "SU": {"type": "number"}, "SL": {"type": "number"}, "P": {"type": "number"}},
        "required": ["F", "SU", "SL", "P"],
        "additionalProperties": False,
    }

def query_llm_gpa(program: str) -> dict:
    """Request and normalize GPA distribution for a program from the LLM."""
    models = MODEL_LIST
    prompt = create_gpa_prompt(program)
    schema = create_gpa_schema()
    attempts = {m: 0 for m in models}
    pool = random.sample(models, len(models))
    while pool:
        m = pool.pop(0)
        attempts[m] += 1
        try:
            logger.info(f"Requesting GPA distribution for {program} using {m}")
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role":"user","content":prompt}],
                response_format={"type":"json_schema","json_schema":{"name":"gpa_distribution","strict":True,"schema":schema}},
                temperature=0.8,
                stream=False,
            )
            raw = resp.choices[0].message.content.strip()
            cleaned = clean_json_response(raw)
            data = json.loads(cleaned)
            dist = data.get("gpa_distribution", data)
            total = sum(dist.values())
            if total <= 0:
                raise ValueError("Empty distribution")
            return {k: float(v)/total*100 for k,v in dist.items()}
        except Exception as e:
            logger.warning(f"GPA error ({m}): {e}")
            if attempts[m] < 2:
                pool.append(m)
            time.sleep(1)
    raise RuntimeError(f"Failed GPA for {program}")

def create_gender_by_program_prompt(program: str) -> str:
    """Build prompt for gender distribution for a given program."""
    return f"""Generate realistic gender distribution for {program} at a university in Trinidad and Tobago.
Provide percentages for F and M that sum to 100. Respond with only JSON."""

def create_gender_by_program_schema() -> dict:
    """Return JSON schema for validating gender distribution responses."""
    return {"type":"object","properties":{"F":{"type":"number"},"M":{"type":"number"}},"required":["F","M"],"additionalProperties":False}

def normalize(dist: dict) -> dict:
    """Normalize distribution values in case of small inacurracies."""
    vals = [float(v) for v in dist.values()]
    total = sum(vals)
    return {k: float(v)/total*100 for k,v in dist.items()} if total>0 else dist

def query_llm_gender_by_program(program: str) -> dict:
    """Request and normalize gender distribution for a program from the LLM."""
    models = MODEL_LIST
    prompt = create_gender_by_program_prompt(program)
    schema = create_gender_by_program_schema()
    attempts = {m:0 for m in models}
    pool = random.sample(models,len(models))
    while pool:
        m = pool.pop(0)
        attempts[m] += 1
        try:
            logger.info(f"Requesting gender for {program} using {m}")
            resp = client.chat.completions.create(
                model=m,
                messages=[{"role":"user","content":prompt}],
                response_format={"type":"json_schema","json_schema":{"name":"gender_by_program","strict":True,"schema":schema}},
                temperature=0.7,
                stream=False,
            )
            raw = resp.choices[0].message.content.strip()
            cleaned = clean_json_response(raw)
            data = json.loads(cleaned)
            raw_dist = data.get("gender_by_program", data)
            # Standardize gender labels to 'F'/'M'
            mapped = {}
            for label, val in raw_dist.items():
                key = label.strip().upper() if isinstance(label, str) else None
                if key and key.startswith('F'):
                    mapped['F'] = mapped.get('F', 0) + float(val)
                elif key and key.startswith('M'):
                    mapped['M'] = mapped.get('M', 0) + float(val)
            return normalize(mapped)
        except Exception as e:
            logger.warning(f"Gender error ({m}): {e}")
            if attempts[m] < 2:
                pool.append(m)
            time.sleep(1)
    raise RuntimeError(f"Failed gender for {program}")

def create_program_distribution_prompt() -> str:
    """Build prompt for program-major distribution across students."""
    return "Generate realistic distribution of students across CS/IT programs. Respond with only JSON."

def get_program_distribution(programmes: list) -> dict:
    """Allocate a total number of students among programs based on distribution percentages."""
    prompt = create_program_distribution_prompt()
    logger.info("Requesting program distribution")
    # Build JSON schema for LLM call
    schema = {
        "type": "object",
        "properties": {p: {"type": "number"} for p in programmes},
        "required": programmes,
        "additionalProperties": False,
    }
    resp = client.chat.completions.create(
        model="google/learnlm-1.5-pro-experimental:free",
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_schema","json_schema":{"name":"program_distribution","strict":True,"schema":schema}},
        temperature=0.7,
        stream=False,
    )
    raw = resp.choices[0].message.content.strip()
    cleaned = clean_json_response(raw)
    data = json.loads(cleaned)
    dist = data.get("program_distribution", data)
    total = sum(dist.values())
    return {k: float(v)/total for k,v in dist.items()}

def generate_gpa(class_of_degree) -> float:
    """Generate a realistic GPA within the range for a class-of-degree code."""
    if class_of_degree == "F":  # First Class
        return round(random.uniform(3.60, 4.30), 2)
    elif class_of_degree == "SU":  # Upper Second Class
        return round(random.uniform(3.00, 3.59), 2)
    elif class_of_degree == "SL":  # Lower Second Class
        return round(random.uniform(2.50, 2.99), 2)
    elif class_of_degree == "P":  # Pass
        return round(random.uniform(2.00, 2.49), 2)
    else:
        return 0.0  # Default case

def select_class_of_degree(program_major, gpa_distributions):
    """Choose class-of-degree based on program-specific GPA distributions."""
    distribution = gpa_distributions[program_major]
    classes = list(distribution.keys())
    weights = [float(v) for v in distribution.values()]
    return random.choices(classes, weights=weights, k=1)[0]

def generate_valid_graduation_term(admit_year, admit_sem, min_years=2, max_years=6):
    """Compute a valid graduation term code and time-to-graduate for an admitted student."""
    min_grad_year = admit_year + min_years
    max_grad_year = admit_year + max_years
     
    if min_grad_year > max_grad_year:
        min_grad_year = max_grad_year
        logger.warning(f"Adjusted graduation year for admit_year {admit_year} to stay within bounds.")

    grad_year = random.randint(min_grad_year, max_grad_year)

    if grad_year == admit_year:
        grad_sem = random.randint(admit_sem + 1, 3)
    else:
        grad_sem = random.randint(1, 3)

    time_to_graduate = (grad_year - admit_year) + (grad_sem - admit_sem) / 3.0

    return f"{grad_year}{grad_sem}0", grad_year, round(time_to_graduate, 2)

def main():
    # Prompt for reproducibility seed
    seed_input = input("Enter random seed for reproducibility (blank for random): ")
    seed = int(seed_input) if seed_input.strip() else random.SystemRandom().randint(0, 2**32 - 1)
    logger.info(f"Using RNG seed: {seed}")
    random.seed(seed)

    # Prompt for number of records to generate
    while True:
        try:
            num_records = int(input("Enter number of records to generate: "))
            if num_records <= 0:
                print("Please enter a positive number")
                continue
            break
        except ValueError:
            print("Please enter a valid integer")

    # Prompt for output CSV filename
    output_file = input("Enter output CSV filename: ")
    if not output_file.endswith('.csv'):
        output_file += '.csv'
    logger.info(f"Generating {num_records} records to {output_file}")

    # Prompt for admission year range
    while True:
        try:
            min_year = int(input("Enter beginning admission year: "))
            max_year = int(input("Enter ending admission year: "))
            if min_year > max_year:
                print("Beginning year cannot be after ending year")
                continue
            break
        except ValueError:
            print("Please enter valid integers for years")

    logger.info("Starting GPA and gender distribution generation")
    logger.info("This script uses OpenRouter for LLM completion.")
    logger.info("Make sure you have set OPENROUTER_API_KEY in your environment or .env file")

    # Define possible programme majors
    programmes = [
        "BSC Computer Science (Major)",
        "BSC Computer Science (Spec)",
        "BSC Computer Science and Management",
        "BSC Information Technology (Major)",
        "BSC Information Technology (Spec)"
    ]

    # Get program-specific GPA distributions from LLM
    try:
        # Get degree class distributions per program
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(query_llm_gpa, program): program for program in programmes}
            gpa_distributions = {}
            for future in as_completed(futures):
                program = futures[future]
                try:
                    gpa_distributions[program] = future.result()
                except Exception as e:
                    logger.error(f"Error generating GPA distribution for {program}: {e}")
        logger.info("Successfully generated degree class distributions for all programs")
        
        # Get gender distributions per program
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(query_llm_gender_by_program, program): program for program in programmes}
            gender_distributions = {}
            for future in as_completed(futures):
                program = futures[future]
                try:
                    gender_distributions[program] = future.result()
                except Exception as e:
                    logger.error(f"Error generating gender distribution for {program}: {e}")
        logger.info("Successfully generated gender distributions for all programs")
        
    except Exception as e:
        logger.error(f"Error generating distributions: {e}")
        logger.error("Script execution failed. Please check your API key and network connection.")
        raise SystemExit(1)  # Exit with error code

    # Get program distribution across majors
    try:
        program_distribution = get_program_distribution(programmes)
        logger.info("Successfully generated program distribution")
    except Exception as e:
        logger.error(f"Error generating program distribution: {e}")
        raise SystemExit(1)  # Exit with error code

    # Allocate students
    students_per_program = {}
    allocated_students = 0
    
    for program in programmes:
        # Calculate student count based on program distribution probability
        students_count = int(program_distribution[program] * num_records)
        students_per_program[program] = students_count
        allocated_students += students_count
    
    # Adjust for rounding errors
    remaining = num_records - allocated_students
    
    if remaining > 0:
        # Distribute remaining students based on fractional parts
        fractional_parts = {
            p: program_distribution[p] * num_records - students_per_program[p] 
            for p in programmes
        }
        # Sort programs by fractional parts (descending)
        sorted_programs = sorted(programmes, key=lambda p: fractional_parts[p], reverse=True)
        
        # Distribute remaining students to programs with largest fractional parts
        for i in range(remaining):
            students_per_program[sorted_programs[i % len(sorted_programs)]] += 1
    elif remaining < 0:
        # Remove excess students based on reverse fractional parts
        fractional_parts = {
            p: program_distribution[p] * num_records - students_per_program[p] 
            for p in programmes
        }
        # Sort programs by fractional parts (ascending)
        sorted_programs = sorted(programmes, key=lambda p: fractional_parts[p])
        
        # Remove students from programs with smallest fractional parts
        for i in range(abs(remaining)):
            if students_per_program[sorted_programs[i % len(sorted_programs)]] > 0:
                students_per_program[sorted_programs[i % len(sorted_programs)]] -= 1
    
    logger.info(f"Distributing {num_records} students among programs based on realistic probabilities:")
    for program, count in students_per_program.items():
        logger.info(f"  {program}: {count} students ({program_distribution[program]*100:.1f}%)")

    with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "TERM_CODE_GRAD", 
            "TERM_CODE_ADMIT", 
            "GENDER",
            "STU_DEGREE_CODE", 
            "FACULTY_CODE", 
            "CLASS_OF_DEGREE_CODE",
            "GPA",
            "TIME_TO_GRADUATE"
        ])

        # Process each program with its allocated number of students
        total_records = 0
        
        for program, student_count in students_per_program.items():
            logger.info(f"Generating {student_count} records for {program}")
            
            for _ in tqdm(range(student_count), desc=f"Generating {program}"):
                # Limit admission year to allow at least 2 years before graduation
                min_years_to_grad = 2
                max_admit = max_year - min_years_to_grad
                if max_admit < min_year:
                    logger.warning("Year range too narrow; using beginning admission year")
                    max_admit = min_year
                admit_year = random.randint(min_year, max_admit)
                
                # All students admitted in Semester 1
                admit_sem = 1
                term_code_admit = f"{admit_year}{admit_sem}0"

                # Generate valid graduation term
                term_code_grad, grad_year, time_to_graduate = generate_valid_graduation_term(admit_year, admit_sem)

                # Determine faculty code based on graduation year
                faculty_code = "SA" if grad_year <= 2012 else "ST"
                
                # Select class of degree based on program
                class_of_degree = select_class_of_degree(program, gpa_distributions)
                
                # Generate GPA based on class of degree
                gpa = generate_gpa(class_of_degree)
                
                # Determine gender based on program's gender distribution
                gender_probs = gender_distributions[program]
                gender = random.choices(
                    list(gender_probs.keys()),
                    weights=list(gender_probs.values()),
                    k=1
                )[0]
                # Normalize gender code to single letter (M or F)
                gender = gender.strip()
                gender = gender[0].upper() if gender else gender
                
                # Write row to CSV
                writer.writerow([
                    term_code_grad,
                    term_code_admit,
                    gender,
                    program,
                    faculty_code,
                    class_of_degree,
                    gpa,
                    time_to_graduate
                ])
                
                total_records += 1
        
        logger.info(f"Generated {total_records} total student records")

if __name__ == "__main__":
    main()