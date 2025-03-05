# Import optimizations
from tqdm import tqdm
import csv
import random
from datetime import date
import faker
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
fake = faker.Faker()


# Query LLMs for demographic distributions
def query_llm(year: int) -> Dict[str, Any]:
    """Query LLM for demographic distributions for a specific year using structured output schema"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"X-Title": "DAFHE Data Generation"},
    )
    
    # List of models to try
    models = [
        "meta-llama/llama-3.3-70b-instruct:free",   # free
        "google/gemini-2.0-pro-exp-02-05:free",     # free
        "google/gemini-2.0-flash-exp:free",         # free
        "openai/gpt-4o-mini",                       # $0.15/M
        "mistralai/ministral-8b"                    # $0.10/M
    ]
    
    # Shuffle models for random rotation
    shuffled_models = random.sample(models, len(models))
    
    # Track attempts per model
    attempt_count = {model: 0 for model in models}
    max_attempts_per_model = 2

    # Define the JSON schema for demographic distributions - simplified with a function
    demographic_schema = create_demographic_schema()

    user_prompt = create_user_prompt(year)

    # Try all models up to max attempts per model
    models_to_try = shuffled_models.copy()
    
    while models_to_try:
        # Get next model to try
        model = models_to_try.pop(0)
        attempt_count[model] += 1
        
        try:
            logger.info(f"Attempting request with model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "demographic_distribution",
                        "strict": True,
                        "schema": demographic_schema
                    }
                },
                max_tokens=800,
                temperature=0.7, # Experiment for realism
                stream=False,
            )
            
            # Validate response has required structure
            validate_response(response, model)
            
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Raw response from {model}: {response_text[:100]}...")
            
            # Try to parse the JSON response
            distributions = json.loads(response_text)

            # Normalize distributions to ensure they sum to 1.0
            normalized = normalize_distributions(distributions)

            logger.info(f"✓ Successfully generated valid distributions for year {year} using {model}")
            return normalized

        except Exception as e:
            logger.warning(f"Error with {model}: {str(e)}")
            
            # If we haven't reached max attempts for this model, add it back to the queue
            if attempt_count[model] < max_attempts_per_model:
                models_to_try.append(model)
                
            time.sleep(1 + random.random())  # Randomized delay

    # If all models failed, use fallback distributions
    logger.error(f"All models failed to generate valid distributions for {year}")
    return create_fallback_distribution()


def create_demographic_schema() -> Dict[str, Any]:
    """Create the JSON schema for demographic distributions"""
    # Create the schema for marital status
    marital_status_props = {
        "type": "object",
        "properties": {
            "Single": {"type": "number"},
            "Married": {"type": "number"},
            "Common-Law": {"type": "number"},
            "Separated": {"type": "number"},
            "Divorced": {"type": "number"},
            "Other": {"type": "number"},
        },
        "required": ["Single", "Married", "Common-Law", "Separated", "Divorced", "Other"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "gender": {
                "type": "object",
                "properties": {
                    "F": {"type": "number", "description": "Proportion of female students (0-1)"},
                    "M": {"type": "number", "description": "Proportion of male students (0-1)"},
                },
                "required": ["F", "M"],
                "additionalProperties": False,
            },
            "religion": {
                "type": "object",
                "properties": {
                    "Christian": {"type": "number", "description": "Proportion of Christian students (0-1)"},
                    "Hindu": {"type": "number", "description": "Proportion of Hindu students (0-1)"},
                    "Muslim": {"type": "number", "description": "Proportion of Muslim students (0-1)"},
                    "None": {"type": "number", "description": "Proportion of students with no religion (0-1)"},
                    "Other": {"type": "number", "description": "Proportion of students with other religions (0-1)"},
                },
                "required": ["Christian", "Hindu", "Muslim", "None", "Other"],
                "additionalProperties": False,
            },
            "marital_status": {
                "type": "object",
                "properties": {
                    "age_16_19": marital_status_props,
                    "age_20_24": marital_status_props,
                    "age_25_34": marital_status_props,
                    "age_35_plus": marital_status_props,
                },
                "required": ["age_16_19", "age_20_24", "age_25_34", "age_35_plus"],
                "additionalProperties": False,
            },
        },
        "required": ["gender", "religion", "marital_status"],
        "additionalProperties": False,
    }


def create_user_prompt(year: int) -> str:
    """Create the prompt for the LLM"""
    return f"""Generate demographic statistics for university CS/IT programs in Trinidad and Tobago for the year {year}.

Include gender distribution (F/M), religion distribution (Christian/Hindu/Muslim/None/Other), and marital status by age group.

For marital status, different age groups have different patterns:
- Age 16-19: Mostly single (>95%)
- Age 20-24: Predominantly single with some married/common-law
- Age 25-34: More diverse distribution
- Age 35+: Higher proportion of married/divorced

Each category must sum EXACTLY to 1.0 (100%). Ensure all values are between 0 and 1."""


def validate_response(response: Any, model: str) -> None:
    """Validate that the response has the expected structure"""
    if not response or not hasattr(response, 'choices') or not response.choices:
        raise ValueError(f"Empty response from {model}")
            
    if not response.choices[0].message or not hasattr(response.choices[0].message, 'content'):
        raise ValueError(f"No message content in response from {model}")
            
    response_text = response.choices[0].message.content.strip()
    if not response_text:
        raise ValueError(f"Empty content from {model}")


def create_fallback_distribution() -> Dict[str, Any]:
    """Create fallback distribution data if LLM calls fail"""
    return {
        "gender": {"F": 0.4, "M": 0.6},
        "religion": {
            "Christian": 0.55, 
            "Hindu": 0.25, 
            "Muslim": 0.12, 
            "None": 0.05, 
            "Other": 0.03
        },
        "marital_status": {
            "age_16_19": {
                "Single": 0.98, "Married": 0.005, "Common-Law": 0.005, 
                "Separated": 0.002, "Divorced": 0.003, "Other": 0.005
            },
            "age_20_24": {
                "Single": 0.85, "Married": 0.07, "Common-Law": 0.05, 
                "Separated": 0.01, "Divorced": 0.01, "Other": 0.01
            },
            "age_25_34": {
                "Single": 0.4, "Married": 0.3, "Common-Law": 0.15, 
                "Separated": 0.07, "Divorced": 0.05, "Other": 0.03
            },
            "age_35_plus": {
                "Single": 0.15, "Married": 0.45, "Common-Law": 0.15, 
                "Separated": 0.1, "Divorced": 0.1, "Other": 0.05
            }
        }
    }


def normalize_distributions(distributions: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize all distributions to ensure they sum to 1.0"""
    # Deep copy to avoid modifying the original
    result = json.loads(json.dumps(distributions))

    # Normalize gender and religion
    for category in ["gender", "religion"]:
        if category in result:
            values = [float(v) for v in result[category].values()]
            total = sum(values)
            if total > 0:
                result[category] = {
                    k: float(v) / total for k, v in result[category].items()
                }

    # Normalize marital status for each age group
    if "marital_status" in result:
        for age_group in ["age_16_19", "age_20_24", "age_25_34", "age_35_plus"]:
            if age_group in result["marital_status"]:
                age_data = result["marital_status"][age_group]
                values = [float(v) for v in age_data.values()]
                total = sum(values)
                if total > 0:
                    result["marital_status"][age_group] = {
                        k: float(v) / total for k, v in age_data.items()
                    }

    return result


def validate_distributions_with_age_groups(distributions: Dict[str, Any]) -> bool:
    """Validate that distributions have the expected structure and values sum to 1.0"""
    # Check basic structure
    required_keys = ["gender", "religion", "marital_status"]
    if not all(key in distributions for key in required_keys):
        return False

    # Validate gender distribution
    gender_keys = {"F", "M"}
    if set(distributions["gender"].keys()) != gender_keys:
        return False

    # Validate religion distribution
    religion_keys = {"Christian", "Hindu", "Muslim", "None", "Other"}
    if not all(k in distributions["religion"] for k in religion_keys):
        return False

    # Validate marital status structure
    age_groups = ["age_16_19", "age_20_24", "age_25_34", "age_35_plus"]
    marital_statuses = {
        "Single",
        "Married",
        "Common-Law",
        "Separated",
        "Divorced",
        "Other",
    }

    for age_group in age_groups:
        if age_group not in distributions["marital_status"]:
            return False

        if set(distributions["marital_status"][age_group].keys()) != marital_statuses:
            return False

    # Check that all distributions sum approximately to 1.0
    if not abs(sum(distributions["gender"].values()) - 1.0) < 0.01:
        return False

    if not abs(sum(distributions["religion"].values()) - 1.0) < 0.01:
        return False

    for age_group in age_groups:
        if (
            not abs(sum(distributions["marital_status"][age_group].values()) - 1.0)
            < 0.01
        ):
            return False

    return True


def precompute_distributions(min_year: int, max_year: int, 
                             retry_count: int = 2) -> Dict[int, Dict[str, Any]]:
    """Pre-compute demographic distributions for all years"""
    distributions = {}
    logger.info("Pre-computing demographic distributions...")

    for year in range(min_year, max_year + 1):
        for attempt in range(retry_count + 1):
            try:
                distributions[year] = query_llm(year)
                break  # Success, break retry loop
            except Exception as e:
                if attempt < retry_count:
                    logger.warning(f"Attempt {attempt+1} failed for year {year}: {e}. Retrying...")
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Critical error: Failed to generate distributions for year {year} after {retry_count+1} attempts."
                    ) from e

    return distributions


def load_data_with_retry(file_path: str, max_retries: int = 3) -> pd.DataFrame:
    """Load data from a CSV file with retry logic"""
    for attempt in range(max_retries):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error loading {file_path}: {e}. Retrying...")
                time.sleep(1)
            else:
                raise RuntimeError(f"Failed to load {file_path} after {max_retries} attempts") from e


def generate_student_ids(num_students: int) -> List[int]:
    """Generate unique student IDs"""
    possible_ids = list(range(318000000, 318999999))
    random.shuffle(possible_ids)
    return possible_ids[:num_students]


def load_country_probabilities() -> Dict[str, float]:
    """Load country probabilities from CSV file"""
    df = pd.read_csv(r"real stats/country_stats_with_probabilities.csv")
    return dict(zip(df["iso3"], df["probability"]))


def prepare_location_data() -> Tuple[
    pd.DataFrame, Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]], List[str], List[float]
]:
    """Cache location data for efficient city selection"""
    worldcities_df = pd.read_csv(r"real stats/worldcities.csv")
    worldcities_df["population"] = pd.to_numeric(
        worldcities_df["population"], errors="coerce"
    ).fillna(0)

    # Load country probabilities and filter cities
    country_probs = load_country_probabilities()
    valid_nations = list(country_probs.keys())
    worldcities_df = worldcities_df[worldcities_df["iso3"].isin(valid_nations)]

    # Pre-compute city weights by country
    city_cache = {}
    nation_probs = []
    valid_nations = []

    for nation in worldcities_df["iso3"].unique():
        nation_cities = worldcities_df[worldcities_df["iso3"] == nation]
        total_pop = nation_cities["population"].sum()

        weights = nation_cities["population"] / total_pop if total_pop > 0 else None
        city_cache[nation] = (nation_cities, weights)
        valid_nations.append(nation)
        nation_probs.append(country_probs.get(nation, 0.0))

    return worldcities_df, city_cache, valid_nations, nation_probs


def generate_dob(admit_year: int) -> str:
    """Generate date of birth based on admission year"""
    if random.random() < 0.90:
        # Most students are 18-24
        age_at_admission = random.randint(18, 24)
    elif random.random() < 0.5:
        # Some younger students
        age_at_admission = random.randint(16, 17)
    else:
        # Some mature students
        age_at_admission = random.randint(25, 45)

    birth_year = admit_year - age_at_admission
    return fake.date_between(
        start_date=date(birth_year, 1, 1), end_date=date(birth_year, 12, 31)
    ).strftime("%Y-%m-%d")


def get_next_term(year: int, sem: int) -> Tuple[int, int]:
    """Get the next semester and year"""
    return (year + 1, 1) if sem == 3 else (year, sem + 1)


def generate_term_codes(admit_year: int, admit_sem: int) -> List[str]:
    """Generate term codes for a student's academic path"""
    # Determine program length with weighted probabilities
    r = random.random()
    total_semesters = 6 if r < 0.60 else 7 if r < 0.85 else 8 if r < 0.95 else 9

    # Generate term codes
    term_codes = []
    year, sem = admit_year, admit_sem
    for _ in range(total_semesters):
        term_codes.append(f"{year}{sem}0")
        year, sem = get_next_term(year, sem)
    return term_codes


def load_degree_probabilities() -> Dict[int, Dict[str, float]]:
    """Load and prepare degree enrollment probabilities"""
    df = pd.read_csv(r"real stats/degree_enrollment.csv")
    prob_data = {}

    for _, row in df.iterrows():
        year = row["year"]
        total = row["total"]

        # Calculate base probabilities
        base_probs = {
            col: row[col] / total
            for col in ["CS-MAJO", "CS-SPEC", "CS-MANA", "IT-MAJO", "IT-SPEC"]
        }

        # Add random variation
        varied_probs = {
            degree: max(0.01, prob * (1 + random.uniform(-0.15, 0.15)))
            for degree, prob in base_probs.items()
        }

        # Normalize
        total_prob = sum(varied_probs.values())
        prob_data[year] = {k: v / total_prob for k, v in varied_probs.items()}

    return prob_data


def get_degree_probabilities(
    year: int, degree_probs: Dict[int, Dict[str, float]]
) -> Tuple[List[str], List[float]]:
    """Get degree probabilities for a specific year"""
    available_years = sorted(degree_probs.keys())

    if year in degree_probs:
        probs = degree_probs[year]
    else:
        nearest_year = min(available_years, key=lambda x: abs(x - year))
        probs = degree_probs[nearest_year]

    degrees = list(probs.keys())
    probabilities = list(probs.values())
    return degrees, probabilities


def get_age_group(dob_str: str, admit_year: int) -> str:
    """Determine age group based on birth date and admission year"""
    birth_year = int(dob_str.split("-")[0])
    age = admit_year - birth_year

    if age <= 19:
        return "age_16_19"
    elif age <= 24:
        return "age_20_24"
    elif age <= 34:
        return "age_25_34"
    else:
        return "age_35_plus"


def main():
    """Main function to execute the script"""
    random.seed(42)
    np.random.seed(42)  # Also set numpy's random seed

    logger.info("Starting student data generation script")
    logger.info("This script uses OpenRouter for LLM completion.")
    logger.info("Make sure you have set OPENROUTER_API_KEY in your environment or .env file")

    # Get parameters (with validation)
    while True:
        try:
            num_students = int(input("Enter the number of students to generate: "))
            if num_students <= 0:
                print("Please enter a positive number")
                continue
                
            min_year = int(input("Enter the beginning year: "))
            max_year = int(input("Enter the ending year: "))
            if min_year > max_year:
                print("Beginning year cannot be after the end year")
                continue
                
            output_file = input("Enter output CSV filename: ")
            if not output_file.endswith('.csv'):
                output_file += '.csv'
            
            break
        except ValueError:
            print("Please enter valid numbers")

    try:
        # Process execution with timing information
        start_time = time.time()
        
        logger.info("Loading location data...")
        worldcities_df, city_cache, valid_nations, nation_probs = prepare_location_data()

        logger.info("Loading degree enrollment data...")
        degree_probs = load_degree_probabilities()

        # Get distributions
        distributions = precompute_distributions(min_year, max_year)

        # Generate student IDs
        student_ids = generate_student_ids(num_students)

        # Generate records
        logger.info("Generating student records...")
        with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "STUDENT_ID", "TERM_CODE_EFF", "TERM_CODE_ADMIT",
                "DATE_OF_BIRTH", "CITY", "STATE", "NATION",
                "GENDER", "RELIGION", "MARITAL_STATUS",
                "FACULTY_CODE", "STU_DEGREE_CODE",
            ])

            for sid in tqdm(student_ids, desc="Generating records"):
                # Basic student attributes
                admit_year = random.randint(min_year, max_year)
                term_code_admit = f"{admit_year}10"  # Fall admission
                admit_sem = 1
                term_codes = generate_term_codes(admit_year, admit_sem)
                dob = generate_dob(admit_year)
                age_group = get_age_group(dob, admit_year)
                faculty_code = "FSA" if int(term_codes[0][:4]) <= 2012 else "FST"

                # Use demographic distributions
                year_dist = distributions[admit_year]

                # Select demographics based on distributions
                gender = random.choices(
                    list(year_dist["gender"].keys()),
                    weights=list(year_dist["gender"].values()),
                    k=1,
                )[0]

                religion = random.choices(
                    list(year_dist["religion"].keys()),
                    weights=list(year_dist["religion"].values()),
                    k=1,
                )[0]

                marital_status = random.choices(
                    list(year_dist["marital_status"][age_group].keys()),
                    weights=list(year_dist["marital_status"][age_group].values()),
                    k=1,
                )[0]

                # Degree selection
                degrees, probabilities = get_degree_probabilities(admit_year, degree_probs)
                degree_code = random.choices(degrees, weights=probabilities, k=1)[0]

                # Location selection
                chosen_nation = random.choices(valid_nations, weights=nation_probs, k=1)[0]
                nation_cities, weights = city_cache[chosen_nation]

                if weights is not None:
                    selected_city = nation_cities.sample(n=1, weights=weights).iloc[0]
                else:
                    selected_city = nation_cities.sample(1).iloc[0]

                city = selected_city["city_ascii"]
                state = selected_city["admin_name"]
                nation_value = selected_city["country"]

                # Write records for each semester
                for term_eff in term_codes:
                    writer.writerow(
                        [
                            sid,
                            term_eff,
                            term_code_admit,
                            dob,
                            city,
                            state,
                            nation_value,
                            gender,
                            religion,
                            marital_status,
                            faculty_code,
                            degree_code,
                        ]
                    )

        elapsed_time = time.time() - start_time
        logger.info(f"Completed successfully in {elapsed_time:.2f} seconds.")
        logger.info(f"Generated data for {num_students} students across {max_year - min_year + 1} years.")
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()