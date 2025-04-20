from tqdm import tqdm
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# see https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs
# === MODEL LIST === 
LLM_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "google/learnlm-1.5-pro-experimental:free",
]

BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'real_stats'))

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
fake = faker.Faker()

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={"X-Title": "DAFHE Data Generation"},
)

def query_llm(year: int, model: str = None, use_historical_data: bool = False, historical_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Query LLM for demographic distributions of a given year."""
    models = LLM_MODELS
    if model is not None:
        models_to_try = [model]
    else:
        models_to_try = random.sample(models, len(models))
    attempt_count = {m: 0 for m in models}
    max_attempts_per_model = 3

    demographic_schema = create_demographic_schema(year, include_degree_distribution=not use_historical_data)
    user_prompt = create_user_prompt(year, use_historical_data, historical_data)

    while models_to_try:
        model_to_use = models_to_try.pop(0)
        attempt_count[model_to_use] += 1
        try:
            logger.info(f"Attempting request with model: {model_to_use}")
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": user_prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "demographic_distribution",
                        "strict": True,
                        "schema": demographic_schema
                    }
                },
                max_tokens=1000,
                temperature=0.9 if not use_historical_data else 0.7,
                stream=False,
            )
            validate_response(response, model_to_use)
            response_text = response.choices[0].message.content.strip()
            logger.info(f"Raw response from {model_to_use}: {response_text[:100]}...")
            distributions = json.loads(response_text)
            normalized = normalize_distributions(distributions)
            if not validate_distributions(normalized, include_degree_distribution=not use_historical_data):
                logger.warning(f"Invalid structure or missing keys in LLM response for year {year}, model {model_to_use}. Retrying.")
                raise ValueError("LLM response did not match expected schema.")
            logger.info(f"✓ Successfully generated valid distributions for year {year} using {model_to_use}")
            return normalized
        except Exception as e:
            logger.warning(f"Error with {model_to_use}: {str(e)}")
            if attempt_count[model_to_use] < max_attempts_per_model:
                models_to_try.append(model_to_use)
            time.sleep(1 + random.random())
    
    # If all models fail
    raise RuntimeError(f"All models failed to generate valid distributions for {year}")


def create_demographic_schema(year: int, include_degree_distribution: bool = False) -> Dict[str, Any]:
    """Create demographic schema for a given year."""
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

    schema = {
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
    
    if include_degree_distribution:
        schema["properties"]["degree_distribution"] = {
            "type": "object",
            "description": f"Distribution of students across degree programs for year {year}",
            "properties": {
                "CS-MAJO": {"type": "number", "description": "Proportion in BSc Computer Science (Major) program (0-1)"},
                "CS-SPEC": {"type": "number", "description": "Proportion in BSc Computer Science (Special) program (0-1)"},
                "CS-MANA": {"type": "number", "description": "Proportion in BSc Computer Science with Management program (0-1)"},
                "IT-MAJO": {"type": "number", "description": "Proportion in BSc Information Technology (Major) program (0-1)"},
                "IT-SPEC": {"type": "number", "description": "Proportion in BSc Information Technology (Special) program (0-1)"},
            },
            "required": ["CS-MAJO", "CS-SPEC", "CS-MANA", "IT-MAJO", "IT-SPEC"],
            "additionalProperties": False,
        }
        schema["required"].append("degree_distribution")
    
    return schema


def create_user_prompt(year: int, use_historical_data: bool = False, historical_data: Optional[Dict[str, Any]] = None) -> str:
    """Create the prompt for the LLM"""
    base_prompt = f"""Generate plausible demographic statistics for university CS/IT programs in Trinidad and Tobago for the year {year}.

Include the following distributions (each must sum to EXACTLY 1.0):
1. Gender distribution (F/M)
2. Religion distribution (Christian/Hindu/Muslim/None/Other)
3. Marital status by age group (age_16_19, age_20_24, age_25_34, age_35_plus)

Note these demographic trends for the region:
- Marital status varies by age: 16-19 (mostly single), 20-24 (predominantly single), 25-34 (more diverse), 35+ (higher married rates)
- Gender ratios in CS/IT programs tend to vary by program and over time

Return ONLY valid JSON matching the required schema. No explanatory text."""
    
    if use_historical_data and historical_data:
        historical_context = ""
        
        # Add gender context if available
        if 'gender' in historical_data and 'gender_ratio' in historical_data['gender']:
            gender_ratio = historical_data['gender']['gender_ratio']
            f_pct = round(gender_ratio.get('F', 0) * 100)
            m_pct = round(gender_ratio.get('M', 0) * 100)
            historical_context += f"\nHistorical gender ratio for CS/IT programs in this period: approximately {f_pct}% female, {m_pct}% male."
        
        # Add program context if available
        if 'program' in historical_data:
            program_dist = historical_data['program']
            # Find the top 2 programs
            if program_dist:
                sorted_programs = sorted(program_dist.items(), key=lambda x: x[1], reverse=True)[:2]
                program_context = ", ".join([f"{prog.replace('-', ' ')}: {round(pct * 100)}%" for prog, pct in sorted_programs])
                historical_context += f"\nMost popular programs: {program_context}."
        
        if historical_context:
            return base_prompt + historical_context
    
    return base_prompt


def validate_response(response: Any, model: str) -> None:
    """Ensure LLM response adheres to the expected schema."""
    if not response or not hasattr(response, 'choices') or not response.choices:
        raise ValueError(f"Empty response from {model}")
            
    if not response.choices[0].message or not hasattr(response.choices[0].message, 'content'):
        raise ValueError(f"No message content in response from {model}")
            
    response_text = response.choices[0].message.content.strip()
    if not response_text:
        raise ValueError(f"Empty content from {model}")


def normalize_distributions(distributions: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize distributions so each sums to 1."""
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

    # Normalize degree distribution if present
    if "degree_distribution" in result:
        values = [float(v) for v in result["degree_distribution"].values()]
        total = sum(values)
        if total > 0:
            result["degree_distribution"] = {
                k: float(v) / total for k, v in result["degree_distribution"].items()
            }

    return result


def validate_distributions(distributions: Dict[str, Any], include_degree_distribution: bool = False) -> bool:
    """Validate distribution structure and total sums."""
    # Check basic structure
    required_keys = ["gender", "religion", "marital_status"]
    if include_degree_distribution:
        required_keys.append("degree_distribution")
        
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
    
    # Validate degree distribution if required
    if include_degree_distribution:
        degree_keys = {"CS-MAJO", "CS-SPEC", "CS-MANA", "IT-MAJO", "IT-SPEC"}
        if not all(k in distributions["degree_distribution"] for k in degree_keys):
            return False
        
        if not abs(sum(distributions["degree_distribution"].values()) - 1.0) < 0.01:
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


def precompute_distributions(
    min_year: int,
    max_year: int,
    retry_count: int = 3,
    use_historical_data: bool = False
) -> Dict[int, Dict[str, Any]]:
    """Precompute LLM demographic distributions across a range of years."""
    logger.info(f"Precomputing demographic distributions for years {min_year}-{max_year} (use_historical_data={use_historical_data})...")
    # Prepare historical data if needed
    historical_data = load_historical_data() if use_historical_data else None
    models = LLM_MODELS
    years = list(range(min_year, max_year + 1))
    assignments = [(year, models[i % len(models)]) for i, year in enumerate(years)]
    distributions: Dict[int, Dict[str, Any]] = {}
    # Parallelize all LLM calls
    max_workers = min(len(assignments), len(models) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {
            executor.submit(
                lambda y, m: (y, query_llm(
                    y,
                    model=m,
                    use_historical_data=use_historical_data,
                    historical_data=get_historical_data_for_year(historical_data, y) if historical_data else None
                )),
                year,
                model
            ): year
            for year, model in assignments
        }
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                _, dist = future.result()
                distributions[year] = dist
                logger.info(f"✓ Generated distribution for {year}")
            except Exception as e:
                logger.error(f"Failed to generate distribution for {year}: {e}")
                raise
    return dict(sorted(distributions.items()))


def load_historical_data() -> Dict[str, Any]:
    """Load historical demographic data from /real_stats."""
    logger.info("Loading historical data from real_stats directory...")
    data = {}
    try:
        deg_data = pd.read_csv(os.path.join(BASE_DATA_DIR, "degree_enrollment_gender.csv"))
        data['gender'] = deg_data
        data['program'] = deg_data
        logger.info("Loaded degree enrollment (with gender) data")
    except Exception as e:
        logger.warning(f"Could not load degree enrollment gender data: {e}")
    try:
        country_data = pd.read_csv(os.path.join(BASE_DATA_DIR, "country_stats_with_probabilities.csv"))
        data['country'] = country_data
        logger.info("Loaded country distribution data")
    except Exception as e:
        logger.warning(f"Could not load country data: {e}")
    return data


def get_historical_data_for_year(historical_data: Dict[str, Any], year: int) -> Dict[str, Any]:
    """Extract relevant historical data for a specific year."""
    if not historical_data:
        return {}
    
    year_data = {}
    
    # Extract gender distribution if available
    if 'gender' in historical_data and not historical_data['gender'].empty:
        gender_df = historical_data['gender']
        
        # Find the closest year in the data
        if year in gender_df['year'].values:
            year_row = gender_df[gender_df['year'] == year]
        else:
            closest_year = min(gender_df['year'].values, key=lambda x: abs(x - year))
            year_row = gender_df[gender_df['year'] == closest_year]
        
        if not year_row.empty:
            # Calculate gender ratios from available data
            gender_info = {}
            
            # For years that have M/F breakdown
            if all(col in year_row.columns for col in ['CS-MAJ-M', 'CS-MAJ-F', 'IT-MAJ-M', 'IT-MAJ-F']):
                total_males = 0
                total_females = 0
                
                for program in ['CS-MAJ', 'CS-MAN', 'IT-MAJ']:
                    if f'{program}-M' in year_row.columns and f'{program}-F' in year_row.columns:
                        males = year_row[f'{program}-M'].iloc[0]
                        females = year_row[f'{program}-F'].iloc[0]
                        
                        if not pd.isna(males) and not pd.isna(females):
                            total_males += males
                            total_females += females
                
                total = total_males + total_females
                if total > 0:
                    gender_info['gender_ratio'] = {
                        'M': total_males / total,
                        'F': total_females / total
                    }
            
            year_data['gender'] = gender_info
    
    # Extract program distribution if available
    if 'program' in historical_data and not historical_data['program'].empty:
        program_df = historical_data['program']
        
        if year in program_df['year'].values:
            program_row = program_df[program_df['year'] == year]
        else:
            closest_year = min(program_df['year'].values, key=lambda x: abs(x - year))
            program_row = program_df[program_df['year'] == closest_year]
        
        if not program_row.empty:
            # Use *-TOTAL columns for program totals
            degree_total_cols = [
                'CS-MAJ-TOTAL', 'CS-SPE-TOTAL', 'CS-MAN-TOTAL', 'IT-MAJ-TOTAL', 'IT-SPE-TOTAL'
            ]
            total = sum([program_row[col].iloc[0] if col in program_row and not pd.isna(program_row[col].iloc[0]) else 0 for col in degree_total_cols])
            program_info = {}
            for col in degree_total_cols:
                prog_key = col.replace('-TOTAL', '').replace('-', '_')
                value = program_row[col].iloc[0] if col in program_row and not pd.isna(program_row[col].iloc[0]) else 0
                program_info[prog_key] = value / total if total > 0 else 0.0
            year_data['program'] = program_info
    
    return year_data


def load_degree_probabilities(variation_level: float = 0.15) -> Dict[int, Dict[str, float]]:
    """Load and optionally vary degree enrollment probabilities."""
    df = pd.read_csv(os.path.join(BASE_DATA_DIR, "degree_enrollment_gender.csv"))
    prob_data = {}
    degree_total_cols = [
        "CS-MAJ-TOTAL", "CS-SPE-TOTAL", "CS-MAN-TOTAL", "IT-MAJ-TOTAL", "IT-SPE-TOTAL"
    ]
    for _, row in df.iterrows():
        year = row["year"]
        # Use sum of *-TOTAL columns for total
        total = sum([row.get(col, 0) if pd.notnull(row.get(col, 0)) else 0 for col in degree_total_cols])
        base_probs = {
            col.replace("-TOTAL", "").replace("-", "_"): (row.get(col, 0) if pd.notnull(row.get(col, 0)) else 0) / total if total > 0 else 0.0
            for col in degree_total_cols
        }
        varied_probs = {
            degree: max(0.01, prob * (1 + random.uniform(-variation_level, variation_level)))
            for degree, prob in base_probs.items()
        }
        total_prob = sum(varied_probs.values())
        prob_data[year] = {k: v / total_prob for k, v in varied_probs.items()}
    return prob_data


def get_degree_probabilities(year: int, degree_probs: Dict[int, Dict[str, float]]) -> Tuple[List[str], List[float]]:
    """Return degree probability list for a specific year."""
    available_years = sorted(degree_probs.keys())
    if year in degree_probs:
        probs = degree_probs[year]
    else:
        nearest_year = min(available_years, key=lambda x: abs(x - year))
        probs = degree_probs[nearest_year]
    degrees = list(probs.keys())
    probabilities = list(probs.values())
    return degrees, probabilities


def load_data_with_retry(file_path: str, max_retries: int = 3) -> pd.DataFrame:
    """Load data with retry on failure."""
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
    """Generate a list of unique student identifiers."""
    possible_ids = list(range(318000000, 318999999))
    random.shuffle(possible_ids)
    return possible_ids[:num_students]


def load_country_probabilities() -> Dict[str, float]:
    """Load country probability distribution from CSV."""
    df = pd.read_csv(os.path.join(BASE_DATA_DIR, "country_stats_with_probabilities.csv"))
    return dict(zip(df["iso3"], df["probability"]))


def prepare_location_data() -> Tuple[
    pd.DataFrame, Dict[str, Tuple[List[Tuple[str, str, str]], Optional[List[float]]]], List[str], List[float]
]:
    """Cache and prepare location/city data."""
    worldcities_df = pd.read_csv(os.path.join(BASE_DATA_DIR, "worldcities.csv"))
    worldcities_df["population"] = pd.to_numeric(
        worldcities_df["population"], errors="coerce"
    ).fillna(0)

    # Load country probabilities and filter cities
    logger.info("Loading country probabilities and filtering cities...")
    country_probs = load_country_probabilities()
    valid_nations = list(country_probs.keys())
    worldcities_df = worldcities_df[worldcities_df["iso3"].isin(valid_nations)]

    # Pre-compute city weights by country
    city_cache = {}
    nation_probs = []
    valid_nations = []

    for nation in worldcities_df["iso3"].unique():
        df_n = worldcities_df[worldcities_df["iso3"] == nation]
        total_pop = df_n["population"].sum()
        # store Python lists for faster sampling
        locations = list(zip(
            df_n["city_ascii"].tolist(),
            df_n["admin_name"].tolist(),
            df_n["country"].tolist()
        ))
        weights = (df_n["population"] / total_pop).tolist() if total_pop > 0 else None
        city_cache[nation] = (locations, weights)
        valid_nations.append(nation)
        nation_probs.append(country_probs.get(nation, 0.0))

    return worldcities_df, city_cache, valid_nations, nation_probs


def generate_dob(admit_year: int) -> str:
    """Generate a realistic date of birth for a student."""
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
    """Compute next academic term code."""
    return (year + 1, 1) if sem == 3 else (year, sem + 1)


def generate_term_codes(admit_year: int, admit_sem: int, summer_rate: float) -> List[str]:
    """Generate sequence of term codes for a student."""
    # Determine program length with weighted probabilities
    r = random.random()
    total_semesters = 6 if r < 0.60 else 7 if r < 0.85 else 8 if r < 0.95 else 9

    # Decide if the student does summer semester
    summer_opt_in = random.random() < summer_rate

    # Generate term codes
    term_codes = []
    year, sem = admit_year, admit_sem
    for _ in range(total_semesters):
        term_codes.append(f"{year}{sem}0")
        if sem == 2 and not summer_opt_in:
            # Skip summer: go to next year's semester 1
            year, sem = year + 1, 1
        else:
            year, sem = get_next_term(year, sem)
    return term_codes


def get_age_group(dob_str: str, admit_year: int) -> str:
    """Determine age group category."""
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
    """Main function to generate and save dataset based on parameters."""
    seed_input = input("Enter random seed for reproducibility (blank for random): ")
    seed = int(seed_input) if seed_input.strip() else random.SystemRandom().randint(0, 2**32 - 1)
    logger.info(f"Using RNG seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    summer_rate = random.random()

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
            
            use_historical_data = input("Use historical data to guide generation? (y/n): ").lower().startswith('y')
                
            output_file = input("Enter output CSV filename: ")
            if not output_file.endswith('.csv'):
                output_file += '.csv'
            
            break
        except ValueError:
            print("Please enter valid numbers")

    try:
        start_time = time.time()
        
        logger.info("Loading location data...")
        worldcities_df, city_cache, valid_nations, nation_probs = prepare_location_data()

        # Get distributions from LLM
        distributions = precompute_distributions(min_year, max_year, use_historical_data=use_historical_data)

        # Generate student IDs
        student_ids = generate_student_ids(num_students)

        # Preload fallback degree probabilities once to optimize performance
        fallback_degree_probs = load_degree_probabilities()

        # Generate student records in bulk via pandas
        logger.info("Generating student records...")
        rows = []
        for sid in tqdm(student_ids, desc="Generating student records"):
            # Basic student attributes
            admit_year = random.randint(min_year, max_year)
            term_code_admit = f"{admit_year}10"
            admit_sem = 1
            term_codes = generate_term_codes(admit_year, admit_sem, summer_rate)
            dob = generate_dob(admit_year)
            age_group = get_age_group(dob, admit_year)
            faculty_code = "FSA" if int(term_codes[0][:4]) <= 2012 else "FST"
            year_dist = distributions[admit_year]
            gender = random.choices(list(year_dist["gender"].keys()), weights=list(year_dist["gender"].values()), k=1)[0]
            religion = random.choices(list(year_dist["religion"].keys()), weights=list(year_dist["religion"].values()), k=1)[0]
            marital_status = random.choices(list(year_dist["marital_status"][age_group].keys()), weights=list(year_dist["marital_status"][age_group].values()), k=1)[0]
            if not use_historical_data and "degree_distribution" in year_dist:
                degree_programs = list(year_dist["degree_distribution"].keys())
                degree_weights = list(year_dist["degree_distribution"].values())
                degree_code = random.choices(degree_programs, weights=degree_weights, k=1)[0]
            else:
                degrees, probabilities = get_degree_probabilities(admit_year, fallback_degree_probs)
                degree_code = random.choices(degrees, weights=probabilities, k=1)[0]
            chosen_nation = random.choices(valid_nations, weights=nation_probs, k=1)[0]
            locations, loc_weights = city_cache[chosen_nation]
            if loc_weights:
                city, state, nation_value = random.choices(locations, weights=loc_weights, k=1)[0]
            else:
                city, state, nation_value = random.choice(locations)
            for term_eff in term_codes:
                rows.append({
                    "STUDENT_ID": sid,
                    "TERM_CODE_EFF": term_eff,
                    "TERM_CODE_ADMIT": term_code_admit,
                    "DATE_OF_BIRTH": dob,
                    "CITY": city,
                    "STATE": state,
                    "NATION": nation_value,
                    "GENDER": gender,
                    "RELIGION": religion,
                    "MARITAL_STATUS": marital_status,
                    "FACULTY_CODE": faculty_code,
                    "STU_DEGREE_CODE": degree_code,
                })
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

        elapsed_time = time.time() - start_time
        logger.info(f"Completed successfully in {elapsed_time:.2f} seconds.")
        logger.info(f"Generated data for {num_students} students across {max_year - min_year + 1} years.")
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        logger.error(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()