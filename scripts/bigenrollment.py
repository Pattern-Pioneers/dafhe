from tqdm import tqdm  # progress bar
import csv
import random
from datetime import date
import faker
import pandas as pd
from typing import Dict
import json
from together import Together  # LLM provider
import os

fake = faker.Faker()


# In case LLM response looks shaky formatwise
def clean_llm_response(response_text: str) -> str:
    # Remove markdown code blocks
    response_text = response_text.strip("`")
    while response_text.startswith("`"):
        response_text = response_text[1:]
    while response_text.endswith("`"):
        response_text = response_text[:-1]
    return response_text.strip()


# Query LLM for demographic distributions
def query_llm(year: int) -> Dict[str, Dict[str, float]]:
    api_key = os.getenv("TOGETHER_API_KEY_dafhe")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY_dafhe environment variable not set")

    client = Together(api_key=api_key)
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Choose a model
    print(f"\nUsing LLM model: {model}")

    messages = [
        {
            "role": "system",
            "content": "You are a data analysis assistant that provides demographic statistics for university enrollment in Trinidad and Tobago. Return only raw JSON without any markdown formatting or code blocks.",
        },
        {
            "role": "user",
            "content": f"""For the year {year}, provide gender, religion, and marital status distributions for university CS/IT programs in Trinidad and Tobago as a JSON object.
            Include only major religions and None/Other categories.
            For marital status, include: Single, Married, Common-Law, Separated, Divorced, Other.
            Return only a raw JSON object like this, without any markdown formatting:
            {{
                "gender": {{"F": 0.25, "M": 0.75}},
                "religion": {{
                    "Christian": 0.35,
                    "Hindu": 0.25,
                    "Muslim": 0.15,
                    "None": 0.10,
                    "Other": 0.15
                }},
                "marital_status": {{
                    "Single": 0.82,
                    "Married": 0.10,
                    "Common-Law": 0.03,
                    "Separated": 0.02,
                    "Divorced": 0.02,
                    "Other": 0.01
                }}
            }}
            Values within each category must sum to 1.0
            """,
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,  # Use model variable
            messages=messages,
            max_tokens=200,
            temperature=0.8,  # TODO experiment with temperatures for variety/realism
            top_p=0.9,
            stream=False,
        )

        print(f"\nLLM Response for year {year}:")
        response_text = response.choices[0].message.content.strip()
        print("Raw response:", response_text)
        print("-" * 50)

        try:  # Clean response before parsing
            cleaned_response = clean_llm_response(response_text)
            distributions = json.loads(cleaned_response)
            if validate_distributions(distributions):
                return distributions
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            raise ValueError(f"Failed to parse LLM response: {e}")

    except Exception as e:
        print(f"Error querying LLM API for year {year}")
        print(f"Error: {str(e)}")
        raise RuntimeError(f"LLM query failed: {e}")


def generate_student_ids(num_students):
    """
    Generate a set of unique 9-digit IDs in the range 318000000 to 318999999.
    """
    possible_ids = list(range(318000000, 318999999))
    random.shuffle(possible_ids)
    return possible_ids[:num_students]


def load_country_probabilities():
    """Load country probabilities from the CSV file"""
    df = pd.read_csv(r"real stats/country_stats_with_probabilities.csv")
    probabilities = dict(zip(df["iso3"], df["probability"]))
    return probabilities


def prepare_location_data():
    """Cache location data processing"""
    worldcities_df = pd.read_csv(r"real stats/worldcities.csv")
    worldcities_df["population"] = pd.to_numeric(
        worldcities_df["population"], errors="coerce"
    ).fillna(0)

    # Load country probabilities
    country_probs = load_country_probabilities()

    # Filter worldcities to only include countries in our probability list
    valid_nations = list(country_probs.keys())
    worldcities_df = worldcities_df[worldcities_df["iso3"].isin(valid_nations)]

    # Pre-compute city selections per country
    city_cache = {}
    nation_probs = []
    valid_nations = []

    for nation in worldcities_df["iso3"].unique():
        nation_cities = worldcities_df[worldcities_df["iso3"] == nation]
        total_pop = nation_cities["population"].sum()

        if total_pop > 0:
            weights = nation_cities["population"] / total_pop
            city_cache[nation] = (nation_cities, weights)
        else:
            city_cache[nation] = (nation_cities, None)

        valid_nations.append(nation)
        nation_probs.append(country_probs.get(nation, 0.0))

    return worldcities_df, city_cache, valid_nations, nation_probs


def precompute_distributions(min_year, max_year):
    """Pre-compute all distributions"""
    distributions = {}
    print("Pre-computing demographic distributions...")
    for year in range(min_year, max_year + 1):
        distributions[year] = query_llm(year)
    return distributions


def generate_dob(admit_year):
    """
    Generate a realistic DOB based on the admission year,
    skewed so 90% of students are enrolling somewhere in the 18-24 age range.
    Returns date in YYYY-MM-DD format.
    """
    if random.random() < 0.90:
        # About 90% of students are between 18 and 24 at admission time
        age_at_admission = random.randint(18, 24)
        birth_year = admit_year - age_at_admission
    else:
        if random.random() < 0.5:
            birth_year = admit_year - random.randint(16, 17)  # 16 or 17
        else:
            birth_year = admit_year - random.randint(25, 45)  # 25 to 45

    start_date = date(birth_year, 1, 1)
    end_date = date(birth_year, 12, 31)
    return fake.date_between(start_date=start_date, end_date=end_date).strftime(
        "%Y-%m-%d"
    )


def get_next_term(year, sem):
    """
    Move to the next semester.
    If sem == 3, move to the next year and reset sem to 1.
    Otherwise, increment sem by 1.
    """
    if sem == 3:
        return year + 1, 1
    else:
        return year, sem + 1


def generate_term_codes(admit_year, admit_sem):
    """
    Generate a list of TERM_CODE_EFF values from admission to completion.
    Typical completion is 6 semesters, but a portion may go up to 9 semesters.
    """
    # Weighted distribution:
    # ~60% -> 6 semesters
    # ~25% -> 7 semesters
    # ~10% -> 8 semesters
    # ~5%  -> 9 semesters
    r = random.random()
    if r < 0.60:
        total_semesters = 6
    elif r < 0.85:
        total_semesters = 7
    elif r < 0.95:
        total_semesters = 8
    else:
        total_semesters = 9

    term_codes = []
    year = admit_year
    sem = admit_sem
    for _ in range(total_semesters):
        term_codes.append(f"{year}{sem}0")  # (YYYY)(Sem)0, e.g. 201610
        year, sem = get_next_term(year, sem)
    return term_codes


def validate_distributions(distributions: Dict[str, Dict[str, float]]) -> bool:
    """
    Validate that both gender and religion distributions are properly formatted and sum to 1.0
    """
    if not isinstance(distributions, dict):
        return False

    # Validate gender distribution
    gender_dist = distributions.get("gender", {})
    if set(gender_dist.keys()) != {"F", "M"}:
        return False
    if not all(isinstance(v, (int, float)) for v in gender_dist.values()):
        return False
    if not abs(sum(gender_dist.values()) - 1.0) < 0.001:
        return False

    # Validate religion distribution
    religion_dist = distributions.get("religion", {})
    if not all(isinstance(v, (int, float)) for v in religion_dist.values()):
        return False
    if not abs(sum(religion_dist.values()) - 1.0) < 0.001:
        return False

    # Validate marital status distribution
    marital_dist = distributions.get("marital_status", {})
    expected_marital_statuses = {
        "Single",
        "Married",
        "Common-Law",
        "Separated",
        "Divorced",
        "Other",
    }
    if set(marital_dist.keys()) != expected_marital_statuses:
        return False
    if not all(isinstance(v, (int, float)) for v in marital_dist.values()):
        return False
    if not abs(sum(marital_dist.values()) - 1.0) < 0.001:
        return False

    return True


def load_degree_probabilities():
    """Load historical degree enrollment data and convert to probabilities with random variation"""
    df = pd.read_csv(r"real stats/degree_enrollment.csv")
    
    # Convert raw counts to base probabilities for each year
    prob_data = {}
    for _, row in df.iterrows():
        year = row["year"]
        total = row["total"]
        
        # Calculate base probabilities
        base_probs = {
            "CS-MAJO": row["CS-MAJO"] / total,
            "CS-SPEC": row["CS-SPEC"] / total,
            "CS-MANA": row["CS-MANA"] / total,
            "IT-MAJO": row["IT-MAJO"] / total,
            "IT-SPEC": row["IT-SPEC"] / total,
        }
        
        # Add random variation within ±15% of original probability
        varied_probs = {}
        for degree, prob in base_probs.items():
            variation = prob * random.uniform(-0.15, 0.15)
            varied_probs[degree] = max(0.01, prob + variation)  # Ensure minimum 1%
            
        # Normalize to ensure probabilities sum to 1
        total_prob = sum(varied_probs.values())
        varied_probs = {k: v/total_prob for k, v in varied_probs.items()}
        
        prob_data[year] = varied_probs
        
    return prob_data


def get_degree_probabilities(year: int, degree_probs: dict) -> tuple:
    """Get degree probabilities for a specific year, using nearest available year if exact match not found"""
    available_years = sorted(degree_probs.keys())

    if year in degree_probs:
        probs = degree_probs[year]
    else:
        # Find nearest year
        nearest_year = min(available_years, key=lambda x: abs(x - year))
        probs = degree_probs[nearest_year]

    degrees = list(probs.keys())
    probabilities = list(probs.values())
    return degrees, probabilities


def main():
    random.seed(42)

    # Prompt user for parameters
    num_students = int(input("Enter the number of students to generate: "))
    min_year = int(input("Enter the minimum year: "))
    max_year = int(input("Enter the maximum year: "))
    output_file = input("Enter output CSV filename: ")

    print("Loading and caching location data...")
    worldcities_df, city_cache, valid_nations, nation_probs = prepare_location_data()

    # Load degree probabilities
    print("Loading degree enrollment data...")
    degree_probs = load_degree_probabilities()

    # Pre-compute all needed distributions
    distributions = precompute_distributions(min_year, max_year)

    # Pre-compute student IDs
    student_ids = generate_student_ids(num_students)

    print("Generating student records...")
    with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "STUDENT_ID",
                "TERM_CODE_EFF",
                "TERM_CODE_ADMIT",
                "DATE_OF_BIRTH",
                "CITY",
                "STATE",
                "NATION",
                "GENDER",
                "RELIGION",
                "MARITAL_STATUS",
                "FACULTY_CODE",
                "STU_DEGREE_CODE",
            ]
        )

        for sid in tqdm(student_ids, desc="Generating records"):
            admit_year = random.randint(min_year, max_year)
            term_code_admit = f"{admit_year}10"
            admit_sem = 1
            term_codes = generate_term_codes(admit_year, admit_sem)
            dob = generate_dob(admit_year)

            # Use cached distributions
            year_dist = distributions[admit_year]
            marital_status = random.choices(
                list(year_dist["marital_status"].keys()),
                weights=list(year_dist["marital_status"].values()),
                k=1,
            )[0]

            faculty_code = "FSA" if int(term_codes[0][:4]) <= 2012 else "FST"

            # Use historical probabilities for degree selection
            degrees, probabilities = get_degree_probabilities(admit_year, degree_probs)
            degree_code = random.choices(degrees, weights=probabilities, k=1)[0]

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

            # Use probability-weighted nation selection
            chosen_nation = random.choices(valid_nations, weights=nation_probs, k=1)[0]
            nation_cities, weights = city_cache[chosen_nation]

            if weights is not None:
                selected_city = nation_cities.sample(n=1, weights=weights).iloc[0]
            else:
                selected_city = nation_cities.sample(1).iloc[0]

            city = selected_city["city_ascii"]
            state = selected_city["admin_name"]
            nation_value = selected_city["country"]

            # Write a record for each semester
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


print("Generating data...")

if __name__ == "__main__":
    main()
