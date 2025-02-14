from tqdm import tqdm
import csv
import random
from datetime import date
import faker
import pandas as pd
from typing import Dict
import json
from together import Together
import os

fake = faker.Faker()

# In case LLM response looks shaky formatwise
def clean_llm_response(response_text: str) -> str:
    """Clean LLM response by removing markdown code blocks and other formatting"""
    # Remove markdown code blocks
    response_text = response_text.strip("`")
    while response_text.startswith("`"):
        response_text = response_text[1:]
    while response_text.endswith("`"):
        response_text = response_text[:-1]
    return response_text.strip()


def query_llm(year: int) -> Dict[str, Dict[str, float]]:
    """
    Get realistic distributions for gender and religion in CS/IT programs for a given year using Together AI.
    Returns a dictionary containing both gender and religion distribution dictionaries.
    """
    api_key = os.getenv("TOGETHER_API_KEY_dafhe")
    if not api_key:
        print("Error: TOGETHER_API_KEY_dafhe environment variable not set")
        return {
            "gender": {"F": 0.6, "M": 0.4},
            "religion": {
                "Christian": 0.35,
                "Hindu": 0.25,
                "Muslim": 0.15,
                "None": 0.10,
                "Other": 0.15,
            },
        }

    client = Together(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": "You are a data analysis assistant that provides demographic statistics for university enrollment in Trinidad and Tobago. Return only raw JSON without any markdown formatting or code blocks.",
        },
        {
            "role": "user",
            "content": f"""For the year {year}, provide both gender and religion distributions for university CS/IT programs in Trinidad and Tobago as a JSON object.
            Include only major religions and None/Other categories.
            Return only a raw JSON object like this, without any markdown formatting:
            {{
                "gender": {{"F": 0.25, "M": 0.75}},
                "religion": {{
                    "Christian": 0.35,
                    "Hindu": 0.25,
                    "Muslim": 0.15,
                    "None": 0.10,
                    "Other": 0.15
                }}
            }}
            Values within each category must sum to 1.0
            """,
        },
    ]

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.6,  # TODO adjust temperatures
            top_p=0.9,
            stream=False,
        )

        print(f"\nLLM Response for year {year}:")
        response_text = response.choices[0].message.content.strip()
        print("Raw response:", response_text)
        print("-" * 50)

        try:
            # Clean response before parsing
            cleaned_response = clean_llm_response(response_text)
            distributions = json.loads(cleaned_response)
            if validate_distributions(distributions):
                return distributions
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")

    except Exception as e:
        print(f"Warning: Together API call failed for year {year}")
        print(f"Error: {str(e)}")

    # Return default distributions if anything fails
    return {
        "gender": {"F": 0.6, "M": 0.4},
        "religion": {
            "Christian": 0.35,
            "Hindu": 0.25,
            "Muslim": 0.15,
            "None": 0.10,
            "Other": 0.15,
        },
    }


def generate_student_ids(num_students):
    """
    Generate a set of unique 9-digit IDs in the range 318000000 to 318999999.
    """
    possible_ids = list(range(318000000, 318999999))
    random.shuffle(possible_ids)
    return possible_ids[:num_students]


def prepare_location_data():
    """Cache location data processing"""
    worldcities_df = pd.read_csv(r"real stats/worldcities.csv")
    worldcities_df["population"] = pd.to_numeric(
        worldcities_df["population"], errors="coerce"
    ).fillna(0)

    country_stats = pd.read_csv(r"real stats\\country_stats.csv")
    country_stats = country_stats[
        ~country_stats["iso3"].isin(["TTO", "UG_TOTAL", "N/A"])
    ]

    valid_nations = country_stats["iso3"].unique()
    worldcities_df = worldcities_df[worldcities_df["iso3"].isin(valid_nations)]

    # Pre-compute city selections per country
    city_cache = {}
    for nation in worldcities_df["iso3"].unique():
        nation_cities = worldcities_df[worldcities_df["iso3"] == nation]
        total_pop = nation_cities["population"].sum()

        if total_pop > 0:
            weights = nation_cities["population"] / total_pop
            city_cache[nation] = (nation_cities, weights)
        else:
            city_cache[nation] = (nation_cities, None)

    return worldcities_df, city_cache


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

    return True


def main():
    random.seed(42)
    num_students = 10000

    print("Loading and caching location data...")
    worldcities_df, city_cache = prepare_location_data()

    # Pre-compute all needed distributions
    distributions = precompute_distributions(2000, 2024)

    # Pre-compute student IDs
    student_ids = generate_student_ids(num_students)

    print("Generating student records...")
    with open("25enrollvLLM.csv", mode="w", newline="", encoding="utf-8") as csv_file:
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
            admit_year = random.randint(2000, 2024)
            term_code_admit = f"{admit_year}10"
            admit_sem = 1
            term_codes = generate_term_codes(admit_year, admit_sem)
            dob = generate_dob(admit_year)
            marital_status = random.choice(
                [
                    "Single",
                    "Married",
                    "Common-Law",
                    "Separated",
                    "Divorced",
                    "Other",
                ]
            )
            faculty_code = "FSA" if int(term_codes[0][:4]) <= 2012 else "FST"
            degree_code = random.choice(
                ["CS-MAJO", "CS-SPEC", "CS-MANA", "IT-MAJO", "IT-SPEC"]
            )

            # Use cached distributions
            year_dist = distributions[admit_year]
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

            # Use cached city data
            chosen_nation = random.choice(list(city_cache.keys()))
            nation_cities, weights = city_cache[chosen_nation]

            if weights is not None:
                selected_city = nation_cities.sample(n=1, weights=weights).iloc[0]
            else:
                selected_city = nation_cities.sample(1).iloc[0]

            city = selected_city["city_ascii"]  # Use ASCII version of city name
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
print("Generated data saved to 25enrollvLLM.csv")

if __name__ == "__main__":
    main()
