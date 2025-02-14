import csv
import random
from datetime import date
import faker
import pandas as pd


def generate_student_ids(num_students):
    """
    Generate a set of unique 9-digit IDs in the range 318000000 to 318999999.
    """
    possible_ids = list(range(318000000, 318999999))
    random.shuffle(possible_ids)
    return possible_ids[:num_students]


fake = faker.Faker()


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


def main():
    random.seed(42)  # For reproducibility
    num_students = 10000  # Adjust the number of students as needed
    student_ids = generate_student_ids(num_students)

    # Load world cities data and compute selection lists
    worldcities_df = pd.read_csv(r"real stats/worldcities.csv")
    # Convert population to numeric, replacing any NaN with 0
    worldcities_df["population"] = pd.to_numeric(
        worldcities_df["population"], errors="coerce"
    ).fillna(0)

    # No longer need these lists since we'll use weighted selection
    # cities = worldcities_df["city"].unique().tolist()
    # states = worldcities_df["admin_name"].unique().tolist()
    nations = worldcities_df["iso3"].unique().tolist()
    countries = worldcities_df["country"].unique().tolist()

    # Load country stats for skewing nationality
    country_stats = pd.read_csv(r"real stats\\country_stats.csv")
    country_stats = country_stats[country_stats["iso3"] != "TTO"]
    country_stats = country_stats[country_stats["iso3"] != "UG_TOTAL"]

    # Define admission year range and other attributes
    min_year_admit = 2000
    max_year_admit = 2024
    marital_status_options = [
        "Single",
        "Married",
        "Common-Law",
        "Separated",
        "Divorced",
        "Other",
    ]
    gender_options = (
        ["F"] * 60 + ["M"] * 40
    )  # TODO: Adjust gender skew to match the floating window approach used in the graduation script
    religion_options = [
        "Other",
        "None",
        "Christian",
        "Seventh-Day Adventist",
        "Pentecostal",
        "Roman Catholic",
        "Muslim",
        "Hindu",
        "Rastafarian",
        "Jehovah's Witness",
        "Presbyterian",
        "Methodist",
        "Anglican",
        "Baptist",
    ]
    degree_codes = ["CS-MAJO", "CS-SPEC", "CS-MANA", "IT-MAJO", "IT-SPEC"]

    with open("25enrollv999.csv", mode="w", newline="", encoding="utf-8") as csv_file:
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

        for sid in student_ids:
            admit_year = random.randint(min_year_admit, max_year_admit)
            term_code_admit = f"{admit_year}10"
            admit_sem = 1
            term_codes = generate_term_codes(admit_year, admit_sem)
            dob = generate_dob(admit_year)
            gender = random.choice(gender_options)
            religion = random.choice(religion_options)
            marital_status = random.choice(marital_status_options)
            faculty_code = "FSA" if int(term_codes[0][:4]) <= 2012 else "FST"
            degree_code = random.choice(degree_codes)

            # Select nation based on real-world skew:
            r = random.random()
            chosen_nation = random.choice(nations)
            nation_cities = worldcities_df[worldcities_df["iso3"] == chosen_nation]
            if not nation_cities.empty:
                # Use population-weighted selection
                total_pop = nation_cities["population"].sum()
                if total_pop > 0:  # Only use weights if we have population data
                    weights = nation_cities["population"] / total_pop
                    selected_city = nation_cities.sample(n=1, weights=weights).iloc[0]
                else:
                    selected_city = nation_cities.sample(1).iloc[0]

                city = selected_city["city_ascii"]  # Use ASCII version of city name
                state = selected_city["admin_name"]
                nation_value = selected_city["country"]
            else:
                # Fallback to global population-weighted selection if no cities found for nation
                total_pop = worldcities_df["population"].sum()
                if total_pop > 0:
                    weights = worldcities_df["population"] / total_pop
                    selected_city = worldcities_df.sample(n=1, weights=weights).iloc[0]
                    city = selected_city["city_ascii"]
                    state = selected_city["admin_name"]
                    nation_value = selected_city["country"]
                else:
                    # Fallback if no population data available
                    selected_city = worldcities_df.sample(1).iloc[0]
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


if __name__ == "__main__":
    main()
