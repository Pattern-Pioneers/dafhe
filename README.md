# DAFHE: Data Analytics for Higher Education

This research project sample provides tools and scripts for generating and analyzing synthetic student demographic datasets for higher education research to uncover trends.

## Features

- Generate synthetic student data .csv with realistic distributions (see `dataset_generator.py`)
- Analyze trends
- Visualize data using ggplot2 and other libraries

## Requirements

- Python 3.7+
- R 4.4.3+
- [OPENROUTER_API_KEY](https://openrouter.ai) (BYOK) in `.env`

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. (Recommended but optional) Place seed data to help ground generator in `/real stats`

3. Generate synthetic data:

   ```bash
   python "dataset_generator.py"
   ```

## References

- World cities dataset <https://simplemaps.com/data/world-cities>
- UWI Reports and Digests <https://sta.uwi.edu/news/reports/default.asp>

### Inspiration for this project methodology

- <https://github.com/lamethods>

---
