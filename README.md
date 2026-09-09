[![DOI](https://zenodo.org/badge/1359688320.svg)](https://doi.org/10.5281/zenodo.22598629)

# international-touch-imu

Analysis code for the programme of papers on international touch match demands and stride mechanics measured with foot-mounted inertial measurement units (PlayerMaker) during the 2025 Trans-Tasman Test Series and associated national sessions.

Author: Kirsten Spencer, Auckland University of Technology, Auckland, New Zealand.

## What is here

| Folder | Paper | What the code reproduces |
|---|---|---|
| `P1_match_demands/` | Match demands of international touch: women's and mixed Trans-Tasman test matches (Science and Medicine in Football, submitted 2026) | Every descriptive value, linear mixed-model estimate, 95% confidence interval, Benjamini-Hochberg adjusted p value and intraclass correlation reported in the paper (16 models: eight outcomes in each of two squads). Release v1.0-P1. |
| `P2_format_comparison/` | Same series, different formats: women's versus mixed-sex international touch match demands (Science and Medicine in Football, submitted 2026) | Every descriptive value, the three model families (24 linear mixed models with 95% confidence intervals, Benjamini-Hochberg adjusted p values and intraclass correlations), the load-share percentages, per-capita indices, relative tempo index and the within-player crossover values reported in the paper. Release v1.0-P2. |

Later releases add the code for the remaining papers (trial-versus-test fidelity, turning demands, peak demands, stride asymmetry thresholds, day-to-day asymmetry).

## Data

No data are stored in this repository. Player performance records were accessed under a data-sharing agreement between Auckland University of Technology and the equipment manufacturer that restricts dissemination. The de-identified datasets the scripts read (one row per player-match; players identified by device number only) are available from the author on reasonable request under that agreement. Each script states the file it expects and the columns it uses.

## Running the scripts

    pip install -r requirements.txt
    python P1_match_demands/P1_supplementary_analysis_code.py path/to/P1_match_level_deidentified.csv
    python P2_format_comparison/P2_supplementary_analysis_code.py path/to/P2_match_level_deidentified.csv path/to/P2_november_trials_deidentified.csv

Each script writes its complete, unedited model output as a text file and its results tables as CSV files; the README.txt in each folder lists them. The outputs as submitted with the papers were produced under Python 3.11.15, pandas 3.0.2, numpy 2.4.4, scipy 1.17.1 and statsmodels 0.15.0; the same values were obtained under Python 3.14 and statsmodels 0.14.6.

## Ethics

Auckland University of Technology Ethics Committee approval 23/342 (Experiences in the Touch NZ community), 19 February 2024 to 19 February 2027. Players appear in all outputs as de-identified device numbers.

## Citing

Please cite the archived release (Zenodo DOI on the release page and in `CITATION.cff`) together with the paper it accompanies.

## Licence

MIT (see `LICENSE`).
