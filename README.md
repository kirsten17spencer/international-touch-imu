# international-touch-imu

Analysis code for the programme of papers on international touch match demands and stride mechanics measured with foot-mounted inertial measurement units (PlayerMaker) during the 2025 Trans-Tasman Test Series and associated national sessions.

Author: Kirsten Spencer, Auckland University of Technology, Auckland, New Zealand.

## What is here

| Folder | Paper | What the code reproduces |
|---|---|---|
| `P1_match_demands/` | Match demands of international touch: women's and mixed Trans-Tasman test matches (Science and Medicine in Football, submitted 2026) | Every descriptive value, linear mixed-model estimate, 95% confidence interval, Benjamini-Hochberg adjusted p value and intraclass correlation reported in the paper (16 models: eight outcomes in each of two squads). |

Later releases add the code for the companion papers (format comparison, trial-versus-test fidelity, turning demands, peak demands, stride asymmetry thresholds, day-to-day asymmetry).

## Data

No data are stored in this repository. Player performance records were accessed under a data-sharing agreement between Auckland University of Technology and the equipment manufacturer that restricts dissemination. The de-identified match-level dataset that the P1 script reads (one row per player-match; players identified by device number only) is available from the author on reasonable request under that agreement. Each script states the file it expects and the columns it uses.

## Running the P1 script

```
pip install -r requirements.txt
python P1_match_demands/P1_supplementary_analysis_code.py path/to/P1_match_level_deidentified.csv
```

It writes `P1_supplementary_statistical_output.txt` (the complete, unedited model output) and `P1_supplementary_results_table.csv`. The output as submitted with the paper was produced under Python 3.11.15, pandas 3.0.2, numpy 2.4.4, scipy 1.17.1 and statsmodels 0.15.0; the same values were obtained under Python 3.14 and statsmodels 0.14.6.

## Ethics

Auckland University of Technology Ethics Committee approval 23/342 (Experiences in the Touch NZ community), 19 February 2024 to 19 February 2027. Players appear in all outputs as de-identified device numbers.

## Citing

Please cite the archived release (Zenodo DOI on the release page and in `kirsten17spencer`) together with the paper it accompanies.

## Licence

MIT (see `LICENSE`).
