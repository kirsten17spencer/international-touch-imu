P1: Match demands of international touch: women's and mixed Trans-Tasman test matches
Science and Medicine in Football, submitted 2026.

P1_supplementary_analysis_code.py
  Input : P1_match_level_deidentified.csv (52 rows, one per player-match; columns
          player_id, squad, day, sex, participation_min, distance_m, work_rate_m_min,
          top_speed_m_s, hid_m, hid_per_min, sprint_distance_m, sprint_count,
          intense_speed_changes, isc_per_min, ball_touches, turns, turns_per_min,
          turn_intensity_au). Available from the author on request; not distributed.
  Output: P1_supplementary_statistical_output.txt (complete unedited model output),
          P1_supplementary_results_table.csv.
  Method: for each squad, eight linear mixed models (outcome ~ day, player random
          intercept, maximum likelihood), two-degree-of-freedom Wald test for day,
          Benjamini-Hochberg adjustment across the eight outcomes within squad,
          day contrasts with 95% confidence intervals, intraclass correlations.
