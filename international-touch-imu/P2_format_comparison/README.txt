P2: Same series, different formats: women's versus mixed-sex international touch match demands
Science and Medicine in Football, submitted 2026. Companion to P1 (same series, same devices).

P2_supplementary_analysis_code.py
  Inputs: P2_match_level_deidentified.csv (52 rows, one per player-match, April 2025 tests;
          columns player_id, sex, analysis_group, format, day, participation_min, distance_m,
          work_rate_m_min, top_speed_m_s, hid_m, hid_per_min, sprint_distance_m, sprint_count,
          intense_speed_changes, isc_per_min, ball_touches, turns, turns_per_min,
          turn_intensity_au) and P2_november_trials_deidentified.csv (12 rows: the six
          crossover players' two November 2024 trial games each; columns player_id, sex,
          format, game, then the same outcome columns). Available from the author on
          request; not distributed.
  Output: P2_supplementary_statistical_output.txt (complete unedited model output),
          P2_supplementary_results_table.csv, P2_supplementary_load_share.csv,
          P2_supplementary_crossover.csv.
  Method: three pre-specified model families, eight linear mixed models each (outcome ~
          group term + day, player random intercept, maximum likelihood): A format
          (Mixed minus Women squad, 52 observations), B women only (Mixed women minus
          Women squad, 42), C sex within the mixed squad (men minus women, 20);
          Benjamini-Hochberg adjustment across the eight outcomes within each family;
          intraclass correlations; load-share percentages, per-capita contribution
          indices, relative tempo index and high-intensity action density within the
          mixed squad; the November-to-April crossover reported descriptively only.
