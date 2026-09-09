#!/usr/bin/env python3
"""
Supplementary analysis code for:
  Match demands of international touch: women's and mixed Trans-Tasman test matches
  (Science and Medicine in Football submission, 2026)

What this script does
  1. Reads the de-identified match-level dataset (one row per player-match; players are
     device numbers; 52 rows: mixed squad 7/7/6 and women's squad 4/14/14 on days 1-3).
  2. Reports descriptive values (mean +/- SD) by squad and by squad x day.
  3. Fits, separately within each squad, a linear mixed model for each of the eight
     modelled outcomes: outcome ~ day (categorical fixed effect, day 1 reference) with a
     random intercept for player, maximum-likelihood fit (statsmodels MixedLM).
  4. Tests the joint (omnibus) day effect with a two-degree-of-freedom Wald test, applies
     Benjamini-Hochberg adjustment across the eight outcomes within each squad, reports the
     day 2 v day 1 and day 3 v day 1 coefficients with 95% confidence intervals and nominal
     p values, the day 3 v day 2 linear contrast, and the intraclass correlation
     (player-intercept variance / total variance).
  5. Writes the complete, unedited model summaries to P1_supplementary_statistical_output.txt
     and the results table to P1_supplementary_results_table.csv.

Software (this run): Python 3.11.15, pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, statsmodels 0.15.0. The
same values were obtained in the project environment (Python 3.14, statsmodels 0.14.6, scipy 1.18)
on 23 August 2026; the versions in use are printed at the top of the output file.
Run:  python P1_supplementary_analysis_code.py  [path/to/P1_match_level_deidentified.csv]
The dataset is available from the corresponding author on reasonable request under the
governing data-sharing agreement.
"""
import sys, io, platform
import numpy as np, pandas as pd
import statsmodels, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

CSV = sys.argv[1] if len(sys.argv) > 1 else "P1_match_level_deidentified.csv"
OUTCOMES = [("distance_m", "Distance (m)"), ("work_rate_m_min", "Work rate (m/min)"), ("top_speed_m_s", "Top speed (m/s)"),
            ("hid_per_min", "High-intensity distance rate (m/min)"), ("sprint_count", "Sprint count"),
            ("isc_per_min", "Intense speed changes per min"), ("turns_per_min", "Turns per min"), ("turn_intensity_au", "Total turn intensity (AU)")]
DESCRIPTIVE = OUTCOMES + [("participation_min", "Participation (min)"), ("hid_m", "High-intensity distance (m)"), ("sprint_distance_m", "Sprint distance (m)"),
                          ("intense_speed_changes", "Intense speed changes (count)"), ("ball_touches", "Ball touches"), ("turns", "Turns")]

d = pd.read_csv(CSV, dtype={"player_id": str})
d["day"] = d["day"].astype(int)
log = io.StringIO()
def P(*a):
    print(*a); print(*a, file=log)

P("=" * 100); P("P1 supplementary statistical output (unedited)")
P(f"Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | statsmodels {statsmodels.__version__}")
P(f"Dataset: {CSV} | rows {len(d)} | players {d.player_id.nunique()}"); P(d.groupby(['squad', 'day']).size().to_string()); P("=" * 100)

P("\nDESCRIPTIVES: mean +/- SD by squad (all player-match observations)")
for col, lab in DESCRIPTIVE:
    g = d.groupby("squad")[col].agg(["mean", "std", "count"])
    P(f"{lab:38s} " + " | ".join(f"{s}: {r['mean']:.2f} +/- {r['std']:.2f} (n={int(r['count'])})" for s, r in g.iterrows()))
P("\nDESCRIPTIVES: mean +/- SD by squad x day")
for col, lab in DESCRIPTIVE:
    g = d.groupby(["squad", "day"])[col].agg(["mean", "std", "count"])
    P(f"{lab:38s} " + " | ".join(f"{s} d{int(dd)}: {r['mean']:.2f} +/- {r['std']:.2f} ({int(r['count'])})" for (s, dd), r in g.iterrows()))

results = []
for squad in ["Mixed", "Women"]:
    ds = d[d.squad == squad].copy()
    P("\n" + "#" * 100); P(f"# SQUAD: {squad}  (players {ds.player_id.nunique()}, player-matches {len(ds)})"); P("#" * 100)
    pj = []
    for col, lab in OUTCOMES:
        m = smf.mixedlm(f"{col} ~ C(day)", ds, groups=ds["player_id"]).fit(reml=False)
        P("\n" + "-" * 100); P(f"OUTCOME: {lab}   [{col} ~ C(day), random intercept player_id, ML]"); P("-" * 100)
        P(m.summary().as_text())
        names = list(m.fe_params.index)
        i2, i3 = names.index("C(day)[T.2]"), names.index("C(day)[T.3]")
        b = m.fe_params.values; V = m.cov_params().values[:len(b), :len(b)]
        # joint 2-df Wald test on the two day coefficients
        L = np.zeros((2, len(b))); L[0, i2] = 1; L[1, i3] = 1
        w = float((L @ b).T @ np.linalg.inv(L @ V @ L.T) @ (L @ b)); p_joint = float(1 - stats.chi2.cdf(w, 2))
        # contrasts
        def con(c):
            est = float(c @ b); se = float(np.sqrt(c @ V @ c)); z = est / se
            return est, est - 1.96 * se, est + 1.96 * se, float(2 * (1 - stats.norm.cdf(abs(z))))
        c21 = np.zeros(len(b)); c21[i2] = 1
        c31 = np.zeros(len(b)); c31[i3] = 1
        c32 = np.zeros(len(b)); c32[i3] = 1; c32[i2] = -1
        e21, e31, e32 = con(c21), con(c31), con(c32)
        var_p = float(m.cov_re.iloc[0, 0]); var_e = float(m.scale); icc = var_p / (var_p + var_e)
        P(f"\nJoint day effect: Wald chi2(2) = {w:.4f}, p = {p_joint:.4f}")
        P(f"Day 2 v day 1: {e21[0]:+.4f} [95% CI {e21[1]:.4f}, {e21[2]:.4f}], p = {e21[3]:.4f}")
        P(f"Day 3 v day 1: {e31[0]:+.4f} [95% CI {e31[1]:.4f}, {e31[2]:.4f}], p = {e31[3]:.4f}")
        P(f"Day 3 v day 2: {e32[0]:+.4f} [95% CI {e32[1]:.4f}, {e32[2]:.4f}], p = {e32[3]:.4f}")
        P(f"Variance components: player intercept {var_p:.4f}, residual {var_e:.4f}; ICC = {icc:.4f}")
        P(f"Converged: {m.converged}; log-likelihood {m.llf:.4f}; AIC {m.aic:.4f}; BIC {m.bic:.4f}")
        pj.append(p_joint)
        results.append(dict(squad=squad, outcome=lab, wald_chi2=w, p_joint=p_joint, icc=icc,
                            d2_v_d1=e21[0], d2_v_d1_lo=e21[1], d2_v_d1_hi=e21[2], d2_v_d1_p=e21[3],
                            d3_v_d1=e31[0], d3_v_d1_lo=e31[1], d3_v_d1_hi=e31[2], d3_v_d1_p=e31[3],
                            d3_v_d2=e32[0], d3_v_d2_lo=e32[1], d3_v_d2_hi=e32[2], d3_v_d2_p=e32[3], converged=m.converged))
    p_bh = multipletests(pj, method="fdr_bh")[1]
    P("\nBenjamini-Hochberg adjustment across the eight omnibus day tests within the squad:")
    for (col, lab), p0, p1 in zip(OUTCOMES, pj, p_bh):
        P(f"  {lab:38s} p = {p0:.4f}  p_BH = {p1:.4f}")
        for r in results:
            if r["squad"] == squad and r["outcome"] == lab: r["p_bh"] = float(p1)

R = pd.DataFrame(results)
R.to_csv("P1_supplementary_results_table.csv", index=False)
open("P1_supplementary_statistical_output.txt", "w").write(log.getvalue())
print("\nWritten: P1_supplementary_statistical_output.txt, P1_supplementary_results_table.csv")
