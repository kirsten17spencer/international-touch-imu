#!/usr/bin/env python3
"""
Supplementary analysis code for:
  Same series, different formats: women's versus mixed-sex international touch match demands
  (Science and Medicine in Football submission, 2026)

What this script does
  1. Reads the de-identified April match-level dataset (one row per player-match; players are
     device numbers; 52 rows in three analysis groups: Mixed men 6 players / 10 observations,
     Mixed women 5 / 10, Women squad 16 / 32) and the de-identified November trial-game rows for
     the six crossover players (12 rows).
  2. Reports descriptive values (mean +/- SD) by analysis group (Table 1).
  3. Fits the three pre-specified model families, one linear mixed model per outcome for each of
     the eight modelled outcomes: outcome ~ group term + day (both categorical fixed effects) with
     a random intercept for player, maximum-likelihood fit (statsmodels MixedLM).
       Family A: format, Mixed squad minus Women squad, all 52 observations.
       Family B: women only, Mixed women minus Women squad, 42 observations.
       Family C: sex within the mixed squad, men minus women, 20 observations.
     Reports the group coefficient with 95% confidence interval and nominal p value (Wald z),
     Benjamini-Hochberg adjustment across the eight outcomes within each family, and the
     intraclass correlation (player-intercept variance / total variance) (Table 2).
  4. Computes the load-share metrics within the mixed squad: men's and women's shares of the
     instrumented squad totals per test and pooled, the per-capita contribution index (share
     divided by observation share), the relative tempo index (player work rate divided by the
     mean work rate of the instrumented players in the same match), and the high-intensity
     action density (Table 3 and text).
  5. Reports the within-player crossover descriptively: per-player means over the two November
     trial games and over the player's April mixed tests (Table 4). The crossover is not modelled.
  6. Writes the complete, unedited model summaries to P2_supplementary_statistical_output.txt and
     the results table to P2_supplementary_results_table.csv.

Software (this run): printed at the top of the output file. The same values were obtained in the
project environment (Python 3.14, statsmodels 0.14.6, scipy 1.18) on 23 August 2026.
Run:  python P2_supplementary_analysis_code.py [P2_match_level_deidentified.csv] [P2_november_trials_deidentified.csv]
Both datasets are available from the corresponding author on reasonable request under the
governing data-sharing agreement.
"""
import sys, io, platform
import numpy as np, pandas as pd
import scipy, statsmodels, statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from decimal import Decimal, ROUND_HALF_UP

def rhu(x, dp):
    """Round half up for display (Python's format rounds half to even)."""
    return str(Decimal(str(float(x))).quantize(Decimal(1) if dp == 0 else Decimal('1.' + '0' * dp), rounding=ROUND_HALF_UP))

CSV = sys.argv[1] if len(sys.argv) > 1 else "P2_match_level_deidentified.csv"
NOV = sys.argv[2] if len(sys.argv) > 2 else "P2_november_trials_deidentified.csv"
OUTCOMES = [("distance_m", "Distance (m)"), ("work_rate_m_min", "Work rate (m/min)"), ("top_speed_m_s", "Top speed (m/s)"),
            ("hid_per_min", "HID per min (m/min)"), ("sprint_count", "Sprint count"), ("isc_per_min", "Intense speed changes per min"),
            ("turns_per_min", "Turns per min"), ("turn_intensity_au", "Turn intensity (AU)")]
DESCRIPTIVE = [("participation_min", "Participation (min)"), ("distance_m", "Distance (m)"), ("work_rate_m_min", "Work rate (m/min)"),
               ("top_speed_m_s", "Top speed (m/s)"), ("hid_per_min", "HID per min (m/min)"), ("sprint_distance_m", "Sprint distance (m)"),
               ("sprint_count", "Sprint count"), ("isc_per_min", "Intense speed changes per min"), ("ball_touches", "Ball touches"),
               ("turns", "Turns"), ("turns_per_min", "Turns per min"), ("turn_intensity_au", "Turn intensity (AU)")]
GROUPS = ["Mixed men", "Mixed women", "Women squad"]

d = pd.read_csv(CSV, dtype={"player_id": str}); d["day"] = d["day"].astype(int)
nov = pd.read_csv(NOV, dtype={"player_id": str})
log = io.StringIO()
def P(*a):
    print(*a); print(*a, file=log)

P("=" * 100); P("P2 supplementary statistical output (unedited)")
P(f"Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | scipy {scipy.__version__} | statsmodels {statsmodels.__version__}")
P(f"April dataset: {CSV} | rows {len(d)} | players {d.player_id.nunique()}")
P(d.groupby(["analysis_group", "day"]).size().to_string())
P(f"November dataset: {NOV} | rows {len(nov)} | players {nov.player_id.nunique()}"); P("=" * 100)

# ---------------------------------------------------------------- 1. descriptives (Table 1)
P("\nTABLE 1. Match-level descriptives by analysis group, mean +/- SD (n players / n observations)")
for g in GROUPS:
    s = d[d.analysis_group == g]; P(f"  {g}: players {s.player_id.nunique()}, observations {len(s)}")
for col, lab in DESCRIPTIVE:
    P(f"{lab:32s} " + " | ".join(f"{g}: {d[d.analysis_group == g][col].mean():.2f} +/- {d[d.analysis_group == g][col].std():.2f}" for g in GROUPS))

# ---------------------------------------------------------------- 2. model families (Table 2)
def family(name, ds, term, ref, label):
    """Fit outcome ~ C(term, reference ref) + C(day) + (1 | player) by ML for each outcome; return result rows."""
    P("\n" + "#" * 100); P(f"# {name}: {label}  (players {ds.player_id.nunique()}, observations {len(ds)})"); P("#" * 100)
    other = [v for v in sorted(ds[term].unique()) if v != ref][0]
    P(f"Coding: {term} reference level = '{ref}'; the coefficient is '{other}' minus '{ref}'; day 1 reference.")
    rows, pvals = [], []
    for col, lab in OUTCOMES:
        f = f"{col} ~ C({term}, Treatment('{ref}')) + C(day)"
        m = smf.mixedlm(f, ds, groups=ds["player_id"]).fit(reml=False)
        P("\n" + "-" * 100); P(f"OUTCOME: {lab}   [{f}, random intercept player_id, ML]"); P("-" * 100)
        P(m.summary().as_text())
        gname = [n for n in m.fe_params.index if n.startswith(f"C({term}")][0]
        b = float(m.fe_params[gname]); se = float(m.bse_fe[gname]); z = b / se
        p = float(2 * (1 - stats.norm.cdf(abs(z)))); zc = stats.norm.ppf(0.975); lo, hi = b - zc * se, b + zc * se
        var_p = float(m.cov_re.iloc[0, 0]); var_e = float(m.scale); icc = var_p / (var_p + var_e)
        P(f"\nGroup contrast ({gname}): b = {b:+.4f} [95% CI {lo:.4f}, {hi:.4f}], SE {se:.4f}, z = {z:.3f}, p = {p:.4f}")
        P(f"Variance components: player intercept {var_p:.4f}, residual {var_e:.4f}; ICC = {icc:.4f}")
        P(f"Converged: {m.converged}; log-likelihood {m.llf:.4f}; AIC {m.aic:.4f}; BIC {m.bic:.4f}")
        pvals.append(p)
        rows.append(dict(family=name, outcome=lab, b=b, ci_low=lo, ci_high=hi, se=se, z=z, p=p, icc=icc, converged=m.converged,
                         var_player=var_p, var_residual=var_e, n_obs=len(ds), n_players=ds.player_id.nunique()))
    p_bh = multipletests(pvals, method="fdr_bh")[1]
    P(f"\nBenjamini-Hochberg adjustment across the eight outcomes within {name}:")
    for (col, lab), p0, p1 in zip(OUTCOMES, pvals, p_bh):
        P(f"  {lab:32s} p = {p0:.4f}  p_BH = {p1:.4f}")
    for r, p1 in zip(rows, p_bh): r["p_bh"] = float(p1)
    return rows

results = []
results += family("Family A", d, "format", "Women", "format contrast, Mixed minus Women squad, all 52 observations")
results += family("Family B", d[d.sex == "F"].copy(), "format", "Women", "women only, Mixed women minus Women squad, 42 observations")
results += family("Family C", d[d.format == "Mixed"].copy(), "sex", "F", "sex within the mixed squad, men minus women, 20 observations")
P("\nTABLE 2. Group coefficients (A and B: Mixed minus Women squad; C: men minus women)")
for fam in ("Family A", "Family B", "Family C"):
    P(f"  {fam}")
    for r in results:
        if r["family"] == fam:
            P(f"    {r['outcome']:32s} b = {r['b']:+.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  p = {r['p']:.4f}  p_BH = {r['p_bh']:.4f}  ICC = {r['icc']:.2f}")

# ---------------------------------------------------------------- 3. load sharing (Table 3)
mx = d[d.format == "Mixed"].copy()
SHARE = [("sprint_count", "Sprints"), ("sprint_distance_m", "Sprint metres"), ("hid_m", "HID metres"), ("intense_speed_changes", "Intense speed changes"),
         ("distance_m", "Distance"), ("turns", "Turns")]
P("\nTABLE 3. Load share within the mixed squad (instrumented players only): men % / women % of the squad total")
for day in (1, 2, 3):
    s = mx[mx.day == day]; nm, nf = (s.sex == "M").sum(), (s.sex == "F").sum()
    P(f"  Day {day} ({nm} men / {nf} women): " + " | ".join(f"{lab} {100 * s[s.sex == 'M'][col].sum() / s[col].sum():.0f}/{100 * s[s.sex == 'F'][col].sum() / s[col].sum():.0f}" for col, lab in SHARE))
nm, nf = (mx.sex == "M").sum(), (mx.sex == "F").sum()
P(f"  All three tests ({nm} men / {nf} women observations):")
share_rows = []
for col, lab in SHARE:
    sm = mx[mx.sex == "M"][col].sum() / mx[col].sum(); sf = 1 - sm
    im, iff = sm / (nm / (nm + nf)), sf / (nf / (nm + nf))
    P(f"    {lab:24s} men {100 * sm:.0f}% / women {100 * sf:.0f}%   per-capita index men {im:.2f} / women {iff:.2f}")
    share_rows.append(dict(metric=lab, men_share_pct=100 * sm, women_share_pct=100 * sf, men_index=im, women_index=iff))
# relative tempo index: player work rate / mean work rate of the instrumented players in the same match (same squad, same day)
d["match_mean_wr"] = d.groupby(["format", "day"])["work_rate_m_min"].transform("mean")
d["relative_tempo"] = d["work_rate_m_min"] / d["match_mean_wr"]
d["hiad_pct"] = 100 * (d["hid_m"] + d["sprint_distance_m"]) / d["distance_m"]
P("  Relative tempo index (player work rate / mean work rate of the instrumented players in the same match) and HIAD % ((HID + sprint metres) / distance):")
for g in GROUPS:
    s = d[d.analysis_group == g]
    P(f"    {g:12s} relative tempo {s.relative_tempo.mean():.2f} +/- {s.relative_tempo.std():.2f}   HIAD {s.hiad_pct.mean():.1f} +/- {s.hiad_pct.std():.1f} %")

# ---------------------------------------------------------------- 4. crossover (Table 4)
P("\nTABLE 4. Within-player crossover, descriptive only: per-player means, November trial games (n = 2) v April mixed tests")
XC = [("distance_m", "Distance (m)", 0), ("work_rate_m_min", "Work rate (m/min)", 1), ("top_speed_m_s", "Top speed (m/s)", 2),
      ("sprint_count", "Sprints", 1), ("turns_per_min", "Turns per min", 2)]
xrows = []
for pid in ["190385", "190358", "190359", "190371", "190364", "190367"]:
    t = nov[nov.player_id == pid]; a = d[d.player_id == pid]
    sexlab = "women's" if t.sex.iloc[0] == "F" else "men's"
    P(f"  {pid}  Nov {sexlab} trial (n = {len(t)}): " + " | ".join(f"{lab} {rhu(t[col].mean(), dp)}" for col, lab, dp in XC))
    P(f"  {'':6s}  Apr mixed test(s) (n = {len(a)}): " + " | ".join(f"{lab} {rhu(a[col].mean(), dp)}" for col, lab, dp in XC))
    xrows.append(dict(player_id=pid, setting=f"Nov trial (n={len(t)})", **{col: t[col].mean() for col, _, _ in XC}))
    xrows.append(dict(player_id=pid, setting=f"Apr mixed tests (n={len(a)})", **{col: a[col].mean() for col, _, _ in XC}))
P("  (Turn metrics are missing for one November men's file; November turn means use the available game.)")

R = pd.DataFrame(results)
R.to_csv("P2_supplementary_results_table.csv", index=False)
pd.DataFrame(share_rows).to_csv("P2_supplementary_load_share.csv", index=False)
pd.DataFrame(xrows).to_csv("P2_supplementary_crossover.csv", index=False)
open("P2_supplementary_statistical_output.txt", "w").write(log.getvalue())
print("\nWritten: P2_supplementary_statistical_output.txt, P2_supplementary_results_table.csv, P2_supplementary_load_share.csv, P2_supplementary_crossover.csv")
