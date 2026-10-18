<<<<<<< HEAD
# Statistics of War: Probability & Statistics Lab Project

This project runs every lab technique (descriptive statistics through ANOVA) on public data covering 401 wars fought between 1816 and 2007.

## Files

| File | What it is |
|---|---|
| `war_stats_project.ipynb` | The Jupyter notebook, already run, with outputs, charts and an interpretation after every test |
| `war_stats_project.html` | The same notebook as a web page. It opens in any browser, so no Python is needed to read it |
| `war_stats_project.py` | Plain-script version of the notebook (jupytext "percent" format) |
| `figures/` | All 14 charts as PNGs, numbered in notebook order |
| `data/raw/` | The original downloaded datasets |
| `data/clean/` | Analysis tables built by the notebook (`wars.csv`, `interstate_participants.csv`, `paired_gdp_growth.csv`, `country_military_burden_growth.csv`) |

## Lab topics covered

| Lab | Section in the notebook | Key result |
|---|---|---|
| 1–2 | Wars per decade, casualty and duration statistics, histograms, box plot, UCDP trend, world map | Mean duration 3.1 years; deaths are extremely right-skewed |
| 3 | Duration vs deaths; military burden vs GDP growth | r ≈ 0.54 (log–log); doubling duration means about 2.4× the deaths. The milex–growth link is weak (r ≈ 0.19) |
| 4 | Multiple regression of log deaths | Adj. R² ≈ 0.44. Duration, war type, era and major-power involvement are all significant |
| 5 | Binomial tests | Initiators win 69% of the time and governments win 71% of civil wars; both differ significantly from 50% |
| 6 | Normal fit of log deaths; Poisson fit of yearly war onsets | Log-normal fits roughly but its right tail is too light. Inter-state war onsets are Poisson (λ ≈ 0.5/yr, χ² p ≈ 0.47) |
| 7–8 | Two-proportion z-test; Welch tests on economic indicators | Military burden is about 2× higher in war years; no significant winner/loser growth gap |
| 9 | Independent t-test; paired t-test | Internationalised civil wars are 3.2× deadlier. GDP growth falls 2.65 pp during war (paired, n = 380) |
| 10 | Chi-square tests of independence | Civil-war type and era are both associated with outcome; initiator × win is also significant |
| 11 | One-way ANOVA and Tukey HSD, Welch ANOVA, Kruskal–Wallis, two-way ANOVA | Classical F p ≈ 0.036, but the robust tests are not significant. This section is a lesson in checking assumptions |

## Data sources

* **Correlates of War** Inter-State War v4 and Intra-State War v4.1, plus National Material Capabilities v6. Taken from the copies shipped in the `peacesciencer` R package (github.com/svmiller/peacesciencer).
* **UCDP/PRIO Armed Conflict Dataset**, also via `peacesciencer`.
* **Maddison Project Database 2020**, via Our World in Data (github.com/owid/owid-datasets).
* **World Bank** GDP in current US$ (github.com/datasets/gdp).
* **Natural Earth** 1:110m country boundaries (github.com/nvkelso/natural-earth-vector).

The official UCDP, COW and World Bank websites could not be reached from the build environment, so the data came from the public GitHub mirrors listed above. They hold the same datasets.

## How to re-run

```bash
pip install pandas scipy statsmodels matplotlib geopandas pyreadr country_converter jupyter
jupyter nbconvert --to notebook --execute war_stats_project.ipynb --inplace
```

The notebook reads from `data/raw/` and writes to `data/clean/` and `figures/`. If you move the folder, change `ROOT` in the first code cell.
=======
# Probability_Statistics
>>>>>>> 4ea5ee1948e65ee76092dae56fdffafd13627279
