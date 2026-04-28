from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd 
from scipy import stats

DATA_FILE = Path("Projekt2data.xlsx")
SNI_FILE = Path("sni-2025-eng-251022.xlsx")
OUTPUT_DIR = Path("output")
RANDOM_SEED = 42
N_BOOTSTRAP = 3000
MAX_ANALYSIS_YEAR = 2024
MIN_TURNOVER_2024 = 250_000
MIN_LAST_EMPLOYEES = 1
MIN_COMPANIES_PER_GROUP = 5
CAGR_WINSOR_LOWER_Q = 0.02
CAGR_WINSOR_UPPER_Q = 0.98
REQUIRE_2023_2024_DATA = True
SNI_PREFIX_OPTIONS = (1, 2, 3)


def _year_columns(df: pd.DataFrame) -> list[int]:
    return [c for c in df.columns if isinstance(c, int) and c <= MAX_ANALYSIS_YEAR]


def _cagr(first_value: float, last_value: float, years: int) -> float:
    if years <= 0 or first_value <= 0 or last_value <= 0:
        return np.nan
    return ((last_value / first_value) ** (1 / years) - 1) * 100


def _metric_cagr(row: pd.Series, year_cols: list[int]) -> dict[str, float]:
    vals = pd.to_numeric(row[year_cols], errors="coerce").dropna()
    if len(vals) < 2:
        return {
            "first_year": np.nan,
            "last_year": np.nan,
            "first_value": np.nan,
            "last_value": np.nan,
            "cagr_pct": np.nan,
        }

    first_year = int(vals.index[0])
    last_year = int(vals.index[-1])
    first_value = float(vals.iloc[0])
    last_value = float(vals.iloc[-1])
    cagr = _cagr(first_value, last_value, last_year - first_year)
    return {
        "first_year": first_year,
        "last_year": last_year,
        "first_value": first_value,
        "last_value": last_value,
        "cagr_pct": cagr,
    }


def bootstrap_mean_ci(
    values: pd.Series,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    mean_value = float(np.mean(arr))
    if arr.size == 1:
        return (mean_value, np.nan, np.nan)

    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True)
    boot_means = samples.mean(axis=1)
    alpha = 1 - confidence
    return (
        mean_value,
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def winsorize_series(series: pd.Series, q_low: float, q_high: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    low = float(s.quantile(q_low))
    high = float(s.quantile(q_high))
    return s.clip(lower=low, upper=high)


def build_company_dataset() -> pd.DataFrame:
    overview = pd.read_excel(DATA_FILE, sheet_name="Company Overview")
    turnover = pd.read_excel(DATA_FILE, sheet_name="Turnover")
    employees = pd.read_excel(DATA_FILE, sheet_name="Employees")

    info = overview[
        ["Org Number", "Legal Name", "SNI Code", "Region", "Classification"]
    ].drop_duplicates("Org Number")

    t_years = _year_columns(turnover)
    e_years = _year_columns(employees)

    t_calc = turnover[["Org Number", *t_years]].copy()
    e_calc = employees[["Org Number", *e_years]].copy()

    t_metrics = t_calc.apply(
        lambda r: _metric_cagr(r, t_years), axis=1, result_type="expand"
    )
    t_metrics.columns = [
        "turnover_first_year",
        "turnover_last_year",
        "turnover_first_value",
        "turnover_last_value",
        "turnover_cagr_pct",
    ]
    t_company = pd.concat([t_calc[["Org Number"]], t_metrics], axis=1)

    e_metrics = e_calc.apply(
        lambda r: _metric_cagr(r, e_years), axis=1, result_type="expand"
    )
    e_metrics.columns = [
        "employees_first_year",
        "employees_last_year",
        "employees_first_value",
        "employees_last_value",
        "employees_cagr_pct",
    ]
    e_company = pd.concat([e_calc[["Org Number"]], e_metrics], axis=1)

    company = info.merge(t_company, on="Org Number", how="inner").merge(
        e_company, on="Org Number", how="inner"
    )
    company = company.dropna(subset=["SNI Code", "turnover_cagr_pct", "employees_cagr_pct"])
    # Require concrete 2023 and 2024 datapoints (turnover + employees).
    t_2324 = turnover[["Org Number", 2023, 2024]].rename(
        columns={2023: "turnover_2023", 2024: "turnover_2024"}
    )
    e_2324 = employees[["Org Number", 2023, 2024]].rename(
        columns={2023: "employees_2023", 2024: "employees_2024"}
    )
    company = company.merge(t_2324, on="Org Number", how="left").merge(
        e_2324, on="Org Number", how="left"
    )
    for col in ["turnover_2023", "turnover_2024", "employees_2023", "employees_2024"]:
        company[col] = pd.to_numeric(company[col], errors="coerce")
    if REQUIRE_2023_2024_DATA:
        company = company[
            company["turnover_2023"].notna()
            & company["turnover_2024"].notna()
            & company["employees_2023"].notna()
            & company["employees_2024"].notna()
        ].copy()
    company = company[
        (company["turnover_2024"] >= MIN_TURNOVER_2024)
        & (company["employees_2024"] >= MIN_LAST_EMPLOYEES)
    ].copy()
    # Robustify against extreme growth outliers.
    company["turnover_cagr_pct"] = winsorize_series(
        company["turnover_cagr_pct"], CAGR_WINSOR_LOWER_Q, CAGR_WINSOR_UPPER_Q
    )
    company["employees_cagr_pct"] = winsorize_series(
        company["employees_cagr_pct"], CAGR_WINSOR_LOWER_Q, CAGR_WINSOR_UPPER_Q
    )

    # Efficiency growth: high turnover CAGR with lower employee CAGR.
    company["efficiency_growth_gap_pct"] = (
        company["turnover_cagr_pct"] - company["employees_cagr_pct"]
    )
    company["cagr_per_employee"] = company["turnover_cagr_pct"] / company[
        "employees_2024"
    ].replace(0, np.nan)
    company["turnover_per_employee_2024"] = company["turnover_2024"] / company[
        "employees_2024"
    ].replace(0, np.nan)
    company["scalability_ratio"] = company["turnover_cagr_pct"] / company[
        "employees_cagr_pct"
    ].replace(0, np.nan)

    # Use only first two SNI digits for coarser categorization.
    sni_digits = (
        company["SNI Code"]
        .astype(str)
        .str.extract(r"(\d{2})", expand=False)
    )
    company["sni_2digit"] = pd.to_numeric(sni_digits, errors="coerce")
    company = company.dropna(subset=["sni_2digit"]).copy()
    company["sni_2digit"] = company["sni_2digit"].astype(int)
    return company


def _expand_division_ranges(sni_section: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for _, row in sni_section.iterrows():
        section = str(row["SNI2025Section"]).strip()
        section_desc = str(row["Description "]).strip()
        raw_interval = str(row["Division (Twodigit)"]).strip()
        parts = [p.strip() for p in raw_interval.split(",")]
        for part in parts:
            if "-" in part:
                start, end = part.split("-")
                for code in range(int(start), int(end) + 1):
                    rows.append(
                        {
                            "sni_2digit": code,
                            "division_interval": part,
                            "section_code": section,
                            "section_desc": section_desc,
                        }
                    )
            else:
                rows.append(
                    {
                        "sni_2digit": int(part),
                        "division_interval": part,
                        "section_code": section,
                        "section_desc": section_desc,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("sni_2digit")


def load_sni_lookup() -> pd.DataFrame:
    section = pd.read_excel(SNI_FILE, sheet_name="Section")
    section_map = _expand_division_ranges(section)
    return section_map[
        ["sni_2digit", "division_interval", "section_code", "section_desc"]
    ].drop_duplicates("sni_2digit")


def summarize_by_sni_interval(grouped: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for interval, g in grouped.groupby("division_interval"):
        if len(g) < MIN_COMPANIES_PER_GROUP:
            continue
        turn_mean, turn_low, turn_high = bootstrap_mean_ci(g["turnover_cagr_pct"])
        emp_mean, emp_low, emp_high = bootstrap_mean_ci(g["employees_cagr_pct"])
        gap_mean, gap_low, gap_high = bootstrap_mean_ci(g["efficiency_growth_gap_pct"])
        cagr_emp_mean, cagr_emp_low, cagr_emp_high = bootstrap_mean_ci(
            g["cagr_per_employee"]
        )
        tpe_mean, tpe_low, tpe_high = bootstrap_mean_ci(g["turnover_per_employee_2024"])
        ratio_mean, ratio_low, ratio_high = bootstrap_mean_ci(g["scalability_ratio"])
        rows.append(
            {
                "division_interval": str(interval),
                "section_code": str(g["section_code"].iloc[0]),
                "section_desc": str(g["section_desc"].iloc[0]),
                "n_companies": int(len(g)),
                "turnover_cagr_mean_pct": turn_mean,
                "turnover_cagr_ci_low": turn_low,
                "turnover_cagr_ci_high": turn_high,
                "employees_cagr_mean_pct": emp_mean,
                "employees_cagr_ci_low": emp_low,
                "employees_cagr_ci_high": emp_high,
                "efficiency_gap_mean_pct": gap_mean,
                "efficiency_gap_ci_low": gap_low,
                "efficiency_gap_ci_high": gap_high,
                "cagr_per_employee_mean": cagr_emp_mean,
                "cagr_per_employee_ci_low": cagr_emp_low,
                "cagr_per_employee_ci_high": cagr_emp_high,
                "turnover_per_employee_2024_mean": tpe_mean,
                "turnover_per_employee_2024_ci_low": tpe_low,
                "turnover_per_employee_2024_ci_high": tpe_high,
                "scalability_ratio_mean": ratio_mean,
                "scalability_ratio_ci_low": ratio_low,
                "scalability_ratio_ci_high": ratio_high,
            }
        )
    return pd.DataFrame(rows).sort_values("efficiency_gap_mean_pct", ascending=False)


def run_hypothesis_tests(company: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    # Paired t-test: is turnover CAGR higher than employee CAGR?
    paired = company[["turnover_cagr_pct", "employees_cagr_pct"]].dropna()
    if len(paired) >= 3:
        t_stat, p_val = stats.ttest_rel(
            paired["turnover_cagr_pct"], paired["employees_cagr_pct"], nan_policy="omit"
        )
        rows.append(
            {
                "test": "Paired t-test: turnover CAGR vs employee CAGR",
                "n_group_1": int(len(paired)),
                "n_group_2": int(len(paired)),
                "statistic": float(t_stat),
                "p_value": float(p_val),
            }
        )

    # Compare SNI sections (official category level from SNI workbook).
    section_groups = [
        g["efficiency_growth_gap_pct"].dropna().to_numpy()
        for _, g in company.groupby("section_code")
        if len(g) >= 2
    ]
    if len(section_groups) >= 2:
        f_sec, p_sec = stats.f_oneway(*section_groups)
        rows.append(
            {
                "test": "One-way ANOVA across SNI sections (official categories)",
                "n_group_1": int(len(section_groups)),
                "n_group_2": np.nan,
                "statistic": float(f_sec),
                "p_value": float(p_sec),
            }
        )

    anova_groups = [
        g["efficiency_growth_gap_pct"].dropna().to_numpy()
        for _, g in company.groupby("sni_2digit")
        if len(g) >= MIN_COMPANIES_PER_GROUP
    ]
    if len(anova_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*anova_groups)
        rows.append(
            {
                "test": f"One-way ANOVA across SNI groups (n>={MIN_COMPANIES_PER_GROUP})",
                "n_group_1": int(len(anova_groups)),
                "n_group_2": np.nan,
                "statistic": float(f_stat),
                "p_value": float(p_val),
            }
        )

    cagr_emp_groups = [
        g["cagr_per_employee"].dropna().to_numpy()
        for _, g in company.groupby("division_interval")
        if len(g) >= MIN_COMPANIES_PER_GROUP
    ]
    if len(cagr_emp_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*cagr_emp_groups)
        rows.append(
            {
                "test": f"One-way ANOVA on CAGR per employee across intervals (n>={MIN_COMPANIES_PER_GROUP})",
                "n_group_1": int(len(cagr_emp_groups)),
                "n_group_2": np.nan,
                "statistic": float(f_stat),
                "p_value": float(p_val),
            }
        )

    tpe_groups = [
        g["turnover_per_employee_2024"].dropna().to_numpy()
        for _, g in company.groupby("division_interval")
        if len(g) >= MIN_COMPANIES_PER_GROUP
    ]
    if len(tpe_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*tpe_groups)
        rows.append(
            {
                "test": f"One-way ANOVA on turnover per employee (2024) across intervals (n>={MIN_COMPANIES_PER_GROUP})",
                "n_group_1": int(len(tpe_groups)),
                "n_group_2": np.nan,
                "statistic": float(f_stat),
                "p_value": float(p_val),
            }
        )
    return pd.DataFrame(rows)


def run_multiple_regression(company: pd.DataFrame) -> pd.DataFrame:
    reg = company.dropna(
        subset=["efficiency_growth_gap_pct", "turnover_last_value", "employees_last_value"]
    ).copy()
    if reg.empty:
        return pd.DataFrame()

    reg["log_turnover_last"] = np.log1p(reg["turnover_last_value"])
    reg["log_employees_last"] = np.log1p(reg["employees_last_value"])
    reg["time_span"] = reg[["turnover_last_year", "turnover_first_year"]].apply(
        lambda r: float(r.iloc[0] - r.iloc[1]), axis=1
    )

    X = reg[["log_turnover_last", "log_employees_last", "time_span"]].to_numpy(dtype=float)
    section_dummies = pd.get_dummies(reg["section_code"], prefix="sec", drop_first=True)
    if not section_dummies.empty:
        X = np.column_stack([X, section_dummies.to_numpy(dtype=float)])
    X = np.column_stack([np.ones(len(reg)), X])
    y = reg["efficiency_growth_gap_pct"].to_numpy(dtype=float)

    x_tx_inv = np.linalg.pinv(X.T @ X)
    beta = x_tx_inv @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float((resid @ resid) / dof)
    cov_beta = x_tx_inv * sigma2
    se = np.sqrt(np.diag(cov_beta))
    t_values = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_values), df=dof)

    terms = ["Intercept", "log_turnover_last", "log_employees_last", "time_span"]
    terms += list(section_dummies.columns)
    out = pd.DataFrame(
        {"term": terms, "coef": beta, "std_err": se, "t_value": t_values, "p_value": p_values}
    )
    out["n_obs"] = n
    out["dof"] = dof
    return out


def run_time_series_trends(company: pd.DataFrame) -> pd.DataFrame:
    turnover = pd.read_excel(DATA_FILE, sheet_name="Turnover")
    employees = pd.read_excel(DATA_FILE, sheet_name="Employees")
    years = _year_columns(turnover)
    ids = company["Org Number"].unique()

    t_long = turnover[turnover["Org Number"].isin(ids)][["Org Number", *years]].melt(
        id_vars="Org Number", var_name="year", value_name="turnover"
    )
    e_long = employees[employees["Org Number"].isin(ids)][["Org Number", *years]].melt(
        id_vars="Org Number", var_name="year", value_name="employees"
    )
    panel = t_long.merge(e_long, on=["Org Number", "year"], how="inner")
    panel = panel.merge(company[["Org Number", "sni_2digit"]], on="Org Number", how="left")
    panel["turnover"] = pd.to_numeric(panel["turnover"], errors="coerce")
    panel["employees"] = pd.to_numeric(panel["employees"], errors="coerce")

    agg = (
        panel.groupby(["sni_2digit", "year"], as_index=False)
        .agg(avg_turnover=("turnover", "mean"), avg_employees=("employees", "mean"))
        .sort_values(["sni_2digit", "year"])
    )
    agg["turnover_yoy_pct"] = agg.groupby("sni_2digit")["avg_turnover"].pct_change() * 100
    agg["employees_yoy_pct"] = agg.groupby("sni_2digit")["avg_employees"].pct_change() * 100
    agg["efficiency_yoy_gap"] = agg["turnover_yoy_pct"] - agg["employees_yoy_pct"]

    rows: list[dict[str, float | int]] = []
    for sni, g in agg.groupby("sni_2digit"):
        valid = g.dropna(subset=["efficiency_yoy_gap"])
        if len(valid) < 4:
            continue
        x = pd.to_numeric(valid["year"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(valid["efficiency_yoy_gap"], errors="coerce").to_numpy(
            dtype=float
        )
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]
        if len(x) < 4 or len(np.unique(x)) < 2:
            continue

        try:
            lr = stats.linregress(x, y)
            slope = float(lr.slope)
            p_value = float(lr.pvalue)
            r_squared = float(lr.rvalue**2)
        except Exception:
            # Fallback for older SciPy/Numpy combos in some local environments.
            slope, intercept = np.polyfit(x, y, 1)
            y_hat = slope * x + intercept
            residual = y - y_hat
            ss_res = float(np.sum(residual**2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            n = len(x)
            if n > 2:
                s_err = np.sqrt(ss_res / (n - 2))
                sxx = float(np.sum((x - np.mean(x)) ** 2))
                if sxx > 0:
                    se_slope = s_err / np.sqrt(sxx)
                    t_stat = float(slope / se_slope) if se_slope > 0 else np.nan
                    p_value = (
                        float(2 * stats.t.sf(np.abs(t_stat), df=n - 2))
                        if np.isfinite(t_stat)
                        else np.nan
                    )
                else:
                    p_value = np.nan
            else:
                p_value = np.nan

        rows.append(
            {
                "sni_2digit": int(sni),
                "n_years": int(len(valid)),
                "trend_slope_pct_per_year": slope,
                "trend_p_value": p_value,
                "trend_r_squared": r_squared,
            }
        )
    return pd.DataFrame(rows).sort_values("trend_slope_pct_per_year", ascending=False)


def _extract_sni_prefix(series: pd.Series, digits: int) -> pd.Series:
    pattern = rf"(\d{{{digits}}})"
    return pd.to_numeric(
        series.astype(str).str.extract(pattern, expand=False), errors="coerce"
    )


def run_sni_prefix_sensitivity(company: pd.DataFrame) -> pd.DataFrame:
    """
    Compare inference sensitivity for different SNI prefix lengths (1,2,3 digits).
    """
    rows: list[dict[str, float | int]] = []
    for digits in SNI_PREFIX_OPTIONS:
        tmp = company.copy()
        tmp["sni_prefix"] = _extract_sni_prefix(tmp["SNI Code"], digits)
        tmp = tmp.dropna(subset=["sni_prefix", "efficiency_growth_gap_pct"]).copy()
        tmp["sni_prefix"] = tmp["sni_prefix"].astype(int)

        grouped = [
            g["efficiency_growth_gap_pct"].dropna().to_numpy()
            for _, g in tmp.groupby("sni_prefix")
            if len(g) >= MIN_COMPANIES_PER_GROUP
        ]
        n_groups = len(grouped)
        if n_groups >= 2:
            f_stat, p_val = stats.f_oneway(*grouped)
        else:
            f_stat, p_val = np.nan, np.nan

        top_group = (
            tmp.groupby("sni_prefix")["efficiency_growth_gap_pct"]
            .mean()
            .sort_values(ascending=False)
        )
        top_prefix = int(top_group.index[0]) if not top_group.empty else np.nan
        top_gap = float(top_group.iloc[0]) if not top_group.empty else np.nan
        rows.append(
            {
                "sni_prefix_digits": digits,
                "n_companies": int(len(tmp)),
                "n_groups_total": int(tmp["sni_prefix"].nunique()),
                "n_groups_ge_min_size": int(n_groups),
                "anova_f_stat_efficiency_gap": float(f_stat) if np.isfinite(f_stat) else np.nan,
                "anova_p_value_efficiency_gap": float(p_val) if np.isfinite(p_val) else np.nan,
                "top_prefix_by_mean_gap": top_prefix,
                "top_prefix_mean_gap_pct": top_gap,
            }
        )
    return pd.DataFrame(rows)


def get_swedish_branch_name(interval: str) -> str:
    try:
        start = int(str(interval).split('-')[0])
        if 1 <= start <= 3: return "Jordbruk & Fiske"
        elif 5 <= start <= 9: return "Mineralutvinning"
        elif 10 <= start <= 33: return "Tillverkning"
        elif 35 <= start <= 39: return "Energi & Miljö"
        elif 41 <= start <= 43: return "Byggverksamhet"
        elif 45 <= start <= 47: return "Handel"
        elif 49 <= start <= 53: return "Transport & Magasinering"
        elif 55 <= start <= 56: return "Hotell & Restaurang"
        elif 58 <= start <= 60: return "Media & Publicering"
        elif 61 <= start <= 63: return "IT & Kommunikation"
        elif 64 <= start <= 66: return "Finans & Försäkring"
        elif start == 68: return "Fastighetsverksamhet"
        elif 69 <= start <= 75: return "Företagstjänster"
        elif 77 <= start <= 82: return "Uthyrning & Service"
        elif 86 <= start <= 88: return "Vård & Omsorg"
        elif 90 <= start <= 93: return "Kultur & Nöje"
        else: return "Övriga tjänster"
    except:
        return "Okänd"


def create_plots(
    summary: pd.DataFrame, company: pd.DataFrame
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    import matplotlib

    if os.getenv("KV_NO_SHOW", "0") == "1":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1_path = OUTPUT_DIR / "plot_sni2_scalability_line.png"
    fig2_path = OUTPUT_DIR / "plot_sni2_efficiency_gap_ci.png"
    fig3_path = OUTPUT_DIR / "plot_company_cagr_scatter.png"
    fig4_path = OUTPUT_DIR / "plot_sni2_cagr_heatmap.png"
    fig5_path = OUTPUT_DIR / "plot_sni2_group_sizes.png"
    fig6_path = OUTPUT_DIR / "plot_efficiency_gap_boxplot.png"
    fig7_path = OUTPUT_DIR / "plot_cagr_distributions_hist.png"
    
    # --- NYA FILVÄGAR FÖR DE TRE NYA PLOTTARNA ---
    fig8_path = OUTPUT_DIR / "plot_sni2_dumbbell_scalability.png"
    fig9_path = OUTPUT_DIR / "plot_sni2_quadrant_scatter.png"
    fig10_path = OUTPUT_DIR / "plot_sni2_scalability_ranking.png"
    fig11_path = OUTPUT_DIR / "plot_sni2_results_table.png"

    # ==========================================
    # BEFINTLIGA PLOTTAR (1-7) - HELT ORÖRDA
    # ==========================================
    
    plot_df = summary.sort_values("efficiency_gap_mean_pct", ascending=False).head(15)
    x = np.arange(len(plot_df))
    labels = [
        f"{str(interval)} : {get_swedish_branch_name(str(interval))}"
        for interval in plot_df["division_interval"]
    ]

    fig1, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(x, plot_df["turnover_cagr_mean_pct"], marker="o", label="Omsättning CAGR")
    ax1.plot(x, plot_df["employees_cagr_mean_pct"], marker="o", label="Anställda CAGR")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("CAGR (%)")
    ax1.set_title("SNI 2-siffrig: Omsättning-CAGR vs Anställda-CAGR (Top 15)")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(fig1_path, dpi=220)

    ci_df = summary.sort_values("efficiency_gap_mean_pct", ascending=True).head(15)
    xerr = np.vstack(
        [
            ci_df["efficiency_gap_mean_pct"] - ci_df["efficiency_gap_ci_low"],
            ci_df["efficiency_gap_ci_high"] - ci_df["efficiency_gap_mean_pct"],
        ]
    )
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    y_labels = [
        f"{str(interval)} : {get_swedish_branch_name(str(interval))}"
        for interval in ci_df["division_interval"]
    ]
    ax2.errorbar(
        ci_df["efficiency_gap_mean_pct"],
        y_labels,
        xerr=xerr,
        fmt="o",
        capsize=4,
    )
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("Efficiency growth gap (%) = Turnover CAGR - Employee CAGR")
    ax2.set_ylabel("SNI 2-siffrig grupp")
    ax2.set_title("SNI-gruppers skalbarhet med 95% bootstrap-KI")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    fig2.tight_layout()
    fig2.savefig(fig2_path, dpi=220)

    scatter_df = company.dropna(
        subset=["turnover_cagr_pct", "employees_cagr_pct", "division_interval"]
    ).copy()
    scatter_df["size"] = np.log1p(pd.to_numeric(scatter_df["turnover_2024"], errors="coerce"))
    scatter_df["bransch_etikett"] = scatter_df["division_interval"].apply(
        lambda x: f"{str(x)} : {get_swedish_branch_name(str(x))}"
    )
    
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    unika_etiketter = scatter_df["bransch_etikett"].unique()
    cmap = plt.get_cmap("tab20")
    
    for i, etikett in enumerate(unika_etiketter):
        subset = scatter_df[scatter_df["bransch_etikett"] == etikett]
        s_values = 20 + 18 * subset["size"].fillna(0)
        ax3.scatter(
            subset["employees_cagr_pct"],
            subset["turnover_cagr_pct"],
            s=s_values,
            alpha=0.55,
            color=cmap(i % 20),
            label=etikett
        )

    min_axis = min(
        np.nanmin(scatter_df["employees_cagr_pct"]),
        np.nanmin(scatter_df["turnover_cagr_pct"]),
    )
    max_axis = max(
        np.nanmax(scatter_df["employees_cagr_pct"]),
        np.nanmax(scatter_df["turnover_cagr_pct"]),
    )
    ax3.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", linewidth=1, color="black", alpha=0.5)
    ax3.set_xlabel("Employee CAGR (%)")
    ax3.set_ylabel("Turnover CAGR (%)")
    ax3.set_title("Bolagsnivå: Omsättningstillväxt vs anställdstillväxt")
    ax3.grid(alpha=0.3, linestyle="--")
    
    leg = ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Bransch")
    for handle in leg.legend_handles:
        handle.set_sizes([50.0])
        
    fig3.tight_layout()
    fig3.savefig(fig3_path, dpi=220, bbox_inches='tight')

    heat_df = (
        summary[
            [
                "division_interval",
                "turnover_cagr_mean_pct",
                "employees_cagr_mean_pct",
                "efficiency_gap_mean_pct",
            ]
        ]
        .set_index("division_interval")
        .sort_values("efficiency_gap_mean_pct", ascending=False)
    )
    heat_values = heat_df.to_numpy(dtype=float)
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    im = ax4.imshow(heat_values, aspect="auto", cmap="RdYlGn")
    ax4.set_xticks(np.arange(heat_df.shape[1]))
    ax4.set_xticklabels(
        ["Turnover CAGR", "Employees CAGR", "Efficiency gap"], rotation=20, ha="right"
    )
    ax4.set_yticks(np.arange(heat_df.shape[0]))
    heat_labels = [f"{str(idx)} : {get_swedish_branch_name(str(idx))}" for idx in heat_df.index]
    ax4.set_yticklabels(heat_labels)
    ax4.set_title("SNI-intervall: jämförelse av tillväxtmått")
    fig4.colorbar(im, ax=ax4, label="Procent")
    fig4.tight_layout()
    fig4.savefig(fig4_path, dpi=220)

    size_df = summary.sort_values("n_companies", ascending=False)
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    x_labels = [f"{str(val)} : {get_swedish_branch_name(str(val))}" for val in size_df["division_interval"]]
    ax5.bar(x_labels, size_df["n_companies"], color="tab:blue", alpha=0.8)
    ax5.set_title("Antal bolag per SNI-intervall i analysurvalet")
    ax5.set_xlabel("SNI-intervall")
    ax5.set_ylabel("Antal bolag")
    ax5.set_xticks(range(len(x_labels)))
    ax5.set_xticklabels(x_labels, rotation=45, ha="right")
    ax5.grid(axis="y", linestyle="--", alpha=0.35)
    fig5.tight_layout()
    fig5.savefig(fig5_path, dpi=220)

    interval_order = (
        summary.sort_values("efficiency_gap_mean_pct", ascending=False)["division_interval"]
        .tolist()
    )
    box_data = [
        company.loc[company["division_interval"] == interval, "efficiency_growth_gap_pct"]
        .dropna()
        .to_numpy()
        for interval in interval_order
    ]
    box_labels = [f"{str(interval)} : {get_swedish_branch_name(str(interval))}" for interval in interval_order]
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    ax6.boxplot(box_data, tick_labels=box_labels, showfliers=True)
    ax6.axhline(0, color="black", linewidth=1)
    ax6.set_title("Fördelning av efficiency growth gap per SNI-intervall")
    ax6.set_xlabel("SNI-intervall")
    ax6.set_ylabel("Efficiency gap (%)")
    ax6.tick_params(axis="x", rotation=45)
    for tick in ax6.get_xticklabels():
        tick.set_ha("right")
    ax6.grid(axis="y", linestyle="--", alpha=0.35)
    fig6.tight_layout()
    fig6.savefig(fig6_path, dpi=220)

    fig7, axes7 = plt.subplots(1, 3, figsize=(15, 4.8))
    axes7[0].hist(company["turnover_cagr_pct"].dropna(), bins=30, color="tab:blue", alpha=0.75)
    axes7[0].set_title("Omsättning CAGR")
    axes7[0].set_xlabel("Procent")
    axes7[0].set_ylabel("Antal bolag")
    axes7[1].hist(company["employees_cagr_pct"].dropna(), bins=30, color="tab:green", alpha=0.75)
    axes7[1].set_title("Anställda CAGR")
    axes7[1].set_xlabel("Procent")
    axes7[2].hist(
        company["efficiency_growth_gap_pct"].dropna(), bins=30, color="tab:purple", alpha=0.75
    )
    axes7[2].set_title("Efficiency gap (Turnover - Employees)")
    axes7[2].set_xlabel("Procent")
    for ax in axes7:
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig7.tight_layout()
    fig7.savefig(fig7_path, dpi=220)


    # ==========================================
    # DE 3 NYA "HJÄLTE"-PLOTTARNA FÖR RAPPORTEN
    # ==========================================

    # FIGUR 8: HANTELDIAGRAMMET (The Dumbbell Plot)
    db_df = summary.sort_values("efficiency_gap_mean_pct", ascending=True)
    db_labels = [f"{str(i)} : {get_swedish_branch_name(str(i))}" for i in db_df["division_interval"]]
    
    fig8, ax8 = plt.subplots(figsize=(10, 8))
    for i, (_, row) in enumerate(db_df.iterrows()):
        t_cagr = row["turnover_cagr_mean_pct"]
        e_cagr = row["employees_cagr_mean_pct"]
        # Rita strecket mellan punkterna (skalbarhetsgapet)
        ax8.plot([e_cagr, t_cagr], [i, i], color="grey", zorder=1, linewidth=2, alpha=0.5)
        # Rita anställda-pricken
        ax8.scatter(e_cagr, i, color="tab:blue", s=100, zorder=2)
        # Rita omsättnings-pricken
        ax8.scatter(t_cagr, i, color="tab:green", s=100, zorder=2)
    
    ax8.set_yticks(range(len(db_df)))
    ax8.set_yticklabels(db_labels)
    ax8.set_xlabel("Genomsnittlig CAGR (%)")
    ax8.set_title("Branschernas skalbarhets-gap: Omsättning vs Anställda")
    ax8.grid(axis="x", linestyle="--", alpha=0.4)
    # Skapa manuell legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=10, label='Anställda CAGR'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markersize=10, label='Omsättning CAGR')
    ]
    ax8.legend(handles=legend_elements, loc='lower right')
    fig8.tight_layout()
    fig8.savefig(fig8_path, dpi=220)


    # FIGUR 9: KVADRANTDIAGRAM (Industry Quadrant Scatter)
    fig9, ax9 = plt.subplots(figsize=(12, 8))
    quad_cmap = plt.get_cmap("tab20")
    
    # Beräkna medelvärden för att dra kvadranter
    mean_e_cagr = summary["employees_cagr_mean_pct"].mean()
    mean_t_cagr = summary["turnover_cagr_mean_pct"].mean()

    for i, (_, row) in enumerate(summary.iterrows()):
        etikett = f"{str(row['division_interval'])} : {get_swedish_branch_name(str(row['division_interval']))}"
        # Storleken baseras på antal bolag
        s_val = 50 + (row["n_companies"] * 10) 
        ax9.scatter(
            row["employees_cagr_mean_pct"],
            row["turnover_cagr_mean_pct"],
            s=s_val,
            alpha=0.7,
            color=quad_cmap(i % 20),
            edgecolor="black",
            label=etikett
        )

    # Rita linjer för kvadranterna
    ax9.axvline(mean_e_cagr, color="black", linestyle="--", alpha=0.5)
    ax9.axhline(mean_t_cagr, color="black", linestyle="--", alpha=0.5)
    
    # Skriv ut text för "Vinnar-kvadranten"
    ax9.text(
        mean_e_cagr * 0.5, 
        ax9.get_ylim()[1] * 0.9, 
        "HÖG SKALBARHET\n(Stark omsättning, lågt personalbehov)", 
        ha="center", va="top", fontsize=10, alpha=0.7, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
    )

    ax9.set_xlabel("Genomsnittlig Anställda CAGR (%)")
    ax9.set_ylabel("Genomsnittlig Omsättning CAGR (%)")
    ax9.set_title("Branschkvadranter: Var växer företagen bäst?")
    ax9.grid(alpha=0.2, linestyle="-")
    
    leg9 = ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Bransch (Storlek = Antal bolag)")
    for handle in leg9.legend_handles:
        handle.set_sizes([50.0])
        
    fig9.tight_layout()
    fig9.savefig(fig9_path, dpi=220, bbox_inches='tight')


    # FIGUR 10: SKALBARHETS-TOPPLISTAN (Stapeldiagram på gapet)
    rank_df = summary.sort_values("efficiency_gap_mean_pct", ascending=True)
    rank_labels = [f"{str(i)} : {get_swedish_branch_name(str(i))}" for i in rank_df["division_interval"]]
    
    fig10, ax10 = plt.subplots(figsize=(10, 8))
    # Färgkodning: Grön om gapet är över 0, röd om det är under
    colors = ["tab:green" if gap >= 0 else "tab:red" for gap in rank_df["efficiency_gap_mean_pct"]]
    
    bars = ax10.barh(rank_labels, rank_df["efficiency_gap_mean_pct"], color=colors, alpha=0.8)
    
    ax10.axvline(0, color="black", linewidth=1.5)
    ax10.set_xlabel("Efficiency Gap (%) (Omsättning CAGR - Anställda CAGR)")
    ax10.set_title("Rankning: Vilka branscher har högst skalbarhet?")
    ax10.grid(axis="x", linestyle="--", alpha=0.4)
    
    # Lägg till procentsiffran i änden av varje stapel för extra tydlighet
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 2 if width > 0 else width - 2
        ha = 'left' if width > 0 else 'right'
        ax10.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.0f}%', 
                  ha=ha, va='center', fontsize=9)

    fig10.tight_layout()
    fig10.savefig(fig10_path, dpi=220)

    # FIGUR 11: FULL RESULTATTABELL (för screenshot i separat plot-flik)
    table_df = summary.sort_values("efficiency_gap_mean_pct", ascending=False).copy()
    table_df["Sektor"] = table_df["division_interval"].map(
        lambda interval: get_swedish_branch_name(str(interval))
    )
    table_df = table_df[
        [
            "division_interval",
            "Sektor",
            "n_companies",
            "turnover_cagr_mean_pct",
            "employees_cagr_mean_pct",
            "efficiency_gap_mean_pct",
        ]
    ].rename(
        columns={
            "division_interval": "SNI-intervall",
            "n_companies": "Antal bolag",
            "turnover_cagr_mean_pct": "Omsättning CAGR (%)",
            "employees_cagr_mean_pct": "Anställda CAGR (%)",
            "efficiency_gap_mean_pct": "Efficiency growth gap (%)",
        }
    )
    for col in [
        "Omsättning CAGR (%)",
        "Anställda CAGR (%)",
        "Efficiency growth gap (%)",
    ]:
        table_df[col] = table_df[col].map(lambda value: f"{value:.2f}")

    fig11_height = max(5, 0.55 * (len(table_df) + 2))
    fig11, ax11 = plt.subplots(figsize=(15, fig11_height))
    ax11.axis("off")
    ax11.set_title(
        "Alla SNI-resultat: omsättning, anställda och efficiency growth gap",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    table = ax11.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2F5597")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#F2F2F2")
    fig11.tight_layout()
    fig11.savefig(fig11_path, dpi=220, bbox_inches="tight")


    if os.getenv("KV_NO_SHOW", "0") != "1":
        plt.show()

    plt.close('all') # Stänger alla figurer rent och snyggt
    
    # Du måste numera returnera alla 11 filvägar
    return fig1_path, fig2_path, fig3_path, fig4_path, fig5_path, fig6_path, fig7_path, fig8_path, fig9_path, fig10_path, fig11_path


def run() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Hittar inte datafilen: {DATA_FILE.resolve()}")
    if not SNI_FILE.exists():
        raise FileNotFoundError(f"Hittar inte SNI-filen: {SNI_FILE.resolve()}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    company = build_company_dataset()
    if company.empty:
        raise ValueError("Inga bolag kvar efter filter.")

    sni_lookup = load_sni_lookup()
    company = company.merge(sni_lookup, on="sni_2digit", how="left")
    company["sni_category"] = (
        company["division_interval"].astype(str)
        + " "
        + company["section_code"].astype(str)
        + " - "
        + company["section_desc"].astype(str)
    )
    summary = summarize_by_sni_interval(company)
    if summary.empty:
        raise ValueError("Inga SNI-grupper kvar efter minsta urval.")

    tests = run_hypothesis_tests(company)
    regression = run_multiple_regression(company)
    trends = run_time_series_trends(company)
    prefix_sensitivity = run_sni_prefix_sensitivity(company)
    
    # Uppdaterad rad för att ta emot 11 figurer
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = create_plots(summary, company)

    company_path = OUTPUT_DIR / "sni2_company_growth_base.csv"
    summary_path = OUTPUT_DIR / "sni2_scalability_summary.csv"
    mapping_audit_path = OUTPUT_DIR / "sni2_mapping_audit.csv"
    tests_path = OUTPUT_DIR / "sni2_hypothesis_tests.csv"
    regression_path = OUTPUT_DIR / "sni2_multiple_regression.csv"
    trends_path = OUTPUT_DIR / "sni2_time_series_trends.csv"
    prefix_sensitivity_path = OUTPUT_DIR / "sni_prefix_sensitivity.csv"

    mapping_audit = company[
        [
            "Org Number",
            "Legal Name",
            "SNI Code",
            "sni_2digit",
            "division_interval",
            "section_code",
            "section_desc",
            "sni_category",
        ]
    ].drop_duplicates(subset=["Org Number"])

    company.to_csv(company_path, index=False)
    summary.to_csv(summary_path, index=False)
    mapping_audit.to_csv(mapping_audit_path, index=False)
    tests.to_csv(tests_path, index=False)
    regression.to_csv(regression_path, index=False)
    trends.to_csv(trends_path, index=False)
    prefix_sensitivity.to_csv(prefix_sensitivity_path, index=False)

    print("\nFråga:")
    print(
        "Vilka SNI-grupper (2-siffrig nivå) visar högst skalbarhet "
        "(hög omsättnings-CAGR men låg employee-CAGR)?"
    )
    print("\nHypotes:")
    print(
        "Programvara/digitala tjänster har högre efficiency growth än "
        "kapital- och personalintensiva segment."
    )
    print("\nUrval:")
    print(f"- Bolag efter storleksfilter: {len(company)}")
    print(f"- Max analysår: {MAX_ANALYSIS_YEAR} (2025 exkluderat)")
    print(
        f"- Krav på datapunkter 2023+2024 (turnover+anställda): {REQUIRE_2023_2024_DATA}"
    )
    print(f"- Min turnover 2024: {MIN_TURNOVER_2024:,.0f} SEK")
    print(f"- Min anställda 2024: {MIN_LAST_EMPLOYEES}")
    print(f"- Winsorize CAGR: {int(CAGR_WINSOR_LOWER_Q*100)}-{int(CAGR_WINSOR_UPPER_Q*100)} percentil")
    print(f"- Min bolag per grupp i summary/ANOVA: {MIN_COMPANIES_PER_GROUP}")
    print(f"- Antal analyserade SNI-grupper: {len(summary)}")

    print("\nFull resultattabell öppnas som separat plot-flik:")
    print("- plot_sni2_results_table: visar alla SNI-intervall med sektor, antal bolag, omsättnings-CAGR, anställda-CAGR och efficiency growth gap.")
    if not tests.empty:
        print("\nHypotestester (p-värden):")
        print(tests.to_string(index=False))
        print("\nSlutsatser från hypotestester:")
        for _, row in tests.iterrows():
            test_name = str(row["test"])
            p_val = float(row["p_value"]) if pd.notna(row["p_value"]) else np.nan
            if np.isfinite(p_val) and p_val < 0.05:
                conclusion = "signifikant skillnad (p < 0.05)"
            elif np.isfinite(p_val):
                conclusion = "ingen statistiskt signifikant skillnad (p >= 0.05)"
            else:
                conclusion = "otillräckligt underlag för slutsats"
            print(f"- {test_name}: {conclusion}.")
    if not regression.empty:
        print("\nMultipel regression (efficiency gap som beroende variabel):")
        print(regression.to_string(index=False))
    if not trends.empty:
        print("\nTidsserie-trender (linjär trend i efficiency YoY-gap):")
        print(trends.head(10).to_string(index=False))
    if not prefix_sensitivity.empty:
        print("\nKänslighetsanalys: SNI-prefix 1/2/3 siffror")
        print(prefix_sensitivity.to_string(index=False))

    print("\nHur plottarna tolkas:")
    print("- plot_sni2_scalability_line: turnover-linje över employee-linje indikerar skalbar tillvaxt.")
    print("- plot_sni2_efficiency_gap_ci: punkter med KI helt over 0 visar robust positiv efficiency gap.")
    print("- plot_company_cagr_scatter: punkter ovanfor diagonalen betyder turnover-tillvaxt > employee-tillvaxt.")
    print("- plot_sni2_cagr_heatmap: hogre/intensivare farg visar hogre medelvaerden for respektive matt.")
    print("- plot_sni2_group_sizes: visar hur mycket data som ligger bakom varje SNI-intervall.")
    print("- plot_efficiency_gap_boxplot: visar median, spridning och outliers per intervall.")
    print("- plot_cagr_distributions_hist: visar totalfordelning av CAGR och efficiency gap i samplet.")
    print("- plot_sni2_dumbbell_scalability (NY): Visar direkt gapet mellan omsättning och personal för varje bransch.")
    print("- plot_sni2_quadrant_scatter (NY): Branscher i övre vänstra hörnet är de riktiga vinnarna.")
    print("- plot_sni2_scalability_ranking (NY): Enkel topplista sorterad på skalbarhet.")
    print("- plot_sni2_results_table (NY): Full tabell över alla SNI-resultat, gjord för screenshot.")

    print("\nFiler sparade:")
    print(f"- {company_path.resolve()}")
    print(f"- {summary_path.resolve()}")
    print(f"- {mapping_audit_path.resolve()}")
    print(f"- {tests_path.resolve()}")
    print(f"- {regression_path.resolve()}")
    print(f"- {trends_path.resolve()}")
    print(f"- {prefix_sensitivity_path.resolve()}")
    print(f"- {f1.resolve()}")
    print(f"- {f2.resolve()}")
    print(f"- {f3.resolve()}")
    print(f"- {f4.resolve()}")
    print(f"- {f5.resolve()}")
    print(f"- {f6.resolve()}")
    print(f"- {f7.resolve()}")
    print(f"- {f8.resolve()}")
    print(f"- {f9.resolve()}")
    print(f"- {f10.resolve()}")
    print(f"- {f11.resolve()}")


if __name__ == "__main__":
    run()