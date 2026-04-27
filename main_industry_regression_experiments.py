from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_FILE = Path("Projekt2data.xlsx")
SNI_FILE = Path("sni-2025-eng-251022.xlsx")
OUTPUT_DIR = Path("output")
RANDOM_SEED = 42
MAX_ANALYSIS_YEAR = 2024
MIN_TURNOVER_2024 = 250_000
MIN_LAST_EMPLOYEES = 1
CAGR_WINSOR_LOWER_Q = 0.02
CAGR_WINSOR_UPPER_Q = 0.98
REQUIRE_2023_2024_DATA = True


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


def winsorize_series(series: pd.Series, q_low: float, q_high: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    low = float(s.quantile(q_low))
    high = float(s.quantile(q_high))
    return s.clip(lower=low, upper=high)


def _safe_log1p(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # Avoid invalid log1p on negative values from noisy accounting fields.
    return np.log1p(s.clip(lower=0))


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


def build_company_dataset() -> pd.DataFrame:
    overview = pd.read_excel(DATA_FILE, sheet_name="Company Overview")
    turnover = pd.read_excel(DATA_FILE, sheet_name="Turnover")
    employees = pd.read_excel(DATA_FILE, sheet_name="Employees")
    sni_section = pd.read_excel(SNI_FILE, sheet_name="Section")

    info = overview[
        [
            "Org Number",
            "Legal Name",
            "SNI Code",
            "Region",
            "Classification",
            "Legal Form",
            "Founded Year",
            "Personnel Costs",
            "Board Total",
            "Board Women",
            "Board Men",
            "Total Funding",
        ]
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

    company["turnover_cagr_pct"] = winsorize_series(
        company["turnover_cagr_pct"], CAGR_WINSOR_LOWER_Q, CAGR_WINSOR_UPPER_Q
    )
    company["employees_cagr_pct"] = winsorize_series(
        company["employees_cagr_pct"], CAGR_WINSOR_LOWER_Q, CAGR_WINSOR_UPPER_Q
    )
    company["efficiency_growth_gap_pct"] = (
        company["turnover_cagr_pct"] - company["employees_cagr_pct"]
    )
    company["turnover_per_employee_2024"] = company["turnover_2024"] / company[
        "employees_2024"
    ].replace(0, np.nan)

    sni_digits = (
        company["SNI Code"].astype(str).str.extract(r"(\d{2})", expand=False)
    )
    company["sni_2digit"] = pd.to_numeric(sni_digits, errors="coerce")
    company = company.dropna(subset=["sni_2digit"]).copy()
    company["sni_2digit"] = company["sni_2digit"].astype(int)

    sni_lookup = _expand_division_ranges(sni_section)
    company = company.merge(sni_lookup, on="sni_2digit", how="left")
    company["time_span"] = (
        company["turnover_last_year"] - company["turnover_first_year"]
    ).astype(float)
    company["log_turnover_2024"] = _safe_log1p(company["turnover_2024"])
    company["log_employees_2024"] = _safe_log1p(company["employees_2024"])
    company["log_turnover_per_employee_2024"] = _safe_log1p(company["turnover_per_employee_2024"])
    for col in [
        "Founded Year",
        "Personnel Costs",
        "Board Total",
        "Board Women",
        "Board Men",
        "Total Funding",
    ]:
        company[col] = pd.to_numeric(company[col], errors="coerce")
    company["firm_age_2024"] = 2024 - company["Founded Year"]
    company.loc[company["firm_age_2024"] < 0, "firm_age_2024"] = np.nan
    company["board_women_share"] = company["Board Women"] / company["Board Total"].replace(
        0, np.nan
    )
    company["log_personnel_costs"] = _safe_log1p(company["Personnel Costs"])
    company["log_total_funding"] = _safe_log1p(company["Total Funding"])
    return company


def inspect_excel_columns() -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    workbook_targets = [
        (DATA_FILE, ["Company Overview", "Turnover", "Employees"]),
        (SNI_FILE, ["Section"]),
    ]
    for file_path, sheets in workbook_targets:
        for sheet in sheets:
            df = pd.read_excel(file_path, sheet_name=sheet, nrows=0)
            for idx, col in enumerate(df.columns):
                rows.append(
                    {
                        "file": file_path.name,
                        "sheet": sheet,
                        "column_idx": int(idx),
                        "column_name": str(col),
                    }
                )
    return pd.DataFrame(rows)


def _weighted_least_squares(
    y: np.ndarray,
    x: np.ndarray,
    term_names: list[str],
    model_name: str,
    weighting_name: str,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(y)
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        valid = np.isfinite(w)
        y = y[valid]
        x = x[valid]
        w = w[valid]
        n = len(y)
        if n == 0:
            return pd.DataFrame()

    sw = np.sqrt(w)
    xw = x * sw[:, None]
    yw = y * sw

    x_tx_inv = np.linalg.pinv(xw.T @ xw)
    beta = x_tx_inv @ (xw.T @ yw)
    resid = y - x @ beta
    k = x.shape[1]
    dof = max(n - k, 1)
    sigma2 = float((w * resid**2).sum() / dof)
    cov_beta = x_tx_inv * sigma2
    se = np.sqrt(np.diag(cov_beta))
    t_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * stats.t.sf(np.abs(t_values), df=dof)

    y_hat = x @ beta
    weighted_mean = float(np.average(y, weights=w))
    ss_res = float((w * (y - y_hat) ** 2).sum())
    ss_tot = float((w * (y - weighted_mean) ** 2).sum())
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    out = pd.DataFrame(
        {
            "model_name": model_name,
            "weighting_name": weighting_name,
            "term": term_names,
            "coef": beta,
            "std_err": se,
            "t_value": t_values,
            "p_value": p_values,
            "n_obs": n,
            "dof": dof,
            "r_squared": r_squared,
        }
    )
    return out


def run_regression_experiments(company: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "efficiency_growth_gap_pct",
        "log_turnover_2024",
        "log_employees_2024",
        "time_span",
        "log_turnover_per_employee_2024",
        "section_code",
        "employees_2024",
        "turnover_2024",
        "firm_age_2024",
        "log_personnel_costs",
        "board_women_share",
        "Region",
        "Classification",
        "Legal Form",
    ]
    reg = company.dropna(subset=base_cols).copy()
    if reg.empty:
        return pd.DataFrame()

    section_dummies = pd.get_dummies(reg["section_code"], prefix="sec", drop_first=True)
    region_dummies = pd.get_dummies(reg["Region"].fillna("Unknown"), prefix="region", drop_first=True)
    class_dummies = pd.get_dummies(
        reg["Classification"].fillna("Unknown"), prefix="class", drop_first=True
    )
    legal_form_dummies = pd.get_dummies(
        reg["Legal Form"].fillna("Unknown"), prefix="legal_form", drop_first=True
    )
    y = reg["efficiency_growth_gap_pct"].to_numpy(dtype=float)

    models: list[tuple[str, list[str]]] = [
        ("base_controls", ["log_turnover_2024", "log_employees_2024", "time_span"]),
        (
            "add_productivity",
            [
                "log_turnover_2024",
                "log_employees_2024",
                "time_span",
                "log_turnover_per_employee_2024",
            ],
        ),
        (
            "extended_company_controls",
            [
                "log_turnover_2024",
                "log_employees_2024",
                "time_span",
                "log_turnover_per_employee_2024",
                "firm_age_2024",
                "log_personnel_costs",
                "board_women_share",
            ],
        ),
    ]

    weight_schemes: list[tuple[str, np.ndarray | None]] = [
        ("ols_equal_weights", None),
        ("wls_by_employees", np.sqrt(reg["employees_2024"].to_numpy(dtype=float))),
        ("wls_by_turnover", np.sqrt(reg["turnover_2024"].to_numpy(dtype=float))),
    ]

    rows: list[pd.DataFrame] = []
    for model_name, vars_in_model in models:
        x_num = reg[vars_in_model].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(reg)), x_num])
        term_names = ["Intercept", *vars_in_model]
        if not section_dummies.empty:
            x = np.column_stack([x, section_dummies.to_numpy(dtype=float)])
            term_names.extend(section_dummies.columns.tolist())
        if model_name == "extended_company_controls":
            if not region_dummies.empty:
                x = np.column_stack([x, region_dummies.to_numpy(dtype=float)])
                term_names.extend(region_dummies.columns.tolist())
            if not class_dummies.empty:
                x = np.column_stack([x, class_dummies.to_numpy(dtype=float)])
                term_names.extend(class_dummies.columns.tolist())
            if not legal_form_dummies.empty:
                x = np.column_stack([x, legal_form_dummies.to_numpy(dtype=float)])
                term_names.extend(legal_form_dummies.columns.tolist())

        for weighting_name, w in weight_schemes:
            rows.append(
                _weighted_least_squares(
                    y=y,
                    x=x,
                    term_names=term_names,
                    model_name=model_name,
                    weighting_name=weighting_name,
                    weights=w,
                )
            )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_weighted_scalability_score_experiment(company: pd.DataFrame) -> pd.DataFrame:
    data = company.dropna(
        subset=["turnover_cagr_pct", "employees_cagr_pct", "efficiency_growth_gap_pct"]
    ).copy()
    if data.empty:
        return pd.DataFrame()

    grid = np.arange(0.5, 3.01, 0.25)
    rows: list[dict[str, float]] = []
    target = data["efficiency_growth_gap_pct"]
    for alpha in grid:
        # score = turnover growth - alpha * employee growth
        score = data["turnover_cagr_pct"] - alpha * data["employees_cagr_pct"]
        corr = score.corr(target)
        rows.append(
            {
                "employee_penalty_alpha": float(alpha),
                "score_mean": float(score.mean()),
                "score_std": float(score.std(ddof=1)),
                "corr_with_efficiency_gap": float(corr) if pd.notna(corr) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("corr_with_efficiency_gap", ascending=False)
    return out


def run() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Hittar inte datafilen: {DATA_FILE.resolve()}")
    if not SNI_FILE.exists():
        raise FileNotFoundError(f"Hittar inte SNI-filen: {SNI_FILE.resolve()}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    columns = inspect_excel_columns()
    company = build_company_dataset()
    regressions = run_regression_experiments(company)
    weighting_experiment = run_weighted_scalability_score_experiment(company)

    columns_path = OUTPUT_DIR / "excel_column_inventory.csv"
    base_path = OUTPUT_DIR / "company_regression_base.csv"
    regressions_path = OUTPUT_DIR / "multiple_regression_experiments.csv"
    weighting_path = OUTPUT_DIR / "weighted_scalability_scores.csv"

    columns.to_csv(columns_path, index=False)
    company.to_csv(base_path, index=False)
    regressions.to_csv(regressions_path, index=False)
    weighting_experiment.to_csv(weighting_path, index=False)

    print("\nRegressionsanalys klar.")
    print(f"- Antal bolag efter samma filterregler: {len(company)}")
    print(f"- Antal kolumner inventerade från Excel: {len(columns)}")

    if not regressions.empty:
        print("\nMultipla regressionsspecifikationer (utdrag):")
        preview = regressions[
            [
                "model_name",
                "weighting_name",
                "term",
                "coef",
                "std_err",
                "p_value",
                "r_squared",
            ]
        ]
        print(preview.head(30).to_string(index=False))

        print("\nSlutsatser (regression):")
        key_terms = [
            "log_turnover_2024",
            "log_employees_2024",
            "time_span",
            "log_turnover_per_employee_2024",
            "firm_age_2024",
            "log_personnel_costs",
            "board_women_share",
        ]
        for term in key_terms:
            term_rows = regressions[regressions["term"] == term].copy()
            if term_rows.empty:
                continue
            n_models = len(term_rows)
            n_sig = int((term_rows["p_value"] < 0.05).sum())
            median_coef = float(term_rows["coef"].median())
            direction = "positiv" if median_coef > 0 else "negativ"
            print(
                f"- {term}: signifikant i {n_sig}/{n_models} modeller, "
                f"median koefficient {median_coef:.3f} ({direction} riktning)."
            )

        model_fit = (
            regressions.groupby(["model_name", "weighting_name"], as_index=False)["r_squared"]
            .first()
            .sort_values("r_squared", ascending=False)
        )
        print("\nModellpassning (R^2):")
        print(model_fit.to_string(index=False))

    if not weighting_experiment.empty:
        print("\nViktnings-test av skalbarhetsscore:")
        print(weighting_experiment.head(10).to_string(index=False))
        best = weighting_experiment.iloc[0]
        print(
            "\nSlutsats (vikter): "
            f"bäst alpha enligt korrelation mot efficiency gap = "
            f"{best['employee_penalty_alpha']:.2f} "
            f"(korrelation {best['corr_with_efficiency_gap']:.4f})."
        )

    print("\nFiler sparade:")
    print(f"- {columns_path.resolve()}")
    print(f"- {base_path.resolve()}")
    print(f"- {regressions_path.resolve()}")
    print(f"- {weighting_path.resolve()}")


if __name__ == "__main__":
    run()
