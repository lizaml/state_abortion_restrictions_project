"""Месячные флаги основных полных и ранних (<20 недель LMP) запретов.

Использует подготовленные restriction_spells.csv, state_coverage.csv и
court_and_timing_events.csv из одной папки policy. Исходные файлы не меняет.
Ноль означает отсутствие соответствующего эпизода в нашей коллекции,
а не отсутствие любых ограничений абортов. Неоднозначность сохраняется как NA.
Требования: pandas, numpy. Пример использования — research_variables.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["TEHC_ST", "reference_year", "MONTHCODE"]
FLAG_COLUMNS = [
    "restriction_flag", "restriction_month_end", "restriction_share",
    "restriction_days_confirmed", "restriction_days_uncertain",
    "policy_uncertain", "policy_covered",
]


def _integer(series):
    """Нормализует только служебные ключи; исходные столбцы не меняются."""
    number = pd.to_numeric(series, errors="coerce")
    return number.where(number.mod(1).eq(0) & number.between(-10000, 10000)).astype("Int64")


def _state(series):
    number = _integer(series)
    return number.where(number.between(1, 99)).astype("string").str.zfill(2)


def build_state_month_restrictions(policy_dir, first_year=2021, last_year=2024):
    """Создает таблицу штат × месяц в пределах заданных календарных лет.

    restriction_flag: 1 = есть хотя бы один подтвержденный день запрета;
    0 = нет учтенных подтвержденных/неоднозначных дней при полном охвате месяца;
    NA = подтвердить ни 1, ни 0 нельзя.

    restriction_share рассчитывается лишь при полном охвате и без неопределенных
    дней. Это доля календарных дней по датам источника, не почасовая экспозиция.
    """
    if not isinstance(first_year, int) or not isinstance(last_year, int) or first_year > last_year:
        raise ValueError("Нужны целые годы: first_year <= last_year.")
    folder = Path(policy_dir)
    spells = pd.read_csv(folder / "restriction_spells.csv", dtype="string")
    coverage = pd.read_csv(folder / "state_coverage.csv", dtype="string")
    events = pd.read_csv(folder / "court_and_timing_events.csv", dtype="string")
    for data in (spells, coverage, events):
        data["state_fips"] = _state(data["state_fips"])
        if data["state_fips"].isna().any():
            raise ValueError("Некорректный код штата в правовых исходниках.")
    if coverage["state_fips"].duplicated().any():
        raise ValueError("В state_coverage должна быть одна строка на штат.")
    states = set(coverage["state_fips"])
    if not set(spells["state_fips"]).issubset(states) or not set(events["state_fips"]).issubset(states):
        raise ValueError("Есть правовые записи для штата без описания охвата.")

    for name in ["start_date", "end_date_exclusive", "alternative_start_date", "alternative_end_date_exclusive"]:
        spells[name] = pd.to_datetime(spells[name], format="%Y-%m-%d", errors="raise")
    if spells[["start_date", "end_date_exclusive"]].isna().any().any():
        raise ValueError("У каждого эпизода должны быть начало и конец.")
    if not spells["start_date"].lt(spells["end_date_exclusive"]).all():
        raise ValueError("Конец эпизода должен быть позже начала.")
    coverage["collection_start"] = pd.to_datetime(coverage["collection_start"], errors="raise")
    coverage["collection_end"] = pd.to_datetime(coverage["collection_end"], errors="raise")
    if coverage[["collection_start", "collection_end"]].isna().any().any():
        raise ValueError("Не заполнен период охвата штата.")
    if not coverage["collection_start"].le(coverage["collection_end"]).all():
        raise ValueError("Некорректный период охвата штата.")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="raise")
    if events["event_date"].isna().any():
        raise ValueError("Не заполнена дата судебного события.")

    dates = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D")
    results = []
    for cov in coverage.itertuples(index=False):
        covered = np.asarray((dates >= cov.collection_start) & (dates <= cov.collection_end))
        definite = np.zeros(len(dates), dtype=bool)
        possible = np.zeros(len(dates), dtype=bool)
        state_spells = spells.loc[spells["state_fips"].eq(cov.state_fips)]

        for row in state_spells.itertuples(index=False):
            primary = np.asarray((dates >= row.start_date) & (dates < row.end_date_exclusive))
            alt_start = row.start_date if pd.isna(row.alternative_start_date) else row.alternative_start_date
            alt_end = row.end_date_exclusive if pd.isna(row.alternative_end_date_exclusive) else row.alternative_end_date_exclusive
            if alt_start >= alt_end:
                raise ValueError(f"Некорректные альтернативные даты: {row.episode_id}")
            alternative = np.asarray((dates >= alt_start) & (dates < alt_end))
            has_alternative = pd.notna(row.alternative_start_date) or pd.notna(row.alternative_end_date_exclusive)
            has_review = pd.notna(row.date_review_note) and bool(row.date_review_note.strip())

            if row.date_basis == "legal_enforcement":
                if row.restriction_type == "total_ban":
                    qualifies = True
                elif row.restriction_type == "gestational_ban":
                    if pd.isna(row.gestational_limit_weeks_lmp):
                        raise ValueError(f"Не указан срок беременности: {row.episode_id}")
                    qualifies = float(row.gestational_limit_weeks_lmp) < 20
                else:
                    raise ValueError(f"Неизвестный тип правового запрета: {row.restriction_type}")
                if not qualifies:
                    continue
                possible |= primary | alternative
                # При альтернативных датах уверены только в пересечении периодов.
                # Примечание без альтернативных дат означает неопределенность всего эпизода.
                if has_alternative:
                    definite |= primary & alternative
                elif not has_review:
                    definite |= primary
            else:
                # Спорная применимость закона или фактическое закрытие клиник.
                # Не превращаем их автоматически ни в правовой запрет, ни в отсутствие запрета.
                possible |= primary | alternative

        # В Вайоминге запрет и TRO пришлись на один день: такого эпизода
        # нет среди интервалов положительной длины, но считать этот день нулем нельзя.
        intraday = events.loc[
            events["state_fips"].eq(cov.state_fips)
            & events["event"].eq("scheduled_trigger_ban_and_same_day_TRO"),
            "event_date",
        ]
        possible |= np.asarray(dates.isin(intraday))
        definite &= covered
        uncertain = (possible & ~definite) | ~covered

        daily = pd.DataFrame({
            "reference_year": dates.year,
            "MONTHCODE": dates.month,
            "definite": definite,
            "uncertain": uncertain,
            "covered": covered,
        })
        monthly = daily.groupby(["reference_year", "MONTHCODE"], sort=True).agg(
            days_in_month=("definite", "size"),
            restriction_days_confirmed=("definite", "sum"),
            restriction_days_uncertain=("uncertain", "sum"),
            end_definite=("definite", "last"),
            end_uncertain=("uncertain", "last"),
            full_coverage=("covered", "all"),
        ).reset_index()
        monthly["TEHC_ST"] = cov.state_fips
        n = monthly["restriction_days_confirmed"]
        u = monthly["restriction_days_uncertain"]
        monthly["restriction_flag"] = pd.Series(pd.NA, index=monthly.index, dtype="Int8")
        monthly.loc[n.gt(0), "restriction_flag"] = 1
        monthly.loc[n.eq(0) & u.eq(0), "restriction_flag"] = 0
        monthly["restriction_month_end"] = monthly["end_definite"].astype("Int8").mask(monthly["end_uncertain"])
        monthly["restriction_share"] = (n / monthly["days_in_month"]).astype("Float64").mask(u.gt(0))
        monthly["policy_uncertain"] = u.gt(0).astype("Int8")
        monthly["policy_covered"] = monthly["full_coverage"].astype("Int8")
        for col in ["restriction_days_confirmed", "restriction_days_uncertain"]:
            monthly[col] = monthly[col].astype("Int8")
        results.append(monthly[KEYS + FLAG_COLUMNS])

    out = pd.concat(results, ignore_index=True)
    out["TEHC_ST"] = out["TEHC_ST"].astype("string")
    if out.duplicated(KEYS).any():
        raise ValueError("В таблице штат × месяц появились повторные ключи.")
    return out


def add_restriction_flag(df, policy_dir, first_year=2021, last_year=2024):
    """Возвращает копию df с семью новыми столбцами; не меняет строки или индекс.

    Использует TEHC_ST, reference_year, MONTHCODE. Статус для неизвестного
    штата, Foreign Country (61), пропущенного/невалидного месяца или года вне
    first_year:last_year остается NA. Значения новых флагов не заполняются нулями.
    """
    missing = set(KEYS) - set(df.columns)
    if missing:
        raise KeyError(f"В df отсутствуют столбцы: {sorted(missing)}")
    collisions = set(FLAG_COLUMNS) & set(df.columns)
    if collisions:
        raise ValueError(f"Флаги уже существуют; явно удалите их перед пересчетом: {sorted(collisions)}")
    monthly = build_state_month_restrictions(policy_dir, first_year, last_year)
    keys = pd.DataFrame({
        "TEHC_ST": _state(df["TEHC_ST"]).array,
        "reference_year": _integer(df["reference_year"]).array,
        "MONTHCODE": _integer(df["MONTHCODE"]).array,
        "_input_order": np.arange(len(df)),
    })
    keys["MONTHCODE"] = keys["MONTHCODE"].where(keys["MONTHCODE"].between(1, 12))
    joined = keys.merge(monthly, on=KEYS, how="left", validate="many_to_one", sort=False)
    joined = joined.sort_values("_input_order", kind="stable")
    if len(joined) != len(df):
        raise RuntimeError("При присоединении изменилось число строк.")
    result = df.copy()
    for column in FLAG_COLUMNS:
        # Позиционное присваивание сохраняет исходный, в том числе повторяющийся, индекс.
        result[column] = joined[column].array
    return result
