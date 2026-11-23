"""
risk_model.py — Монте-Карло моделювання фінансового ризику інвестиційного портфеля.

Курсова робота: «Моделювання фінансового ризику методом Монте-Карло»
Дисципліна: Інтелектуальний аналіз даних та фінансове моделювання

Усі функції є спільними для ноутбука та Streamlit-застосунку.
Імена змінних і функцій — англійською; коментарі, docstring, підписи — українською.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.stats as stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# БЛОК НАЛАШТУВАНЬ — усі параметри в одному місці
# ─────────────────────────────────────────────
CONFIG = {
    # Активи та їхні ваги в портфелі
    "tickers": ["SPY", "VGK", "TLT", "GLD"],
    "weights": [0.40, 0.20, 0.25, 0.15],
    # Вартість портфеля (умовна, у доларовому еквіваленті)
    # 1 000 000 грн / 41 грн/$ ≈ 24 390 $
    # Примітка: валютний ризик гривні у базовій моделі не моделюється
    "portfolio_value": 24_390.0,
    # Період завантаження даних
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    # Параметри моделювання
    "n_simulations": 100_000,         # базова кількість сценаріїв (вебзастосунок)
    # Кількість сценаріїв для підсумкових таблиць курсової — обрана за правилом
    # зупинки (для розподілу Стьюдента CVaR стабілізується лише з N ≈ 1 000 000)
    "n_simulations_report": 1_000_000,
    "seed": 42,                        # seed генератора (відтворюваність)
    "holding_periods": [1, 10],        # горизонти в торгових днях
    "confidence_levels": [0.95, 0.99], # рівні довіри для VaR/CVaR
    # Параметри розподілу Стьюдента
    # Оцінено методом максимальної правдоподібності за денними дохідностями
    # портфеля (estimate_student_df): ν̂ ≈ 3,83 → округлено до 4
    "student_df": 4,
    # Параметри стрес-сценарію
    "stress_target_corr": 0.90,        # цільова кореляція між ризиковими активами
    "stress_vol_multiplier": 1.50,     # множник волатильності
    # Аналіз чутливості — множники волатильності
    "sensitivity_vol_multipliers": [0.8, 1.0, 1.2, 1.5],
    # Сітка N для аналізу збіжності
    "convergence_n_grid": [1_000, 2_000, 5_000, 10_000, 20_000,
                            50_000, 100_000, 200_000, 500_000,
                            1_000_000, 2_000_000],
    "convergence_threshold": 0.01,     # правило зупинки: < 1 %
    # Кількість торгових днів у році (для річної ануалізації)
    "trading_days_per_year": 252,
    # Шляхи для збереження результатів
    "data_dir": "data",
    "figures_dir": "figures",
    "results_dir": "results",
}


# ─────────────────────────────────────────────
# РОЗДІЛ 2.1 — ПІДГОТОВКА ВИХІДНИХ ДАНИХ
# ─────────────────────────────────────────────

def load_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_path: str = "data/prices.csv",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Завантажує скориговані ціни закриття (Adjusted Close) з yfinance
    та зберігає їх у кеш-файл CSV.

    Якщо файл кешу вже існує і force_download=False — читає з диска,
    щоб результати не залежали від наявності інтернету.

    Parameters
    ----------
    tickers : список тікерів ETF
    start_date, end_date : межі часового ряду (рядки YYYY-MM-DD)
    cache_path : шлях до CSV-файлу кешу
    force_download : якщо True — завжди завантажує з мережі

    Returns
    -------
    DataFrame з датами як індекс і тікерами як стовпці
    """
    cache = Path(cache_path)
    if cache.exists() and not force_download:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"[Дані] Завантажено з кешу: {cache_path}")
        return prices

    import yfinance as yf
    print(f"[Дані] Завантаження з yfinance: {tickers}, {start_date}–{end_date}…")
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    # Вибираємо стовпець Close (після auto_adjust це скоригована ціна)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices.dropna(how="all", inplace=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    print(f"[Дані] Збережено в кеш: {cache_path}  ({len(prices)} рядків)")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Обчислює щоденні логарифмічні дохідності та видаляє пропуски.

    r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : DataFrame цін закриття

    Returns
    -------
    DataFrame денних лог-дохідностей (без рядків з NaN)
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns.dropna(inplace=True)
    return log_returns


def descriptive_stats(returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Розраховує описову статистику для кожного активу.

    Показники: середня денна і річна дохідність, денна та річна волатильність,
    мінімум, максимум, асиметрія (skewness), ексцес (excess kurtosis),
    статистика і p-value тесту Жарка–Бера (перевірка нормальності).

    Parameters
    ----------
    returns : DataFrame денних лог-дохідностей
    trading_days : кількість торгових днів на рік (для ануалізації)

    Returns
    -------
    DataFrame з рядками = активи, стовпцями = статистики
    """
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        jb_stat, jb_p = stats.jarque_bera(r)
        rows.append({
            "Актив": col,
            "Середня денна дохідність": r.mean(),
            "Річна дохідність": r.mean() * trading_days,
            "Денна волатильність": r.std(),
            "Річна волатильність": r.std() * np.sqrt(trading_days),
            "Мінімум": r.min(),
            "Максимум": r.max(),
            "Асиметрія": float(stats.skew(r)),
            "Ексцес": float(stats.kurtosis(r)),
            "Жарк-Бера (статистика)": jb_stat,
            "Жарк-Бера (p-value)": jb_p,
        })
    df = pd.DataFrame(rows).set_index("Актив")
    return df


def portfolio_weights_valid(weights: list[float]) -> np.ndarray:
    """
    Нормує ваги до суми 1.0 і перевіряє їхню коректність.

    Parameters
    ----------
    weights : список вагових часток

    Returns
    -------
    ndarray нормованих ваг
    """
    w = np.array(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("Ваги не можуть бути від'ємними.")
    total = w.sum()
    if total == 0:
        raise ValueError("Сума ваг не може дорівнювати нулю.")
    return w / total


# ─────────────────────────────────────────────
# РОЗДІЛ 2.2 — МОДЕЛЬ МОНТЕ-КАРЛО
# ─────────────────────────────────────────────

def estimate_parameters(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Оцінює параметри розподілу дохідностей за вибіркою.

    Повертає:
    - mu  : вектор середніх денних дохідностей (n,)
    - sigma_vec : вектор денних волатильностей (n,)
    - cov : коваріаційна матриця денних дохідностей (n, n)

    Parameters
    ----------
    returns : DataFrame денних лог-дохідностей

    Returns
    -------
    (mu, sigma_vec, cov)
    """
    mu = returns.mean().values
    sigma_vec = returns.std().values
    cov = returns.cov().values
    return mu, sigma_vec, cov


def cholesky_decompose(cov: np.ndarray) -> np.ndarray:
    """
    Виконує розклад Холецького коваріаційної матриці: Σ = L · Lᵀ.

    Якщо матриця не є строго додатно визначеною — додає невелику
    регуляризацію (nugget) і повторює спробу.

    Parameters
    ----------
    cov : коваріаційна матриця (n, n)

    Returns
    -------
    L : нижньотрикутна матриця Холецького (n, n)
    """
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Додаємо мінімальну регуляризацію для чисельної стабільності
        nugget = 1e-8 * np.eye(cov.shape[0])
        L = np.linalg.cholesky(cov + nugget)
        print("[Cholesky] Додано регуляризацію nugget=1e-8 для чисельної стабільності.")
    return L


def simulate_normal(
    mu: np.ndarray,
    cov: np.ndarray,
    n_sim: int,
    holding_period: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Генерує N сценаріїв кумулятивної лог-дохідності за h днів
    з багатовимірного нормального розподілу через розклад Холецького.

    Алгоритм:
      1. Масштабуємо: mu_h = mu * h,  cov_h = cov * h
      2. L = cholesky(cov_h)
      3. Генеруємо ε ~ N(0, I), розмір (N, n)
      4. R = mu_h + ε @ L.T  — корельовані дохідності

    Parameters
    ----------
    mu : вектор середніх денних дохідностей (n,)
    cov : коваріаційна матриця денних дохідностей (n, n)
    n_sim : кількість сценаріїв
    holding_period : горизонт утримання в торгових днях
    seed : початкове значення генератора

    Returns
    -------
    returns_sim : масив сценаріїв дохідностей (N, n)
    """
    h = holding_period
    mu_h = mu * h
    cov_h = cov * h
    L = cholesky_decompose(cov_h)

    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    eps = rng.standard_normal((n_sim, n_assets))   # (N, n) ~ N(0, I)
    returns_sim = mu_h + eps @ L.T                  # (N, n), коваріація → cov_h
    return returns_sim


def simulate_student(
    mu: np.ndarray,
    cov: np.ndarray,
    n_sim: int,
    holding_period: int,
    df: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """
    Генерує N сценаріїв кумулятивної лог-дохідності за h днів
    з багатовимірного розподілу Стьюдента (важкі хвости).

    Алгоритм (багатовимірний t-розподіл):
      1. Z = ε · Lᵀ, ε ~ N(0, I) — корельований нормальний вектор
      2. W ~ χ²(ν) — одна спільна величина для всіх активів у сценарії
      3. R = μ·h + Z · sqrt((ν − 2) / W)

    Множник sqrt((ν − 2)/W) робить коваріацію сценаріїв рівною Σ·h
    (бо E[(ν − 2)/W] = 1), тож порівняння з нормальним розподілом
    відбувається за однакової волатильності — відрізняються лише хвости.
    Спільний W означає, що екстремальні значення в активах настають
    одночасно (хвостова залежність), як це буває в кризу.

    Parameters
    ----------
    mu : вектор середніх денних дохідностей (n,)
    cov : коваріаційна матриця денних дохідностей (n, n)
    n_sim : кількість сценаріїв
    holding_period : горизонт утримання в торгових днях
    df : ступені свободи ν (ν > 2 для скінченної дисперсії)
    seed : початкове значення генератора

    Returns
    -------
    returns_sim : масив сценаріїв дохідностей (N, n)
    """
    if df <= 2:
        raise ValueError("Ступені свободи ν мають бути > 2 (для скінченної дисперсії).")
    h = holding_period
    mu_h = mu * h
    cov_h = cov * h
    L = cholesky_decompose(cov_h)

    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    eps = rng.standard_normal((n_sim, n_assets))     # ε ~ N(0, I)
    z = eps @ L.T                                    # Cov(z) = cov_h
    w_chi2 = rng.chisquare(df, size=n_sim)           # W ~ χ²(ν)
    scale = np.sqrt((df - 2) / w_chi2)[:, None]      # E[scale²] = 1
    returns_sim = mu_h + z * scale
    return returns_sim


def estimate_student_df(
    returns: pd.DataFrame,
    weights: np.ndarray,
) -> float:
    """
    Оцінює кількість ступенів свободи ν розподілу Стьюдента методом
    максимальної правдоподібності за денними дохідностями портфеля.

    Parameters
    ----------
    returns : DataFrame денних лог-дохідностей активів
    weights : ваги активів

    Returns
    -------
    nu_hat : оцінка ν
    """
    w = portfolio_weights_valid(list(weights))
    r_port = np.log(np.exp(returns.values) @ w)
    nu_hat, _, _ = stats.t.fit(r_port)
    return float(nu_hat)


def portfolio_losses(
    returns_sim: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float,
) -> np.ndarray:
    """
    Перетворює матрицю сценаріїв дохідностей на вектор збитків портфеля.

    Вартість портфеля в кінці горизонту:
        V_T = V0 * Σ_i w_i * exp(R_i)
    Збиток (від'ємне значення = прибуток):
        L = V0 − V_T

    Parameters
    ----------
    returns_sim : масив дохідностей (N, n)
    weights : нормовані ваги активів (n,)
    portfolio_value : початкова вартість портфеля V0

    Returns
    -------
    losses : вектор збитків (N,), > 0 означає фактичний збиток
    """
    w = portfolio_weights_valid(list(weights))
    # Вартість у кожному сценарії: V_T = V0 * Σ w_i * exp(R_i)
    portfolio_returns = np.sum(w * np.exp(returns_sim), axis=1) - 1.0
    losses = -portfolio_value * portfolio_returns   # збиток > 0
    return losses


# ─────────────────────────────────────────────
# РОЗДІЛ 2.3 — ОЦІНЮВАННЯ РИЗИКУ
# ─────────────────────────────────────────────

def compute_var_cvar(
    losses: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Оцінює VaR і CVaR за вибіркою змодельованих збитків.

    VaR_α = α-квантиль розподілу збитків
    CVaR_α = середнє збитків, що перевищують VaR_α (Expected Shortfall)

    Parameters
    ----------
    losses : вектор збитків (N,)
    confidence_level : рівень довіри α (0.95 або 0.99)

    Returns
    -------
    (var, cvar) : пара оцінок у тих самих одиницях, що й losses
    """
    var = float(np.quantile(losses, confidence_level))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def historical_var_cvar(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    holding_period: int,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Оцінює VaR і CVaR за фактичними (історичними) дохідностями.

    Для h = 1: використовує денні дохідності напряму.
    Для h > 1: використовує ковзні h-денні кумулятивні дохідності
               (сума логарифмічних дохідностей за h послідовних днів).

    Parameters
    ----------
    returns : DataFrame денних лог-дохідностей
    weights : нормовані ваги активів
    portfolio_value : початкова вартість портфеля
    holding_period : горизонт h
    confidence_level : рівень довіри α

    Returns
    -------
    (var, cvar) : пара оцінок
    """
    w = portfolio_weights_valid(list(weights))
    if holding_period == 1:
        r_h = returns.values
    else:
        # Ковзні h-денні суми лог-дохідностей
        r_h = np.array([
            returns.values[i:i + holding_period].sum(axis=0)
            for i in range(len(returns) - holding_period + 1)
        ])
    losses = portfolio_losses(r_h, w, portfolio_value)
    return compute_var_cvar(losses, confidence_level)


def parametric_var_cvar(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float,
    holding_period: int,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Аналітичні (параметричні) VaR і CVaR за нормальним розподілом.

    Формули для портфеля:
        μ_p = w·μ·h,   σ_p = sqrt(wᵀ Σ h w)
        VaR_α   = −V0·(μ_p + σ_p·Φ⁻¹(1−α))
        CVaR_α  = −V0·(μ_p − σ_p·φ(Φ⁻¹(α))/(1−α))

    Parameters
    ----------
    mu : вектор середніх денних дохідностей
    cov : коваріаційна матриця
    weights : ваги активів
    portfolio_value : V0
    holding_period : h
    confidence_level : α

    Returns
    -------
    (var, cvar) : аналітичні оцінки
    """
    w = portfolio_weights_valid(list(weights))
    h = holding_period
    mu_p = float(w @ mu) * h
    sigma_p = float(np.sqrt(w @ cov @ w)) * np.sqrt(h)

    alpha = confidence_level
    z = stats.norm.ppf(alpha)
    phi_z = stats.norm.pdf(z)

    var  = portfolio_value * (-mu_p + sigma_p * z)
    cvar = portfolio_value * (-mu_p + sigma_p * phi_z / (1 - alpha))
    return float(var), float(cvar)


def build_results_table(
    returns: pd.DataFrame,
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float,
    holding_periods: list[int],
    confidence_levels: list[float],
    n_sim: int,
    student_df: int,
    seed: int,
) -> pd.DataFrame:
    """
    Будує зведену таблицю VaR і CVaR за трьома підходами:
    нормальний МК, Стьюдента МК, історичний.

    Стовпці: горизонт, α, метод, VaR ($), CVaR ($), VaR (%), CVaR (%)

    Parameters
    ----------
    (всі вхідні параметри задокументовані вище)

    Returns
    -------
    DataFrame з результатами
    """
    rows = []
    for h in holding_periods:
        # 1) Нормальний МК
        sim_norm = simulate_normal(mu, cov, n_sim, h, seed)
        losses_norm = portfolio_losses(sim_norm, weights, portfolio_value)
        # 2) Стьюдент МК
        sim_stud = simulate_student(mu, cov, n_sim, h, student_df, seed)
        losses_stud = portfolio_losses(sim_stud, weights, portfolio_value)
        # 3) Параметричний (аналітичний нормальний)
        # 4) Історичний

        for alpha in confidence_levels:
            var_n, cvar_n = compute_var_cvar(losses_norm, alpha)
            var_s, cvar_s = compute_var_cvar(losses_stud, alpha)
            var_p, cvar_p = parametric_var_cvar(mu, cov, weights,
                                                portfolio_value, h, alpha)
            var_h, cvar_h = historical_var_cvar(returns, weights,
                                                portfolio_value, h, alpha)
            for method, var, cvar in [
                ("Нормальний (МК)", var_n, cvar_n),
                ("Стьюдент (МК)", var_s, cvar_s),
                ("Параметричний", var_p, cvar_p),
                ("Історичний", var_h, cvar_h),
            ]:
                rows.append({
                    "Горизонт (днів)": h,
                    "Рівень довіри α": alpha,
                    "Метод": method,
                    "VaR ($)": round(var, 2),
                    "CVaR ($)": round(cvar, 2),
                    "VaR (%)": round(var / portfolio_value * 100, 2),
                    "CVaR (%)": round(cvar / portfolio_value * 100, 2),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# АНАЛІЗ ЗБІЖНОСТІ
# ─────────────────────────────────────────────

def convergence_study(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float,
    n_grid: list[int],
    holding_period: int,
    confidence_level: float,
    student_df: int,
    seed: int,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Вивчає збіжність оцінки CVaR залежно від кількості сценаріїв N.

    Правило зупинки: N* — найменше N сітки, починаючи з якого відносна
    зміна CVaR між сусідніми значеннями сітки стабільно менша за threshold:
        |CVaR(N_k) − CVaR(N_{k−1})| / |CVaR(N_{k−1})| < 1 % для всіх k ≥ k*

    Parameters
    ----------
    n_grid : список значень N (зростаючий порядок)
    threshold : відносний поріг зупинки (0.01 = 1 %)

    Returns
    -------
    DataFrame зі стовпцями: N, CVaR_norm, CVaR_stud, RelChange_norm, RelChange_stud
    """
    rows = []
    cvar_prev_n, cvar_prev_s = None, None

    for n in n_grid:
        sim_n = simulate_normal(mu, cov, n, holding_period, seed)
        losses_n = portfolio_losses(sim_n, weights, portfolio_value)
        _, cvar_n = compute_var_cvar(losses_n, confidence_level)

        sim_s = simulate_student(mu, cov, n, holding_period, student_df, seed)
        losses_s = portfolio_losses(sim_s, weights, portfolio_value)
        _, cvar_s = compute_var_cvar(losses_s, confidence_level)

        rel_n = abs(cvar_n - cvar_prev_n) / abs(cvar_prev_n) if cvar_prev_n else None
        rel_s = abs(cvar_s - cvar_prev_s) / abs(cvar_prev_s) if cvar_prev_s else None

        rows.append({
            "N": n,
            "CVaR (нормальний)": round(cvar_n, 2),
            "CVaR (Стьюдент)": round(cvar_s, 2),
            "Відн. зміна (норм.)": round(rel_n, 4) if rel_n is not None else None,
            "Відн. зміна (Стьюд.)": round(rel_s, 4) if rel_s is not None else None,
        })
        cvar_prev_n, cvar_prev_s = cvar_n, cvar_s

    df = pd.DataFrame(rows)

    def _stable_from(col: str):
        """Найменше N, починаючи з якого ВСІ подальші відносні зміни < threshold."""
        changes = df[col].tolist()
        for i in range(1, len(changes)):
            tail = changes[i:]
            if all(c is not None and not pd.isna(c) and c < threshold for c in tail):
                return int(df["N"].iloc[i])
        return None

    min_n_norm = _stable_from("Відн. зміна (норм.)")
    min_n_stud = _stable_from("Відн. зміна (Стьюд.)")
    df.attrs["min_n_normal"] = min_n_norm
    df.attrs["min_n_student"] = min_n_stud
    return df


# ─────────────────────────────────────────────
# АНАЛІЗ ЧУТЛИВОСТІ
# ─────────────────────────────────────────────

def sensitivity_volatility(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float,
    holding_period: int,
    confidence_level: float,
    vol_multipliers: list[float],
    n_sim: int,
    student_df: int,
    seed: int,
) -> pd.DataFrame:
    """
    Аналізує чутливість VaR і CVaR до зміни волатильності.

    Масштабує коваріаційну матрицю: Σ_scaled = D · Σ · D,
    де D = diag(multiplier), що еквівалентно множенню σ_i на multiplier.

    Parameters
    ----------
    vol_multipliers : список множників волатильності (напр. [0.8, 1.0, 1.2, 1.5])

    Returns
    -------
    DataFrame з результатами для кожного множника
    """
    rows = []
    for mult in vol_multipliers:
        # Масштабування: σ → σ·m  ⟹  cov → cov · m²
        cov_scaled = cov * (mult ** 2)
        sim_n = simulate_normal(mu, cov_scaled, n_sim, holding_period, seed)
        losses_n = portfolio_losses(sim_n, weights, portfolio_value)
        var_n, cvar_n = compute_var_cvar(losses_n, confidence_level)

        sim_s = simulate_student(mu, cov_scaled, n_sim, holding_period, student_df, seed)
        losses_s = portfolio_losses(sim_s, weights, portfolio_value)
        var_s, cvar_s = compute_var_cvar(losses_s, confidence_level)

        rows.append({
            "Множник σ": mult,
            "VaR (норм., $)": round(var_n, 2),
            "CVaR (норм., $)": round(cvar_n, 2),
            "VaR (Стьюд., $)": round(var_s, 2),
            "CVaR (Стьюд., $)": round(cvar_s, 2),
        })
    return pd.DataFrame(rows)


def sensitivity_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    portfolio_value: float,
    holding_period: int,
    confidence_level: float,
    weight_scenarios: dict[str, list[float]],
    n_sim: int,
    student_df: int,
    seed: int,
) -> pd.DataFrame:
    """
    Аналізує чутливість VaR і CVaR до зміни розподілу ваг портфеля.

    Parameters
    ----------
    weight_scenarios : словник {назва_сценарію: [w1, w2, w3, w4]}

    Returns
    -------
    DataFrame з результатами для кожного сценарію ваг
    """
    rows = []
    for scenario_name, w in weight_scenarios.items():
        weights_arr = portfolio_weights_valid(w)
        sim_s = simulate_student(mu, cov, n_sim, holding_period, student_df, seed)
        losses_s = portfolio_losses(sim_s, weights_arr, portfolio_value)
        var_s, cvar_s = compute_var_cvar(losses_s, confidence_level)

        rows.append({
            "Сценарій ваг": scenario_name,
            "SPY": round(weights_arr[0], 2),
            "VGK": round(weights_arr[1], 2),
            "TLT": round(weights_arr[2], 2),
            "GLD": round(weights_arr[3], 2),
            "VaR ($)": round(var_s, 2),
            "CVaR ($)": round(cvar_s, 2),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# СТРЕС-СЦЕНАРІЙ «ЗБІГ КОРЕЛЯЦІЙ»
# ─────────────────────────────────────────────

def build_stress_cov(
    cov: np.ndarray,
    target_corr: float = 0.90,
    vol_multiplier: float = 1.50,
) -> tuple[np.ndarray, bool]:
    """
    Будує стресову коваріаційну матрицю:
    1. Підвищує попарні кореляції до target_corr
    2. Масштабує волатильності на vol_multiplier

    Якщо результуюча матриця не є додатно визначеною — коригує
    до найближчої PD-матриці методом Хігема (scipy).

    Parameters
    ----------
    cov : базова коваріаційна матриця
    target_corr : цільова кореляція між активами
    vol_multiplier : множник волатильностей

    Returns
    -------
    (cov_stress, was_corrected) : матриця і прапорець корекції
    """
    n = cov.shape[0]
    sigma = np.sqrt(np.diag(cov))

    # Нова кореляційна матриця: target_corr скрізь, крім діагоналі
    C_stress = np.full((n, n), target_corr)
    np.fill_diagonal(C_stress, 1.0)

    # Нові волатильності
    sigma_stress = sigma * vol_multiplier

    # Коваріаційна матриця: Σ_stress = D · C_stress · D
    D = np.diag(sigma_stress)
    cov_stress = D @ C_stress @ D

    # Перевірка на positive definiteness
    eigenvalues = np.linalg.eigvalsh(cov_stress)
    was_corrected = False
    if np.any(eigenvalues <= 0):
        # Метод Хігема: проєкція на конус PD-матриць
        cov_stress = _nearest_pd(cov_stress)
        was_corrected = True
        print("[Стрес] Матриця скоригована до найближчої PD методом Хігема.")

    return cov_stress, was_corrected


def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """
    Знаходить найближчу симетричну додатно визначену матрицю.
    Алгоритм Хігема (2002): ітеративна проєкція.

    Parameters
    ----------
    A : симетрична матриця

    Returns
    -------
    найближча PD-матриця
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    # Невелика регуляризація
    mineig = np.min(np.real(np.linalg.eigvals(A3)))
    if mineig < 0:
        A3 += (-mineig * np.eye(A.shape[0]) + 1e-8)
    return A3


def stress_test(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    portfolio_value: float,
    holding_period: int,
    confidence_levels: list[float],
    n_sim: int,
    student_df: int,
    seed: int,
    target_corr: float = 0.90,
    vol_multiplier: float = 1.50,
) -> pd.DataFrame:
    """
    Порівнює VaR і CVaR базового і стресового сценаріїв.

    Стресовий сценарій:
    - Усі попарні кореляції між активами = target_corr
    - Волатильності × vol_multiplier

    Parameters
    ----------
    (параметри задокументовані вище)

    Returns
    -------
    DataFrame порівняльної таблиці
    """
    cov_stress, corrected = build_stress_cov(cov, target_corr, vol_multiplier)
    rows = []

    for alpha in confidence_levels:
        # Базовий сценарій
        sim_b = simulate_student(mu, cov, n_sim, holding_period, student_df, seed)
        losses_b = portfolio_losses(sim_b, weights, portfolio_value)
        var_b, cvar_b = compute_var_cvar(losses_b, alpha)

        # Стресовий сценарій
        sim_s = simulate_student(mu, cov_stress, n_sim, holding_period, student_df, seed)
        losses_s = portfolio_losses(sim_s, weights, portfolio_value)
        var_s, cvar_s = compute_var_cvar(losses_s, alpha)

        rows.append({
            "α": alpha,
            "VaR базовий ($)": round(var_b, 2),
            "CVaR базовий ($)": round(cvar_b, 2),
            "VaR стрес ($)": round(var_s, 2),
            "CVaR стрес ($)": round(cvar_s, 2),
            "Зміна VaR (%)": round((var_s - var_b) / abs(var_b) * 100, 1),
            "Зміна CVaR (%)": round((cvar_s - cvar_b) / abs(cvar_b) * 100, 1),
            "Корекція PD": "Так" if corrected else "Ні",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# БЕКТЕСТИНГ (необов'язковий блок)
# ─────────────────────────────────────────────

def backtest_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence_level: float,
    window: int = 250,
) -> dict:
    """
    Бектестинг одноденного VaR на ковзному вікні.

    Для кожного дня t у тестовому розрізі (останні 252 дні):
    - оцінює VaR за попередніми `window` спостереженнями (параметрично і історично)
    - фіксує перевищення: 1, якщо фактичний збиток > VaR

    Тест Купієця (Unconditional Coverage):
        LR = -2·ln[p^x·(1-p)^(n-x)] + 2·ln[(x/n)^x·(1-x/n)^(n-x)]
        H0: p_exceedance = 1 - α
        LR ~ χ²(1)

    Parameters
    ----------
    window : розмір ковзного вікна (торгових днів)
    confidence_level : рівень довіри α

    Returns
    -------
    Словник з результатами бектестингу та тесту Купієця
    """
    w = portfolio_weights_valid(list(weights))
    r_port = (returns @ w).values
    n_test = 252
    total = len(r_port)
    if total < window + n_test:
        raise ValueError(f"Недостатньо даних: потрібно {window + n_test}, є {total}")

    test_start = total - n_test
    var_hist_list, var_param_list, actual_losses = [], [], []

    mu_all, _, cov_all = estimate_parameters(returns)

    for t in range(test_start, total):
        window_r = r_port[t - window: t]
        # Порт-дохідність кожного дня у вікні
        mu_w = window_r.mean()
        sigma_w = window_r.std()

        # Параметричний VaR (нормальний, одноденний)
        z = stats.norm.ppf(confidence_level)
        var_p = portfolio_value * (-mu_w + sigma_w * z)

        # Історичний VaR
        port_losses_w = -portfolio_value * window_r
        var_h = float(np.quantile(port_losses_w, confidence_level))

        # Фактичний збиток за день t
        actual_loss = -portfolio_value * r_port[t]

        var_param_list.append(var_p)
        var_hist_list.append(var_h)
        actual_losses.append(actual_loss)

    actual_losses = np.array(actual_losses)
    var_param_arr = np.array(var_param_list)
    var_hist_arr = np.array(var_hist_list)

    exc_param = int((actual_losses > var_param_arr).sum())
    exc_hist = int((actual_losses > var_hist_arr).sum())
    n = len(actual_losses)
    expected_exc = int(round(n * (1 - confidence_level)))

    def kupiec_lr(n_exc, n_obs, alpha):
        """LR-статистика тесту Купієця (unconditional coverage)."""
        p_hat = n_exc / n_obs
        p0 = 1 - alpha
        if n_exc == 0:
            lr = -2 * (n_obs * np.log(1 - p0))
        elif n_exc == n_obs:
            lr = -2 * (n_obs * np.log(p0))
        else:
            lr = -2 * (n_exc * np.log(p0) + (n_obs - n_exc) * np.log(1 - p0))
            lr -= -2 * (n_exc * np.log(p_hat) + (n_obs - n_exc) * np.log(1 - p_hat))
        return float(lr)

    lr_p = kupiec_lr(exc_param, n, confidence_level)
    lr_h = kupiec_lr(exc_hist, n, confidence_level)
    pval_p = float(stats.chi2.sf(lr_p, df=1))
    pval_h = float(stats.chi2.sf(lr_h, df=1))

    return {
        "n_test_days": n,
        "expected_exceedances": expected_exc,
        "confidence_level": confidence_level,
        "parametric": {
            "exceedances": exc_param,
            "kupiec_lr": round(lr_p, 4),
            "kupiec_pvalue": round(pval_p, 4),
            "var_series": var_param_arr,
        },
        "historical": {
            "exceedances": exc_hist,
            "kupiec_lr": round(lr_h, 4),
            "kupiec_pvalue": round(pval_h, 4),
            "var_series": var_hist_arr,
        },
        "actual_losses": actual_losses,
    }


# ─────────────────────────────────────────────
# ГРАФІКИ — matplotlib (для ноутбука та PNG)
# ─────────────────────────────────────────────

def _setup_matplotlib_ukrainian():
    """Налаштовує matplotlib для коректного відображення кирилиці."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує теплову карту кореляційної матриці дохідностей.

    Рис. 2.2 курсової роботи.

    Parameters
    ----------
    returns : DataFrame денних лог-дохідностей
    save_path : якщо вказано — зберігає PNG 300 dpi

    Returns
    -------
    Figure matplotlib
    """
    _setup_matplotlib_ukrainian()
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Коефіцієнт кореляції", fontsize=10)

    tickers = list(corr.columns)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_yticklabels(tickers, fontsize=11)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")

    ax.set_title("Рис. 2.2 – Кореляційна матриця денних лог-дохідностей", fontsize=12, pad=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Рисунок] Збережено: {save_path}")
    return fig


def plot_returns_hist_qq(
    returns: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує гістограми денних дохідностей з нормальною кривою та QQ-графіки.

    Рис. 2.2 курсової роботи.
    «Товсті хвости» видно на QQ-графіку як відхилення від прямої.

    Parameters
    ----------
    returns : DataFrame денних лог-дохідностей
    save_path : якщо вказано — зберігає PNG 300 dpi

    Returns
    -------
    Figure matplotlib
    """
    _setup_matplotlib_ukrainian()
    tickers = list(returns.columns)
    n = len(tickers)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))

    for i, ticker in enumerate(tickers):
        r = returns[ticker].dropna()
        mu_r, sigma_r = r.mean(), r.std()

        # Гістограма
        ax_hist = axes[0, i]
        ax_hist.hist(r, bins=60, density=True, color="steelblue",
                     alpha=0.65, edgecolor="none", label="Факт.")
        x = np.linspace(r.min(), r.max(), 300)
        ax_hist.plot(x, stats.norm.pdf(x, mu_r, sigma_r),
                     "r-", lw=1.8, label="Норм.")
        ax_hist.set_title(ticker, fontsize=12)
        ax_hist.set_xlabel("Дохідність", fontsize=10)
        ax_hist.set_ylabel("Щільність", fontsize=10)
        if i == 0:
            ax_hist.legend(fontsize=9)

        # QQ-графік
        ax_qq = axes[1, i]
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        ax_qq.scatter(osm, osr, s=5, color="steelblue", alpha=0.5)
        line_x = np.array([osm[0], osm[-1]])
        ax_qq.plot(line_x, slope * line_x + intercept, "r-", lw=1.8)
        ax_qq.set_title(f"{ticker} — QQ-графік", fontsize=11)
        ax_qq.set_xlabel("Теоретичні квантилі", fontsize=10)
        ax_qq.set_ylabel("Вибіркові квантилі", fontsize=10)

    fig.suptitle(
        "Рис. 2.1 – Розподіл денних дохідностей: гістограми та QQ-графіки",
        fontsize=12, y=1.01
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Рисунок] Збережено: {save_path}")
    return fig


def plot_loss_distribution(
    losses_normal: np.ndarray,
    losses_student: np.ndarray,
    portfolio_value: float,
    confidence_levels: list[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує гістограму розподілу змодельованих збитків портфеля
    для нормального розподілу і розподілу Стьюдента.

    Вертикальні лінії — VaR і CVaR для кожного рівня довіри.
    Рис. 2.3 курсової роботи.

    Parameters
    ----------
    losses_normal : збитки з нормального МК
    losses_student : збитки зі Стьюдент МК
    portfolio_value : V0 (для підпису осі у %)
    confidence_levels : рівні довіри для VaR/CVaR
    save_path : якщо вказано — зберігає PNG

    Returns
    -------
    Figure matplotlib
    """
    _setup_matplotlib_ukrainian()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    titles = ["Нормальний розподіл", f"Розподіл Стьюдента (ν={CONFIG['student_df']})"]
    colors_line = {0.95: ("navy", "cornflowerblue"),
                   0.99: ("darkred", "salmon")}

    # Спільний діапазон осі X для обох панелей (0,05–99,95 перцентилі),
    # щоб поодинокі екстремальні сценарії t-розподілу не стискали графік
    both = np.concatenate([losses_normal, losses_student])
    x_lo, x_hi = np.quantile(both, [0.0005, 0.9995])
    bins = np.linspace(x_lo, x_hi, 121)

    for ax, losses, title in zip(axes, [losses_normal, losses_student], titles):
        ax.hist(losses[(losses >= x_lo) & (losses <= x_hi)], bins=bins, density=True,
                color="steelblue", alpha=0.5, edgecolor="none", label="Збитки")
        ax.set_xlim(x_lo, x_hi)

        for alpha in confidence_levels:
            var, cvar = compute_var_cvar(losses, alpha)
            c_var, c_cvar = colors_line[alpha]
            ax.axvline(var, color=c_var, lw=1.8, ls="--",
                       label=f"VaR {int(alpha*100)}%  = {var:,.0f} $")
            ax.axvline(cvar, color=c_cvar, lw=1.8, ls="-.",
                       label=f"CVaR {int(alpha*100)}% = {cvar:,.0f} $")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Збиток ($)", fontsize=11)
        ax.set_ylabel("Щільність", fontsize=11)
        ax.legend(fontsize=8.5)

    fig.suptitle(
        "Рис. 2.3 – Розподіл змодельованих збитків портфеля з лініями VaR і CVaR",
        fontsize=12
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Рисунок] Збережено: {save_path}")
    return fig


def plot_convergence(
    conv_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує графік залежності CVaR від кількості сценаріїв N.

    Вісь X — логарифмічна шкала. Показує стабілізацію оцінки
    при збільшенні N (правило зупинки).
    Рис. 2.4 курсової роботи.

    Parameters
    ----------
    conv_df : DataFrame з колонками N, CVaR (нормальний), CVaR (Стьюдент)
    save_path : якщо вказано — зберігає PNG

    Returns
    -------
    Figure matplotlib
    """
    _setup_matplotlib_hungarian()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(conv_df["N"], conv_df["CVaR (нормальний)"],
            "o-", color="steelblue", lw=1.8, ms=6, label="Нормальний")
    ax.plot(conv_df["N"], conv_df["CVaR (Стьюдент)"],
            "s-", color="tomato", lw=1.8, ms=6, label=f"Стьюдент (ν={CONFIG['student_df']})")

    # Позначити мінімальне достатнє N
    min_n_n = conv_df.attrs.get("min_n_normal")
    min_n_s = conv_df.attrs.get("min_n_student")
    if min_n_n:
        row = conv_df[conv_df["N"] == min_n_n].iloc[0]
        ax.axvline(min_n_n, color="steelblue", ls=":", lw=1.2, alpha=0.7)
        ax.annotate(f"N*={min_n_n:,}", xy=(min_n_n, row["CVaR (нормальний)"]),
                    xytext=(min_n_n * 1.3, row["CVaR (нормальний)"] * 1.02),
                    fontsize=9, color="steelblue", arrowprops=dict(arrowstyle="->", color="steelblue"))
    if min_n_s:
        row = conv_df[conv_df["N"] == min_n_s].iloc[0]
        ax.axvline(min_n_s, color="tomato", ls=":", lw=1.2, alpha=0.7)
        ax.annotate(f"N*={min_n_s:,}", xy=(min_n_s, row["CVaR (Стьюдент)"]),
                    xytext=(min_n_s * 0.25, row["CVaR (Стьюдент)"] * 1.05),
                    fontsize=9, color="tomato", arrowprops=dict(arrowstyle="->", color="tomato"))

    ax.set_xscale("log")
    ax.set_xlabel("Кількість сценаріїв N (лог. шкала)", fontsize=11)
    ax.set_ylabel("CVaR₉₉ ($)", fontsize=11)
    ax.set_title("Рис. 2.4 – Збіжність оцінки CVaR залежно від кількості сценаріїв", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Рисунок] Збережено: {save_path}")
    return fig


def _setup_matplotlib_hungarian():
    """Аліас для _setup_matplotlib_ukrainian (виправлення опечатки)."""
    _setup_matplotlib_ukrainian()


def plot_sensitivity(
    sens_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує графік чутливості CVaR до зміни волатильності.

    Рис. 2.5 курсової роботи.

    Parameters
    ----------
    sens_df : DataFrame з колонками «Множник σ», CVaR (норм., $), CVaR (Стьюд., $)
    save_path : якщо вказано — зберігає PNG

    Returns
    -------
    Figure matplotlib
    """
    _setup_matplotlib_ukrainian()
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sens_df["Множник σ"], sens_df["CVaR (норм., $)"],
            "o-", color="steelblue", lw=1.8, ms=7, label="Нормальний")
    ax.plot(sens_df["Множник σ"], sens_df["CVaR (Стьюд., $)"],
            "s-", color="tomato", lw=1.8, ms=7, label=f"Стьюдент (ν={CONFIG['student_df']})")

    ax.axvline(1.0, color="gray", ls=":", lw=1.2, label="Базовий рівень")
    ax.set_xlabel("Множник волатильності", fontsize=11)
    ax.set_ylabel("CVaR₉₉ ($)", fontsize=11)
    ax.set_title("Рис. 2.5 – Чутливість CVaR до зміни волатильності", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Рисунок] Збережено: {save_path}")
    return fig


def plot_backtest(
    backtest_result: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Будує графік бектестингу: фактичні збитки vs VaR.

    Parameters
    ----------
    backtest_result : словник, повернутий backtest_var()
    save_path : якщо вказано — зберігає PNG

    Returns
    -------
    Figure matplotlib
    """
    _setup_matplotlib_ukrainian()
    actual = backtest_result["actual_losses"]
    var_p = backtest_result["parametric"]["var_series"]
    var_h = backtest_result["historical"]["var_series"]
    n = len(actual)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(range(n), actual, color="steelblue", lw=0.8, alpha=0.7, label="Факт. збиток")
    ax.plot(range(n), var_p, color="tomato", lw=1.5, ls="--", label="VaR параметр.")
    ax.plot(range(n), var_h, color="darkgreen", lw=1.5, ls="-.", label="VaR історичний")

    exc_p_idx = np.where(actual > var_p)[0]
    exc_h_idx = np.where(actual > var_h)[0]
    ax.scatter(exc_p_idx, actual[exc_p_idx], color="tomato", s=20, zorder=5, label=f"Перевищення парам. ({len(exc_p_idx)})")
    ax.scatter(exc_h_idx, actual[exc_h_idx], color="darkgreen", s=20, marker="^", zorder=5, label=f"Перевищення істор. ({len(exc_h_idx)})")

    alpha = backtest_result["confidence_level"]
    ax.set_title(f"Рис. 2.6 – Бектестинг VaR {int(alpha*100)}% (ковзне вікно 250 днів)", fontsize=12)
    ax.set_xlabel("Торгових днів (тестовий розріз)", fontsize=11)
    ax.set_ylabel("Збиток ($)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[Рисунок] Збережено: {save_path}")
    return fig


# ─────────────────────────────────────────────
# ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ─────────────────────────────────────────────

def save_tables_to_excel(tables: dict[str, pd.DataFrame], path: str):
    """
    Зберігає кілька DataFrame на окремих аркушах XLSX.

    Parameters
    ----------
    tables : словник {назва_аркуша: DataFrame}
    path : шлях до XLSX-файлу
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=True)
    print(f"[Таблиці] Збережено: {path}")


def save_csv(df: pd.DataFrame, path: str):
    """
    Зберігає DataFrame у CSV з двома знаками після коми.

    Parameters
    ----------
    df : таблиця результатів
    path : шлях до файлу
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, float_format="%.2f")
    print(f"[CSV] Збережено: {path}")
