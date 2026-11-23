"""
generate_outputs.py — Генерує всі рисунки, таблиці та підсумковий файл.
Запуск: python generate_outputs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # без GUI, для збереження файлів
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import risk_model as rm

# ── Параметри ──────────────────────────────────────────
CFG    = rm.CONFIG
N      = CFG.get("n_simulations_report", CFG["n_simulations"])
SEED   = CFG["seed"]
NU     = CFG["student_df"]
PV     = CFG["portfolio_value"]
TICKERS = CFG["tickers"]

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

print("=" * 60)
print("ГЕНЕРАЦІЯ ВИХОДІВ — Монте-Карло ризик-менеджмент")
print("=" * 60)

# ── 1. Дані ─────────────────────────────────────────────
print("\n[1/8] Завантаження даних...")
prices = rm.load_prices(
    tickers=TICKERS,
    start_date=CFG["start_date"],
    end_date=CFG["end_date"],
    cache_path="data/prices.csv",
)
returns = rm.compute_returns(prices)
mu, sigma_vec, cov = rm.estimate_parameters(returns)
WEIGHTS = rm.portfolio_weights_valid(CFG["weights"])
NU_HAT = rm.estimate_student_df(returns, WEIGHTS)
print(f"       Оцінка ν (ММП, портфель): {NU_HAT:.2f} → у моделі ν = {NU}")

print(f"       Даних: {len(returns)} торгових днів ({returns.index[0].date()} → {returns.index[-1].date()})")

# ── 2. Описова статистика (Таблиця 2.1) ─────────────────
print("\n[2/8] Таблиця 2.1 — описова статистика...")
stats_df = rm.descriptive_stats(returns, trading_days=CFG["trading_days_per_year"])
rm.save_csv(stats_df, "results/table_2_2_statistics.csv")
print(stats_df.round(4).to_string())

# ── 3. Рис. 2.1 — Теплова карта кореляцій ───────────────
print("\n[3/8] Рис. 2.2 — теплова карта кореляцій...")
fig = rm.plot_correlation_heatmap(returns, save_path="figures/fig_2_2_correlation_heatmap.png")
plt.close(fig)

# ── 4. Рис. 2.2 — Гістограми + QQ ───────────────────────
print("[4/8] Рис. 2.1 — гістограми + QQ-графіки...")
fig = rm.plot_returns_hist_qq(returns, save_path="figures/fig_2_1_returns_hist_qq.png")
plt.close(fig)

# ── 5. Симуляція + Рис. 2.3 + Таблиця 2.2 ───────────────
print("\n[5/8] Симуляція та Таблиця 2.2 — VaR/CVaR за методами...")
sim_norm_1  = rm.simulate_normal(mu, cov, N, 1,  SEED)
sim_stud_1  = rm.simulate_student(mu, cov, N, 1,  NU, SEED)
sim_norm_10 = rm.simulate_normal(mu, cov, N, 10, SEED)
sim_stud_10 = rm.simulate_student(mu, cov, N, 10, NU, SEED)

losses_norm_1  = rm.portfolio_losses(sim_norm_1,  WEIGHTS, PV)
losses_stud_1  = rm.portfolio_losses(sim_stud_1,  WEIGHTS, PV)
losses_norm_10 = rm.portfolio_losses(sim_norm_10, WEIGHTS, PV)
losses_stud_10 = rm.portfolio_losses(sim_stud_10, WEIGHTS, PV)

fig = rm.plot_loss_distribution(
    losses_norm_1, losses_stud_1, PV,
    CFG["confidence_levels"],
    save_path="figures/fig_2_3_loss_distribution.png"
)
plt.close(fig)

results_df = rm.build_results_table(
    returns=returns, mu=mu, cov=cov,
    weights=WEIGHTS, portfolio_value=PV,
    holding_periods=CFG["holding_periods"],
    confidence_levels=CFG["confidence_levels"],
    n_sim=N, student_df=NU, seed=SEED,
)
rm.save_csv(results_df, "results/table_2_3_var_cvar.csv")
print(results_df.to_string(index=False))

# ── 6. Збіжність (Таблиця 2.3 + Рис. 2.4) ──────────────
print("\n[6/8] Аналіз збіжності CVaR (N-сітка, може зайняти ~1-2 хв)...")
conv_df = rm.convergence_study(
    mu=mu, cov=cov, weights=WEIGHTS, portfolio_value=PV,
    n_grid=CFG["convergence_n_grid"],
    holding_period=1, confidence_level=0.99,
    student_df=NU, seed=SEED,
    threshold=CFG["convergence_threshold"],
)
rm.save_csv(conv_df, "results/table_2_4_convergence.csv")
print(conv_df.to_string(index=False))
print(f"  → Мін. N (нормальний):  {conv_df.attrs.get('min_n_normal')}")
print(f"  → Мін. N (Стьюдент):    {conv_df.attrs.get('min_n_student')}")

fig = rm.plot_convergence(conv_df, save_path="figures/fig_2_4_convergence.png")
plt.close(fig)

# ── 7. Чутливість + Стрес (Таблиці 2.4, 2.5 + Рис. 2.5) ─
print("\n[7/8] Аналіз чутливості та стрес-сценарій...")
sens_df = rm.sensitivity_volatility(
    mu=mu, cov=cov, weights=WEIGHTS, portfolio_value=PV,
    holding_period=1, confidence_level=0.99,
    vol_multipliers=CFG["sensitivity_vol_multipliers"],
    n_sim=N, student_df=NU, seed=SEED,
)
rm.save_csv(sens_df, "results/table_2_5_sensitivity_vol.csv")
print("Чутливість σ:")
print(sens_df.to_string(index=False))

weight_scenarios = {
    "Базовий (40/20/25/15)":          [0.40, 0.20, 0.25, 0.15],
    "Акціє-орієнтований (60/30/5/5)": [0.60, 0.30, 0.05, 0.05],
    "Захисний (20/10/50/20)":         [0.20, 0.10, 0.50, 0.20],
    "Рівноважний (25/25/25/25)":      [0.25, 0.25, 0.25, 0.25],
}
sens_w_df = rm.sensitivity_weights(
    mu=mu, cov=cov, portfolio_value=PV,
    holding_period=1, confidence_level=0.99,
    weight_scenarios=weight_scenarios,
    n_sim=N, student_df=NU, seed=SEED,
)
rm.save_csv(sens_w_df, "results/table_2_7_sensitivity_weights.csv")

fig = rm.plot_sensitivity(sens_df, save_path="figures/fig_2_5_sensitivity.png")
plt.close(fig)

stress_df = rm.stress_test(
    mu=mu, cov=cov, weights=WEIGHTS, portfolio_value=PV,
    holding_period=1,
    confidence_levels=CFG["confidence_levels"],
    n_sim=N, student_df=NU, seed=SEED,
    target_corr=CFG["stress_target_corr"],
    vol_multiplier=CFG["stress_vol_multiplier"],
)
rm.save_csv(stress_df, "results/table_2_8_stress.csv")
print("\nСтрес-сценарій:")
print(stress_df.to_string(index=False))


# ── 7b. Чутливість до ν і розклад стрес-ефекту (Таблиці 2.6, 2.8b) ──
nu_rows = []
for nu in [3, 4, 6, 12, 30]:
    L_nu = rm.portfolio_losses(rm.simulate_student(mu, cov, N, 1, nu, SEED), WEIGHTS, PV)
    v95, c95 = rm.compute_var_cvar(L_nu, 0.95); v99, c99 = rm.compute_var_cvar(L_nu, 0.99)
    nu_rows.append({"ν": nu, "VaR 95% ($)": round(v95, 2), "CVaR 95% ($)": round(c95, 2),
                    "VaR 99% ($)": round(v99, 2), "CVaR 99% ($)": round(c99, 2)})
L_n = rm.portfolio_losses(rm.simulate_normal(mu, cov, N, 1, SEED), WEIGHTS, PV)
v95, c95 = rm.compute_var_cvar(L_n, 0.95); v99, c99 = rm.compute_var_cvar(L_n, 0.99)
nu_rows.append({"ν": "∞ (нормальний)", "VaR 95% ($)": round(v95, 2), "CVaR 95% ($)": round(c95, 2),
                "VaR 99% ($)": round(v99, 2), "CVaR 99% ($)": round(c99, 2)})
nu_df = pd.DataFrame(nu_rows)
rm.save_csv(nu_df, "results/table_2_6_sensitivity_nu.csv")
print("Чутливість до ν:"); print(nu_df.to_string(index=False))

dec_rows = []
for name, tc, vm in [("Базовий", None, 1.0),
                     ("Лише кореляції → 0,9", CFG["stress_target_corr"], 1.0),
                     ("Лише волатильність × 1,5", None, CFG["stress_vol_multiplier"]),
                     ("Кореляції і волатильність разом", CFG["stress_target_corr"], CFG["stress_vol_multiplier"])]:
    if tc is None:
        c_use = cov * vm ** 2
    else:
        c_use, _ = rm.build_stress_cov(cov, tc, vm)
    L_s = rm.portfolio_losses(rm.simulate_student(mu, c_use, N, 1, NU, SEED), WEIGHTS, PV)
    v99, c99 = rm.compute_var_cvar(L_s, 0.99)
    dec_rows.append({"Сценарій": name, "VaR 99% ($)": round(v99, 2), "CVaR 99% ($)": round(c99, 2)})
stress_dec_df = pd.DataFrame(dec_rows)
base_c = stress_dec_df.loc[0, "CVaR 99% ($)"]
stress_dec_df["Зміна CVaR (%)"] = ((stress_dec_df["CVaR 99% ($)"] / base_c - 1) * 100).round(1)
rm.save_csv(stress_dec_df, "results/table_2_8b_stress_decomposition.csv")
print("Розклад стрес-ефекту:"); print(stress_dec_df.to_string(index=False))

# ── 8. XLSX + Підсумок ───────────────────────────────────
print("\n[8/8] Збереження XLSX та підсумкового файлу...")
rm.save_tables_to_excel(
    tables={
        "2.2 Статистика":    stats_df,
        "2.3 VaR CVaR":      results_df,
        "2.4 Збіжність":     conv_df,
        "2.5 Чутливість σ":  sens_df,
        "2.6 Чутливість ν":  nu_df,
        "2.7 Ваги":          sens_w_df,
        "2.8 Стрес":         stress_df,
        "2.8b Стрес-розклад": stress_dec_df,
    },
    path="results/all_tables.xlsx",
)

# ── Підсумковий результат ─────────────────────────────────
print("\n" + "=" * 60)
print("ПІДСУМОК КЛЮЧОВИХ РЕЗУЛЬТАТІВ")
print("=" * 60)

# Збираємо ключові числа для results_summary.md
var95_n, cvar95_n = rm.compute_var_cvar(losses_norm_1, 0.95)
var99_n, cvar99_n = rm.compute_var_cvar(losses_norm_1, 0.99)
var95_s, cvar95_s = rm.compute_var_cvar(losses_stud_1, 0.95)
var99_s, cvar99_s = rm.compute_var_cvar(losses_stud_1, 0.99)
var95_n10, cvar95_n10 = rm.compute_var_cvar(losses_norm_10, 0.95)
var99_n10, cvar99_n10 = rm.compute_var_cvar(losses_norm_10, 0.99)
var95_s10, cvar95_s10 = rm.compute_var_cvar(losses_stud_10, 0.95)
var99_s10, cvar99_s10 = rm.compute_var_cvar(losses_stud_10, 0.99)

min_n_norm = conv_df.attrs.get("min_n_normal")
min_n_stud = conv_df.attrs.get("min_n_student")

stress_row_99 = stress_df[stress_df["α"] == 0.99].iloc[0]

dev95 = (cvar95_s - cvar95_n) / abs(cvar95_n) * 100
dev99 = (cvar99_s - cvar99_n) / abs(cvar99_n) * 100

# Бектестинг (якщо вистачає даних)
bt_text = ""
try:
    bt = rm.backtest_var(returns, WEIGHTS, PV, 0.95, 250)
    fig = rm.plot_backtest(bt, save_path="figures/fig_2_6_backtest.png")
    plt.close(fig)
    bt_text = f"""
## 6. Бектестинг VaR 95% (вікно 250 днів)
- Тестовий розріз: {bt['n_test_days']} днів
- Очікувана кількість перевищень: {bt['expected_exceedances']}
- Параметричний VaR: {bt['parametric']['exceedances']} перевищень | LR={bt['parametric']['kupiec_lr']:.3f} | p={bt['parametric']['kupiec_pvalue']:.3f}
- Історичний VaR: {bt['historical']['exceedances']} перевищень | LR={bt['historical']['kupiec_lr']:.3f} | p={bt['historical']['kupiec_pvalue']:.3f}
- **Висновок:** {'H0 не відхиляється (модель калібрована)' if bt['parametric']['kupiec_pvalue'] > 0.05 else 'H0 відхиляється (модель перевищує очікувану частоту)'}
"""
except Exception as e:
    bt_text = f"\n## 6. Бектестинг\nНе виконано: {e}\n"
    print(f"  Бектестинг: {e}")

# ── Запис results_summary.md ─────────────────────────────
summary = f"""# Підсумок результатів моделювання — Розділ 2.3
Курсова робота: «Моделювання фінансового ризику методом Монте-Карло»

---

## Параметри моделі
- **Портфель:** SPY ({WEIGHTS[0]:.0%}), VGK ({WEIGHTS[1]:.0%}), TLT ({WEIGHTS[2]:.0%}), GLD ({WEIGHTS[3]:.0%})
- **Вартість портфеля:** ${PV:,.0f} (~1 000 000 грн)
- **Даних:** {len(returns)} торгових днів ({returns.index[0].date()} → {returns.index[-1].date()})
- **Кількість сценаріїв N:** {N:,} | seed = {SEED}
- **Розподіл Стьюдента:** ν = {NU} (оцінка ММП ν̂ = {NU_HAT:.2f})

---

## 1. VaR і CVaR за методами (горизонт h = 1 день)

| Метод | VaR 95% ($) | CVaR 95% ($) | VaR 99% ($) | CVaR 99% ($) |
|-------|------------|-------------|------------|-------------|
| Нормальний (МК) | {var95_n:.2f} | {cvar95_n:.2f} | {var99_n:.2f} | {cvar99_n:.2f} |
| Стьюдент ν={NU} (МК) | {var95_s:.2f} | {cvar95_s:.2f} | {var99_s:.2f} | {cvar99_s:.2f} |

**Відхилення Стьюдент від нормального:**
- CVaR 95%: {dev95:+.1f}%
- CVaR 99%: {dev99:+.1f}%

**Висновок:** Розподіл Стьюдента (ν={NU}) дає вищі оцінки CVaR на {dev99:.1f}% (α=99%), що кількісно показує, наскільки припущення нормальності **занижує** крайні втрати.

---

## 2. VaR і CVaR за методами (горизонт h = 10 днів)

| Метод | VaR 95% ($) | CVaR 95% ($) | VaR 99% ($) | CVaR 99% ($) |
|-------|------------|-------------|------------|-------------|
| Нормальний (МК) | {var95_n10:.2f} | {cvar95_n10:.2f} | {var99_n10:.2f} | {cvar99_n10:.2f} |
| Стьюдент ν={NU} (МК) | {var95_s10:.2f} | {cvar95_s10:.2f} | {var99_s10:.2f} | {cvar99_s10:.2f} |

---

## 3. Збіжність CVaR₉₉

- **Мінімальне достатнє N (нормальний):** {min_n_norm}
- **Мінімальне достатнє N (Стьюдент):**   {min_n_stud}
- **Правило зупинки:** найменше N, з якого відносна зміна CVaR між сусідніми точками сітки стабільно < 1%
- **Висновок:** При N ≥ {min_n_stud or min_n_norm} оцінка CVaR стабілізується. Базовий N = {N:,} є достатнім.

---

## 4. Чутливість CVaR₉₉ до зміни волатильності (Стьюдент, h=1)

{sens_df.to_markdown(index=False)}

**Висновок:** При збільшенні волатильності на 50% (множник 1.5) CVaR зростає приблизно пропорційно. Розподіл Стьюдента чутливіший до хвостових ризиків.

---

## 5. Стрес-сценарій «збіг кореляцій» (кореляції → {CFG['stress_target_corr']}, σ × {CFG['stress_vol_multiplier']})

| α | CVaR базовий ($) | CVaR стрес ($) | Зміна (%) |
|---|-----------------|---------------|----------|
| 95% | {stress_df.iloc[0]['CVaR базовий ($)']:.2f} | {stress_df.iloc[0]['CVaR стрес ($)']:.2f} | {stress_df.iloc[0]['Зміна CVaR (%)']:+.1f}% |
| 99% | {stress_row_99['CVaR базовий ($)']:.2f} | {stress_row_99['CVaR стрес ($)']:.2f} | {stress_row_99['Зміна CVaR (%)']:+.1f}% |

**Корекція PD:** {stress_df.iloc[0]['Корекція PD']}
**Висновок:** В умовах ринкової кризи («збіг кореляцій») CVaR зростає на ~{stress_row_99['Зміна CVaR (%)']:.0f}%, що підтверджує важливість стрес-тестування.
{bt_text}
---

## Перелік рисунків

| Файл | Зміст |
|------|-------|
| figures/fig_2_2_correlation_heatmap.png | Рис. 2.2 — Теплова карта кореляцій |
| figures/fig_2_1_returns_hist_qq.png | Рис. 2.1 — Гістограми + QQ |
| figures/fig_2_3_loss_distribution.png | Рис. 2.3 — Розподіл збитків, VaR/CVaR |
| figures/fig_2_4_convergence.png | Рис. 2.4 — Збіжність CVaR |
| figures/fig_2_5_sensitivity.png | Рис. 2.5 — Чутливість CVaR |
| figures/fig_2_6_backtest.png | Рис. 2.6 — Бектестинг (якщо є) |

---
*Усі числа отримані запуском коду. Seed={SEED}. Дата генерації: {pd.Timestamp.today().date()}.*
"""

Path("results/results_summary.md").write_text(summary, encoding="utf-8")
print(summary)

print("\n" + "=" * 60)
print("ГОТОВО! Усі файли збережено.")
print(f"  figures/  — {len(list(Path('figures').glob('*.png')))} PNG-рисунків")
print(f"  results/  — {len(list(Path('results').glob('*')))} файлів (CSV + XLSX + MD)")
print("=" * 60)
