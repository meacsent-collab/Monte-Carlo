"""
app.py — Вебзастосунок Streamlit для демонстрації моделі Монте-Карло ризик-менеджменту.

Курсова робота: «Моделювання фінансового ризику методом Монте-Карло»
Дисципліна: Інтелектуальний аналіз даних та фінансове моделювання
Мова інтерфейсу: українська

Запуск: streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st

# Додаємо кореневу директорію до шляху для імпорту risk_model
sys.path.insert(0, str(Path(__file__).parent))

import risk_model as rm


# ─────────────────────────────────────────────
# КОНФІГУРАЦІЯ СТОРІНКИ ТА ТЕМНИЙ ФІНТЕХ-СТИЛЬ
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Монте-Карло | Оцінка фінансового ризику",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Глибоке темне тло з легким індиго-сяйвом */
.stApp {
    background: radial-gradient(circle at 20% 0%, #151C36 0%, #0B0E17 55%, #090B12 100%);
    color: #F1F5F9;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1320px;
}

/* Примусовий світлий текст для елементів Streamlit */
.stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #E2E8F0;
}

/* Верхня панель Streamlit */
header[data-testid="stHeader"] {
    background-color: rgba(11, 14, 23, 0.92) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
}

/* Бічна панель */
section[data-testid="stSidebar"] {
    background-color: #0F1322;
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #CBD5E1 !important;
}

/* Expander-и: темносиній текст на світлому фоні з неоновим зеленим обідком */
div[data-testid="stExpander"] {
    background-color: rgba(20, 25, 43, 0.82);
    border: 1px solid rgba(34, 197, 94, 0.45);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stExpander"] details summary {
    background-color: #F8FAFC !important;
    border-bottom: 1.5px solid #22C55E !important;
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.22);
    border-radius: 10px 10px 0 0;
}

div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span,
div[data-testid="stExpander"] details summary svg {
    color: #0F172A !important;
    fill: #0F172A !important;
    font-weight: 600 !important;
}

/* Поля вводу та випадаючі списки: зелений неон ободок + темносиній текст */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: #F8FAFC !important;
    border: 1.5px solid #22C55E !important;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.25) !important;
    border-radius: 9px !important;
    color: #0F172A !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] svg,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
input, textarea {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 500 !important;
}

/* Теги обраних активів у multiselect */
span[data-baseweb="tag"] {
    background-color: #1E3A8A !important;
    border: 1px solid #22C55E !important;
}

span[data-baseweb="tag"] span,
span[data-baseweb="tag"] svg {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Випадаючий список (popover) */
div[data-baseweb="popover"] ul,
ul[role="listbox"] {
    background-color: #F8FAFC !important;
    border: 1.5px solid #22C55E !important;
}

ul[role="listbox"] li,
ul[role="listbox"] li span {
    color: #0F172A !important;
}

/* Кнопки завантаження */
.stDownloadButton > button {
    background: rgba(99, 102, 241, 0.14);
    border: 1px solid rgba(99, 102, 241, 0.45);
    color: #C7D2FE !important;
    border-radius: 9px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stDownloadButton > button:hover {
    background: rgba(99, 102, 241, 0.26);
    border-color: #818CF8;
    color: #FFFFFF !important;
}

/* Верхній заголовок (Hero) */
.hero-card {
    background: linear-gradient(135deg, rgba(22, 29, 54, 0.92) 0%, rgba(15, 20, 36, 0.96) 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    padding: 1.45rem 1.75rem;
    border-radius: 16px;
    margin-bottom: 1.25rem;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #93C5FD;
    margin-bottom: 0.45rem;
}

.hero-title {
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.75rem 0;
    color: #FFFFFF !important;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(148, 163, 184, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.22);
    color: #E2E8F0 !important;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.79rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}

.pill-stress {
    background: rgba(244, 63, 94, 0.18);
    border-color: rgba(251, 113, 133, 0.55);
    color: #FECDD3 !important;
    font-weight: 600;
}

/* Картки KPI з тонкою градієнтною рамкою */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.95rem;
    margin: 0.75rem 0 1.25rem 0;
}

.kpi-card {
    position: relative;
    background: rgba(18, 22, 36, 0.92);
    border-radius: 14px;
    padding: 1.05rem 1.2rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
}

.kpi-card.accent-blue {
    border: 1px solid rgba(99, 102, 241, 0.45);
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.08);
}

.kpi-card.accent-red {
    border: 1px solid rgba(244, 63, 94, 0.48);
    box-shadow: 0 4px 20px rgba(244, 63, 94, 0.08);
}

.kpi-card.accent-slate {
    border: 1px solid rgba(148, 163, 184, 0.3);
}

.kpi-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}

.kpi-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.6rem;
    border-radius: 8px;
    font-size: 0.74rem;
    font-weight: 600;
}

.badge-blue {
    background: rgba(99, 102, 241, 0.16);
    color: #93C5FD !important;
}

.badge-red {
    background: rgba(244, 63, 94, 0.16);
    color: #FDA4AF !important;
}

.badge-slate {
    background: rgba(148, 163, 184, 0.16);
    color: #CBD5E1 !important;
}

.kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFFFFF !important;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.82rem;
    color: #94A3B8 !important;
    margin-top: 0.35rem;
    font-variant-numeric: tabular-nums;
}

.kpi-sub.danger {
    color: #FDA4AF !important;
    font-weight: 600;
}

/* Пунктирні картки (як на скріншоті 1: Рішення / Використання) */
.dashed-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.15rem;
    margin: 0.8rem 0 1.4rem 0;
}

.dashed-card {
    background: rgba(17, 20, 32, 0.88);
    border-radius: 16px;
    padding: 1.35rem 1.5rem;
}

.dashed-card.blue {
    border: 1.5px dashed rgba(99, 102, 241, 0.65);
}

.dashed-card.rose {
    border: 1.5px dashed rgba(244, 63, 94, 0.65);
}

.dashed-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.28rem 0.75rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.85rem;
}

.dashed-pill.blue {
    background: rgba(99, 102, 241, 0.15);
    color: #818CF8 !important;
}

.dashed-pill.rose {
    background: rgba(244, 63, 94, 0.15);
    color: #FB7185 !important;
}

.dashed-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #F8FAFC !important;
    margin-bottom: 0.85rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.sq-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.sq-list li {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    font-size: 0.92rem;
    color: #CBD5E1 !important;
    margin-bottom: 0.6rem;
    line-height: 1.45;
}

.sq-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    margin-top: 0.42rem;
    flex-shrink: 0;
}

.sq-dot.blue { background-color: #6366F1; }
.sq-dot.rose { background-color: #F43F5E; }

/* 4 картки з градієнтною рамкою (як на скріншоті 2: Коли впроваджувати?) */
.glow-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem;
    margin: 0.75rem 0 1.4rem 0;
}

.glow-card {
    background: linear-gradient(145deg, rgba(24, 26, 40, 0.95), rgba(16, 18, 28, 0.98));
    border-radius: 16px;
    padding: 1.25rem 1.3rem;
    border: 1px solid transparent;
    background-clip: padding-box;
    position: relative;
    box-shadow: 0 0 0 1px rgba(236, 72, 153, 0.35), 0 8px 24px rgba(0, 0, 0, 0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glow-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 0 1px rgba(236, 72, 153, 0.65), 0 10px 28px rgba(236, 72, 153, 0.12);
}

.glow-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(236, 72, 153, 0.2));
    border: 1px solid rgba(236, 72, 153, 0.32);
    color: #F9A8D4 !important;
    margin-bottom: 0.75rem;
}

.glow-title {
    font-size: 1rem;
    font-weight: 600;
    color: #F8FAFC !important;
    margin-bottom: 0.4rem;
}

.glow-text {
    font-size: 0.86rem;
    color: #94A3B8 !important;
    line-height: 1.5;
}

/* Інформаційний банер порівняння */
.insight-banner {
    background: rgba(30, 58, 138, 0.22);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-left: 4px solid #6366F1;
    border-radius: 10px;
    padding: 0.9rem 1.15rem;
    margin: 0.5rem 0 1.25rem 0;
    color: #E0E7FF !important;
    font-size: 0.92rem;
    line-height: 1.5;
}

.insight-banner.warning {
    background: rgba(136, 19, 55, 0.24);
    border-color: rgba(244, 63, 94, 0.4);
    border-left-color: #F43F5E;
    color: #FFE4E6 !important;
}

/* Заголовки секцій */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.12rem;
    font-weight: 600;
    color: #F8FAFC !important;
    margin: 1.1rem 0 0.25rem 0;
}

.section-desc {
    font-size: 0.88rem;
    color: #94A3B8 !important;
    margin-bottom: 0.9rem;
}

/* Вкладки Streamlit */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background-color: rgba(18, 22, 36, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.14);
    padding: 0.32rem;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.48rem 1rem;
    font-weight: 600;
    font-size: 0.87rem;
    color: #94A3B8 !important;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(59, 130, 246, 0.2)) !important;
    border: 1px solid rgba(99, 102, 241, 0.45) !important;
    color: #FFFFFF !important;
}

/* Футер */
.footer-note {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(148, 163, 184, 0.14);
    font-size: 0.82rem;
    color: #64748B !important;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Мінімалістичні векторні піктограмки (Lucide-стиль)
ICONS = {
    "chart": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "shield": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "flame": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>',
    "target": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "cpu": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
    "layers": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "lock": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    "activity": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
}

# Єдина темна палітра для графіків Plotly
COLOR_NORMAL = "#60A5FA"   # світлий індиго-синій
COLOR_STUDENT = "#FB7185"  # неоновий коралово-рожевий
COLOR_ACCENT = "#34D399"   # м'ятно-зелений
COLOR_MUTED = "#94A3B8"    # сріблясто-сірий


def style_plotly(fig: go.Figure, height: int = 400) -> go.Figure:
    """Застосовує єдиний темний фінансовий стиль до графіків Plotly."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color="#CBD5E1"),
        title_font=dict(size=14, color="#F8FAFC"),
        margin=dict(l=48, r=24, t=54, b=44),
        paper_bgcolor="rgba(17, 21, 36, 0.88)",
        plot_bgcolor="rgba(11, 14, 23, 0.65)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(17, 21, 36, 0.7)",
            font=dict(color="#E2E8F0"),
        ),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#F8FAFC", font_size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)", zeroline=False)
    return fig


# ─────────────────────────────────────────────
# КЕШОВАНІ ФУНКЦІЇ (УСЯ ЛОГІКА З risk_model.py)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Завантаження історичних цін активів…")
def cached_load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Кешує завантаження цін, щоб уникати зайвих запитів до yfinance."""
    return rm.load_prices(list(tickers), start, end)


@st.cache_data(show_spinner="Генерація сценаріїв Монте-Карло…")
def cached_simulate(
    mu_arr: tuple,
    cov_flat: tuple,
    cov_shape: tuple,
    n_sim: int,
    h: int,
    df_student: int,
    seed: int,
    stress: bool,
    target_corr: float,
    vol_mult: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Кешує генерацію нормальних та t-сценаріїв."""
    mu = np.array(mu_arr)
    cov = np.array(cov_flat).reshape(cov_shape)
    was_corrected = False

    if stress:
        cov_use, was_corrected = rm.build_stress_cov(cov, target_corr, vol_mult)
    else:
        cov_use = cov

    sim_norm = rm.simulate_normal(mu, cov_use, n_sim, h, seed)
    sim_stud = rm.simulate_student(mu, cov_use, n_sim, h, df_student, seed)
    return sim_norm, sim_stud, was_corrected


@st.cache_data(show_spinner="Розрахунок порівняльної таблиці…")
def cached_results_table(
    returns_values: np.ndarray,
    returns_cols: tuple,
    returns_idx: tuple,
    mu_arr: tuple,
    cov_flat: tuple,
    cov_shape: tuple,
    weights_arr: tuple,
    portfolio_value: float,
    holding_periods: tuple,
    n_sim: int,
    df_student: int,
    seed: int,
) -> pd.DataFrame:
    """Кешує зведену таблицю VaR і CVaR за всіма підходами."""
    ret_df = pd.DataFrame(returns_values, index=pd.to_datetime(returns_idx), columns=list(returns_cols))
    mu = np.array(mu_arr)
    cov = np.array(cov_flat).reshape(cov_shape)
    w = np.array(weights_arr)
    return rm.build_results_table(
        returns=ret_df,
        mu=mu,
        cov=cov,
        weights=w,
        portfolio_value=portfolio_value,
        holding_periods=list(holding_periods),
        confidence_levels=[0.95, 0.99],
        n_sim=n_sim,
        student_df=df_student,
        seed=seed,
    )


@st.cache_data(show_spinner="Аналіз збіжності CVaR…")
def cached_convergence(
    mu_arr: tuple,
    cov_flat: tuple,
    cov_shape: tuple,
    weights_arr: tuple,
    portfolio_value: float,
    n_grid: tuple,
    h: int,
    alpha: float,
    df_student: int,
    seed: int,
) -> tuple[pd.DataFrame, object, object]:
    """Кешує дослідження збіжності та повертає (DataFrame, min_n_norm, min_n_stud)."""
    mu = np.array(mu_arr)
    cov = np.array(cov_flat).reshape(cov_shape)
    w = np.array(weights_arr)
    conv_df = rm.convergence_study(
        mu=mu,
        cov=cov,
        weights=w,
        portfolio_value=portfolio_value,
        n_grid=list(n_grid),
        holding_period=h,
        confidence_level=alpha,
        student_df=df_student,
        seed=seed,
        threshold=rm.CONFIG["convergence_threshold"],
    )
    return (
        conv_df,
        conv_df.attrs.get("min_n_normal", "Не досягнуто"),
        conv_df.attrs.get("min_n_student", "Не досягнуто"),
    )


@st.cache_data(show_spinner="Аналіз чутливості та стрес-тестування…")
def cached_sensitivity_and_stress(
    mu_arr: tuple,
    cov_flat: tuple,
    cov_shape: tuple,
    weights_arr: tuple,
    portfolio_value: float,
    h: int,
    n_sim: int,
    df_student: int,
    seed: int,
    target_corr: float,
    vol_mult: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Кешує аналіз чутливості до волатильності, ваг і стрес-тест."""
    mu = np.array(mu_arr)
    cov = np.array(cov_flat).reshape(cov_shape)
    w = np.array(weights_arr)
    n_eff = min(n_sim, 50_000)

    sens_vol = rm.sensitivity_volatility(
        mu=mu,
        cov=cov,
        weights=w,
        portfolio_value=portfolio_value,
        holding_period=h,
        confidence_level=0.99,
        vol_multipliers=[0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        n_sim=n_eff,
        student_df=df_student,
        seed=seed,
    )

    n_assets = len(w)
    if n_assets == 4:
        weight_scenarios = {
            "Поточний вибір": list(w),
            "Акціє-орієнтований (60/30/5/5)": [0.60, 0.30, 0.05, 0.05],
            "Захисний (20/10/50/20)": [0.20, 0.10, 0.50, 0.20],
            "Рівноважний (25/25/25/25)": [0.25, 0.25, 0.25, 0.25],
        }
    else:
        equal_w = [1.0 / n_assets] * n_assets
        tilt_first = [0.50] + [0.50 / (n_assets - 1)] * (n_assets - 1)
        tilt_last = [0.50 / (n_assets - 1)] * (n_assets - 1) + [0.50]
        weight_scenarios = {
            "Поточний вибір": list(w),
            "Рівноважний": equal_w,
            "Фокус на 1-му активі (50%)": tilt_first,
            "Фокус на останньому активі (50%)": tilt_last,
        }

    sens_w = rm.sensitivity_weights(
        mu=mu,
        cov=cov,
        portfolio_value=portfolio_value,
        holding_period=h,
        confidence_level=0.99,
        weight_scenarios=weight_scenarios,
        n_sim=n_eff,
        student_df=df_student,
        seed=seed,
    )

    stress_df = rm.stress_test(
        mu=mu,
        cov=cov,
        weights=w,
        portfolio_value=portfolio_value,
        holding_period=h,
        confidence_levels=[0.95, 0.99],
        n_sim=n_eff,
        student_df=df_student,
        seed=seed,
        target_corr=target_corr,
        vol_multiplier=vol_mult,
    )
    return sens_vol, sens_w, stress_df


# ─────────────────────────────────────────────
# БІЧНА ПАНЕЛЬ — НАЛАШТУВАННЯ МОДЕЛІ
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Параметри моделі")

    with st.expander("Склад і ваги портфеля", expanded=True):
        all_tickers = ["SPY", "VGK", "TLT", "GLD", "EEM", "QQQ", "IWM", "AGG"]
        selected_tickers = st.multiselect(
            "Активи (ETF)",
            all_tickers,
            default=rm.CONFIG["tickers"],
            help="Оберіть щонайменше 2 активи",
        )
        if len(selected_tickers) < 2:
            st.error("Оберіть щонайменше 2 активи.")
            st.stop()

        default_weights = rm.CONFIG["weights"]
        raw_weights = []
        for i, ticker in enumerate(selected_tickers):
            default_pct = int(round(default_weights[i] * 100)) if i < len(default_weights) else int(round(100 / len(selected_tickers)))
            w_val = st.slider(
                f"Вага {ticker} (%)",
                min_value=0,
                max_value=100,
                value=default_pct,
                step=1,
            )
            raw_weights.append(float(w_val))

        weights_arr = rm.portfolio_weights_valid(raw_weights)
        weights_str = " · ".join(f"{t} {w:.0%}" for t, w in zip(selected_tickers, weights_arr))
        st.caption(f"Нормовані ваги: **{weights_str}**")

    with st.expander("Капітал та історичний період", expanded=True):
        col_uah, col_fx = st.columns(2)
        portfolio_uah = col_uah.number_input(
            "Портфель (грн)",
            min_value=10_000,
            max_value=100_000_000,
            value=1_000_000,
            step=50_000,
        )
        fx_rate = col_fx.number_input(
            "Курс грн/$",
            min_value=10.0,
            max_value=100.0,
            value=41.0,
            step=0.5,
            format="%.1f",
        )
        portfolio_value = float(portfolio_uah) / float(fx_rate)
        st.caption(f"Доларовий еквівалент: **${portfolio_value:,.0f}**")

        col_d1, col_d2 = st.columns(2)
        start_date = col_d1.text_input("Початок", value=rm.CONFIG["start_date"])
        end_date = col_d2.text_input("Кінець", value=rm.CONFIG["end_date"])

    with st.expander("Параметри симуляції", expanded=True):
        holding_period = st.selectbox("Горизонт утримання h (днів)", [1, 5, 10, 20], index=0)
        alpha = st.selectbox(
            "Рівень довіри α",
            [0.95, 0.99],
            index=1,
            format_func=lambda x: f"{int(x * 100)} %",
        )
        df_student = st.slider(
            "Ступені свободи ν (Стьюдент)",
            min_value=3,
            max_value=50,
            value=rm.CONFIG["student_df"],
        )
        n_sim = st.select_slider(
            "Кількість сценаріїв N",
            options=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000],
            value=rm.CONFIG["n_simulations"],
        )
        seed = st.number_input(
            "Seed (відтворюваність)",
            min_value=0,
            max_value=99999,
            value=rm.CONFIG["seed"],
        )

    with st.expander("Стрес-сценарій «збіг кореляцій»", expanded=False):
        stress_on = st.toggle("Активувати стрес-режим", value=False)
        target_corr = st.slider(
            "Цільова кореляція ρ",
            min_value=0.50,
            max_value=0.99,
            value=rm.CONFIG["stress_target_corr"],
            step=0.01,
            disabled=not stress_on,
        )
        vol_mult = st.slider(
            "Множник волатильності",
            min_value=1.0,
            max_value=3.0,
            value=rm.CONFIG["stress_vol_multiplier"],
            step=0.1,
            disabled=not stress_on,
        )


# ─────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ ДАНИХ ТА РОЗРАХУНОК
# ─────────────────────────────────────────────

CACHE_CSV = str(Path(__file__).parent / "data" / "prices.csv")

try:
    if Path(CACHE_CSV).exists():
        prices_all = pd.read_csv(CACHE_CSV, index_col=0, parse_dates=True)
        missing = [t for t in selected_tickers if t not in prices_all.columns]
        if missing:
            prices_all = cached_load_prices(tuple(selected_tickers), start_date, end_date)
        prices = prices_all[[t for t in selected_tickers if t in prices_all.columns]].dropna()
    else:
        prices = cached_load_prices(tuple(selected_tickers), start_date, end_date)
except Exception as e:
    st.error(f"Помилка завантаження даних: {e}")
    st.stop()

try:
    prices = prices[
        (prices.index >= pd.Timestamp(start_date))
        & (prices.index <= pd.Timestamp(end_date))
    ]
    prices = prices[[t for t in selected_tickers if t in prices.columns]].dropna()
except Exception as e:
    st.error(f"Некоректний формат дат: {e}")
    st.stop()

if len(prices) < 30:
    st.error("Замало торгових днів для обраного діапазону дат (потрібно щонайменше 30).")
    st.stop()

returns = rm.compute_returns(prices)
mu, sigma_vec, cov = rm.estimate_parameters(returns)

sim_norm, sim_stud, stress_pd_corrected = cached_simulate(
    mu_arr=tuple(mu),
    cov_flat=tuple(cov.flatten()),
    cov_shape=cov.shape,
    n_sim=n_sim,
    h=holding_period,
    df_student=df_student,
    seed=int(seed),
    stress=stress_on,
    target_corr=target_corr,
    vol_mult=vol_mult,
)

losses_norm = rm.portfolio_losses(sim_norm, weights_arr, portfolio_value)
losses_stud = rm.portfolio_losses(sim_stud, weights_arr, portfolio_value)


# ─────────────────────────────────────────────
# ВЕРХНІЙ ЗАГОЛОВОК (HERO HEADER)
# ─────────────────────────────────────────────

stress_pill_html = (
    f'<span class="pill pill-stress">{ICONS["flame"]} СТРЕС: ρ → {target_corr:.2f}, σ × {vol_mult:.1f}</span>'
    if stress_on
    else f'<span class="pill">{ICONS["shield"]} Базовий режим</span>'
)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-kicker">{ICONS["activity"]} Курсова робота · Інтелектуальний аналіз даних та фінансове моделювання</div>
        <div class="hero-title">Моделювання фінансового ризику інвестиційного портфеля методом Монте-Карло</div>
        <div class="hero-pills">
            <span class="pill">{ICONS["layers"]} {weights_str}</span>
            <span class="pill">Капітал: {portfolio_uah:,.0f} грн (${portfolio_value:,.0f})</span>
            <span class="pill">Горизонт: {holding_period} дн.</span>
            <span class="pill">α = {int(alpha * 100)}%</span>
            <span class="pill">N = {n_sim:,} · ν = {df_student} · seed = {int(seed)}</span>
            {stress_pill_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# ВКЛАДКИ ЗАСТОСУНКУ
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Дані та кореляції",
    "Оцінка ризику (VaR / CVaR)",
    "Збіжність моделі",
    "Чутливість і стрес-тест",
    "Методологія",
])


# ═════════════════════════════════════════════
# ВКЛАДКА 1 — ДАНІ ТА КОРЕЛЯЦІЇ
# ═════════════════════════════════════════════

with tab1:
    st.markdown(
        f'<div class="section-title">{ICONS["chart"]} 2.1. Вихідні дані та статистичний аналіз активів</div>'
        f'<div class="section-desc">Період вибірки: {prices.index[0].date()} — {prices.index[-1].date()} '
        f'({len(returns):,} торгових днів). Валютний ризик гривні у базовій моделі не моделюється.</div>',
        unsafe_allow_html=True,
    )

    norm_prices = (prices / prices.iloc[0]) * 100.0
    port_cum = pd.Series(
        100.0 * np.exp((returns.values @ weights_arr).cumsum()),
        index=returns.index,
        name="Портфель",
    )
    fig_perf = go.Figure()
    palette_assets = ["#64748B", "#34D399", "#FBBF24", "#A78BFA", "#38BDF8", "#A3E635", "#F472B6", "#FB923C"]
    for idx_t, t in enumerate(selected_tickers):
        fig_perf.add_trace(
            go.Scatter(
                x=norm_prices.index,
                y=norm_prices[t],
                mode="lines",
                name=t,
                line=dict(width=1.5, color=palette_assets[idx_t % len(palette_assets)]),
                opacity=0.72,
            )
        )
    fig_perf.add_trace(
        go.Scatter(
            x=port_cum.index,
            y=port_cum.values,
            mode="lines",
            name="Портфель (зважений)",
            line=dict(width=2.8, color="#F8FAFC"),
        )
    )
    fig_perf.update_layout(
        title="Динаміка нормованої вартості активів та портфеля (база = 100)",
        xaxis_title="Дата",
        yaxis_title="Індекс (база = 100)",
    )
    st.plotly_chart(style_plotly(fig_perf, height=340), use_container_width=True)

    st.markdown(f'<div class="section-title">{ICONS["layers"]} Таблиця 2.1 — Описова статистика денних лог-дохідностей</div>', unsafe_allow_html=True)
    stats_df = rm.descriptive_stats(returns, trading_days=rm.CONFIG["trading_days_per_year"])
    st.dataframe(
        stats_df.style.format({
            "Середня денна дохідність": "{:.5f}",
            "Річна дохідність": "{:.2%}",
            "Денна волатильність": "{:.4f}",
            "Річна волатильність": "{:.2%}",
            "Мінімум": "{:.4f}",
            "Максимум": "{:.4f}",
            "Асиметрія": "{:.3f}",
            "Ексцес": "{:.2f}",
            "Жарк-Бера (статистика)": "{:,.1f}",
            "Жарк-Бера (p-value)": "{:.4e}",
        }),
        use_container_width=True,
    )
    st.download_button(
        "Завантажити Таблицю 2.1 (CSV)",
        stats_df.to_csv(float_format="%.5f").encode("utf-8"),
        file_name="table_2_1_statistics.csv",
        mime="text/csv",
    )

    col_corr, col_dist = st.columns([1, 1.35], gap="large")

    with col_corr:
        st.markdown('<div class="section-title">Рис. 2.1 — Кореляційна матриця</div>', unsafe_allow_html=True)
        corr = returns.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            labels={"color": "ρ"},
        )
        fig_corr.update_layout(title="Кореляція денних лог-дохідностей")
        st.plotly_chart(style_plotly(fig_corr, height=390), use_container_width=True)

    with col_dist:
        st.markdown('<div class="section-title">Рис. 2.2 — Гістограма та QQ-графік (перевірка хвостів)</div>', unsafe_allow_html=True)
        ticker_sel = st.selectbox("Оберіть актив для перевірки нормальності", selected_tickers, label_visibility="collapsed")
        r_sel = returns[ticker_sel].dropna()
        mu_r, sigma_r = float(r_sel.mean()), float(r_sel.std())
        x_range = np.linspace(float(r_sel.min()), float(r_sel.max()), 300)
        (osm, osr), (slope, intercept, _) = stats.probplot(r_sel, dist="norm")

        fig_dist = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(f"Гістограма {ticker_sel} vs Нормальний", f"QQ-графік {ticker_sel}"),
            horizontal_spacing=0.12,
        )
        fig_dist.add_trace(
            go.Histogram(
                x=r_sel,
                histnorm="probability density",
                nbinsx=55,
                name="Факт.",
                marker_color=COLOR_NORMAL,
                opacity=0.6,
            ),
            row=1,
            col=1,
        )
        fig_dist.add_trace(
            go.Scatter(
                x=x_range,
                y=stats.norm.pdf(x_range, mu_r, sigma_r),
                mode="lines",
                name="Норм. щільність",
                line=dict(color=COLOR_STUDENT, width=2),
            ),
            row=1,
            col=1,
        )
        fig_dist.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode="markers",
                name="Квантилі",
                marker=dict(size=4, color=COLOR_NORMAL, opacity=0.65),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        line_x = np.array([osm[0], osm[-1]])
        fig_dist.add_trace(
            go.Scatter(
                x=line_x,
                y=slope * line_x + intercept,
                mode="lines",
                name="Теоретична пряма",
                line=dict(color=COLOR_STUDENT, width=1.8, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig_dist.update_xaxes(title_text="Дохідність", row=1, col=1)
        fig_dist.update_xaxes(title_text="Теоретичні квантилі", row=1, col=2)
        fig_dist.update_yaxes(title_text="Щільність", row=1, col=1)
        fig_dist.update_yaxes(title_text="Вибіркові квантилі", row=1, col=2)
        st.plotly_chart(style_plotly(fig_dist, height=390), use_container_width=True)


# ═════════════════════════════════════════════
# ВКЛАДКА 2 — ОЦІНКА РИЗИКУ (VaR / CVaR)
# ═════════════════════════════════════════════

with tab2:
    if stress_on:
        pd_note = " (матрицю скориговано до найближчої PD)" if stress_pd_corrected else ""
        st.markdown(
            f'<div class="insight-banner warning">'
            f'<strong>Активовано стрес-сценарій «збіг кореляцій»:</strong> '
            f'попарні кореляції піднято до <strong>{target_corr:.2f}</strong>, '
            f'волатильності помножено на <strong>{vol_mult:.1f}</strong>{pd_note}.'
            f'</div>',
            unsafe_allow_html=True,
        )

    var_n, cvar_n = rm.compute_var_cvar(losses_norm, alpha)
    var_s, cvar_s = rm.compute_var_cvar(losses_stud, alpha)

    diff_var_pct = (var_s - var_n) / abs(var_n) * 100 if var_n != 0 else 0.0
    diff_cvar_pct = (cvar_s - cvar_n) / abs(cvar_n) * 100 if cvar_n != 0 else 0.0

    # Картки KPI із піктограмками та бейджами
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card accent-blue">
                <div class="kpi-top">
                    <span class="kpi-badge badge-blue">{ICONS["chart"]} VaR {int(alpha*100)}% · Нормальний</span>
                </div>
                <div class="kpi-value">{var_n * fx_rate:,.0f} грн</div>
                <div class="kpi-sub">${var_n:,.0f} · {var_n / portfolio_value * 100:.2f}% портфеля</div>
            </div>
            <div class="kpi-card accent-red">
                <div class="kpi-top">
                    <span class="kpi-badge badge-red">{ICONS["flame"]} VaR {int(alpha*100)}% · Стьюдент (ν={df_student})</span>
                </div>
                <div class="kpi-value">{var_s * fx_rate:,.0f} грн</div>
                <div class="kpi-sub danger">${var_s:,.0f} · {var_s / portfolio_value * 100:.2f}% ({diff_var_pct:+.1f}% до норм.)</div>
            </div>
            <div class="kpi-card accent-blue">
                <div class="kpi-top">
                    <span class="kpi-badge badge-blue">{ICONS["shield"]} CVaR {int(alpha*100)}% · Нормальний</span>
                </div>
                <div class="kpi-value">{cvar_n * fx_rate:,.0f} грн</div>
                <div class="kpi-sub">${cvar_n:,.0f} · {cvar_n / portfolio_value * 100:.2f}% портфеля</div>
            </div>
            <div class="kpi-card accent-red">
                <div class="kpi-top">
                    <span class="kpi-badge badge-red">{ICONS["target"]} CVaR {int(alpha*100)}% · Стьюдент (ν={df_student})</span>
                </div>
                <div class="kpi-value">{cvar_s * fx_rate:,.0f} грн</div>
                <div class="kpi-sub danger">${cvar_s:,.0f} · {cvar_s / portfolio_value * 100:.2f}% ({diff_cvar_pct:+.1f}% до норм.)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="insight-banner">'
        f'<strong>Порівняння хвостів розподілу (h = {holding_period} дн., α = {int(alpha*100)}%):</strong> '
        f'припущення нормальності занижує очікувані крайні втрати (CVaR) на <strong>{diff_cvar_pct:.1f}%</strong> '
        f'({(cvar_s - cvar_n) * fx_rate:,.0f} грн / ${cvar_s - cvar_n:,.0f}) порівняно з розподілом Стьюдента (ν = {df_student}).'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="section-title">{ICONS["chart"]} Рис. 2.3 — Розподіл змодельованих втрат портфеля</div>', unsafe_allow_html=True)

    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Histogram(
            x=losses_norm * fx_rate,
            histnorm="probability density",
            nbinsx=130,
            name="Нормальний розподіл",
            marker_color=COLOR_NORMAL,
            opacity=0.45,
        )
    )
    fig_loss.add_trace(
        go.Histogram(
            x=losses_stud * fx_rate,
            histnorm="probability density",
            nbinsx=130,
            name=f"Розподіл Стьюдента (ν={df_student})",
            marker_color=COLOR_STUDENT,
            opacity=0.45,
        )
    )

    vlines = [
        (var_n * fx_rate, COLOR_NORMAL, "dash", f"VaR норм. ({var_n * fx_rate:,.0f} грн)", 0.96),
        (cvar_n * fx_rate, COLOR_NORMAL, "dot", f"CVaR норм. ({cvar_n * fx_rate:,.0f} грн)", 0.84),
        (var_s * fx_rate, COLOR_STUDENT, "dash", f"VaR Стьюд. ({var_s * fx_rate:,.0f} грн)", 0.72),
        (cvar_s * fx_rate, COLOR_STUDENT, "dot", f"CVaR Стьюд. ({cvar_s * fx_rate:,.0f} грн)", 0.60),
    ]
    for x_val, color, dash, label, y_pos in vlines:
        fig_loss.add_vline(x=x_val, line_dash=dash, line_color=color, line_width=2)
        fig_loss.add_annotation(
            x=x_val,
            y=y_pos,
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="left",
            xshift=6,
            font=dict(size=11, color=color),
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )

    fig_loss.update_layout(
        title=f"Порівняння розподілів втрат (h = {holding_period} дн., α = {int(alpha * 100)}%, N = {n_sim:,})",
        xaxis_title="Втрати портфеля (грн)",
        yaxis_title="Щільність імовірності",
        barmode="overlay",
    )
    st.plotly_chart(style_plotly(fig_loss, height=440), use_container_width=True)

    st.markdown(f'<div class="section-title">{ICONS["layers"]} Таблиця 2.2 — Порівняння підходів оцінювання VaR і CVaR</div>', unsafe_allow_html=True)
    results_df = cached_results_table(
        returns_values=returns.values,
        returns_cols=tuple(returns.columns),
        returns_idx=tuple(returns.index.astype(str)),
        mu_arr=tuple(mu),
        cov_flat=tuple(cov.flatten()),
        cov_shape=cov.shape,
        weights_arr=tuple(weights_arr),
        portfolio_value=portfolio_value,
        holding_periods=(1, 10) if holding_period in (1, 10) else (1, holding_period),
        n_sim=n_sim,
        df_student=df_student,
        seed=int(seed),
    ).copy()
    results_df.insert(5, "VaR (грн)", (results_df["VaR ($)"] * fx_rate).round(0))
    results_df.insert(6, "CVaR (грн)", (results_df["CVaR ($)"] * fx_rate).round(0))

    st.dataframe(
        results_df.style.format({
            "Рівень довіри α": "{:.2f}",
            "VaR ($)": "{:,.2f}",
            "CVaR ($)": "{:,.2f}",
            "VaR (грн)": "{:,.0f}",
            "CVaR (грн)": "{:,.0f}",
            "VaR (%)": "{:.2f}%",
            "CVaR (%)": "{:.2f}%",
        }),
        use_container_width=True,
    )
    st.download_button(
        "Завантажити Таблицю 2.2 (CSV)",
        results_df.to_csv(index=False, float_format="%.2f").encode("utf-8"),
        file_name="table_2_2_var_cvar.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════
# ВКЛАДКА 3 — ЗБІЖНІСТЬ МОДЕЛІ
# ═════════════════════════════════════════════

with tab3:
    st.markdown(
        f'<div class="section-title">{ICONS["target"]} Аналіз збіжності оцінки CVaR та правило зупинки</div>'
        '<div class="section-desc">Кількість сценаріїв N вважається достатньою, '
        'коли подальше подвоєння N змінює оцінку CVaR менш ніж на 1%.</div>',
        unsafe_allow_html=True,
    )

    conv_alpha = st.radio(
        "Рівень довіри для перевірки збіжності",
        [0.95, 0.99],
        index=1,
        format_func=lambda x: f"α = {int(x * 100)}%",
        horizontal=True,
    )

    n_grid_app = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000)
    conv_df, min_n_norm, min_n_stud = cached_convergence(
        mu_arr=tuple(mu),
        cov_flat=tuple(cov.flatten()),
        cov_shape=cov.shape,
        weights_arr=tuple(weights_arr),
        portfolio_value=portfolio_value,
        n_grid=n_grid_app,
        h=holding_period,
        alpha=conv_alpha,
        df_student=df_student,
        seed=int(seed),
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card accent-blue">
                <div class="kpi-top">
                    <span class="kpi-badge badge-blue">{ICONS["cpu"]} Достатнє N* · Нормальний</span>
                </div>
                <div class="kpi-value">{f"{min_n_norm:,}" if isinstance(min_n_norm, int) else min_n_norm}</div>
                <div class="kpi-sub">Поріг стабілізації: &lt; 1.0%</div>
            </div>
            <div class="kpi-card accent-red">
                <div class="kpi-top">
                    <span class="kpi-badge badge-red">{ICONS["cpu"]} Достатнє N* · Стьюдент (ν={df_student})</span>
                </div>
                <div class="kpi-value">{f"{min_n_stud:,}" if isinstance(min_n_stud, int) else min_n_stud}</div>
                <div class="kpi-sub">Важкі хвости потребують більше сценаріїв</div>
            </div>
            <div class="kpi-card accent-slate">
                <div class="kpi-top">
                    <span class="kpi-badge badge-slate">{ICONS["lock"]} Поточне N у моделі</span>
                </div>
                <div class="kpi-value">{n_sim:,}</div>
                <div class="kpi-sub">{"Умова зупинки виконана" if isinstance(min_n_stud, int) and n_sim >= min_n_stud else "Перевірте таблицю нижче"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_conv = go.Figure()
    fig_conv.add_trace(
        go.Scatter(
            x=conv_df["N"],
            y=conv_df["CVaR (нормальний)"] * fx_rate,
            mode="lines+markers",
            name="Нормальний",
            line=dict(color=COLOR_NORMAL, width=2.4),
            marker=dict(size=8),
        )
    )
    fig_conv.add_trace(
        go.Scatter(
            x=conv_df["N"],
            y=conv_df["CVaR (Стьюдент)"] * fx_rate,
            mode="lines+markers",
            name=f"Стьюдент (ν={df_student})",
            line=dict(color=COLOR_STUDENT, width=2.4),
            marker=dict(size=8, symbol="square"),
        )
    )
    if isinstance(min_n_norm, int):
        fig_conv.add_vline(
            x=min_n_norm,
            line_dash="dot",
            line_color=COLOR_NORMAL,
            annotation_text=f"N* норм = {min_n_norm:,}",
            annotation_position="top right",
        )
    if isinstance(min_n_stud, int):
        fig_conv.add_vline(
            x=min_n_stud,
            line_dash="dot",
            line_color=COLOR_STUDENT,
            annotation_text=f"N* Стьюд = {min_n_stud:,}",
            annotation_position="bottom right",
        )

    fig_conv.update_layout(
        title=f"Рис. 2.4 — Збіжність CVaR{int(conv_alpha * 100)} залежно від кількості сценаріїв N (h = {holding_period} дн.)",
        xaxis_title="Кількість сценаріїв N (логарифмічна шкала)",
        yaxis_title="CVaR (грн)",
        xaxis_type="log",
    )
    st.plotly_chart(style_plotly(fig_conv, height=410), use_container_width=True)

    st.markdown('<div class="section-title">Таблиця 2.3 — Показники збіжності CVaR</div>', unsafe_allow_html=True)
    conv_disp = conv_df.copy()
    conv_disp["CVaR норм. (грн)"] = (conv_disp["CVaR (нормальний)"] * fx_rate).round(0)
    conv_disp["CVaR Стьюд. (грн)"] = (conv_disp["CVaR (Стьюдент)"] * fx_rate).round(0)
    st.dataframe(
        conv_disp.style.format({
            "N": "{:,}",
            "CVaR (нормальний)": "{:,.2f}",
            "CVaR (Стьюдент)": "{:,.2f}",
            "CVaR норм. (грн)": "{:,.0f}",
            "CVaR Стьюд. (грн)": "{:,.0f}",
            "Відн. зміна (норм.)": "{:.2%}",
            "Відн. зміна (Стьюд.)": "{:.2%}",
        }, na_rep="—"),
        use_container_width=True,
    )
    st.download_button(
        "Завантажити Таблицю 2.3 (CSV)",
        conv_disp.to_csv(index=False, float_format="%.4f").encode("utf-8"),
        file_name="table_2_3_convergence.csv",
        mime="text/csv",
    )


# ═════════════════════════════════════════════
# ВКЛАДКА 4 — ЧУТЛИВІСТЬ І СТРЕС-ТЕСТ
# ═════════════════════════════════════════════

with tab4:
    sens_vol, sens_w, stress_df = cached_sensitivity_and_stress(
        mu_arr=tuple(mu),
        cov_flat=tuple(cov.flatten()),
        cov_shape=cov.shape,
        weights_arr=tuple(weights_arr),
        portfolio_value=portfolio_value,
        h=holding_period,
        n_sim=n_sim,
        df_student=df_student,
        seed=int(seed),
        target_corr=target_corr,
        vol_mult=vol_mult,
    )

    col_sv, col_sw = st.columns(2, gap="large")

    with col_sv:
        st.markdown(f'<div class="section-title">{ICONS["activity"]} Рис. 2.5 — Чутливість CVaR₉₉ до волатильності</div>', unsafe_allow_html=True)
        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Scatter(
                x=sens_vol["Множник σ"],
                y=sens_vol["CVaR (норм., $)"] * fx_rate,
                mode="lines+markers",
                name="Нормальний",
                line=dict(color=COLOR_NORMAL, width=2.4),
                marker=dict(size=8),
            )
        )
        fig_sens.add_trace(
            go.Scatter(
                x=sens_vol["Множник σ"],
                y=sens_vol["CVaR (Стьюд., $)"] * fx_rate,
                mode="lines+markers",
                name=f"Стьюдент (ν={df_student})",
                line=dict(color=COLOR_STUDENT, width=2.4),
                marker=dict(size=8, symbol="square"),
            )
        )
        fig_sens.add_vline(x=1.0, line_dash="dot", line_color=COLOR_MUTED, annotation_text="База (1.0×)")
        fig_sens.update_layout(
            title="Вплив масштабування σ на CVaR₉₉",
            xaxis_title="Множник волатильності",
            yaxis_title="CVaR₉₉ (грн)",
        )
        st.plotly_chart(style_plotly(fig_sens, height=370), use_container_width=True)
        st.dataframe(
            sens_vol.style.format({
                "Множник σ": "{:.1f}×",
                "VaR (норм., $)": "{:,.2f}",
                "CVaR (норм., $)": "{:,.2f}",
                "VaR (Стьюд., $)": "{:,.2f}",
                "CVaR (Стьюд., $)": "{:,.2f}",
            }),
            use_container_width=True,
        )

    with col_sw:
        st.markdown(f'<div class="section-title">{ICONS["layers"]} Чутливість CVaR₉₉ до структури ваг портфеля</div>', unsafe_allow_html=True)
        fig_w = go.Figure()
        fig_w.add_trace(
            go.Bar(
                y=sens_w["Сценарій ваг"],
                x=sens_w["VaR ($)"] * fx_rate,
                name="VaR 99%",
                orientation="h",
                marker_color=COLOR_NORMAL,
                opacity=0.88,
            )
        )
        fig_w.add_trace(
            go.Bar(
                y=sens_w["Сценарій ваг"],
                x=sens_w["CVaR ($)"] * fx_rate,
                name="CVaR 99%",
                orientation="h",
                marker_color=COLOR_STUDENT,
                opacity=0.88,
            )
        )
        fig_w.update_layout(
            title="Порівняння альтернативних алокацій (Стьюдент)",
            xaxis_title="Ризик (грн)",
            barmode="group",
        )
        st.plotly_chart(style_plotly(fig_w, height=370), use_container_width=True)
        st.dataframe(
            sens_w.style.format({
                "VaR ($)": "{:,.2f}",
                "CVaR ($)": "{:,.2f}",
            }),
            use_container_width=True,
        )

    st.markdown(f'<div class="section-title">{ICONS["flame"]} Таблиця 2.5 — Стрес-сценарій «збіг кореляцій»</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-desc">Порівняння базового ринку зі сценарієм кризи: '
        f'попарні кореляції → {target_corr:.2f}, волатильності × {vol_mult:.1f} (горизонт {holding_period} дн.).</div>',
        unsafe_allow_html=True,
    )

    col_st1, col_st2 = st.columns([1.2, 1], gap="large")
    with col_st1:
        stress_disp = stress_df.copy()
        stress_disp["CVaR базовий (грн)"] = (stress_disp["CVaR базовий ($)"] * fx_rate).round(0)
        stress_disp["CVaR стрес (грн)"] = (stress_disp["CVaR стрес ($)"] * fx_rate).round(0)
        st.dataframe(
            stress_disp.style.format({
                "α": "{:.2f}",
                "VaR базовий ($)": "{:,.2f}",
                "CVaR базовий ($)": "{:,.2f}",
                "VaR стрес ($)": "{:,.2f}",
                "CVaR стрес ($)": "{:,.2f}",
                "CVaR базовий (грн)": "{:,.0f}",
                "CVaR стрес (грн)": "{:,.0f}",
                "Зміна VaR (%)": "{:+.1f}%",
                "Зміна CVaR (%)": "{:+.1f}%",
            }),
            use_container_width=True,
        )
        st.download_button(
            "Завантажити Таблицю 2.5 (CSV)",
            stress_disp.to_csv(index=False, float_format="%.2f").encode("utf-8"),
            file_name="table_2_5_stress.csv",
            mime="text/csv",
        )

    with col_st2:
        fig_stress = go.Figure()
        fig_stress.add_trace(
            go.Bar(
                x=[f"α = {int(a * 100)}%" for a in stress_df["α"]],
                y=stress_df["CVaR базовий ($)"] * fx_rate,
                name="Базовий сценарій",
                marker_color=COLOR_NORMAL,
            )
        )
        fig_stress.add_trace(
            go.Bar(
                x=[f"α = {int(a * 100)}%" for a in stress_df["α"]],
                y=stress_df["CVaR стрес ($)"] * fx_rate,
                name="Стрес-сценарій",
                marker_color=COLOR_STUDENT,
            )
        )
        fig_stress.update_layout(
            title="CVaR: базовий vs стрес-сценарій (грн)",
            xaxis_title="Рівень довіри",
            yaxis_title="CVaR (грн)",
            barmode="group",
        )
        st.plotly_chart(style_plotly(fig_stress, height=320), use_container_width=True)

    with st.expander("Бектестинг одноденного VaR на ковзному вікні 250 днів (тест Купієця)"):
        try:
            bt = rm.backtest_var(returns, weights_arr, portfolio_value, confidence_level=0.95, window=250)
            c_bt1, c_bt2, c_bt3 = st.columns(3)
            c_bt1.metric("Очікувано перевищень (5%)", bt["expected_exceedances"])
            c_bt2.metric(
                "Параметричний VaR",
                f"{bt['parametric']['exceedances']} перевищ.",
                f"p-value = {bt['parametric']['kupiec_pvalue']:.3f}",
            )
            c_bt3.metric(
                "Історичний VaR",
                f"{bt['historical']['exceedances']} перевищ.",
                f"p-value = {bt['historical']['kupiec_pvalue']:.3f}",
            )

            actual_uah = bt["actual_losses"] * fx_rate
            var_p_uah = bt["parametric"]["var_series"] * fx_rate
            var_h_uah = bt["historical"]["var_series"] * fx_rate
            days_idx = list(range(len(actual_uah)))

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=days_idx, y=actual_uah, mode="lines", name="Фактичний збиток", line=dict(color="#64748B", width=1.2)))
            fig_bt.add_trace(go.Scatter(x=days_idx, y=var_p_uah, mode="lines", name="VaR 95% (парам.)", line=dict(color=COLOR_NORMAL, width=2, dash="dash")))
            fig_bt.add_trace(go.Scatter(x=days_idx, y=var_h_uah, mode="lines", name="VaR 95% (істор.)", line=dict(color=COLOR_ACCENT, width=2, dash="dot")))
            exc_idx = np.where(actual_uah > var_p_uah)[0]
            fig_bt.add_trace(
                go.Scatter(
                    x=exc_idx,
                    y=actual_uah[exc_idx],
                    mode="markers",
                    name="Перевищення (парам.)",
                    marker=dict(color=COLOR_STUDENT, size=7),
                )
            )
            fig_bt.update_layout(
                title="Рис. 2.6 — Бектестинг одноденного VaR 95% (останні 252 торгові дні)",
                xaxis_title="Торговий день тестового вікна",
                yaxis_title="Збиток (грн)",
            )
            st.plotly_chart(style_plotly(fig_bt, height=340), use_container_width=True)
        except Exception as e:
            st.info(f"Бектестинг недоступний для поточної довжини ряду: {e}")


# ═════════════════════════════════════════════
# ВКЛАДКА 5 — МЕТОДОЛОГІЯ (ІЗ ГРАДІЄНТНИМИ ТА ПУНКТИРНИМИ КАРТКАМИ)
# ═════════════════════════════════════════════

with tab5:
    st.markdown(f'<div class="section-title">{ICONS["cpu"]} Ключові наукові та програмні елементи моделі</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glow-grid">
            <div class="glow-card">
                <div class="glow-icon">{ICONS["chart"]}</div>
                <div class="glow-title">Паралельні розподіли</div>
                <div class="glow-text">Одночасне моделювання нормального закону та розподілу Стьюдента (ν = {df_student}) демонструє заниження хвостового ризику.</div>
            </div>
            <div class="glow-card">
                <div class="glow-icon">{ICONS["flame"]}</div>
                <div class="glow-title">Збіг кореляцій</div>
                <div class="glow-text">Кризовий стрес-тест піднімає кореляції до 0.90 і волатильність ×1.5 із корекцією Хігема до PD-матриці.</div>
            </div>
            <div class="glow-card">
                <div class="glow-icon">{ICONS["target"]}</div>
                <div class="glow-title">Правило зупинки</div>
                <div class="glow-text">Автоматичний пошук найменшого N, після якого відносна зміна CVaR при подвоєнні вибірки не перевищує 1%.</div>
            </div>
            <div class="glow-card">
                <div class="glow-icon">{ICONS["lock"]}</div>
                <div class="glow-title">Повна відтворюваність</div>
                <div class="glow-text">Фіксований генератор <code>numpy.random.default_rng(seed)</code> і локальний кеш цін гарантують ідентичність результатів.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="dashed-grid">
            <div class="dashed-card blue">
                <div class="dashed-pill blue">{ICONS["shield"]} Переваги підходу</div>
                <div class="dashed-title">Чому метод Монте-Карло</div>
                <ul class="sq-list">
                    <li><span class="sq-dot blue"></span><span>Враховує нелінійність експоненційної дохідності та «товсті хвости» активів</span></li>
                    <li><span class="sq-dot blue"></span><span>Відтворює міжринкові кореляції через розклад Холецького Σ = L·Lᵀ</span></li>
                    <li><span class="sq-dot blue"></span><span>Дозволяє гнучко задавати кризові стрес-сценарії та довільні ваги портфеля</span></li>
                </ul>
            </div>
            <div class="dashed-card rose">
                <div class="dashed-pill rose">{ICONS["layers"]} Обмеження моделі</div>
                <div class="dashed-title">Припущення та ризики</div>
                <ul class="sq-list">
                    <li><span class="sq-dot rose"></span><span>Параметри μ та Σ оцінюються за 5-річною історією і вважаються стаціонарними</span></li>
                    <li><span class="sq-dot rose"></span><span>Валютний ризик гривні та транзакційні витрати в базовій моделі не враховуються</span></li>
                    <li><span class="sq-dot rose"></span><span>Точність оцінки хвостів залежить від кількості сценаріїв N і параметра ν</span></li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns([1.4, 1], gap="large")

    with col_m1:
        st.markdown('<div class="section-title">Математична формалізація</div>', unsafe_allow_html=True)
        st.markdown("""
        **1. Оцінювання параметрів** (за денними логарифмічними дохідностями $r_t = \\ln(P_t/P_{t-1})$):
        вектор середніх $\\hat{\\mu}$ та коваріаційна матриця $\\hat{\\Sigma}$.

        **2. Розклад Холецького:**
        $$\\hat{\\Sigma} = L \\cdot L^\\top, \\quad Z_{\\text{corr}} = \\varepsilon \\cdot L^\\top$$

        **3. Сценарії за горизонт $h$ днів:**
        - **Нормальний:** $R_j = \\mu h + \\varepsilon_j L_h^\\top$, де $\\varepsilon_j \\sim N(0, I)$.
        - **Стьюдента:** $R_j = \\mu h + \\frac{t_j}{\\sqrt{\\nu / (\\nu - 2)}} L_h^\\top$, де $t_j \\sim t_\\nu(0, I)$.

        **4. Втрати та метрики ризику:**
        $$V_T = V_0 \\sum_{i} w_i \\exp(R_i), \\quad L = V_0 - V_T$$
        $$\\text{VaR}_\\alpha = F_L^{-1}(\\alpha), \\quad \\text{CVaR}_\\alpha = \\mathbb{E}[L \\mid L \\ge \\text{VaR}_\\alpha]$$
        """)

    with col_m2:
        st.markdown('<div class="section-title">Активна конфігурація</div>', unsafe_allow_html=True)
        cfg_df = pd.DataFrame({
            "Параметр": [
                "Активи портфеля",
                "Нормовані ваги",
                "Вартість портфеля",
                "Курс перерахунку",
                "Горизонт h",
                "Рівень довіри α",
                "Ступені свободи ν",
                "Кількість сценаріїв N",
                "Seed",
                "Стрес-режим",
            ],
            "Значення": [
                ", ".join(selected_tickers),
                weights_str,
                f"{portfolio_uah:,.0f} грн (${portfolio_value:,.0f})",
                f"{fx_rate:.1f} грн/$",
                f"{holding_period} торг. дн.",
                f"{int(alpha * 100)}%",
                f"{df_student}",
                f"{n_sim:,}",
                f"{int(seed)}",
                f"Увімкнено (ρ={target_corr:.2f}, σ×{vol_mult:.1f})" if stress_on else "Вимкнено",
            ],
        })
        st.dataframe(cfg_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────
# ДИСКЛЕЙМЕР
# ─────────────────────────────────────────────

st.markdown(
    '<div class="footer-note">'
    'Навчальна модель, не є інвестиційною рекомендацією. '
    'Усі розрахунки виконуються модулем <code>risk_model.py</code>.'
    '</div>',
    unsafe_allow_html=True,
)