"""
app.py — Streamlit chat interface for CONDUCTOR.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import copy
import difflib
import html
import pandas as pd
import functools
import json
import logging
import os
import pathlib
import re
import time
import uuid
from datetime import datetime, timezone

import httpx
import streamlit as st

from agent.config import (
    BASE_URL,
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
    GEMINI_THINKING_LEVELS,
    HTTP_TIMEOUT,
    last_grid_constants_status,
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_OUTPUT_RESERVE,
    ANTHROPIC_EFFORT,
    ANTHROPIC_EFFORTS,
    ANTHROPIC_MODEL,
    OLLAMA_TIMEOUT_S,
    OPENAI_API,
    OPENAI_APIS,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
    OPENAI_REASONING_EFFORTS,
)
from agent.config import (
    fetch_grid_constants,
    grid_limits,
    is_network_change,
    network_fingerprint,
)
from agent.hardware import advice as hw_advice
from agent.hardware import detect as detect_hardware
from agent.hardware import num_ctx_warning, recommended_num_ctx
from agent import loop as _loop_module
from agent import model_check as _model_check
from agent import recommender as _recommender
from agent import update_check as _update_check
from agent import timeline as _tl
from agent import generator_table as _gen_table
from agent import data_view as _dv
from agent import trace as _trace
from agent import staleness as _staleness
from agent import report as _report
from agent.loop import run_agent_turn
from agent.providers import (
    active_provider_name,
    available_providers,
    describe_provider,
    get_provider,
    reset_providers,
)
from agent.providers.ollama import probe as ollama_probe
from agent.providers.openai import OpenAIProvider as _OpenAIProvider
from agent.providers.openai import probe as openai_probe
from agent import network_map as _network_map
from agent.renderers import RENDERER_MAP, render_network_diff, render_network_map
from agent import tools as _tools_module
from agent.tools import get_current_timestamp

_APP_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _APP_DIR.parent
_DATA_FILES_DIR = _REPO_ROOT / "data_files"
_SYSTEMS_DIR = _REPO_ROOT / "systems"
# Brand marks (assets/). The colour mark sits on the light app surfaces; the
# round app icon, legible on light and dark browser chrome, is the tab icon.
_LOGO_PATH = _APP_DIR / "assets" / "conductor-mark.png"
_APP_ICON_PATH = _APP_DIR / "assets" / "conductor-app-icon.png"
# CONDUCTOR answers under its colour mark (the dark round icon looked heavy
# on the white cards); the operator keeps Streamlit's avatar.
_AVATARS = {"assistant": str(_LOGO_PATH)}

_PROVIDER_LABELS = {
    "google": "Google Gemini API",
    "openai": "OpenAI API (or compatible)",
    "anthropic": "Anthropic API (Claude)",
    "ollama": "Ollama (local)",
}


def _active_model_label() -> str:
    """Model id of the selected backend, for the info panel."""
    try:
        return get_provider().model
    except Exception:  # noqa: BLE001 — never let the badge break the page
        return "unknown"


_BRAND_NAME = "CONDUCTOR"
_BRAND_TAGLINE = "Ask, test and decide: a solver-grounded AI assistant for distribution grids"
_BRAND_CAPTION = (
    "Grid studies in plain language — security, N-1, probabilistic risk, flexibility and "
    "hosting capacity — changes to the grid tested as proposals, and every figure traced "
    "to the solver that produced it."
)


def _image_as_data_uri(path: pathlib.Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=_BRAND_NAME,
    page_icon=str(_APP_ICON_PATH),
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(42, 114, 232, 0.07), transparent 26%),
            linear-gradient(180deg, #fbfcfe 0%, #f5f7fb 100%);
    }
    .conductor-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .conductor-brand img {
        display: block;
        height: auto;
        flex-shrink: 0;
    }
    .conductor-brand-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 0.15rem;
    }
    .conductor-brand-title {
        margin: 0;
        color: #2e3140;
        font-size: 2.5rem;
        line-height: 1.0;
        font-weight: 800;
        letter-spacing: 0.01em;
    }
    .conductor-brand-tagline {
        margin: 0;
        color: #6e7484;
        font-size: 1rem;
        line-height: 1.35;
        max-width: 960px;
    }
    .conductor-brand-support {
        margin: 0.6rem 0 1.2rem 0;
        color: #5c6272;
        font-size: 1rem;
        line-height: 1.55;
        max-width: 1040px;
    }
    .conductor-hero-shell {
        padding: 0.45rem 0 0.65rem 0;
        margin-bottom: 0.35rem;
    }
    .conductor-sidebar-copy {
        color: #5c6272;
        line-height: 1.55;
        margin-top: 0.35rem;
    }
    .conductor-sidebar-label {
        color: #7a8090;
        text-transform: uppercase;
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin: 0.8rem 0 0.35rem 0;
    }
    /* ---- Visual system --------------------------------------------------
       One set of surfaces, borders and radii for the whole app. Selectors
       target Streamlit's data-testid attributes (checked on 1.63). */
    :root {
        --cd-ink: #2E3140;
        --cd-muted: #6E7484;
        --cd-border: #E3E7EF;
        --cd-surface: #FFFFFF;
        --cd-brand: #1D6FE8;
        --cd-brand-soft: #EEF4FF;
        --cd-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 1px 3px rgba(16, 24, 40, 0.04);
    }
    [data-testid="stSidebar"] {
        background: var(--cd-surface);
        border-right: 1px solid var(--cd-border);
    }
    /* Sidebar: no empty band above the first section, tighter dividers. */
    [data-testid="stSidebarHeader"] { height: 2.25rem; min-height: 2.25rem; padding-bottom: 0; }
    [data-testid="stSidebarUserContent"] { padding-top: 0; }
    [data-testid="stSidebar"] hr { margin: 0.5rem 0 0.25rem 0; }
    [data-testid="stSidebar"] .conductor-sidebar-label:first-of-type { margin-top: 0.2rem; }
    /* Operating point card. */
    .op-when { display: flex; justify-content: space-between; align-items: baseline; }
    .op-date { font-weight: 600; color: var(--cd-ink); font-size: 0.9rem; }
    .op-time { font-weight: 700; color: var(--cd-ink); font-size: 1.4rem;
               font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
    .op-strip { position: relative; margin: 0.15rem 0 0.4rem; }
    .op-track { position: relative; height: 6px; background: #EEF0F4; border-radius: 3px; margin: 4px 0; }
    .op-fill { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
    .op-cursor { position: absolute; top: -3px; bottom: -3px; width: 2px; margin-left: -1px;
                 background: var(--cd-ink); border-radius: 1px; }
    .op-scale { display: flex; justify-content: space-between; font-size: 0.68rem;
                color: var(--cd-muted); margin: -0.3rem 0 0.25rem; font-variant-numeric: tabular-nums; }
    .st-key-op_card [data-testid="stSliderThumbValue"] { display: none; }
    /* Three datasets on one row in the narrow sidebar. */
    .st-key-op_mode [data-testid="stButtonGroup"] > div { flex-wrap: nowrap; }
    .st-key-op_mode button { flex: 1 1 0; min-width: 0; padding: 0.2rem 0.35rem; min-height: 2rem; }
    .st-key-op_mode button p { font-size: 0.8rem; }
    .op-legend { display: flex; justify-content: space-between; gap: 0.4rem;
                 font-size: 0.72rem; color: var(--cd-muted); line-height: 1.55; }
    .op-legend b { color: var(--cd-ink); font-weight: 600; }
    .op-dim { opacity: 0.55; }
    /* Example queries: a left-aligned list to scan, not a stack of buttons. */
    .st-key-examples_list { gap: 0.1rem; }
    [class*="st-key-eq_"] button {
        justify-content: flex-start;
        width: 100%;
        padding: 0.3rem 0.45rem;
        border-radius: 8px;
        color: var(--cd-ink);
    }
    [class*="st-key-eq_"] button:hover { background: var(--cd-brand-soft); color: var(--cd-brand); }
    [class*="st-key-eq_"] button p { text-align: left; font-size: 0.86rem; line-height: 1.35; }
    [class*="st-key-eq_"] button > div,
    [class*="st-key-eq_"] button > div > span { align-items: flex-start; }
    [class*="st-key-eq_"] button [data-testid="stIconMaterial"] { color: var(--cd-brand); margin-top: 0.1rem; }
    /* Answers on white cards; the operator's own messages on a brand tint. */
    [data-testid="stChatMessage"] {
        background: var(--cd-surface);
        border: 1px solid var(--cd-border);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: var(--cd-shadow);
        margin-bottom: 0.75rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--cd-brand-soft);
        border-color: #D6E4FF;
        box-shadow: none;
    }
    [data-testid="stChatMessageContent"] p { line-height: 1.6; }
    /* Each chart in its own soft card. */
    [data-testid="stPlotlyChart"] {
        background: var(--cd-surface);
        border: 1px solid var(--cd-border);
        border-radius: 12px;
        padding: 0.35rem;
    }
    [data-testid="stExpander"] details {
        border-radius: 12px;
        border-color: var(--cd-border);
        background: var(--cd-surface);
    }
    [data-testid="stExpander"] summary { font-weight: 600; }
    /* The question box as a raised card. */
    [data-testid="stForm"] {
        padding: 0.5rem 0.7rem;
        background: var(--cd-surface);
        border: 1px solid var(--cd-border);
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
    }
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        font-weight: 600;
    }
    [data-testid="stBaseButton-secondary"] {
        border-color: var(--cd-border);
    }
    [data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--cd-brand);
        color: var(--cd-brand);
    }
    [data-testid="stTab"] { font-weight: 600; }
    [data-testid="stAlert"] { border-radius: 10px; }
    [data-testid="stCaptionContainer"] { color: var(--cd-muted); }
    /* Once a conversation has started, the hero shrinks to a single line so
       the answers, not the branding, fill the screen. */
    .conductor-hero-compact .conductor-brand img { width: 58px !important; }
    .conductor-hero-compact .conductor-brand-title { font-size: 1.75rem; }
    .conductor-hero-compact .conductor-brand-tagline,
    .conductor-hero-compact .conductor-brand-support { display: none; }
    .conductor-hero-compact { padding: 0; margin-bottom: 0; }
    /* The AI agent's label: a different accent from tool output, because its
       text is model reasoning, not a solver result. */
    .conductor-ai-comment-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #5b3fc4;
        margin-bottom: 0.2rem;
    }
    /* The AI agent panel stays in view beside a long conversation. Both
       columns stretch to the row's height, so the panel can stick inside its
       own; without support it simply sits at the top of its column. */
    .st-key-chat_layout [data-testid="stHorizontalBlock"] { align-items: stretch !important; }
    [data-testid="stColumn"]:has(.st-key-ai_panel) > [data-testid="stVerticalBlock"],
    [data-testid="stColumn"]:has(.st-key-ai_panel_closed) > [data-testid="stVerticalBlock"] { height: 100%; }
    [data-testid="stLayoutWrapper"]:has(> .st-key-ai_panel), .st-key-ai_panel,
    [data-testid="stLayoutWrapper"]:has(> .st-key-ai_panel_closed), .st-key-ai_panel_closed {
        position: sticky; top: 4rem;
    }
    .st-key-ai_panel { max-height: calc(100vh - 7rem); overflow-y: auto; padding-right: 0.25rem; }
    /* The collapsed panel's launcher: a round button in the AI agent's purple. */
    .st-key-ai_panel_closed { align-items: center; }
    .st-key-agent_open button, .st-key-agent_open_flag button {
        position: relative;
        width: 2.9rem; height: 2.9rem; min-height: 2.9rem;
        padding: 0; border-radius: 50%;
        border: 1.5px solid #8E5CD9;
        background: rgba(107, 79, 216, 0.10);
        box-shadow: 0 2px 8px rgba(91, 63, 196, 0.18);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .st-key-agent_open button p, .st-key-agent_open_flag button p { font-size: 1.35rem; line-height: 1; }
    .st-key-agent_open button:hover, .st-key-agent_open_flag button:hover {
        transform: scale(1.06); box-shadow: 0 4px 14px rgba(91, 63, 196, 0.30);
    }
    .st-key-agent_open_flag button { background: #6B4FD8; border-color: #5b3fc4; }
    /* The panel's hide button, in the same style: smaller, outlined. */
    .st-key-agent_hide button {
        width: 2.2rem; height: 2.2rem; min-height: 2.2rem;
        padding: 0; border-radius: 50%;
        border: 1.5px solid #8E5CD9;
        background: rgba(107, 79, 216, 0.08);
        color: #5b3fc4;
        box-shadow: 0 1px 5px rgba(91, 63, 196, 0.15);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .st-key-agent_hide button:hover {
        transform: scale(1.06); color: #4a2fb0;
        background: rgba(107, 79, 216, 0.16); box-shadow: 0 3px 10px rgba(91, 63, 196, 0.28);
    }
    .st-key-agent_open_flag button::after {
        content: ""; position: absolute; top: 1px; right: 1px;
        width: 0.7rem; height: 0.7rem; border-radius: 50%;
        background: #E8A317; border: 2px solid #ffffff;
    }
    [class*="st-key-aibox_"] {
        background: rgba(107, 79, 216, 0.06);
        border-left: 3px solid #8E5CD9;
        border-radius: 8px;
        padding: 0.75rem 1rem 0.8rem 1rem;
        margin: 0.25rem 0 0.85rem 0;
        gap: 0.5rem;
    }
    /* The label keeps its own line: Streamlit tightens the space between the
       two markdown elements, and the label sat on the first line of text. */
    [class*="st-key-aibox_"] .conductor-ai-comment-label {
        display: block; margin: 0; padding-bottom: 0.35rem; line-height: 1.2;
    }
    [class*="st-key-aibox_"] [data-testid="stMarkdownContainer"] p { line-height: 1.55; margin: 0 0 0.55rem 0; }
    [class*="st-key-aibox_"] [data-testid="stMarkdownContainer"] > *:last-child,
    [class*="st-key-aibox_"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }
    [class*="st-key-aibox_"] [data-testid="stMarkdownContainer"] ul,
    [class*="st-key-aibox_"] [data-testid="stMarkdownContainer"] ol { margin: 0.1rem 0 0.55rem 0; padding-left: 1.2rem; }
    /* The waiting line in the chat box's place: centred, with a spinner. */
    .st-key-chat_busy { min-height: 2.9rem; justify-content: center; }
    .st-key-chat_busy [data-testid="stMarkdownContainer"] p { margin: 0; }
    .conductor-busy {
        display: flex; align-items: center; gap: 0.75rem;
        min-height: 2.9rem; padding: 0 0.35rem;
        color: #5b6170; font-size: 0.98rem; line-height: 1.3;
    }
    .conductor-stale {
        font-size: 0.8rem; color: #8a5a00; background: rgba(240, 170, 30, 0.12);
        border: 1px solid rgba(240, 170, 30, 0.35); border-radius: 999px; padding: 0.12rem 0.6rem;
    }
    .conductor-eq-group {
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700;
        color: #9aa0b0; line-height: 1.4; padding: 0.45rem 0.45rem 0.15rem;
    }
    .st-key-examples_list [data-testid="stMarkdownContainer"]:has(.conductor-eq-group) { margin-bottom: 0; }
    .conductor-spin {
        flex: none; width: 1.05rem; height: 1.05rem; border-radius: 50%;
        border: 2.5px solid rgba(29, 111, 232, 0.18); border-top-color: #1D6FE8;
        animation: conductor-spin 0.9s linear infinite;
    }
    .conductor-busy-agent .conductor-spin { border-color: rgba(107, 79, 216, 0.18); border-top-color: #6B4FD8; }
    .conductor-busy-agent { color: #5b3fc4; }
    @keyframes conductor-spin { to { transform: rotate(360deg); } }
    /* Discussion as a chat: the operator's messages as right-aligned bubbles. */
    .conductor-you-row { display: flex; justify-content: flex-end; margin: 0.25rem 0 0.5rem 0; }
    .conductor-you {
        max-width: 85%;
        background: #EEF3FC; border: 1px solid #DCE6F7; border-radius: 14px 14px 4px 14px;
        padding: 0.5rem 0.85rem; line-height: 1.45; color: var(--cd-ink);
    }
    .conductor-sent { color: var(--cd-muted); font-size: 0.85rem; margin: 0.1rem 0 0.4rem 0; }
    .conductor-study-title { font-weight: 600; }
    /* The question box stays at the bottom of the panel while the thread scrolls. */
    .st-key-adv_input {
        position: sticky; bottom: 0; z-index: 2;
        background: #f1f2fa;     /* the page under the advisor panel's tint */
        padding: 0.5rem 0 0.25rem 0; gap: 0.1rem;
        box-shadow: 0 -8px 12px -10px rgba(30, 40, 60, 0.18);
    }
    /* Who each box belongs to: CONDUCTOR (blue) runs studies; the advisor
       (violet, tinted panel) discusses and suggests. */
    .conductor-who {
        display: inline-block; white-space: nowrap;
        background: var(--cd-brand-soft); color: var(--cd-brand);
        border: 1px solid rgba(29, 111, 232, 0.25); border-radius: 0.7rem;
        padding: 0.2rem 0.6rem; font-size: 0.8rem; font-weight: 700;
    }
    .st-key-ai_panel {
        background: rgba(107, 79, 216, 0.035);
        border: 1px solid rgba(107, 79, 216, 0.16); border-radius: 14px;
        padding: 0.6rem 0.75rem 0.4rem 0.75rem;
    }
    /* The generator tables: plain text rows, a hairline between them. */
    [class*="st-key-genrow_"] { border-top: 1px solid var(--cd-border); padding-top: 0.3rem; }
    .gen-head { font-size: 0.8rem; font-weight: 600; color: var(--cd-muted); }
    .gen-cell { font-size: 0.92rem; }
    .gen-changed {
        margin-left: 0.4rem; font-size: 0.7rem; font-weight: 600; color: var(--cd-brand);
        background: var(--cd-brand-soft); border-radius: 0.6rem; padding: 0.05rem 0.45rem;
    }
    [class*="st-key-gentable_"] [data-testid="stButton"] button { min-height: 1.6rem; padding: 0 0.4rem; }
    /* The System tab's assumption chips: small pills, filled when changed. */
    .st-key-sys_chips button {
        border-radius: 999px; min-height: 1.9rem; padding: 0.1rem 0.85rem;
    }
    .st-key-sys_chips button p { font-size: 0.85rem; }
    /* An advisor card: title, why, and a small hand-off to CONDUCTOR — tight. */
    [class*="st-key-card_"] { gap: 0.1rem; }
    /* Its title row never wraps: the title shrinks (and wraps its text). */
    [class*="st-key-row_"] { flex-wrap: nowrap !important; }
    [class*="st-key-row_"] > div:first-child { flex: 1 1 0 !important; min-width: 0 !important; }
    [class*="st-key-card_"] [data-testid="stCaptionContainer"] p { margin-bottom: 0; }
    [class*="st-key-card_"] [data-testid="stButton"] button {
        min-height: 1.6rem; padding: 0.05rem 0.65rem; border-radius: 0.8rem;
        color: var(--cd-brand); border-color: rgba(29, 111, 232, 0.35); background: var(--cd-surface);
    }
    [class*="st-key-card_"] [data-testid="stButton"] button p { font-size: 0.78rem; white-space: nowrap; }
    [class*="st-key-card_"] [data-testid="stButton"] button:hover { background: var(--cd-brand-soft); }
    .conductor-ai-badge {
        display: inline-block; white-space: nowrap;
        background: rgba(107, 79, 216, 0.10);
        color: #5b3fc4;
        border-radius: 0.6rem;
        padding: 0.05rem 0.5rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-right: 0.4rem;
    }
    @media (max-width: 900px) {
        .conductor-brand-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# First-run setup — Gemini API key
# ---------------------------------------------------------------------------
_ENV_PATH = _APP_DIR / ".env"


def _render_brand_block(title: str, caption: str | None = None, *, image_width: int = 140) -> None:
    """Render the shared logo + title treatment used across the app."""
    logo_uri = _image_as_data_uri(_LOGO_PATH)
    st.markdown(
        f"""
        <div class="conductor-brand">
            <img src="{logo_uri}" alt="{_BRAND_NAME} logo" style="width:{image_width}px;" />
            <div class="conductor-brand-text">
                <h1 class="conductor-brand-title">{title}</h1>
                {f'<p class="conductor-brand-tagline">{caption}</p>' if caption else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_main_hero(compact: bool = False) -> None:
    logo_uri = _image_as_data_uri(_LOGO_PATH)
    shell = "conductor-hero-shell conductor-hero-compact" if compact else "conductor-hero-shell"
    st.markdown(
        f"""
        <div id="conductor-hero" class="{shell}">
            <div class="conductor-brand">
                <img src="{logo_uri}" alt="{_BRAND_NAME} logo" style="width:150px;" />
                <div class="conductor-brand-text">
                    <h1 class="conductor-brand-title">{_BRAND_NAME}</h1>
                    <p class="conductor-brand-tagline">{_BRAND_TAGLINE}</p>
                </div>
            </div>
            <p class="conductor-brand-support">
                Ask about security, contingencies, risk, flexibility or hosting capacity — or try a change to
                the grid and compare it with today. CONDUCTOR runs the studies, and every figure comes from a solver
                and is traced; the advisor beside the conversation helps you think results through.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # `st.iframe` with an HTML string: a same-origin iframe that runs
    # JavaScript — what `st.components.v1.html` did, which is deprecated and
    # past its announced removal date. The script reaches the page through
    # `window.parent` as before. (`st.html` was tried first and dropped the
    # script.) Height 1, not 0: `st.iframe` rejects a zero height. Verified
    # in Chrome: header, Upload Network / Upload Data / Info, upload panel.
    st.iframe(
        """
        <script>
        const parentWindow = window.parent;
        const parentDoc = parentWindow.document;
        const previousCleanup = parentWindow.__conductorHeroCleanup;
        const styleId = 'conductor-floating-brand-style';
        const brandId = 'conductor-floating-brand';
        const baseUrl = 'BASE_URL_PLACEHOLDER';
        // Guard: a broken cleanup left by a prior (e.g. hot-reloaded) iframe must
        // not abort this script before the brand + buttons are rebuilt.
        if (typeof previousCleanup === 'function') {
            try { previousCleanup(); }
            catch (e) { (parentWindow.console || console).error('[conductor] previousCleanup failed:', e); }
        }

        const ensureBrandStyle = () => {
            let style = parentDoc.getElementById(styleId);
            if (!style) {
                style = parentDoc.createElement('style');
                style.id = styleId;
                parentDoc.head.appendChild(style);
            }
            style.textContent = `
                #${brandId} {
                    position: fixed;
                    left: 3.15rem;
                    top: 0;
                    height: 60px;
                    z-index: 9999999;
                    display: flex;
                    align-items: center;
                    gap: 0.4rem;
                    max-width: calc(100vw - 12rem);
                    opacity: 1;
                    pointer-events: none;
                    color: #2e3140;
                    white-space: nowrap;
                }
                #${brandId} img {
                    width: 38px;
                    height: auto;
                    display: block;
                    filter: saturate(1.05);
                }
                #${brandId} .cb-name {
                    display: block;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    font-size: 1.2rem;
                    line-height: 1;
                    font-weight: 800;
                    letter-spacing: 0.025em;
                    color: #2e3140;
                    font-family: inherit;
                    padding: 0.26rem 0.62rem 0.3rem 0.62rem;
                    border-radius: 999px;
                    background: rgba(255,255,255,0.64);
                    border: 1px solid rgba(46,49,64,0.08);
                    backdrop-filter: blur(10px);
                }
                .cb-sep {
                    width: 1px;
                    height: 1.3rem;
                    background: rgba(46,49,64,0.18);
                    margin: 0 0.2rem;
                    flex-shrink: 0;
                }
                .cb-upload-btn {
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 0.92rem;
                    font-weight: 600;
                    color: #4a5270;
                    padding: 0.22rem 0.52rem;
                    border-radius: 6px;
                    transition: background 0.12s, color 0.12s;
                    white-space: nowrap;
                    pointer-events: auto;
                    font-family: inherit;
                    letter-spacing: 0.01em;
                    line-height: 1;
                }
                .cb-upload-btn:hover { background: rgba(46,49,64,0.07); color: #2e3140; }
                .cb-panel {
                    position: fixed;
                    top: 64px;
                    z-index: 9999998;
                    background: #fff;
                    border: 1px solid rgba(46,49,64,0.13);
                    border-radius: 10px;
                    padding: 1rem 1.1rem;
                    box-shadow: 0 6px 28px rgba(0,0,0,0.11);
                    width: 310px;
                    display: none;
                    flex-direction: column;
                    gap: 0.6rem;
                    pointer-events: auto;
                }
                .cb-panel.is-open { display: flex; }
                /* The upload and info panels as centred pop-ups over a dimmed page. */
                .cb-panel.cb-modal {
                    top: 50% !important; left: 50% !important; transform: translate(-50%, -50%);
                    width: min(860px, 94vw); max-height: 86vh; overflow-y: auto;
                    padding: 1.2rem 1.5rem 1.3rem; gap: 0.75rem; border-radius: 14px;
                    box-shadow: 0 18px 60px rgba(15, 20, 35, 0.28);
                }
                #cb-info-panel.cb-modal { width: min(520px, 94vw); }
                .cb-backdrop {
                    position: fixed; inset: 0; z-index: 9999997; background: rgba(15, 20, 35, 0.32);
                    display: none; pointer-events: auto;
                }
                .cb-backdrop.is-open { display: block; }
                .cb-panel-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
                .cb-panel-head .cb-panel-title { font-size: 1.02rem; }
                .cb-close {
                    border: none; background: transparent; font-size: 1.05rem; color: #5c6272; cursor: pointer;
                    width: 2rem; height: 2rem; border-radius: 50%; line-height: 1;
                }
                .cb-close:hover { background: rgba(46,49,64,0.08); color: #2e3140; }
                .cb-template {
                    display: flex; gap: 0.6rem; align-items: flex-start; margin-top: 0.35rem;
                    padding: 0.6rem 0.75rem; border: 1px dashed rgba(26,115,232,0.45); border-radius: 10px;
                    background: rgba(26,115,232,0.04); color: #3c4152 !important; text-decoration: none !important;
                    font-size: 0.78rem; line-height: 1.35;
                }
                .cb-template:hover { background: rgba(26,115,232,0.09); }
                .cb-template strong { color: #1a73e8; font-size: 0.82rem; }
                .cb-template-icon { font-size: 1.05rem; color: #1a73e8; line-height: 1.2; }
                .cb-modal-grid { display: grid; grid-template-columns: minmax(260px, 1fr) 1.25fr; gap: 1.6rem; align-items: start; }
                .cb-modal-main { display: flex; flex-direction: column; gap: 0.6rem; }
                .cb-modal-side .cb-fmt-guide { border-top: none; padding-top: 0; margin-top: 0; }
                .cb-modal-side { border-left: 1px solid rgba(46,49,64,0.08); padding-left: 1.4rem; }
                @media (max-width: 760px) {
                    .cb-modal-grid { grid-template-columns: 1fr; }
                    .cb-modal-side { border-left: none; padding-left: 0; }
                }
                .cb-panel-title { font-size: 0.88rem; font-weight: 700; color: #2e3140; }
                .cb-panel input[type=file] { font-size: 0.8rem; color: #4a5270; }
                .cb-check {
                    display: flex;
                    align-items: flex-start;
                    gap: 0.4rem;
                    font-size: 0.77rem;
                    color: #5c6272;
                    cursor: pointer;
                    line-height: 1.35;
                }
                .cb-check input { margin-top: 0.15rem; flex-shrink: 0; cursor: pointer; }
                .cb-submit {
                    background: #1D6FE8;
                    color: #fff;
                    border: none;
                    border-radius: 6px;
                    padding: 0.4rem 0.9rem;
                    font-size: 0.82rem;
                    font-weight: 600;
                    cursor: pointer;
                    font-family: inherit;
                    transition: background 0.12s;
                    align-self: flex-start;
                }
                .cb-submit:hover { background: #1d5fd4; }
                .cb-submit:disabled { background: #a0aec0; cursor: not-allowed; }
                .cb-status { font-size: 0.78rem; color: #5c6272; min-height: 1rem; }
                .cb-status.ok { color: #1a7f4b; }
                .cb-status.err { color: #c0392b; }
                .cb-info-btn { font-size: 0.8rem; opacity: 0.7; padding: 0.22rem 0.52rem; }
                .cb-info-btn:hover { opacity: 1; background: rgba(46,49,64,0.07); }
                .cb-info-row { display: flex; flex-direction: column; gap: 0.25rem; }
                .cb-info-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; color: #9aa0b0; font-weight: 700; }
                .cb-info-value { font-size: 0.85rem; color: #2e3140; font-weight: 600; word-break: break-all; }
                #cb-info-panel { width: 380px; }
                .cb-cap-list { margin: 0; padding: 0 0 0 1.1rem; display: flex; flex-direction: column; gap: 0.38rem; }
                .cb-cap-list li { font-size: 0.78rem; color: #4a5270; line-height: 1.4; }
                .cb-cap-list li strong { color: #2e3140; }
                .cb-fmt-guide { border-top: 1px solid rgba(46,49,64,0.08); padding-top: 0.55rem; margin-top: 0.1rem; display: flex; flex-direction: column; gap: 0.3rem; }
                .cb-fmt-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; color: #9aa0b0; font-weight: 700; margin-bottom: 0.1rem; }
                .cb-fmt-row { display: flex; align-items: flex-start; gap: 0.45rem; }
                .cb-fmt-tag { font-family: monospace; font-size: 0.73rem; font-weight: 700; color: #1D6FE8; background: rgba(29,111,232,0.09); padding: 0.05rem 0.32rem; border-radius: 4px; flex-shrink: 0; margin-top: 0.08rem; }
                .cb-fmt-desc { font-size: 0.74rem; color: #5c6272; line-height: 1.4; }
                .cb-fmt-desc code { font-family: monospace; font-size: 0.7rem; background: rgba(46,49,64,0.07); padding: 0.02rem 0.25rem; border-radius: 3px; }
                .cb-csv-example { font-family: monospace; font-size: 0.67rem; background: rgba(46,49,64,0.06); border-radius: 4px; padding: 0.4rem 0.5rem; margin: 0.1rem 0 0; color: #2e3140; overflow-x: auto; line-height: 1.5; white-space: pre; }
                .cb-warn-box { background: #fff8e1; border: 1px solid #f0c040; border-radius: 5px; padding: 0.45rem 0.55rem; font-size: 0.75rem; color: #7a5a00; line-height: 1.5; }
                .cb-warn-box strong { color: #5a3e00; }
                .cb-reload-btn { margin-top: 0.45rem; width: 100%; padding: 0.35rem 0; font-size: 0.78rem; font-weight: 600; background: #1D6FE8; color: #fff; border: none; border-radius: 5px; cursor: pointer; }
                .cb-reload-btn:hover { background: #1d5fd4; }
                .cb-dl-btn { background: #fff; color: #1D6FE8; border: 1.5px solid #1D6FE8; }
                .cb-dl-btn:hover { background: #f0f7ff; }
                .cb-bus-chips { display: flex; flex-wrap: wrap; gap: 0.25rem; max-height: 7rem; overflow-y: auto; padding: 0.1rem 0; }
                .cb-bus-chip { font-family: monospace; font-size: 0.67rem; background: rgba(29,111,232,0.09); color: #1D6FE8; border-radius: 3px; padding: 0.05rem 0.28rem; white-space: nowrap; }
                .cb-success-box { background: #f0f7ff; border: 1px solid #9dc8f5; border-radius: 5px; padding: 0.45rem 0.55rem; font-size: 0.75rem; color: #1a3a5c; line-height: 1.5; }
                .cb-ds-toggle { display: flex; gap: 0.25rem; background: rgba(46,49,64,0.07); border-radius: 7px; padding: 0.2rem; margin-bottom: 0.55rem; }
                .cb-ds-tab { flex: 1; border: none; background: transparent; color: #5a5f6c; font-size: 0.78rem; font-weight: 600; padding: 0.32rem 0; border-radius: 5px; cursor: pointer; transition: background 0.12s, color 0.12s; }
                .cb-ds-tab:hover { color: #2e3140; }
                .cb-ds-tab-active { background: #fff; color: #1D6FE8; box-shadow: 0 1px 2px rgba(46,49,64,0.15); }
                .cb-ds-section { margin-bottom: 0.6rem; }
                .cb-ds-sub { display: block; font-size: 0.69rem; color: #7a7f8c; margin-bottom: 0.35rem; }
                .cb-adv-card { border-top: 1px solid rgba(46,49,64,0.08); margin-top: 0.2rem; padding-top: 0.55rem; }
                .cb-adv-actions { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.35rem 0 0.2rem; }
                .cb-chip-btn {
                    border: 1px solid rgba(46,49,64,0.2);
                    background: #fff;
                    color: #2e3140;
                    border-radius: 16px;
                    padding: 0.2rem 0.55rem;
                    font-size: 0.72rem;
                    font-weight: 600;
                    cursor: pointer;
                }
                .cb-chip-btn:hover { background: rgba(46,49,64,0.06); }
                .cb-adv-upload-wrap { display: none; margin-top: 0.35rem; }

                .cb-drawer-overlay {
                    position: fixed;
                    inset: 0;
                    background: rgba(16, 22, 36, 0.35);
                    z-index: 9999997;
                    display: none;
                }
                .cb-drawer-overlay.is-open { display: block; }
                .cb-drawer {
                    position: fixed;
                    top: 0;
                    right: 0;
                    height: 100vh;
                    width: var(--cb-drawer-w, min(460px, 92vw));
                    min-width: 340px;
                    max-width: 95vw;
                    background: #fff;
                    border-left: 1px solid rgba(46,49,64,0.14);
                    box-shadow: -12px 0 30px rgba(0,0,0,0.15);
                    z-index: 9999998;
                    transform: translateX(100%);
                    transition: transform 0.18s ease;
                    display: flex;
                    flex-direction: column;
                }
                .cb-drawer.is-open { transform: translateX(0); }
                .cb-drawer-head {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0.8rem 0.95rem;
                    border-bottom: 1px solid rgba(46,49,64,0.12);
                }
                .cb-drawer-controls {
                    display: flex;
                    align-items: center;
                    gap: 0.4rem;
                    font-size: 0.72rem;
                    color: #5c6272;
                    margin-right: auto;
                    margin-left: 0.8rem;
                }
                .cb-drawer-controls input[type=range] { width: 120px; }
                .cb-drawer-wval {
                    min-width: 2.2rem;
                    text-align: right;
                    font-variant-numeric: tabular-nums;
                    color: #2e3140;
                    font-weight: 600;
                }
                .cb-drawer-title { font-size: 0.86rem; font-weight: 700; color: #2e3140; }
                .cb-drawer-close {
                    border: none;
                    background: transparent;
                    font-size: 1rem;
                    line-height: 1;
                    color: #5c6272;
                    cursor: pointer;
                }
                .cb-drawer-body {
                    overflow-y: auto;
                    overflow-x: hidden;
                    padding: 0.8rem 0.95rem 1rem;
                    display: flex;
                    flex-direction: column;
                    gap: 0.65rem;
                    box-sizing: border-box;
                }
                .cb-drawer-body .cb-fmt-guide,
                .cb-drawer-body .cb-fmt-row,
                .cb-drawer-body .cb-fmt-desc,
                .cb-drawer-body .cb-csv-example { width: 100%; box-sizing: border-box; }
                .cb-drawer-body .cb-fmt-row { align-items: flex-start; }
                .cb-drawer-body .cb-fmt-desc { min-width: 0; overflow-wrap: anywhere; word-break: break-word; }
                .cb-drawer-body .cb-fmt-desc code {
                    white-space: normal;
                    overflow-wrap: anywhere;
                    word-break: break-word;
                }
                .cb-drawer-body .cb-csv-example {
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    word-break: break-word;
                }
                @media (max-width: 900px) {
                    #${brandId} { left: 2.75rem; max-width: calc(100vw - 8.5rem); }
                    #${brandId} img { width: 24px; }
                    #${brandId} .cb-name { font-size: 0.88rem; padding: 0.22rem 0.5rem 0.24rem 0.5rem; }
                }
            `;
        };

        const findHeaderHost = () => (
            parentDoc.querySelector('header[data-testid="stHeader"]')
            || parentDoc.querySelector('[data-testid="stHeader"]')
            || null
        );

        const ensureFloatingBrand = () => {
            let brand = parentDoc.getElementById(brandId);
            if (!brand) {
                brand = parentDoc.createElement('div');
                brand.id = brandId;
                parentDoc.body.appendChild(brand);
            }
            // Always refresh innerHTML so buttons survive Streamlit re-runs
            brand.innerHTML = `
                <img alt="" />
                <span class="cb-name"></span>
                <div class="cb-sep"></div>
                <button class="cb-upload-btn" id="cb-net-btn">Upload Network</button>
                <button class="cb-upload-btn" id="cb-data-btn">Upload Data</button>
                <button class="cb-upload-btn cb-info-btn" id="cb-info-btn">ℹ Info</button>
            `;
            brand.querySelector('img').src = 'DATA_URI_PLACEHOLDER';
            brand.querySelector('.cb-name').textContent = 'BRAND_NAME_PLACEHOLDER';
            return brand;
        };

        let _headerH = 60;

        // Compute the pixel value we want for brand's left edge
        const computeBrandLeft = () => {
            // Check every sidebar-related selector — use the largest right edge found
            const sidebarSelectors = [
                'section[data-testid="stSidebar"]',
                '[data-testid="stSidebarContent"]',
                '[data-testid="stSidebar"]',
            ];
            let maxRight = 0;
            for (const sel of sidebarSelectors) {
                const el = parentDoc.querySelector(sel);
                if (el) {
                    const r = el.getBoundingClientRect();
                    if (r.right > maxRight) maxRight = r.right;
                }
            }
            if (maxRight > 50) return (maxRight + 10) + 'px';

            // Sidebar closed — place brand after the toggle button
            const header = findHeaderHost();
            if (!header) return '3.15rem';
            const headerRect = header.getBoundingClientRect();
            const leftBtns = Array.from(header.querySelectorAll('button')).filter(b => {
                const r = b.getBoundingClientRect();
                return r.width > 0 && r.right < headerRect.left + headerRect.width / 2;
            });
            if (leftBtns.length > 0) {
                const rightmost = Math.max(...leftBtns.map(b => b.getBoundingClientRect().right));
                return (rightmost + 10) + 'px';
            }
            return '3.15rem';
        };

        // rAF loop: updates only when the value changes, tracks sidebar animations in real time
        let _rafId = null;
        let _lastLeft = '';
        const alignLoop = () => {
            const brand = parentDoc.getElementById(brandId);
            if (brand) {
                const next = computeBrandLeft();
                if (next !== _lastLeft) {
                    brand.style.left = next;
                    _lastLeft = next;
                }
            }
            _rafId = parentWindow.requestAnimationFrame(alignLoop);
        };

        const alignBrandLeft = () => {}; // kept for alignBrand() calls below

        const alignBrand = () => {
            const brand = parentDoc.getElementById(brandId);
            const header = findHeaderHost();
            if (!brand || !header) return;
            const rect = header.getBoundingClientRect();
            if (rect.height > 0) {
                _headerH = rect.height;
                brand.style.top = rect.top + 'px';
                brand.style.height = rect.height + 'px';
                parentDoc.querySelectorAll('.cb-panel:not(.cb-modal)').forEach(p => {
                    p.style.top = (rect.top + rect.height + 4) + 'px';
                });
            }
            alignBrandLeft();
        };

        const alignBrandWithRetry = (n) => {
            const h = findHeaderHost();
            const r = h ? h.getBoundingClientRect() : null;
            if (r && r.height > 0) { alignBrand(); }
            else if (n > 0) { setTimeout(() => alignBrandWithRetry(n - 1), 100); }
        };

        const ensureUploadControls = () => {
            // Remove stale panels and old document-level click handler from prior runs
            ['cb-net-panel', 'cb-data-panel', 'cb-info-panel', 'cb-adm-overlay', 'cb-adm-drawer', 'cb-backdrop'].forEach(id => {
                const el = parentDoc.getElementById(id);
                if (el) el.remove();
            });
            if (typeof parentWindow.__conductorCloseAll === 'function') {
                parentDoc.removeEventListener('click', parentWindow.__conductorCloseAll);
            }

            const netPanel = parentDoc.createElement('div');
            netPanel.id = 'cb-net-panel';
            netPanel.className = 'cb-panel cb-modal';
            netPanel.innerHTML = `
                <div class="cb-panel-head"><div class="cb-panel-title">Upload Network</div>
                    <button type="button" class="cb-close" aria-label="Close">✕</button></div>
                <div class="cb-modal-grid"><div class="cb-modal-main">
                <input type="file" id="cb-net-file" accept=".m,.json,.xlsx,.uct" />
                <label class="cb-check">
                    <input type="checkbox" id="cb-net-convert" checked />
                    Start generators at a fixed power factor (pf 1.0); unticked, they keep voltage control. Change each unit afterwards in the System tab.
                </label>
                <label class="cb-check" style="display:block; margin-top:0.35rem;">
                    Transformer tap policy
                    <select id="cb-net-tap-policy" style="display:block; width:100%; margin-top:0.25rem;">
                        <option value="current" selected>Keep current taps from uploaded file</option>
                        <option value="neutral">Force neutral taps (tap_pos = tap_neutral)</option>
                    </select>
                </label>
                <button id="cb-net-submit" class="cb-submit">Upload &amp; Load</button>
                <div id="cb-net-status" class="cb-status"></div>
                </div><div class="cb-modal-side">
                <div class="cb-fmt-guide">
                    <div class="cb-fmt-title">Supported formats</div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.m</span><span class="cb-fmt-desc">MATPOWER case file — IEEE cases, pglib-opf, or any compatible model</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.json</span><span class="cb-fmt-desc">pandapower JSON — export with <code>pp.to_json(net, "file.json")</code></span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.xlsx</span><span class="cb-fmt-desc">pandapower Excel — export with <code>pp.to_excel(net, "file.xlsx")</code></span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.uct</span><span class="cb-fmt-desc">UCTE/CGMES exchange — ENTSO-E standard for European TSO networks</span></div>
                </div>
                <div class="cb-adv-card">
                    <div class="cb-fmt-title">Advanced OPF admittance (expert)</div>
                    <div class="cb-fmt-desc">Hidden by default. Use only when replacing backend-generated OPF admittance with externally prepared CSV databases.</div>
                    <div class="cb-adv-actions">
                        <button type="button" class="cb-chip-btn" id="cb-adm-guide-open">Open guide</button>
                        <button type="button" class="cb-chip-btn" id="cb-adm-dl-core">Core template</button>
                        <button type="button" class="cb-chip-btn" id="cb-adm-dl-meta">Meta template</button>
                        <button type="button" class="cb-chip-btn" id="cb-adm-toggle">Show advanced upload</button>
                    </div>
                    <div class="cb-adv-upload-wrap" id="cb-adm-upload-wrap">
                        <input type="file" id="cb-adm-core-file" accept=".csv" />
                        <input type="file" id="cb-adm-meta-file" accept=".csv" style="margin-top:0.25rem;" />
                        <button id="cb-adm-submit" class="cb-submit" style="margin-top:0.35rem;">Upload advanced admittance</button>
                        <div id="cb-adm-status" class="cb-status"></div>
                    </div>
                </div>
                </div></div>
            `;
            parentDoc.body.appendChild(netPanel);

            const admOverlay = parentDoc.createElement('div');
            admOverlay.id = 'cb-adm-overlay';
            admOverlay.className = 'cb-drawer-overlay';
            parentDoc.body.appendChild(admOverlay);

            const admDrawer = parentDoc.createElement('div');
            admDrawer.id = 'cb-adm-drawer';
            admDrawer.className = 'cb-drawer';
            admDrawer.innerHTML = `
                <div class="cb-drawer-head">
                    <div class="cb-drawer-title">Advanced OPF admittance guide</div>
                    <div class="cb-drawer-controls">
                        <span>Width</span>
                        <input type="range" id="cb-adm-width" min="360" max="920" step="10" value="460" />
                        <span class="cb-drawer-wval" id="cb-adm-width-val">460</span>
                    </div>
                    <button type="button" class="cb-drawer-close" id="cb-adm-guide-close" aria-label="Close">✕</button>
                </div>
                <div class="cb-drawer-body">
                    <div class="cb-fmt-desc"><strong>When to use</strong><br>
                        Use this only for expert workflows where OPF results must match an external/legacy admittance pipeline.
                        Typical cases: reproducibility against prior studies, custom tap/admittance conventions, or validated precomputed N-1 databases.
                        For normal operation, upload only the network and let backend admittance generation run automatically.
                    </div>

                    <div class="cb-fmt-guide" style="margin-top:0.1rem;">
                        <div class="cb-fmt-title">What this upload replaces</div>
                        <div class="cb-fmt-desc">Uploading core+meta CSVs replaces in-memory OPF admittance databases:</div>
                        <div class="cb-fmt-desc">• Full case admittance (<code>db_full</code>)</div>
                        <div class="cb-fmt-desc">• N-1 line admittance (<code>db_n1_line</code>)</div>
                        <div class="cb-fmt-desc">• N-1 transformer admittance (<code>db_n1_trafo</code>)</div>
                        <div class="cb-fmt-desc">It does <strong>not</strong> upload timeseries and does <strong>not</strong> change the simulation clock.</div>
                    </div>

                    <div class="cb-fmt-guide" style="margin-top:0.2rem;">
                        <div class="cb-fmt-title">Which tools use these values</div>
                        <div class="cb-fmt-desc">Used by OPF-based tools:</div>
                        <div class="cb-fmt-desc">• Flexibility optimize (N-0 OPF)</div>
                        <div class="cb-fmt-desc">• Robust OPF (heuristic + scenario paths)</div>
                        <div class="cb-fmt-desc">• Contingency optimize (N-1 OPF)</div>
                        <div class="cb-fmt-desc" style="margin-top:0.25rem;">Not used by runpp-only tools:</div>
                        <div class="cb-fmt-desc">• Real-time RSA snapshot</div>
                        <div class="cb-fmt-desc">• Worst-case timestamp scan</div>
                        <div class="cb-fmt-desc">• Contingency simulate-all screening</div>
                    </div>

                    <div class="cb-fmt-guide" style="margin-top:0; padding-top:0; border-top:none;">
                        <div class="cb-fmt-title">Required files (2 CSVs)</div>
                        <div class="cb-fmt-row"><span class="cb-fmt-tag">core CSV</span><span class="cb-fmt-desc">Columns: <code>outage_scope,outage_index,section,from_bus,to_bus,tap,value</code></span></div>
                        <div class="cb-fmt-row"><span class="cb-fmt-tag">sections</span><span class="cb-fmt-desc"><code>Yff_r,Yff_i,Yft_r,Yft_i</code></span></div>
                        <pre class="cb-csv-example">outage_scope,outage_index,section,from_bus,to_bus,tap,value
full,,Yff_r,3,33,0,6.693097
full,,Yft_i,3,33,0,-17.483221
line,0,Yff_r,2,34,0,8.747084</pre>
                    </div>

                    <div class="cb-fmt-guide" style="margin-top:0.2rem;">
                        <div class="cb-fmt-row"><span class="cb-fmt-tag">meta CSV</span><span class="cb-fmt-desc">Columns: <code>outage_scope,outage_index,meta_type,from_bus,to_bus,trafo_index,tap,value</code></span></div>
                        <div class="cb-fmt-row"><span class="cb-fmt-tag">meta_type</span><span class="cb-fmt-desc"><code>TAPS,trafo_defaults,trafo_ranges,branch_to_trafo</code></span></div>
                        <div class="cb-fmt-row"><span class="cb-fmt-tag">scope</span><span class="cb-fmt-desc"><code>outage_scope</code> must be <code>full</code>, <code>line</code>, or <code>trafo</code>. At least one <code>full</code> entry is required.</span></div>
                        <pre class="cb-csv-example">outage_scope,outage_index,meta_type,from_bus,to_bus,trafo_index,tap,value
full,,TAPS,,,,0,
full,,trafo_defaults,,,0,,3
full,,branch_to_trafo,3,33,0,,</pre>
                    </div>

                    <div class="cb-fmt-guide" style="margin-top:0.2rem;">
                        <div class="cb-fmt-title">Construction checklist</div>
                        <div class="cb-fmt-desc">1) Build from a known-good baseline.</div>
                        <div class="cb-fmt-desc">2) Keep bus and outage indices aligned with the currently loaded network.</div>
                        <div class="cb-fmt-desc">3) Validate one OPF run before batch studies.</div>
                    </div>
                </div>
            `;
            parentDoc.body.appendChild(admDrawer);

            const dataPanel = parentDoc.createElement('div');
            dataPanel.id = 'cb-data-panel';
            dataPanel.className = 'cb-panel cb-modal';
            dataPanel.innerHTML = `
                <div class="cb-panel-head"><div class="cb-panel-title">Upload Time-series Data (.csv)</div>
                    <button type="button" class="cb-close" aria-label="Close">✕</button></div>
                <div class="cb-modal-grid"><div class="cb-modal-main">
                <div class="cb-ds-toggle">
                    <button type="button" id="cb-tab-meas" class="cb-ds-tab cb-ds-tab-active">Measurements</button>
                    <button type="button" id="cb-tab-fc" class="cb-ds-tab">Forecasts</button>
                </div>
                <div class="cb-ds-section" id="cb-sec-meas">
                    <div class="cb-ds-sub">Historical actuals · drives the simulation clock.</div>
                    <input type="file" id="cb-data-file" accept=".csv" />
                    <button id="cb-data-submit" class="cb-submit">Upload measurements</button>
                    <div id="cb-data-status" class="cb-status"></div>
                </div>
                <div class="cb-ds-section" id="cb-sec-fc" style="display:none">
                    <div class="cb-ds-sub">Look-ahead · read-only, used for planning / rescheduling.</div>
                    <input type="file" id="cb-fc-file" accept=".csv" />
                    <button id="cb-fc-submit" class="cb-submit">Upload forecasts</button>
                    <div id="cb-fc-status" class="cb-status"></div>
                </div>
                <a class="cb-template" href="${baseUrl}/api/data/template" download>
                    <span class="cb-template-icon">⬇</span>
                    <span><strong>Download a template for this network</strong><br>
                    Every substation name it expects, the loaded data's first hour as example values, and the
                    optional reactive columns. Fill it in and upload it back.</span></a>
                </div><div class="cb-modal-side">
                <div class="cb-fmt-guide">
                    <div class="cb-fmt-desc" style="margin-bottom:0.4rem">Both datasets use the <strong>same CSV format</strong>. Measurements replace what the clock runs on now; forecasts are read-only look-ahead data for planning tools.</div>
                    <div class="cb-fmt-title">Required columns</div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">timestamp</span><span class="cb-fmt-desc">Datetime string parseable by pandas — e.g. <code>2024-01-01 00:15:00</code>. Any regular interval; 15 min recommended.</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">substation_name</span><span class="cb-fmt-desc">Bus name from the loaded network. Fuzzy-matched — partial or prefix names are fine.</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">production_mw</span><span class="cb-fmt-desc">Total generator output at that bus in MW. Use <code>0.0</code> for load-only buses.</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">consumption_mw</span><span class="cb-fmt-desc">Total load at that bus in MW. Use <code>0.0</code> for generation-only buses.</span></div>
                    <div class="cb-fmt-title" style="margin-top:0.45rem">Optional columns — reactive power</div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">consumption_mvar</span><span class="cb-fmt-desc">Reactive demand in Mvar (positive draws, negative injects). When given, it is always used; the load power factor only fills empty cells.</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">production_mvar</span><span class="cb-fmt-desc">Reactive generation in Mvar, same sign as production_mw. Used for units at a fixed power factor or fixed Q; a voltage-controlled unit decides its own.</span></div>
                    <div class="cb-fmt-desc" style="margin-top:0.2rem">Leave a cell empty when not measured. The upload is checked (coverage, implausible power factors, kvar instead of Mvar, -9999 markers) — see the Data tab.</div>
                    <div class="cb-fmt-title" style="margin-top:0.45rem">Example</div>
                    <pre class="cb-csv-example">timestamp,substation_name,production_mw,consumption_mw
2024-01-01 00:00:00,Bus 1,2.5,1.2
2024-01-01 00:00:00,Bus 2,0.0,3.4
2024-01-01 00:15:00,Bus 1,2.3,1.4
2024-01-01 00:15:00,Bus 2,0.0,3.6</pre>
                    <div class="cb-fmt-desc" style="margin-top:0.3rem">Long format — one row per timestamp × bus. Comma-separated UTF-8; Excel BOM exports accepted.</div>
                </div>
                </div></div>
            `;
            parentDoc.body.appendChild(dataPanel);

            const infoPanel = parentDoc.createElement('div');
            infoPanel.id = 'cb-info-panel';
            infoPanel.className = 'cb-panel cb-modal';
            infoPanel.innerHTML = `
                <div class="cb-panel-head"><div class="cb-panel-title">About CONDUCTOR</div>
                    <button type="button" class="cb-close" aria-label="Close">✕</button></div>
                <div class="cb-info-row">
                    <span class="cb-info-label">LLM Model</span>
                    <span class="cb-info-value">LLM_MODEL_PLACEHOLDER</span>
                </div>
                <div class="cb-info-row">
                    <span class="cb-info-label">Provider</span>
                    <span class="cb-info-value">LLM_PROVIDER_PLACEHOLDER</span>
                </div>
                <div class="cb-info-row">
                    <span class="cb-info-label">Backend</span>
                    <span class="cb-info-value">${baseUrl}</span>
                </div>
                <div style="height:1px;background:rgba(46,49,64,0.1);margin:0.2rem 0"></div>
                <div class="cb-info-label" style="margin-bottom:0.3rem">Capabilities</div>
                <ul class="cb-cap-list">
                    <li><strong>Deterministic Security Assessment</strong> — N-0 power-flow, voltage &amp; loading checks at the current operating point</li>
                    <li><strong>N-1 Contingency Analysis</strong> — single-outage screening for lines, transformers, and generators</li>
                    <li><strong>Probabilistic Risk Assessment</strong> — Monte Carlo risk indices (ENS, LOLP, overload probability) under uncertainty</li>
                    <li><strong>Robust Corrective Dispatch (OPF)</strong> — worst-case redispatch with explicit uncertainty margins</li>
                    <li><strong>Flexibility Envelopes</strong> — feasible import/export power bounds per bus or zone</li>
                    <li><strong>Hosting Capacity Analysis</strong> — maximum DER penetration per bus without constraint violations</li>
                    <li><strong>KPI Evaluation</strong> — composite grid health scores (voltage quality, loading margin, loss index)</li>
                    <li><strong>Time-series Scanning</strong> — multi-timestamp horizon sweeps for violations and trend detection</li>
                    <li><strong>Custom Network Loading</strong> — MATPOWER .m, pandapower JSON / Excel, and UCTE files</li>
                    <li><strong>Custom Time-series Data</strong> — CSV upload with per-substation production &amp; consumption</li>
                </ul>
            `;
            parentDoc.body.appendChild(infoPanel);

            const backdrop = parentDoc.createElement('div');
            backdrop.id = 'cb-backdrop';
            backdrop.className = 'cb-backdrop';
            parentDoc.body.appendChild(backdrop);

            const positionUnder = (panel, btn) => {
                // A centred pop-up needs no position; a drop-down sits under its button.
                if (panel.classList.contains('cb-modal')) { backdrop.classList.add('is-open'); return; }
                const r = btn.getBoundingClientRect();
                panel.style.left = r.left + 'px';
            };

            const closeAll = () => {
                netPanel.classList.remove('is-open');
                dataPanel.classList.remove('is-open');
                infoPanel.classList.remove('is-open');
                backdrop.classList.remove('is-open');
            };
            [netPanel, dataPanel, infoPanel].forEach(panel => {
                panel.querySelectorAll('.cb-close').forEach(b => b.addEventListener('click', e => {
                    e.stopPropagation(); closeAll();
                }));
            });
            if (typeof parentWindow.__conductorEscape === 'function') {
                parentDoc.removeEventListener('keydown', parentWindow.__conductorEscape);
            }
            parentWindow.__conductorEscape = e => { if (e.key === 'Escape') closeAll(); };
            parentDoc.addEventListener('keydown', parentWindow.__conductorEscape);
            parentWindow.__conductorCloseAll = closeAll;

            const netBtn = parentDoc.getElementById('cb-net-btn');
            const dataBtn = parentDoc.getElementById('cb-data-btn');
            const infoBtn = parentDoc.getElementById('cb-info-btn');
            const openAdmGuide = () => {
                admOverlay.classList.add('is-open');
                admDrawer.classList.add('is-open');
            };
            const closeAdmGuide = () => {
                admOverlay.classList.remove('is-open');
                admDrawer.classList.remove('is-open');
            };

            netBtn && netBtn.addEventListener('click', e => {
                e.stopPropagation();
                const was = netPanel.classList.contains('is-open');
                closeAll();
                if (!was) { positionUnder(netPanel, netBtn); netPanel.classList.add('is-open'); }
            });
            dataBtn && dataBtn.addEventListener('click', e => {
                e.stopPropagation();
                const was = dataPanel.classList.contains('is-open');
                closeAll();
                if (!was) { positionUnder(dataPanel, dataBtn); dataPanel.classList.add('is-open'); }
            });
            infoBtn && infoBtn.addEventListener('click', e => {
                e.stopPropagation();
                const was = infoPanel.classList.contains('is-open');
                closeAll();
                if (!was) { positionUnder(infoPanel, infoBtn); infoPanel.classList.add('is-open'); }
            });

            parentDoc.addEventListener('click', closeAll);
            netPanel.addEventListener('click', e => e.stopPropagation());
            dataPanel.addEventListener('click', e => e.stopPropagation());
            infoPanel.addEventListener('click', e => e.stopPropagation());
            admDrawer.addEventListener('click', e => e.stopPropagation());
            admOverlay.addEventListener('click', closeAdmGuide);

            const _dlText = (filename, text) => {
                const blob = new Blob([text], { type: 'text/csv' });
                const url = (parentWindow.URL || parentWindow.webkitURL).createObjectURL(blob);
                const a = parentDoc.createElement('a');
                a.href = url;
                a.download = filename;
                parentDoc.body.appendChild(a);
                a.click();
                parentDoc.body.removeChild(a);
                (parentWindow.URL || parentWindow.webkitURL).revokeObjectURL(url);
            };

            const coreTemplate = [
                'outage_scope,outage_index,section,from_bus,to_bus,tap,value',
                'full,,Yff_r,3,33,0,0.0',
                'full,,Yff_i,3,33,0,0.0',
                'full,,Yft_r,3,33,0,0.0',
                'full,,Yft_i,3,33,0,0.0',
            ].join(String.fromCharCode(10));
            const metaTemplate = [
                'outage_scope,outage_index,meta_type,from_bus,to_bus,trafo_index,tap,value',
                'full,,TAPS,,,,0,',
                'full,,trafo_defaults,,,0,,0',
                'full,,trafo_ranges,,,0,0,',
                'full,,branch_to_trafo,3,33,0,,',
            ].join(String.fromCharCode(10));

            parentDoc.getElementById('cb-adm-guide-open').addEventListener('click', openAdmGuide);
            parentDoc.getElementById('cb-adm-guide-close').addEventListener('click', closeAdmGuide);

            const widthKey = 'conductor_adm_drawer_w';
            const widthInput = parentDoc.getElementById('cb-adm-width');
            const widthVal = parentDoc.getElementById('cb-adm-width-val');
            const _applyDrawerWidth = (raw) => {
                const n = Math.max(360, Math.min(920, parseInt(raw, 10) || 460));
                admDrawer.style.width = n + 'px';
                widthInput.value = String(n);
                widthVal.textContent = String(n);
                try { parentWindow.localStorage.setItem(widthKey, String(n)); } catch (_) {}
            };
            try {
                const saved = parentWindow.localStorage.getItem(widthKey);
                _applyDrawerWidth(saved || 460);
            } catch (_) {
                _applyDrawerWidth(460);
            }
            widthInput.addEventListener('input', (e) => _applyDrawerWidth(e.target.value));

            parentDoc.getElementById('cb-adm-dl-core').addEventListener('click', () => _dlText('admittance_core_template.csv', coreTemplate));
            parentDoc.getElementById('cb-adm-dl-meta').addEventListener('click', () => _dlText('admittance_meta_template.csv', metaTemplate));
            parentDoc.getElementById('cb-adm-toggle').addEventListener('click', () => {
                const wrap = parentDoc.getElementById('cb-adm-upload-wrap');
                const btn = parentDoc.getElementById('cb-adm-toggle');
                const open = wrap.style.display === 'block';
                wrap.style.display = open ? 'none' : 'block';
                btn.textContent = open ? 'Show advanced upload' : 'Hide advanced upload';
            });

            parentDoc.getElementById('cb-net-submit').addEventListener('click', async () => {
                const file = parentDoc.getElementById('cb-net-file').files[0];
                const convert = parentDoc.getElementById('cb-net-convert').checked;
                const tapPolicy = parentDoc.getElementById('cb-net-tap-policy').value;
                const status = parentDoc.getElementById('cb-net-status');
                const btn = parentDoc.getElementById('cb-net-submit');
                if (!file) { status.textContent = 'Select a file first (.m, .json, .xlsx, .uct).'; status.className = 'cb-status err'; return; }
                status.textContent = 'Uploading…'; status.className = 'cb-status'; btn.disabled = true;
                const fd = new FormData();
                fd.append('file', file, file.name);
                fd.append('convert_gen_to_sgen', convert ? 'true' : 'false');
                fd.append('tap_policy', tapPolicy);
                try {
                    const resp = await fetch(baseUrl + '/api/network/upload', { method: 'POST', body: fd });
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.detail || resp.statusText);

                    const activeBuses = data.active_bus_names || data.bus_names || [];
                    const allBuses    = data.bus_names || [];
                    const ex1 = activeBuses[0] || 'Bus_0';
                    const ex2 = activeBuses[1] || 'Bus_1';

                    status.innerHTML = '';
                    status.className = 'cb-status';

                    // Success summary
                    const box = parentDoc.createElement('div');
                    box.className = 'cb-success-box';
                    box.innerHTML = '✅ <strong>' + data.n_buses + ' buses · ' + data.n_lines + ' lines · ' + data.n_trafos + ' trafos</strong> · '
                        + data.n_timestamps + ' synthetic timestamps generated.';
                    status.appendChild(box);

                    // CSV guide
                    const guide = parentDoc.createElement('div');
                    guide.className = 'cb-fmt-guide';
                    guide.style.marginTop = '0.5rem';

                    const guideTitle = parentDoc.createElement('div');
                    guideTitle.className = 'cb-fmt-title';
                    guideTitle.textContent = 'How to prepare your CSV data';
                    guide.appendChild(guideTitle);

                    const exPre = parentDoc.createElement('pre');
                    exPre.className = 'cb-csv-example';
                    exPre.textContent = `timestamp,substation_name,production_mw,consumption_mw
2024-01-01 00:00:00,${ex1},2.5,1.2
2024-01-01 00:00:00,${ex2},0.0,3.4
2024-01-01 00:15:00,${ex1},2.3,1.4
2024-01-01 00:15:00,${ex2},0.0,3.6`;
                    guide.appendChild(exPre);

                    const busTitle = parentDoc.createElement('div');
                    busTitle.className = 'cb-fmt-title';
                    busTitle.style.marginTop = '0.4rem';
                    const showBuses = activeBuses.length > 0 ? activeBuses : allBuses;
                    busTitle.textContent = (activeBuses.length > 0
                        ? 'Buses with loads or generators (' + activeBuses.length + ')'
                        : 'All buses (' + allBuses.length + ')')
                        + ' — use these as substation_name:';
                    guide.appendChild(busTitle);

                    const chips = parentDoc.createElement('div');
                    chips.className = 'cb-bus-chips';
                    showBuses.forEach(name => {
                        const chip = parentDoc.createElement('span');
                        chip.className = 'cb-bus-chip';
                        chip.textContent = name;
                        chips.appendChild(chip);
                    });
                    guide.appendChild(chips);
                    status.appendChild(guide);

                    // Build downloadable CSV with all active bus names
                    const csvBuses = showBuses;
                    const tsList = ['2024-01-01 00:00:00', '2024-01-01 00:15:00', '2024-01-01 00:30:00', '2024-01-01 00:45:00'];
                    let csvLines = ['timestamp,substation_name,production_mw,consumption_mw'];
                    tsList.forEach(ts => { csvBuses.forEach(bn => { csvLines.push(ts + ',' + bn + ',0.0,0.0'); }); });
                    const csvBlob = csvLines.join(String.fromCharCode(10));

                    const dlBtn = parentDoc.createElement('button');
                    dlBtn.className = 'cb-reload-btn cb-dl-btn';
                    dlBtn.textContent = 'Download example CSV';
                    dlBtn.addEventListener('click', () => {
                        const blob = new Blob([csvBlob], { type: 'text/csv' });
                        const url = (parentWindow.URL || parentWindow.webkitURL).createObjectURL(blob);
                        const a = parentDoc.createElement('a');
                        a.href = url; a.download = 'example_timeseries.csv';
                        parentDoc.body.appendChild(a); a.click(); parentDoc.body.removeChild(a);
                        (parentWindow.URL || parentWindow.webkitURL).revokeObjectURL(url);
                    });
                    status.appendChild(dlBtn);

                    const reloadBtn = parentDoc.createElement('button');
                    reloadBtn.className = 'cb-reload-btn';
                    reloadBtn.textContent = 'Reload page';
                    reloadBtn.addEventListener('click', () => parentWindow.location.reload());
                    status.appendChild(reloadBtn);
                    btn.disabled = false;
                } catch (err) {
                    status.textContent = '❌ ' + err.message;
                    status.className = 'cb-status err';
                    btn.disabled = false;
                }
            });

            parentDoc.getElementById('cb-adm-submit').addEventListener('click', async () => {
                const coreFile = parentDoc.getElementById('cb-adm-core-file').files[0];
                const metaFile = parentDoc.getElementById('cb-adm-meta-file').files[0];
                const status = parentDoc.getElementById('cb-adm-status');
                const btn = parentDoc.getElementById('cb-adm-submit');
                if (!coreFile || !metaFile) {
                    status.textContent = 'Select both CSV files (core + meta).';
                    status.className = 'cb-status err';
                    return;
                }

                const send = async (overwrite) => {
                    const fd = new FormData();
                    fd.append('core_file', coreFile, coreFile.name);
                    fd.append('meta_file', metaFile, metaFile.name);
                    if (overwrite) fd.append('overwrite', 'true');
                    const resp = await fetch(baseUrl + '/api/network/upload_advanced_admittance', { method: 'POST', body: fd });
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.detail || resp.statusText);
                    return data;
                };

                status.textContent = 'Uploading advanced admittance…';
                status.className = 'cb-status';
                btn.disabled = true;
                try {
                    let data = await send(false);
                    if (data && data.status === 'confirm_required') {
                        if (!parentWindow.confirm(data.message + '  Replace it?')) {
                            status.textContent = 'Upload cancelled — existing admittance kept.';
                            status.className = 'cb-status';
                            btn.disabled = false;
                            return;
                        }
                        data = await send(true);
                    }
                    const s = data.summary || {};
                    status.innerHTML = '✅ Advanced admittance uploaded. '
                        + 'full scalars=' + (s.n_full_scalars ?? '?')
                        + ', line outages=' + (s.n_line_outages ?? '?')
                        + ', trafo outages=' + (s.n_trafo_outages ?? '?') + '.';
                    status.className = 'cb-status ok';
                } catch (err) {
                    status.textContent = '❌ ' + err.message;
                    status.className = 'cb-status err';
                } finally {
                    btn.disabled = false;
                }
            });

            // Segmented toggle: show one upload form at a time (Measurements / Forecasts).
            const tabMeas = parentDoc.getElementById('cb-tab-meas');
            const tabFc = parentDoc.getElementById('cb-tab-fc');
            const secMeas = parentDoc.getElementById('cb-sec-meas');
            const secFc = parentDoc.getElementById('cb-sec-fc');
            const selectDataset = (which) => {
                const isMeas = which === 'meas';
                secMeas.style.display = isMeas ? '' : 'none';
                secFc.style.display = isMeas ? 'none' : '';
                tabMeas.classList.toggle('cb-ds-tab-active', isMeas);
                tabFc.classList.toggle('cb-ds-tab-active', !isMeas);
            };
            tabMeas.addEventListener('click', () => selectDataset('meas'));
            tabFc.addEventListener('click', () => selectDataset('fc'));

            // POST a dataset; if the backend asks to confirm an overwrite of
            // existing user-uploaded data, prompt and retry with overwrite=true.
            // Returns the success payload, or null if the user declined.
            const postData = async (file, kind) => {
                const send = async (overwrite) => {
                    const fd = new FormData();
                    fd.append('file', file, file.name);
                    fd.append('kind', kind);
                    if (overwrite) fd.append('overwrite', 'true');
                    const resp = await fetch(baseUrl + '/api/data/upload', { method: 'POST', body: fd });
                    const data = await resp.json();
                    if (!resp.ok) throw new Error(data.detail || resp.statusText);
                    return data;
                };
                let data = await send(false);
                if (data && data.status === 'confirm_required') {
                    if (!parentWindow.confirm(data.message + '  Replace it?')) return null;
                    data = await send(true);
                }
                return data;
            };

            parentDoc.getElementById('cb-data-submit').addEventListener('click', async () => {
                const file = parentDoc.getElementById('cb-data-file').files[0];
                const status = parentDoc.getElementById('cb-data-status');
                const btn = parentDoc.getElementById('cb-data-submit');
                if (!file) { status.textContent = 'Select a .csv file first.'; status.className = 'cb-status err'; return; }
                status.textContent = 'Uploading…'; status.className = 'cb-status'; btn.disabled = true;
                try {
                    const data = await postData(file, 'measurements');
                    if (!data) { status.textContent = 'Upload cancelled — existing data kept.'; status.className = 'cb-status'; btn.disabled = false; return; }
                    const unmatched = data.unmatched_buses || [];
                    if (unmatched.length === 0) {
                        status.innerHTML = '✅ ' + data.n_timestamps + ' timestamps (' + data.first_timestamp + ' → ' + data.last_timestamp + '). Reloading…';
                        status.className = 'cb-status ok';
                        setTimeout(() => parentWindow.location.reload(), 1500);
                    } else {
                        status.innerHTML = '';
                        const box = parentDoc.createElement('div');
                        box.className = 'cb-warn-box';
                        box.innerHTML = '✅ <strong>' + data.n_timestamps + ' timestamps loaded</strong> (' + data.first_timestamp + ' → ' + data.last_timestamp + ').<br>'
                            + '⚠️ <strong>' + unmatched.length + ' bus' + (unmatched.length > 1 ? 'es' : '') + ' had no matching CSV row</strong> — base-case values kept:<br>'
                            + unmatched.map(b => '&nbsp;&nbsp;• ' + b).join('<br>');
                        const reloadBtn = parentDoc.createElement('button');
                        reloadBtn.className = 'cb-reload-btn';
                        reloadBtn.textContent = 'Reload page';
                        reloadBtn.addEventListener('click', () => parentWindow.location.reload());
                        status.appendChild(box);
                        status.appendChild(reloadBtn);
                        status.className = 'cb-status';
                        btn.disabled = false;
                    }
                } catch (err) {
                    status.textContent = '❌ ' + err.message;
                    status.className = 'cb-status err';
                    btn.disabled = false;
                }
            });

            // Forecast upload — read-only look-ahead data. Does NOT touch the
            // simulation clock, so no page reload is needed on success.
            parentDoc.getElementById('cb-fc-submit').addEventListener('click', async () => {
                const file = parentDoc.getElementById('cb-fc-file').files[0];
                const status = parentDoc.getElementById('cb-fc-status');
                const btn = parentDoc.getElementById('cb-fc-submit');
                if (!file) { status.textContent = 'Select a .csv file first.'; status.className = 'cb-status err'; return; }
                status.textContent = 'Uploading…'; status.className = 'cb-status'; btn.disabled = true;
                try {
                    const data = await postData(file, 'forecasts');
                    if (!data) { status.textContent = 'Upload cancelled — existing forecast kept.'; status.className = 'cb-status'; btn.disabled = false; return; }
                    status.innerHTML = '✅ <strong>Forecast loaded</strong> — ' + data.n_timestamps
                        + ' timestamps (' + data.first_timestamp + ' → ' + data.last_timestamp + ').<br>'
                        + 'Planning tools can now use <code>data_source="forecasts"</code>.';
                    status.className = 'cb-status ok';
                    btn.disabled = false;
                } catch (err) {
                    status.textContent = '❌ ' + err.message;
                    status.className = 'cb-status err';
                    btn.disabled = false;
                }
            });
        };

        // Run each init phase independently so a failure in one (e.g. panel
        // wiring) can never prevent the floating brand + buttons from rendering.
        const _safe = (label, fn) => {
            try { fn(); }
            catch (e) { (parentWindow.console || console).error('[conductor] ' + label + ' failed:', e); }
        };
        _safe('ensureBrandStyle', ensureBrandStyle);
        _safe('ensureFloatingBrand', ensureFloatingBrand);
        _safe('ensureUploadControls', ensureUploadControls);
        _safe('alignBrandWithRetry', () => alignBrandWithRetry(20));
        _safe('alignLoop', alignLoop);

        const resizeObserver = new ResizeObserver(alignBrand);
        const headerForObs = findHeaderHost();
        if (headerForObs) resizeObserver.observe(headerForObs);

        parentWindow.__conductorHeroCleanup = () => {
            resizeObserver.disconnect();
            if (_rafId) parentWindow.cancelAnimationFrame(_rafId);
        };
        </script>
        """.replace("DATA_URI_PLACEHOLDER", logo_uri).replace("BRAND_NAME_PLACEHOLDER", _BRAND_NAME).replace("BASE_URL_PLACEHOLDER", BASE_URL).replace("LLM_MODEL_PLACEHOLDER", _active_model_label()).replace("LLM_PROVIDER_PLACEHOLDER", _PROVIDER_LABELS.get(active_provider_name(), active_provider_name())),
        height=1,
    )


def _write_env(updates: dict[str, str]) -> None:
    """Merge key=value pairs into .env, preserving unrelated lines and comments."""
    lines = _ENV_PATH.read_text(encoding="utf-8").splitlines() if _ENV_PATH.exists() else []
    remaining = dict(updates)

    merged = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in remaining:
            merged.append(f"{key}={remaining.pop(key)}")
        else:
            merged.append(line)
    merged.extend(f"{k}={v}" for k, v in remaining.items())

    _ENV_PATH.write_text("\n".join(merged) + "\n", encoding="utf-8")
    # Inject into the running process so the change takes effect on the next rerun.
    os.environ.update(updates)
    reset_providers()


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _provider_configured(provider: str) -> bool:
    """Whether `provider` already has what it needs, so starting is one click."""
    if provider == "ollama":
        return bool(os.environ.get("OLLAMA_MODEL", "").strip())
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return _has_api_key()


def _render_google_setup() -> None:
    st.markdown(
        """
        #### Google Gemini API

        The **free tier** at Google AI Studio is sufficient — no payment needed.

        1. Go to [https://aistudio.google.com/](https://aistudio.google.com/)
        2. Sign in with any Google account
        3. Click **Get API key** → **Create API key**
        4. Copy the key and paste it below
        """
    )
    with st.form("api_key_setup"):
        key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy…",
            help="Your key is saved locally to .env and never sent anywhere else.",
        )
        thinking = _gemini_thinking_picker(key="setup_gemini_thinking")
        submitted = st.form_submit_button("Save & Start", type="primary", width="stretch")

    if submitted:
        key = key_input.strip()
        mc = _model_check.check("google", api_key=key) if key else None
        if not key:
            st.error("Please paste a valid API key before continuing.")
        elif mc is not None and mc.status == _model_check.KEY_REJECTED:
            # No model field on this form: a missing model is reported on the
            # next screen, where it can be changed. Only a bad key stops here.
            st.error(_model_problem_text(mc))
        else:
            _write_env({
                "LLM_PROVIDER": "google",
                "GEMINI_API_KEY": key,
                "GEMINI_THINKING_LEVEL": thinking,
            })
            st.session_state._launched = True
            st.success("API key saved! Starting the assistant…")
            st.rerun()


_GEMINI_DEFAULT_LABEL = "model default"


def _gemini_thinking_picker(*, key: str) -> str:
    """Gemini reasoning choice. Returns "" for the model's own budget.

    Gemini's ladder has four rungs to OpenAI's five, and an extra "leave it to
    the model" position that OpenAI has no equivalent for — so this is spelled
    out separately rather than forced onto a shared scale.
    """
    options = [_GEMINI_DEFAULT_LABEL, *GEMINI_THINKING_LEVELS]
    raw = os.environ.get("GEMINI_THINKING_LEVEL")
    current = (GEMINI_THINKING_LEVEL if raw is None else raw).strip().lower()
    value = current if current in GEMINI_THINKING_LEVELS else _GEMINI_DEFAULT_LABEL
    picked = st.select_slider(
        "Reasoning",
        options=options,
        value=value,
        key=key,
        help="Gemini reasons by default at a budget it chooses. Pin a level to "
             "trade answer quality against speed and cost. Models that predate "
             "thinking levels ignore this.",
    )
    return "" if picked == _GEMINI_DEFAULT_LABEL else picked


def _openai_effort_picker(*, key: str) -> str:
    """Reasoning-effort choice, shared by the setup and settings screens."""
    options = list(OPENAI_REASONING_EFFORTS)
    current = (os.environ.get("OPENAI_REASONING_EFFORT") or OPENAI_REASONING_EFFORT)
    index = options.index(current) if current in options else options.index("medium")
    return st.select_slider(
        "Reasoning effort",
        options=options,
        value=options[index],
        key=key,
        help="How hard the model thinks before answering. A turn plans several "
             "tool calls and then reasons over the results, so lowering this "
             "trades answer quality for speed and cost. Ignored by models that "
             "do not reason.",
    )


def _anthropic_effort_picker(*, key: str) -> str:
    """Effort choice, shared by the setup and settings screens.

    Anthropic's ladder has no "none" rung — thinking is adaptive and effort
    sets its depth — so this starts at `low` where the OpenAI picker starts at
    `none`. Forcing both onto one scale would misrepresent either.
    """
    options = list(ANTHROPIC_EFFORTS)
    current = (os.environ.get("ANTHROPIC_EFFORT") or ANTHROPIC_EFFORT)
    index = options.index(current) if current in options else options.index("medium")
    return st.select_slider(
        "Effort",
        options=options,
        value=options[index],
        key=key,
        help="How hard the model thinks before answering. A turn plans several "
             "tool calls and then reasons over the results, so lowering this "
             "trades answer quality for speed and cost. `max` earns its cost "
             "only on hard problems.",
    )


def _render_anthropic_setup() -> None:
    st.markdown(
        """
        #### Anthropic API (Claude)

        Paid per token — there is no free tier — so an account with credit is
        needed. If you would rather not pay, **Ollama** runs the same agent
        locally at no cost.

        1. Go to [console.anthropic.com](https://console.anthropic.com/settings/keys)
        2. Click **Create Key** and copy it
        3. Paste it below
        """
    )
    with st.form("anthropic_setup"):
        key_input = st.text_input(
            "Anthropic API key",
            type="password",
            placeholder="sk-ant-…",
            help="Your key is saved locally to .env and never sent anywhere else.",
        )
        model_input = st.text_input(
            "Model",
            value=os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL,
            help="Must support tool calling — CONDUCTOR drives the grid entirely "
                 "through tools.",
        )
        effort = _anthropic_effort_picker(key="setup_anthropic_effort")
        submitted = st.form_submit_button(
            "Save & Start", type="primary", width="stretch"
        )

    if not submitted:
        return

    key = key_input.strip()
    if not key:
        st.error("Please paste a valid API key before continuing.")
        return

    # Same reason as the OpenAI setup: a typo or a retired model is otherwise
    # only discovered on the first question.
    if _model_rejected("anthropic", model_input.strip() or ANTHROPIC_MODEL, api_key=key):
        return

    _write_env({
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": key,
        "ANTHROPIC_MODEL": model_input.strip() or ANTHROPIC_MODEL,
        "ANTHROPIC_EFFORT": effort,
    })
    os.environ.update({
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": key,
        "ANTHROPIC_MODEL": model_input.strip() or ANTHROPIC_MODEL,
        "ANTHROPIC_EFFORT": effort,
    })
    reset_providers()
    st.session_state._launched = True
    st.rerun()


def _render_anthropic_settings(*, key_prefix: str) -> bool:
    """Anthropic options, editable at any time. Returns True if saved."""
    with st.form(f"{key_prefix}_anthropic_settings"):
        model = st.text_input(
            "Model",
            value=os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL,
            help="Must support tool calling.",
        )
        key = st.text_input(
            "API key (leave blank to keep the current one)",
            type="password", placeholder="sk-ant-…",
        )
        effort = _anthropic_effort_picker(key=f"{key_prefix}_anthropic_effort")
        thinking = st.checkbox(
            "Adaptive thinking",
            value=(os.environ.get("ANTHROPIC_THINKING", "1").strip().lower()
                   not in ("0", "false", "no")),
            key=f"{key_prefix}_anthropic_thinking",
            help="The model decides when and how much to think, with the effort "
                 "above setting the depth. Turning it off is only accepted at "
                 "effort `high` or below, and is rarely worth it — lowering the "
                 "effort is the better way to spend less.",
        )
        if st.form_submit_button("Save settings", type="primary", width="stretch"):
            updates = {
                "LLM_PROVIDER": "anthropic",
                "ANTHROPIC_MODEL": model.strip() or ANTHROPIC_MODEL,
                "ANTHROPIC_EFFORT": effort,
                "ANTHROPIC_THINKING": "1" if thinking else "0",
            }
            if key.strip():
                updates["ANTHROPIC_API_KEY"] = key.strip()
            if _model_rejected("anthropic", updates["ANTHROPIC_MODEL"],
                               api_key=key.strip() or None):
                return False
            _write_env(updates)
            return True
    return False


def _render_openai_setup() -> None:
    st.markdown(
        """
        #### OpenAI API

        Paid per token — there is no free tier — so an account with credit is
        needed. If you would rather not pay, **Ollama** runs the same agent
        locally at no cost.

        1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        2. Click **Create new secret key** and copy it
        3. Paste it below

        **Using a different provider?** This option also drives anything
        speaking the OpenAI chat-completions dialect — Kimi, DeepSeek, Qwen,
        Groq, Mistral, OpenRouter, Azure OpenAI, a self-hosted vLLM server.
        Paste that provider's key here, name its model, and set its **Base
        URL** below. The model must support tool calling.
        """
    )
    with st.form("openai_setup"):
        key_input = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-…",
            help="Your key is saved locally to .env and never sent anywhere else.",
        )
        model_input = st.text_input(
            "Model",
            value=os.environ.get("OPENAI_MODEL") or OPENAI_MODEL,
            help="Must support tool calling — CONDUCTOR drives the grid entirely "
                 "through tools.",
        )
        effort = _openai_effort_picker(key="setup_openai_effort")
        # Here rather than only in Model settings: the probe below runs against
        # this URL, so without it a key for a compatible provider is checked
        # against OpenAI's own endpoint and rejected before the app ever starts.
        base_url_input = st.text_input(
            "Base URL",
            value=os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL,
            help="Leave as-is for OpenAI. Change it to use another provider "
                 "speaking the same dialect — Kimi, DeepSeek, Qwen, Groq, "
                 "OpenRouter, Azure OpenAI, or a local vLLM server.",
        )
        submitted = st.form_submit_button(
            "Save & Start", type="primary", width="stretch"
        )

    if not submitted:
        return

    key = key_input.strip()
    if not key:
        st.error("Please paste a valid API key before continuing.")
        return

    base_url = base_url_input.strip() or OPENAI_BASE_URL

    # Check the key and the model now rather than letting the first question
    # fail: a typo here is otherwise only discovered mid-conversation.
    info = openai_probe(api_key=key, base_url=base_url)
    if not info["reachable"]:
        st.error(info["error"] or f"Could not reach `{base_url}` with that key.")
        return
    if info["models"] and model_input.strip() not in info["models"]:
        st.error(
            f"This key cannot reach `{model_input.strip()}`. "
            f"Available: {', '.join(info['models'][:15])}"
        )
        return

    _write_env({
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": key,
        "OPENAI_MODEL": model_input.strip(),
        "OPENAI_REASONING_EFFORT": effort,
        "OPENAI_BASE_URL": base_url,
    })
    st.session_state._launched = True
    st.success("API key saved! Starting the assistant…")
    st.rerun()


def _render_openai_settings(*, key_prefix: str) -> bool:
    """OpenAI options, editable at any time. Returns True if saved."""
    with st.form(f"{key_prefix}_openai_settings"):
        model = st.text_input(
            "Model",
            value=os.environ.get("OPENAI_MODEL") or OPENAI_MODEL,
            help="Must support tool calling.",
        )
        key = st.text_input(
            "API key (leave blank to keep the current one)",
            type="password", placeholder="sk-…",
        )
        effort = _openai_effort_picker(key=f"{key_prefix}_openai_effort")
        api = st.selectbox(
            "Endpoint",
            options=list(OPENAI_APIS),
            index=list(OPENAI_APIS).index(
                (os.environ.get("OPENAI_API") or OPENAI_API)
                if (os.environ.get("OPENAI_API") or OPENAI_API) in OPENAI_APIS
                else "auto"
            ),
            format_func=lambda a: {
                "auto": "auto — Responses when reasoning is on",
                "chat": "chat/completions",
                "responses": "responses",
            }.get(a, a),
            key=f"{key_prefix}_openai_api",
            help="Newer reasoning models refuse function tools together with "
                 "reasoning on chat/completions, and CONDUCTOR uses tools on "
                 "every turn — so reasoning needs /v1/responses. Most "
                 "OpenAI-compatible servers implement only chat/completions.",
        )
        base_url = st.text_input(
            "Base URL",
            value=os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL,
            help="Change this to use any OpenAI-compatible endpoint — Kimi, "
                 "DeepSeek, Qwen, Groq, OpenRouter, Azure OpenAI, a local "
                 "vLLM server.",
        )
        if st.form_submit_button("Save settings", type="primary", width="stretch"):
            updates = {
                "LLM_PROVIDER": "openai",
                "OPENAI_MODEL": model.strip(),
                "OPENAI_REASONING_EFFORT": effort,
                "OPENAI_API": api,
                "OPENAI_BASE_URL": base_url.strip() or OPENAI_BASE_URL,
            }
            if key.strip():
                updates["OPENAI_API_KEY"] = key.strip()
            if _model_rejected("openai", model.strip(), api_key=key.strip() or None,
                               base_url=base_url.strip() or OPENAI_BASE_URL):
                return False
            _write_env(updates)
            return True
    return False


def _prompt_token_estimate() -> int:
    """Roughly how many tokens the system prompt and tool schemas occupy."""
    try:
        from agent.providers.schema import to_json_schema_tools
        from agent.system_prompt import get_system_prompt
        from agent.tool_schemas import TOOLS

        blob = json.dumps(to_json_schema_tools(TOOLS)) + get_system_prompt()
        return len(blob) // 4
    except Exception:  # noqa: BLE001
        return 25_000


def _render_ollama_settings(*, key_prefix: str) -> bool:
    """
    Every Ollama option, editable at any time. Returns True if saved.

    Sizing guidance is derived from the detected machine rather than hard-coded,
    since what is generous on a workstation is impossible on a small laptop.
    """
    hw = detect_hardware()
    required = _prompt_token_estimate()
    current_ctx = int(os.environ.get("OLLAMA_NUM_CTX") or OLLAMA_NUM_CTX)

    for note in hw_advice(hw):
        st.caption(note)

    info = ollama_probe()
    installed = [m["name"] for m in info["models"] if m["tools"] is not False]
    if not installed:
        installed = [os.environ.get("OLLAMA_MODEL") or ""]
    current_model = os.environ.get("OLLAMA_MODEL") or installed[0]

    with st.form(f"{key_prefix}_ollama_settings"):
        model = st.selectbox(
            "Model",
            options=installed,
            index=installed.index(current_model) if current_model in installed else 0,
        )

        suggested = recommended_num_ctx(hw, required)
        num_ctx = st.number_input(
            f"Context window — prompt needs ~{required:,} tokens, suggested {suggested:,}",
            min_value=4096, max_value=1_048_576, value=current_ctx, step=4096,
            help=(
                "Ollama defaults to ~4096 and silently discards anything beyond the "
                "window, dropping the system prompt first. CONDUCTOR refuses "
                "over-long prompts instead. Larger windows use more memory but, "
                "measured, do not slow processing down."
            ),
        )
        warning = num_ctx_warning(int(num_ctx), required, hw)
        if warning:
            st.warning(warning)

        col_a, col_b = st.columns(2)
        with col_a:
            keep_alive = st.text_input(
                "Keep model loaded for",
                value=os.environ.get("OLLAMA_KEEP_ALIVE") or OLLAMA_KEEP_ALIVE,
                help=(
                    "How long Ollama keeps the model and its prompt cache in memory. "
                    "The cache is what makes questions after the first one fast, so "
                    "short values make every question pay the full startup cost."
                ),
            )
            think_default = (os.environ.get("OLLAMA_THINK", "true").lower() == "true")
            think = st.checkbox(
                "Allow the model to reason before answering",
                value=think_default,
                help=(
                    "On by default, matching the hosted path — Gemini also reasons, "
                    "it just doesn't show the trace. Turning this off trades answer "
                    "quality for speed."
                ),
            )
        with col_b:
            timeout_s = st.number_input(
                "Request timeout (s)",
                min_value=30, max_value=3600,
                value=int(float(os.environ.get("OLLAMA_TIMEOUT_S") or OLLAMA_TIMEOUT_S)),
                step=30,
                help="Local generation is slow, especially while the model loads.",
            )
            output_reserve = st.number_input(
                "Tokens reserved for the reply",
                min_value=512, max_value=32768,
                value=int(os.environ.get("OLLAMA_OUTPUT_RESERVE") or OLLAMA_OUTPUT_RESERVE),
                step=512,
                help="Held back from the context window so the answer has room.",
            )

        host = st.text_input(
            "Ollama host",
            value=os.environ.get("OLLAMA_HOST") or OLLAMA_HOST,
        )

        if st.form_submit_button("Save settings", type="primary", width="stretch"):
            _write_env({
                "LLM_PROVIDER": "ollama",
                "OLLAMA_MODEL": model,
                "OLLAMA_NUM_CTX": str(int(num_ctx)),
                "OLLAMA_KEEP_ALIVE": keep_alive.strip(),
                "OLLAMA_THINK": "true" if think else "false",
                "OLLAMA_TIMEOUT_S": str(int(timeout_s)),
                "OLLAMA_OUTPUT_RESERVE": str(int(output_reserve)),
                "OLLAMA_HOST": host.strip(),
            })
            return True
    return False


def _render_google_settings(*, key_prefix: str) -> bool:
    """Gemini options, editable at any time. Returns True if saved."""
    with st.form(f"{key_prefix}_google_settings"):
        model = st.text_input(
            "Model",
            value=os.environ.get("GEMINI_MODEL") or GEMINI_MODEL,
            help="Free-tier Gemma models cap input well below this agent's "
                 "per-call overhead; a Flash model is the safe choice.",
        )
        thinking = _gemini_thinking_picker(key=f"{key_prefix}_gemini_thinking")
        key = st.text_input(
            "API key (leave blank to keep the current one)",
            type="password", placeholder="AIzaSy…",
        )
        if st.form_submit_button("Save settings", type="primary", width="stretch"):
            updates = {
                "LLM_PROVIDER": "google",
                "GEMINI_MODEL": model.strip(),
                "GEMINI_THINKING_LEVEL": thinking,
            }
            if key.strip():
                updates["GEMINI_API_KEY"] = key.strip()
            if _model_rejected("google", model.strip() or GEMINI_MODEL,
                               api_key=key.strip() or None):
                return False
            _write_env(updates)
            return True
    return False


def _warm_up_if_needed() -> None:
    """
    Pay the cold-start cost here, with an explanation, rather than inside the
    user's first question.

    The outcome is stashed in session_state rather than rendered directly: this
    runs immediately before `st.rerun()`, which discards anything written here.
    """
    if active_provider_name() != "ollama":
        return
    provider = get_provider()
    if not hasattr(provider, "warm_up"):
        return

    from agent.system_prompt import get_system_prompt
    from agent.tool_schemas import TOOLS

    # Always warm up rather than asking whether it's needed. `/api/ps` reports
    # Ollama's keep_alive bookkeeping, not real residency — it still claims a
    # model is loaded after the weights have been evicted — so skipping on its
    # word produced a confident "already loaded" that was simply false. Warming
    # an already-warm model costs ~0.5s (measured), which is not worth a lie.
    with st.spinner(
        f"Preparing `{provider.model}`. If the model needs loading this can take "
        "several minutes; if it is already in memory it will be quick. Questions "
        "after this are much faster."
    ):
        elapsed, error = provider.warm_up(TOOLS, get_system_prompt())

    if error:
        st.session_state._warm_up_note = (
            "warning",
            f"Warm-up did not complete ({error}). Your first question will be "
            "slow, but it will still work.",
        )
    elif elapsed < 5:
        # Fast means the prefix was genuinely cached — the elapsed time is the
        # evidence, unlike anything the server claims about residency.
        st.session_state._warm_up_note = (
            "info", f"`{provider.model}` was already warm ({elapsed:.1f}s)."
        )
    else:
        st.session_state._warm_up_note = (
            "success", f"Model loaded and ready in {elapsed:.0f}s."
        )


def _show_warm_up_note() -> None:
    """Render the warm-up outcome once, after the rerun that followed it."""
    note = st.session_state.pop("_warm_up_note", None)
    if not note:
        return
    level, message = note
    {"success": st.success, "warning": st.warning, "info": st.caption}[level](message)


def _render_ollama_setup() -> None:
    st.markdown(
        """
        #### Local model via Ollama

        Runs entirely on this machine — no API key, and no grid data leaves it.
        """
    )

    info = ollama_probe()

    if not info["reachable"]:
        st.error(
            f"Cannot reach Ollama at `{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}`."
        )
        st.markdown(
            "Install it from [ollama.com](https://ollama.com/download), then start it:\n\n"
            "```\nollama serve\n```"
        )
        if st.button("Check again", width="stretch"):
            st.rerun()
        return

    st.success(f"Ollama {info['version']} is running.")

    tool_models = [m for m in info["models"] if m["tools"] is not False]
    if not tool_models:
        if info["models"]:
            st.warning(
                "None of the installed models support tool calling. CONDUCTOR drives "
                "the grid entirely through tools, so it needs one that does."
            )
        else:
            st.warning("No models are installed yet.")
        st.markdown(
            "Pull a tool-capable model, for example:\n\n"
            "```\nollama pull gemma4:12b-mlx\n```\n\n"
            "On Apple Silicon the `-mlx` tags run noticeably faster."
        )
        if st.button("Check again", width="stretch"):
            st.rerun()
        return

    def _label(model: dict) -> str:
        caveat = "" if model["tools"] else "  ⚠︎ tool support unconfirmed"
        return f"{model['name']}  ({model['size_gb']} GB){caveat}"

    with st.form("ollama_setup"):
        choice = st.selectbox(
            "Model",
            options=[m["name"] for m in tool_models],
            format_func=lambda n: _label(next(m for m in tool_models if m["name"] == n)),
        )
        num_ctx = st.number_input(
            "Context window (num_ctx)",
            min_value=8192,
            max_value=1_048_576,
            value=int(os.environ.get("OLLAMA_NUM_CTX", OLLAMA_NUM_CTX)),
            step=8192,
            help=(
                "This agent's system prompt and tool schemas are ~25k tokens before "
                "any conversation. Ollama defaults to ~4096 and silently discards "
                "anything beyond the window, so this must be set generously. Larger "
                "windows use more memory."
            ),
        )
        submitted = st.form_submit_button("Save & Start", type="primary", width="stretch")

    if submitted:
        _write_env({
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": choice,
            "OLLAMA_NUM_CTX": str(int(num_ctx)),
        })
        st.session_state._launched = True
        st.success(f"Using {choice}. Starting the assistant…")
        st.rerun()


# One entry per backend, so adding a provider means adding its two screens and
# a row here rather than editing every if/else that names a backend.
_SETUP_RENDERERS = {
    "google": _render_google_setup,
    "openai": _render_openai_setup,
    "anthropic": _render_anthropic_setup,
    "ollama": _render_ollama_setup,
}
_SETTINGS_RENDERERS = {
    "google": _render_google_settings,
    "openai": _render_openai_settings,
    "anthropic": _render_anthropic_settings,
    "ollama": _render_ollama_settings,
}


# ---------------------------------------------------------------------------
# Model check — does the configured model still exist?
#
# Hosted providers retire models on their own schedule, and a `.env` written
# months ago pins whatever was current then. Asking the provider before the
# app starts turns "the first question fails" into a message on this screen.
# The check never changes the model; see `agent/model_check.py`.
# ---------------------------------------------------------------------------

_CHECKED_PROVIDERS = ("openai", "anthropic", "google")


def _model_check_key(provider: str) -> tuple:
    """What a check result depends on, so a changed setting is re-checked."""
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL") or OPENAI_MODEL
        key = os.environ.get("OPENAI_API_KEY", "")
        where = os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
    elif provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        where = os.environ.get("ANTHROPIC_BASE_URL", "")
    else:
        model = os.environ.get("GEMINI_MODEL") or GEMINI_MODEL
        key = os.environ.get("GEMINI_API_KEY", "")
        where = ""
    return (provider, model, key[-6:], where)


def _cached_model_check(provider: str):
    """One check per configuration per browser session — not one per rerun."""
    cache = st.session_state.setdefault("_model_checks", {})
    key = _model_check_key(provider)
    if key not in cache:
        with st.spinner("Checking the model with the provider…"):
            cache[key] = _model_check.check(provider)
    return cache[key]


def _model_problem_text(mc) -> str:
    if mc.status == _model_check.KEY_REJECTED:
        return f"**The API key was not accepted.** {mc.message}"
    text = (f"**`{mc.model}` is not available** — the provider may have retired "
            "it, or this key cannot use it. Starting now would fail on the first "
            "question.")
    if mc.alternatives:
        text += ("\n\nAvailable instead (same family first): "
                 + ", ".join(f"`{a}`" for a in mc.alternatives) + ".")
    return text + "\n\nPick another model under **Change settings**."


def _model_rejected(provider: str, model: str, **kwargs) -> bool:
    """Check a model before saving it. True — and an error shown — if it
    cannot be used. A check that cannot reach the provider does not block."""
    mc = _model_check.check(provider, model, **kwargs)
    if mc is not None and mc.blocks_start:
        st.error(_model_problem_text(mc))
        return True
    return False


def _fetch_scenarios() -> dict | None:
    """What is loaded and what can be (Resume, the demos), from the backend."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.get(f"{BASE_URL}/api/scenario")
            r.raise_for_status()
            return r.json()
    except Exception:  # noqa: BLE001
        return None


def _switch_scenario(name: str) -> None:
    """Load a scenario; the caches and the conversation start afresh (they
    described the previous one). Errors are kept for the picker to show."""
    try:
        with httpx.Client(timeout=max(HTTP_TIMEOUT, 120)) as client:
            r = client.post(f"{BASE_URL}/api/scenario", json={"name": name})
        if r.status_code >= 400:
            st.session_state._scenario_error = r.json().get("detail", r.text)
            return
    except Exception as exc:  # noqa: BLE001
        st.session_state._scenario_error = str(exc)
        return
    st.cache_data.clear()
    if st.session_state.get("_launched"):
        _reset_conversation_state()
        st.session_state._scenario_note = f"Loaded **{r.json().get('label')}** — a new conversation has started."


def _render_scenario_picker(*, key: str, immediate: bool) -> None:
    """Resume or a demo. On the launch screen a choice loads at once; in the
    sidebar it waits for Load (it starts a new conversation)."""
    data = _fetch_scenarios()
    error = st.session_state.pop("_scenario_error", None)
    if error:
        st.warning(error)
    if data is None:
        st.caption("The backend is not responding, so the scenario cannot be chosen yet.")
        return
    options = {o["name"]: o for o in data["options"]}
    names = list(options)
    widget = f"scenario_{key}"
    if st.session_state.get(widget) not in names:
        st.session_state[widget] = data["current"]

    def _on_change() -> None:
        chosen = st.session_state[widget]
        if not options[chosen]["available"]:
            st.session_state._scenario_error = options[chosen]["description"]
            st.session_state[widget] = data["current"]
        elif immediate and chosen != data["current"]:
            _switch_scenario(chosen)

    st.radio("Scenario", names, key=widget, format_func=lambda n: options[n]["label"], label_visibility="collapsed",
             captions=[options[n]["description"] for n in names], on_change=_on_change)
    if not immediate:
        chosen = st.session_state[widget]
        st.button("Load", key=f"{widget}_load", type="primary", width="stretch",
                  disabled=chosen == data["current"] or not options[chosen]["available"],
                  help="Loads it now and starts a new conversation.", on_click=_switch_scenario, args=(chosen,))
        st.caption(f"Loaded now: **{data['label']}**. Demos never replace your uploads.")


def _render_ready_to_start(provider: str) -> None:
    """
    The already-configured case: confirm what will be used and get out of the way.

    Starting is a single click — no re-entering settings that are already saved.
    """
    mc = _cached_model_check(provider) if provider in _CHECKED_PROVIDERS else None
    if mc is not None and mc.blocks_start:
        st.error(_model_problem_text(mc))
        if st.button("Check again", width="stretch"):
            st.session_state.pop("_model_checks", None)
            st.rerun()
        with st.expander("Change settings", expanded=True):
            if _SETTINGS_RENDERERS[provider](key_prefix="launch"):
                st.success("Saved.")
                st.rerun()
        return

    if provider == "google":
        key = os.environ.get("GEMINI_API_KEY", "")
        raw = os.environ.get("GEMINI_THINKING_LEVEL")
        level = (GEMINI_THINKING_LEVEL if raw is None else raw).strip().lower()
        reasoning = f"`{level}`" if level in GEMINI_THINKING_LEVELS else "model default"
        st.success(
            f"Ready — model `{os.environ.get('GEMINI_MODEL') or GEMINI_MODEL}`, "
            f"reasoning {reasoning}, key `…{key[-4:]}` loaded from `.env`."
        )
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL") or OPENAI_MODEL
        base = os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
        where = "" if base == OPENAI_BASE_URL else f" via `{base}`"
        effort = os.environ.get("OPENAI_REASONING_EFFORT") or OPENAI_REASONING_EFFORT
        reasoning = f", reasoning `{effort}`" if effort else ""
        # The endpoint is derived from the other settings under "auto", so
        # naming it here is the only way the user can see which one applies.
        #
        # Built fresh rather than read off get_provider(): that instance is
        # memoised, so it reports the settings in force when it was first
        # constructed, not the ones on screen. A bare provider costs nothing —
        # no client, no request.
        endpoint = ("/v1/responses" if _OpenAIProvider().uses_responses
                    else "/v1/chat/completions")
        st.success(
            f"Ready — model `{model}`{reasoning} via `{endpoint}`, "
            f"key `…{key[-4:]}` loaded from `.env`{where}."
        )
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL
        effort = os.environ.get("ANTHROPIC_EFFORT") or ANTHROPIC_EFFORT
        thinking = (os.environ.get("ANTHROPIC_THINKING", "1").strip().lower()
                    not in ("0", "false", "no"))
        # Effort is meaningless without saying whether thinking is on at all —
        # the same reason the OpenAI branch names its endpoint.
        depth = (f"effort `{effort}`" if effort else "default effort")
        if not thinking:
            depth += ", thinking off"
        st.success(
            f"Ready — model `{model}`, {depth}, "
            f"key `…{key[-4:]}` loaded from `.env`."
        )
    else:
        model = os.environ.get("OLLAMA_MODEL", "")
        info = ollama_probe()
        if not info["reachable"]:
            st.error("Ollama is not running. Start it with `ollama serve`, then re-check.")
            if st.button("Check again", width="stretch"):
                st.rerun()
            return
        if model not in {m["name"] for m in info["models"]}:
            st.error(f"Model `{model}` is no longer installed. Pull it or pick another below.")
            _render_ollama_setup()
            return
        st.success(
            f"Ready — `{model}` on Ollama {info['version']}, "
            f"context {int(os.environ.get('OLLAMA_NUM_CTX') or OLLAMA_NUM_CTX):,} tokens."
        )

    if mc is not None and mc.status == _model_check.UNVERIFIED:
        st.warning(
            f"Could not confirm `{mc.model}` with the provider ({mc.message}). "
            "Starting is still possible; a retired model would fail on the "
            "first question."
        )
    elif mc is not None and mc.newer:
        st.info(
            f"A newer **{_model_check.family(mc.model)}** model is available: "
            f"`{mc.newer}`. CONDUCTOR keeps using `{mc.model}` until you change "
            "it under **Change settings** — session logs record which model "
            "answered, so a switch is always your decision."
        )

    if st.button("Start CONDUCTOR", type="primary", width="stretch"):
        _write_env({"LLM_PROVIDER": provider})
        _warm_up_if_needed()
        st.session_state._launched = True
        st.rerun()

    with st.expander("Change settings"):
        saved = _SETTINGS_RENDERERS.get(provider, _render_ollama_settings)(
            key_prefix="launch"
        )
        if saved:
            st.success("Saved.")
            st.rerun()


# The launch screen shows on every start of the app — picking a backend is an
# explicit, per-session decision rather than a one-time setup step. The flag is
# session-scoped, so it survives Streamlit reruns but not an app restart.
if not st.session_state.get("_launched"):
    _render_brand_block(_BRAND_NAME, _BRAND_TAGLINE, image_width=120)
    st.caption(_BRAND_CAPTION)
    # Side by side: what to load, and how to run the assistant — both on one
    # screen, with Start at the end of the second.
    _col_load, _col_model = st.columns(2, gap="large", border=True)
    with _col_load:
        st.markdown("### 1 · Choose what to load")
        _render_scenario_picker(key="launch", immediate=True)
        st.divider()
        st.caption("Switch any time from the sidebar (**Settings → Scenario**). To work on your own grid, "
                   "start with any option and use **Upload Network** / **Upload Data** in the header — "
                   "they become what *Resume* loads. Demos never replace your uploads.")
    _col_model.markdown("### 2 · Choose how to run the assistant")

    _CHOICES = {
        "Google Gemini API": "google",
        "OpenAI API (or compatible)": "openai",
        "Anthropic API": "anthropic",
        "Local model (Ollama)": "ollama",
    }
    _labels = list(_CHOICES)
    _current = active_provider_name()
    _index = next((i for i, l in enumerate(_labels) if _CHOICES[l] == _current), 0)

    _picked = _CHOICES[
        _col_model.radio(
            "Backend",
            _labels,
            index=_index,
            captions=[
                "Hosted, needs a free API key, fastest to set up.",
                "Hosted, paid per token — also Kimi, DeepSeek, Groq and any "
                "other OpenAI-compatible endpoint.",
                "Hosted, paid per token, Claude models.",
                "Fully local and private, needs a tool-capable model.",
            ],
            label_visibility="collapsed",
        )
    ]

    with _col_model:
        st.divider()
        if _provider_configured(_picked):
            _render_ready_to_start(_picked)
        else:
            _SETUP_RENDERERS[_picked]()

    st.stop()  # Do not render the rest of the app until a backend is chosen

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Report how the model warm-up went. Written just before the rerun that brought
# us here, so it could not have been rendered on the launch screen itself.
_show_warm_up_note()


def _warn_if_grid_identity_unverified() -> None:
    """
    Say so when the agent doesn't actually know which network is loaded.

    The system prompt falls back to constants for a *different* grid when the
    backend is unreachable. The model is instructed not to describe the grid in
    that state, but the user deserves to see it directly rather than inferring
    it from hedged answers.
    """
    status = last_grid_constants_status()
    if status is None or status.live:
        return
    st.error(
        f"**Backend unreachable — the loaded network cannot be identified.**\n\n"
        f"Analyses and grid details shown may be wrong. The assistant has been "
        f"told not to describe the network until the backend responds again "
        f"(`{BASE_URL}`).\n\n"
        f"Reason: {status.error}"
    )

if "current_ts" not in st.session_state:
    ts_result = get_current_timestamp()
    st.session_state.current_ts = ts_result.get(
        "current_timestamp", ts_result.get("timestamp", "—")
    )

# Display messages: [{"role": "user"|"assistant", "text": str, "charts": list|None}]
# Separate from `history` (the provider-neutral message list — see agent/providers/base.py).
# Charts are stored here so they survive re-runs.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Identifies this conversation in AI Agent Suggestion records, so a suggestion
# and what the operator did with it can be read together from the log.
if "ui_conversation_id" not in st.session_state:
    st.session_state.ui_conversation_id = uuid.uuid4().hex[:12]

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
# (query, icon) — one icon per study, so the list can be scanned by topic.
EXAMPLE_QUERIES: list[tuple[str, str]] = [
    ("Is the grid secure at the current timestamp?", ":material/verified_user:"),
    ("Find the worst-case timestamp in this week's measurements.", ":material/trending_down:"),
    ("Will the grid stay secure across the forecast horizon?", ":material/schedule:"),
    ("Which single N-1 outage is the most critical right now?", ":material/account_tree:"),
    ("Run a probabilistic risk assessment under load and generation uncertainty.",
     ":material/casino:"),
    ("Find a robust corrective dispatch to clear any violations.", ":material/tune:"),
    ("What is the hosting capacity at the most heavily loaded bus?", ":material/solar_power:"),
    ("Evaluate the flexibility KPIs for the current operating point.", ":material/speed:"),
    ("Is the network secure at its nominal operating point, and how do the synthetic data "
     "compare with it?", ":material/push_pin:"),
]

# Questions that fit what is loaded, listed first under "For this scenario".
SCENARIO_QUERIES: dict[str, list[tuple[str, str]]] = {
    "resume": [
        ("Summarise the loaded network and data: what is there, and is anything missing or suspicious?",
         ":material/summarize:"),
        ("What does the data-quality report say about the uploaded data, and does any of it affect the "
         "studies?", ":material/fact_check:"),
    ],
    "ieee14": [
        ("How sensitive are the voltages to the load power factor? Compare 0.90, 0.95 and 1.0 at the "
         "current timestamp.", ":material/tune:"),
        ("Is the network secure at its nominal operating point, and how do the synthetic data compare "
         "with it?", ":material/push_pin:"),
    ],
    "ieee14_reactive": [
        ("What does the data-quality report say about the reactive measurements, and how are the gaps "
         "filled?", ":material/fact_check:"),
        ("Which loads have the lowest power factor right now, and does that matter for the voltages?",
         ":material/bolt:"),
        ("With the measured reactive demand, when is the worst moment of the week, and is the grid "
         "still secure then?", ":material/trending_down:"),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_timeline() -> dict:
    """Fetch the measurement timeline + forecast horizon for the scrubber."""
    import httpx
    from agent.config import BASE_URL, HTTP_TIMEOUT
    try:
        r = httpx.get(f"{BASE_URL}/api/time/timeline", timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"timestamps": [], "n": 0, "current_timestamp": None, "forecast": {}}


def _jump_to_timestamp(ts: str) -> str:
    """Jump the simulation clock directly to `ts` (any direction, instant)."""
    import httpx
    from agent.config import BASE_URL, HTTP_TIMEOUT
    try:
        r = httpx.post(f"{BASE_URL}/api/time/advance", json={"target_timestamp": ts}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json().get("new_timestamp", ts)
    except Exception:
        return st.session_state.current_ts


def _reset_if_network_changed() -> str | None:
    """
    Start a fresh conversation when the backend is serving a different network.

    Prior history holds the *old* grid's tool results and the assistant's own
    statements about them. Carrying that into a new network lets the model blend
    two grids — the system prompt would describe the new one while its memory
    describes the other. Correct constants do not help if the memory is wrong.

    Returns the new network's name when a reset happened, else None.
    """
    status = fetch_grid_constants()
    previous = st.session_state.get("_network_fingerprint")
    changed = is_network_change(previous, status)

    # Only record a fingerprint from a real fetch; the fallback constants
    # describe a different grid and would look like a swap on recovery.
    if status.live:
        st.session_state._network_fingerprint = network_fingerprint(status.values)

    if not changed:
        return None

    _reset_conversation_state()
    return status.values.get("name") or "the loaded network"


def _dedupe_tool_results(results):
    """Drop repeated identical tool results, keeping the first of each.

    Observed live: asked for a percentage, the agent called
    `get_current_conditions` a second time to obtain the operands, and the
    operator saw the same conditions table rendered twice. The two payloads
    were byte-identical, so the second told them nothing and invited reading
    it as a second measurement.

    Identity is (tool name, serialised result). Two calls that returned
    anything different — a different timestamp, a different element, a
    re-solve after a topology change — are separate analyses and both render.
    A result that will not serialise is never treated as a duplicate, since
    the safe failure here is showing a chart twice rather than hiding one.
    """
    seen: set[tuple[str, str]] = set()
    kept = []
    for name, result in results:
        try:
            key = (name, json.dumps(result, sort_keys=True, default=str))
        except (TypeError, ValueError):
            kept.append((name, result))
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append((name, result))
    return kept


def _reset_conversation_state() -> None:
    """Clear chat memory so the next prompt starts a fresh LLM conversation."""
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state._box_value = ""
    st.session_state._box_n = st.session_state.get("_box_n", 0) + 1
    st.session_state.pop("_box_pending", None)
    st.session_state.pop("_suggestion_used", None)
    st.session_state.pop("discussions", None)
    st.session_state.pop("_advisor_request", None)
    st.session_state.pop("system_ts", None)
    st.session_state.ui_conversation_id = uuid.uuid4().hex[:12]
    _tools_module._last_tool_results.clear()


def _path_within(path: pathlib.Path, base: pathlib.Path) -> bool:
    """Return True when path is inside base (or equal)."""
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _read_json_file(path: pathlib.Path) -> dict | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _collect_uploaded_cleanup_groups() -> dict[str, list[pathlib.Path]]:
    """Collect grouped uploaded artifacts that can be safely deleted by user choice."""
    groups: dict[str, list[pathlib.Path]] = {}
    seen: set[pathlib.Path] = set()

    def _add(group_name: str, candidate: pathlib.Path | None) -> None:
        if candidate is None:
            return
        p = candidate.resolve()
        if not p.is_file():
            return
        if not (_path_within(p, _DATA_FILES_DIR) or _path_within(p, _SYSTEMS_DIR)):
            return
        if p in seen:
            return
        groups.setdefault(group_name, []).append(p)
        seen.add(p)

    upload_sentinel = _SYSTEMS_DIR / "last_uploaded.json"
    upload_meta = _read_json_file(upload_sentinel)
    if upload_sentinel.is_file():
        _add("Uploaded network restore metadata", upload_sentinel)
    if isinstance(upload_meta, dict):
        net_path = pathlib.Path(str(upload_meta.get("network_path", "")))
        if net_path:
            _add("Current uploaded network file", net_path)

    ts_sentinel = _DATA_FILES_DIR / "last_uploaded_timeseries.json"
    ts_meta = _read_json_file(ts_sentinel)
    if ts_sentinel.is_file() and isinstance(ts_meta, dict) and ts_meta.get("source") == "uploaded":
        _add("Uploaded measurements CSV + restore metadata", ts_sentinel)
        _add(
            "Uploaded measurements CSV + restore metadata",
            pathlib.Path(str(ts_meta.get("csv_path", ""))),
        )

    fc_sentinel = _DATA_FILES_DIR / "last_uploaded_forecast.json"
    fc_meta = _read_json_file(fc_sentinel)
    if fc_sentinel.is_file() and isinstance(fc_meta, dict) and fc_meta.get("source") == "uploaded":
        _add("Uploaded forecast CSV + restore metadata", fc_sentinel)
        _add(
            "Uploaded forecast CSV + restore metadata",
            pathlib.Path(str(fc_meta.get("csv_path", ""))),
        )

    for p in sorted(_DATA_FILES_DIR.glob("uploaded_admittance*.csv")):
        _add("Advanced admittance uploads", p)
    for p in sorted(_DATA_FILES_DIR.glob("*admittance*upload*.csv")):
        _add("Advanced admittance uploads", p)

    for p in sorted(_DATA_FILES_DIR.glob("uploaded_*.csv")):
        _add("Other uploaded-looking CSV files", p)
    for p in sorted(_SYSTEMS_DIR.glob("uploaded_*")):
        if p.suffix.lower() in {".m", ".json", ".xlsx", ".uct"}:
            _add("Other uploaded-looking network files", p)

    # Show non-default network files as optional removable artifacts.
    protected_system_files = {
        ".gitkeep",
        "pglib_opf_case14_ieee.m",
        "pglib_opf_case30_ieee.m",
        "pandapower_network_flex.xlsx",
    }
    for p in sorted(_SYSTEMS_DIR.iterdir()):
        if not p.is_file() or p.name in protected_system_files:
            continue
        if p.suffix.lower() in {".m", ".json", ".xlsx", ".uct"}:
            _add("Other network files in systems (non-default)", p)

    return groups


def _delete_uploaded_artifacts(paths: list[pathlib.Path]) -> tuple[int, list[str]]:
    """Delete selected files inside data_files/systems and return (count, errors)."""
    deleted = 0
    errors: list[str] = []
    deduped = []
    seen: set[pathlib.Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            deduped.append(rp)
            seen.add(rp)

    for p in deduped:
        if not (_path_within(p, _DATA_FILES_DIR) or _path_within(p, _SYSTEMS_DIR)):
            errors.append(f"Skipped outside managed folders: {p}")
            continue
        if not p.exists() or not p.is_file():
            continue
        try:
            p.unlink()
            deleted += 1
        except Exception as exc:
            errors.append(f"Could not remove {p.name}: {exc}")
    return deleted, errors


def _reset_active_backend_profile(profile_name: str = "pglib_case14") -> tuple[bool, str]:
    """Request backend to reload active in-memory state from a YAML profile."""
    import httpx

    try:
        r = httpx.post(
            f"{BASE_URL}/api/network/reset_active",
            json={"profile_name": profile_name},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json() if r.content else {}
        if str(payload.get("status", "")).lower() == "success":
            return True, f"Active backend reset to profile '{profile_name}'."
        return False, f"Reset endpoint returned unexpected payload: {payload}"
    except Exception as exc:
        return False, str(exc)


_CHARTS_PER_ROW = 2      # wider charts: three to a row cut every title off
_VISIBLE_CHART_ROWS = 2  # beyond this, charts go under "More charts"


def _polish(fig):
    """Presentation applied to every chart at draw time.

    Two things that cost readability across many renderers: legends drawn
    over the plot area, and a marker on every point of a week-long series
    (672 markers make a solid band). Applied here rather than in each renderer
    so they cannot drift apart; a renderer that positions its own legend keeps it.
    """
    try:
        legend = fig.layout.legend
        axis2 = fig.layout.to_plotly_json().get("yaxis2") or {}
        # Only charts with a second y-axis: there the default legend, outside
        # on the right, sits on top of that axis. Elsewhere it is fine.
        if axis2.get("overlaying") and legend.orientation is None \
                and legend.x is None and legend.y is None:
            fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.42,
                                      "xanchor": "left", "x": 0, "font": {"size": 11}},
                              margin={"b": 130})
    except Exception:  # noqa: BLE001 — polish must never cost a chart
        pass
    try:
        # One chart style for every figure, whichever renderer built it: many
        # never set a template and rendered on Plotly's grey default. Explicit
        # layout settings a renderer made still win over the template.
        fig.update_layout(template="conductor")
        # Streamlit's frontend paints its own secondary background into the
        # plot area even with theme=None; only an explicit value survives.
        if fig.layout.plot_bgcolor is None:
            fig.update_layout(plot_bgcolor="white")
        if fig.layout.paper_bgcolor is None:
            fig.update_layout(paper_bgcolor="white")
    except Exception:  # noqa: BLE001
        pass
    try:
        for trace in fig.data:
            mode = getattr(trace, "mode", None)
            x = getattr(trace, "x", None)
            if mode == "lines+markers" and x is not None and len(x) > 150:
                trace.mode = "lines"
    except Exception:  # noqa: BLE001 — polish must never cost a chart
        pass
    return fig


def _status_of(fig) -> dict | None:
    """The status a renderer returned in place of a chart, if any."""
    meta = getattr(getattr(fig, "layout", None), "meta", None)
    return meta if isinstance(meta, dict) and "conductor_status" in meta else None


def _non_converged_note(result: dict) -> str | None:
    """A warning when a scan skipped ticks whose power flow did not converge.

    Scans used to drop those ticks without a word; they now report them, and
    the operator should see it beside the charts — a tick with no converged
    power flow is often the most stressed moment in the window.
    """
    stamps = list(result.get("non_converged_timestamps") or [])
    n = result.get("n_non_converged") or 0
    for scenario in result.get("scenarios") or []:
        if isinstance(scenario, dict):
            n += scenario.get("n_non_converged") or 0
            stamps += scenario.get("non_converged_timestamps") or []
    if not n:
        return None
    shown = ", ".join(sorted(set(str(t)[:16] for t in stamps))[:3])
    more = " …" if len(set(stamps)) > 3 else ""
    return (f"⚠️ {n} tick(s) had no converged power flow and are not in these results — "
            f"often the most stressed moments{': ' + shown + more if shown else ''}.")


def _chart_caption(tool_name: str, result: dict) -> str:
    """Which analysis these charts come from, and which moment they describe."""
    parts = [f"**{_TOOL_LABELS.get(tool_name, tool_name.replace('_', ' '))}**"]
    when = result.get("timestamp") or result.get("worst_timestamp")
    start, end = result.get("window_start"), result.get("window_end")
    if when:
        parts.append(str(when)[:16])
    elif start and end:
        parts.append(f"{str(start)[:16]} → {str(end)[:16]}")
    if str(result.get("data_source", "")).startswith("forecast"):
        parts.append("forecast")
    return " · ".join(parts)


def _table_of(fig) -> tuple[str, str, "pd.DataFrame"] | None:
    """A figure that is only a table, as (title, subtitle, rows); else None.

    A Plotly table is a picture: its cells cannot be selected or copied. Such
    figures are shown with `st.dataframe` instead — select and copy (Ctrl/Cmd+C),
    sort, search, download as CSV — while renderers keep returning figures.
    """
    try:
        if len(fig.data) != 1 or fig.data[0].type != "table":
            return None
        table = fig.data[0]
        headers = [re.sub(r"<[^>]+>", "", str(h)) for h in table.header.values]
        frame = pd.DataFrame({h: list(col) for h, col in zip(headers, table.cells.values)})
        title, _, sub = (fig.layout.title.text or "").partition("<br>")
        return re.sub(r"<[^>]+>", "", title), re.sub(r"<[^>]+>", "", sub), frame
    except Exception:  # noqa: BLE001 — fall back to drawing the figure
        return None


def render_charts(tool_results: list) -> None:
    """
    Render Plotly charts for a list of (tool_name, result) tuples.
    Works both for live results and stored results replayed from session state.

    A renderer may return one figure, a list (one row), or a list of lists
    (several rows) — or nothing at all, which several do on purpose (no
    violations to attribute, nothing infeasible to draw). An empty result used
    to reach `st.columns(0)`, which raises; and because the message is stored
    before its charts are drawn, every later rerun raised too and the whole
    conversation view was gone until the chat was reset. One failing chart now
    costs that chart, never the conversation.

    Presentation, for an operator reading the answer rather than the charts:
    each tool's charts carry a caption naming the analysis and its moment;
    status figures ("no violations", "infeasible") become one-line messages;
    charts wrap at two per row, and only the first rows are shown — the rest
    wait under "More charts" instead of pushing the conversation off screen.
    """
    for chart_idx, (tool_name, result) in enumerate(tool_results):
        renderer = RENDERER_MAP.get(tool_name)
        if renderer is None or not isinstance(result, dict) or "error" in result:
            continue

        try:
            figs = renderer(result)
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "Renderer for %s failed.", tool_name, exc_info=True)
            label = _TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))
            st.caption(f"⚠️ The chart for {label} could not be drawn; the answer above is unaffected.")
            continue

        if figs is None or (isinstance(figs, list) and not figs):
            continue
        if not isinstance(figs, list):
            figs = [figs]
        flat = [fig for row in figs for fig in (row if isinstance(row, list) else [row])
                if fig is not None]
        if not flat:
            continue

        st.caption(_chart_caption(tool_name, result))
        note = _non_converged_note(result)
        if note:
            st.warning(note)
        statuses = [s for s in (_status_of(fig) for fig in flat) if s]
        tables = [t for t in (_table_of(fig) for fig in flat) if t]
        charts = [fig for fig in flat if not _status_of(fig) and not _table_of(fig)]
        for status in {s["conductor_status"]: s for s in statuses}.values():
            show = {"ok": st.success, "error": st.error}.get(status["tone"], st.info)
            show(status["conductor_status"])
        for title, subtitle, frame in tables:
            if title:
                st.markdown(f"**{title}**")
            if subtitle:
                st.caption(subtitle)
            st.dataframe(frame, hide_index=True, width="stretch",
                         height=min(38 + 35 * len(frame), 458))

        rows = [charts[i:i + _CHARTS_PER_ROW] for i in range(0, len(charts), _CHARTS_PER_ROW)]
        visible, hidden = rows[:_VISIBLE_CHART_ROWS], rows[_VISIBLE_CHART_ROWS:]

        def _draw(row_list, offset):
            for row_idx, row in enumerate(row_list, start=offset):
                cols = st.columns(_CHARTS_PER_ROW if len(rows) > 1 else len(row))
                for fig_idx, (col, fig) in enumerate(zip(cols, row)):
                    with col:
                        # theme=None: Streamlit's own chart theme would
                        # override the "conductor" template (fonts, colours).
                        st.plotly_chart(_polish(fig), width="stretch", theme=None,
                                        key=f"chart_{id(tool_results)}_{chart_idx}_{row_idx}_{fig_idx}")

        _draw(visible, 0)
        if hidden:
            n_hidden = sum(len(r) for r in hidden)
            with st.expander(f"More charts ({n_hidden})"):
                _draw(hidden, len(visible))


# ---------------------------------------------------------------------------
# Chart injection helper
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Operating point — which data, at what time (logic in agent/timeline.py)
#
# The dataset is chosen first, then the time within that dataset's own
# timestamps. Measured moves the simulation clock; Forecast is query-only;
# Compare (where both share a timestamp) asks for forecast against actual.
# ---------------------------------------------------------------------------
_DATASET_COLOURS = {_tl.MEASURED: "#1D6FE8", _tl.FORECAST: "#8E5CD9"}
_OP_CAPTIONS = {
    _tl.MEASURED: "Moves the simulation clock.",
    _tl.FORECAST: "Query-only — the simulation clock does not move.",
    _tl.COMPARE: "Forecast against measured — moves the simulation clock.",
}


def _op_step(key: str, ticks: tuple, n: int) -> None:
    st.session_state[key] = _tl.step(ticks, st.session_state.get(key), n)


def _coverage_strip(timeline: _tl.Timeline, cursor: str | None, mode: str) -> str:
    """Bars for where each dataset has data, the cursor across them, a legend."""
    cov = _tl.coverage(timeline, cursor)
    bars, legend = [], []
    for row in cov["rows"]:
        colour = _DATASET_COLOURS[row["mode"]]
        active = mode in (row["mode"], _tl.COMPARE)
        width = max(row["end"] - row["start"], 0.008) * 100
        span = f"{_tl.LABELS[row['mode']]}: {row['first'][:16]} → {row['last'][:16]}"
        bars.append(
            f"<div class='op-track' title='{html.escape(span)}'><div class='op-fill' "
            f"style='left:{row['start'] * 100:.2f}%;width:{width:.2f}%;background:{colour};"
            f"opacity:{1 if active else 0.35}'></div></div>"
        )
        meta = " · ".join(x for x in (row["resolution"], row["source"]) if x)
        legend.append(
            f"<div class='op-legend{'' if active else ' op-dim'}'>"
            f"<span><span style='color:{colour}'>●</span> <b>{_tl.LABELS[row['mode']]}</b></span>"
            f"<span>{html.escape(meta)}</span></div>"
        )
    if not timeline.forecast:
        legend.append("<div class='op-legend op-dim'><span><span style='color:#8E5CD9'>●</span> "
                      "<b>Forecast</b></span><span>none loaded</span></div>")
    cursor_html = ("" if cov["cursor"] is None else
                   f"<div class='op-cursor' style='left:{cov['cursor'] * 100:.2f}%'></div>")
    ends = [r["first"] for r in cov["rows"]] + [r["last"] for r in cov["rows"]]
    scale = (f"<div class='op-scale'><span>{min(ends)[5:10]}</span><span>{max(ends)[5:10]}</span></div>"
             if ends else "")
    return f"<div class='op-strip'>{''.join(bars)}{cursor_html}</div>{scale}{''.join(legend)}"


def _nominal_switch(timeline: _tl.Timeline) -> bool:
    """The nominal point: beside the datasets, not on the timeline (it has no
    time). Off by itself when the loaded network has no usable one."""
    ss = st.session_state
    if not timeline.nominal_available:
        ss.op_nominal = False
    return st.toggle(
        "Nominal point", key="op_nominal", disabled=not timeline.nominal_available,
        help=("The network's own operating point as loaded — the case file's loads and generator "
              "set-points, which the synthetic data are built from. No time: studies run on those "
              "values, and the simulation clock stays where it is."
              if timeline.nominal_available else
              f"Not available for this network. {timeline.nominal_reason}"))


def _render_operating_point() -> None:
    ss = st.session_state
    timeline = _tl.Timeline.from_payload(_fetch_timeline())
    modes = timeline.modes()
    if not modes:
        st.warning("Timeline unavailable — the backend is not responding.")
        return

    if ss.get("op_nominal") and timeline.nominal_available:
        st.markdown("<div class='op-when'><span class='op-time'>Nominal</span></div>",
                    unsafe_allow_html=True)
        st.caption("The network's own values as loaded — no time. Studies use the case's own "
                   "operating point"
                   + (f"; the simulation clock stays at {timeline.clock[5:16]}." if timeline.clock else "."))
        _nominal_switch(timeline)
        ss.op_view = {"mode": _tl.NOMINAL, "ts": None}
        return

    # Which data.
    if ss.get("op_mode") not in modes:
        ss.op_mode = modes[0]
    if len(modes) > 1:
        st.segmented_control("Dataset", modes, format_func=_tl.LABELS.get, key="op_mode",
                             required=True, label_visibility="collapsed", width="stretch")
    mode = ss.op_mode
    ticks = timeline.ticks(mode)
    follows_clock = mode in (_tl.MEASURED, _tl.COMPARE) and bool(timeline.measured)

    # At what time: keep the moment across a dataset switch (nearest tick), and
    # follow the clock when a tool moved it while the view is on measured data.
    key = f"op_time_{mode}"
    target = ss.get(key)
    if ss.get("_op_mode_last") != mode or target not in ticks:
        target = _tl.nearest(ticks, ss.get("op_time") or timeline.clock)
    if (follows_clock and timeline.clock != ss.get("_op_clock_last")
            and timeline.clock in ticks):
        target = timeline.clock
    ss[key] = target

    when = _tl._parse(target)
    st.markdown(
        f"<div class='op-when'><span class='op-date'>{when:%a} {when.day} {when:%b %Y}</span>"
        f"<span class='op-time'>{when:%H:%M}</span></div>",
        unsafe_allow_html=True,
    )
    res = _tl.resolution(ticks) or "tick"
    c_prev, c_slide, c_next = st.columns([1, 5, 1], vertical_alignment="center", gap="xsmall")
    c_prev.button("", icon=":material/chevron_left:", type="tertiary", key="op_prev",
                  help=f"Back {res}", on_click=_op_step, args=(key, ticks, -1))
    value = c_slide.select_slider("Time", options=ticks, key=key,
                                  format_func=lambda t: t[5:16], label_visibility="collapsed")
    c_next.button("", icon=":material/chevron_right:", type="tertiary", key="op_next",
                  help=f"Forward {res}", on_click=_op_step, args=(key, ticks, 1))

    st.markdown(_coverage_strip(timeline, value, mode), unsafe_allow_html=True)
    caption = _OP_CAPTIONS[mode]
    if not timeline.measured:
        caption = "No measurements loaded — the simulation clock is unavailable."
    elif mode == _tl.FORECAST and timeline.clock:
        caption = f"Query-only — the simulation clock stays at {timeline.clock[5:16]}."
    st.caption(caption)

    # Measured data moves the backend clock; a forecast cannot hold it.
    if follows_clock and value in timeline.measured and value != timeline.clock:
        _jump_to_timestamp(value)
        ss.current_ts = value
        ss._op_clock_last = value  # our own move: not a tool's, so do not snap back
    else:
        ss._op_clock_last = timeline.clock
    ss._op_mode_last = mode
    ss.op_time = value
    ss.op_view = {"mode": mode, "ts": value}
    _nominal_switch(timeline)


# ---------------------------------------------------------------------------
# Network proposals (sandbox)
#
# The agent builds proposals; the operator always sees what it has built and
# can take any edit back. The base network cannot be touched, so no edit needs
# a confirmation — but which grid a picture shows is always stated.
# ---------------------------------------------------------------------------
def _list_proposals() -> list[dict] | None:
    """Every proposal, or None when the backend cannot be read."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.get(f"{BASE_URL}/api/proposals")
            r.raise_for_status()
            return r.json().get("proposals") or []
    except Exception:  # noqa: BLE001 — the card says so
        return None


def _proposal_action(path: str) -> None:
    """DELETE a proposal or one of its edits; errors are shown, not raised."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.delete(f"{BASE_URL}{path}")
        if r.status_code >= 400:
            st.session_state._proposal_error = r.json().get("detail", r.text)
    except Exception as exc:  # noqa: BLE001
        st.session_state._proposal_error = str(exc)


def _render_proposals() -> None:
    proposals = _list_proposals()
    if proposals is None:
        st.caption("Proposals could not be read from the backend.")
        return
    error = st.session_state.pop("_proposal_error", None)
    if error:
        st.warning(error)
    if not proposals:
        st.caption("None yet. Ask, for example: *What if we connect a 50 MW data centre at "
                   "Bus 9?* The loaded network is never modified.")
        return
    for prop in proposals:
        pid = prop["id"]
        summary = prop.get("summary") or {}
        with st.expander(f"{pid} · {summary.get('n_edits', len(prop.get('edits', [])))} edit(s)",
                         icon=":material/edit_note:", key=f"prop_{pid}"):
            for n, text in enumerate(summary.get("edits") or [], start=1):
                c_text, c_del = st.columns([0.86, 0.14], vertical_alignment="center")
                c_text.caption(f"{n}. {text}")
                c_del.button("", icon=":material/close:", type="tertiary", key=f"prop_{pid}_rm{n}",
                             help="Remove this edit", on_click=_proposal_action,
                             args=(f"/api/proposals/{pid}/edits/{n}",))
            st.button("Discard proposal", icon=":material/delete:", width="stretch",
                      key=f"prop_{pid}_discard", on_click=_proposal_action,
                      args=(f"/api/proposals/{pid}",))


def _fetch_assumptions() -> dict | None:
    """The session's study assumptions from the backend, read on every run so
    a change the agent made shows here at once. None when unreachable."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.get(f"{BASE_URL}/api/assumptions")
            r.raise_for_status()
            return r.json()
    except Exception:  # noqa: BLE001 — the setting says so
        return None


# Read once per script run: every answer's staleness check compares with it.
_RUN_MEMO: dict = {}


def _settings_now() -> dict | None:
    if "settings" not in _RUN_MEMO:
        _RUN_MEMO["settings"] = _staleness.current(_fetch_assumptions())
    return _RUN_MEMO["settings"]


def _rerun_answer(question: str, why: list[str]) -> None:
    """Ask the question again, as a new turn under the current settings."""
    st.session_state._pending_turn = _staleness.rerun_prompt(question, why)
    st.session_state.pop("_attempts_turn", None)


def _render_staleness(msg: dict, idx: int) -> None:
    """A line under an answer whose studies ran under other settings, and a
    way to run it again."""
    why = _staleness.reasons(msg.get("context"), _settings_now())
    if not why:
        return
    question = next((m["text"] for m in reversed(st.session_state.messages[:idx]) if m["role"] == "user"), None)
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"stale_{idx}"):
        st.markdown(f"<span class='conductor-stale'>⏱ Computed under other settings: "
                    f"{html.escape('; '.join(why))}</span>", unsafe_allow_html=True, width="content")
        if question:
            st.button("↻ Rerun with the current settings", key=f"rerun_{idx}", type="tertiary",
                      disabled=_ui_busy(), on_click=_rerun_answer, args=(question, why),
                      help="Asks CONDUCTOR the same question again, as a new turn, so its studies run "
                           "under today's settings. The earlier answer stays for comparison.")


def _report_charts_html(tool_results: list) -> str:
    """One answer's charts for the report, as `render_charts` draws them:
    caption, status lines, tables, then the figures (interactive)."""
    out = []
    for tool_name, result in tool_results or []:
        renderer = RENDERER_MAP.get(tool_name)
        if renderer is None or not isinstance(result, dict) or "error" in result:
            continue
        try:
            figs = renderer(result)
        except Exception:  # noqa: BLE001 — one chart, never the report
            continue
        figs = figs if isinstance(figs, list) else [figs] if figs is not None else []
        flat = [f for row in figs for f in (row if isinstance(row, list) else [row]) if f is not None]
        if not flat:
            continue
        out.append(f"<div class='chart-caption'>{_report.markdown(_chart_caption(tool_name, result))}</div>")
        note = _non_converged_note(result)
        if note:
            out.append(f"<div class='status error'>{html.escape(note)}</div>")
        charts = []
        for fig in flat:
            status, table = _status_of(fig), _table_of(fig)
            if status:
                tone = {"ok": "", "error": " error"}.get(status.get("tone"), " info")
                out.append(f"<div class='status{tone}'>{html.escape(status['conductor_status'])}</div>")
            elif table:
                title, subtitle, frame = table
                out.append((f"<p><strong>{html.escape(title)}</strong></p>" if title else "")
                           + (f"<p class='limits'>{html.escape(subtitle)}</p>" if subtitle else "")
                           + frame.to_html(index=False, border=0))
            else:
                try:
                    fig = _polish(fig)
                    fig.update_layout(width=None, autosize=True)       # fill the report's column
                    charts.append(fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%",
                                              config={"displaylogo": False, "responsive": True}))
                except Exception:  # noqa: BLE001
                    continue
        if charts:
            out.append(f"<div class='charts'>{''.join(charts)}</div>")
    return "".join(out)


def _data_line(grid: dict) -> str | None:
    """The loaded series in one line, for the report."""
    parts = []
    for kind in ("measurements", "forecasts"):
        info = grid.get(kind) if isinstance(grid.get(kind), dict) else {}
        if info.get("n_timestamps"):
            parts.append(f"{kind} ({info.get('source')}): {info['n_timestamps']} ticks, "
                         f"{str(info.get('first_timestamp'))[:16]} → {str(info.get('last_timestamp'))[:16]}")
    return "; ".join(parts) or None


def _session_report(messages: list, model_label: str) -> bytes:
    """The conversation and what it ran on, as one standalone HTML file. Runs
    when the download is clicked (no Streamlit calls in here)."""
    from plotly.offline import get_plotlyjs

    assumptions = _fetch_assumptions()
    now = _staleness.current(assumptions)
    scenario = (_fetch_scenarios() or {}).get("label")
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            grid = client.get(f"{BASE_URL}/api/grid_constants").json()
            quality = (client.get(f"{BASE_URL}/api/data/quality").json().get("measurements") or {}).get("summary")
    except Exception:  # noqa: BLE001 — the report says less, it still downloads
        grid, quality = {}, None
    meta = {
        "title": f"CONDUCTOR session — {grid.get('name') or 'grid study'}",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scenario": scenario, "network": grid.get("name"), "model": model_label,
        "data": _data_line(grid),
        "assumptions": assumptions, "quality": quality, "proposals": _list_proposals(),
    }
    page = _report.build(messages, meta, charts_html=_report_charts_html, plotly_js=get_plotlyjs(),
                         stale=lambda m: _staleness.reasons(m.get("context"), now))
    return page.encode("utf-8")


def _set_load_pf(value: float | None) -> None:
    """Apply (or with None, reset) the load power factor; errors are shown."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.post(f"{BASE_URL}/api/assumptions", json={"load_power_factor": value})
        if r.status_code >= 400:
            st.session_state._pf_error = r.json().get("detail", r.text)
    except Exception as exc:  # noqa: BLE001
        st.session_state._pf_error = str(exc)


_LOAD_PF_EXPLAINED = """
The measured (and forecast) data give each substation's **active power only** — MW of
consumption and production. A load's **reactive power** is therefore an assumption:
**Q = P · tan(acos pf)**, lagging (the load draws reactive power).

**What it changes:** every study that uses a measured moment — the security assessment,
N-1, probabilistic risk, hosting capacity, the OPF and flexibility studies, the scans over
time. A lower power factor means more reactive power drawn through the network: lower
voltages, higher currents (line and transformer loading), more losses, more reactive power
imported from the external grid, and a harder starting point for the OPF.

**What it does not change:** the data themselves (nothing is regenerated), and the
**nominal point**, which keeps the reactive power written in the network file.

CONDUCTOR can also run **one study** at another power factor without changing this
setting, and every result states the value it used.
"""


def _render_load_pf(assumptions: dict) -> None:
    """The load power factor every study uses."""
    pf = assumptions.get("load_power_factor")
    if pf is None:
        st.caption("Load power factor unavailable — the backend is not responding.")
        return
    changed = pf.get("source") != "default"
    st.markdown(f"##### Loads — reactive power · pf {pf['value']:.2f}" + (" (changed)" if changed else ""))
    c_set, c_why = st.columns([1, 2], gap="large")
    with c_set, st.container(border=True, key="sys_load_pf"):
        error = st.session_state.pop("_pf_error", None)
        if error:
            st.warning(error)
        # Keyed by the current value: a change made by CONDUCTOR resets the box.
        key = f"pf_input_{pf['value']:.4f}"
        st.number_input("Load power factor (lagging)", min_value=0.5, max_value=1.0,
                        value=float(pf["value"]), step=0.01, format="%.2f", key=key)
        c_apply, c_reset = st.columns(2)
        c_apply.button("Apply", key="pf_apply", width="stretch", type="primary",
                       disabled=st.session_state.get(key) == pf["value"],
                       on_click=lambda: _set_load_pf(float(st.session_state[key])))
        c_reset.button("Reset", key="pf_reset", width="stretch", help=f"Back to {pf['default']:.2f}",
                       disabled=not changed, on_click=lambda: _set_load_pf(None))
        st.caption(f"Default {pf['default']:.2f}. " + ("Set for this session and kept for this network."
                                                         if changed else "In use since the network was loaded."))
        measured = (assumptions.get("reactive_measured") or {}).get("measurements") or {}
        if measured.get("consumption"):
            st.info("The measured data carry reactive demand: it is used wherever it is given, and this "
                    "power factor only fills the gaps (see *Data quality* in the Data tab).",
                    icon=":material/sensors:")
    with c_why:
        st.markdown(_LOAD_PF_EXPLAINED)


def _env_reflection() -> str:
    """CONDUCTOR_REFLECTION, as the loop reads it: off, warn or inform."""
    return _loop_module.normalise_mode(_loop_module.REFLECTION_MODE)


def _reflection_mode() -> str:
    """The answer-check mode for this turn: the operator's switch, or the
    research setting from the environment when the switch is off."""
    if st.session_state.get("answer_review"):
        return "inform"
    return "warn" if _env_reflection() == "warn" else "off"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _update_status() -> dict | None:
    """Whether GitHub has a newer version: the launcher's answer when it gave
    one, else asked once here (read-only, never blocks for long, None when
    it cannot tell)."""
    checked, status = _update_check.from_env()
    if checked:
        return status
    return _update_check.check(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


with st.sidebar:
    _update_notice = _update_check.message(_update_status())
    if _update_notice:
        st.info(_update_notice, icon=":material/system_update:")
    # The brand is already in the page header and the hero; a third copy here
    # pushed the controls an operator uses — the clock first — below the fold.
    # The most used action first.
    if st.button("New conversation", icon=":material/add_comment:", width="stretch",
                 type="secondary", key="new_conversation"):
        _reset_conversation_state()
        st.rerun()
    st.download_button(
        "Export session report", icon=":material/download:", width="stretch", key="export_report",
        data=functools.partial(_session_report, list(st.session_state.messages), _active_model_label()),
        file_name=f"conductor_session_{datetime.now():%Y%m%d_%H%M}.html", mime="text/html",
        on_click="ignore", disabled=not st.session_state.messages,
        help="One HTML file that opens anywhere, offline: the scenario, network, assumptions and proposals, "
             "then each question and answer with its charts and evidence. For co-authors and appendices.")

    st.markdown("<div class='conductor-sidebar-label'>Operating point</div>", unsafe_allow_html=True)
    with st.container(border=True, key="op_card"):
        _render_operating_point()

    st.markdown("<div class='conductor-sidebar-label'>Proposals</div>", unsafe_allow_html=True)
    _render_proposals()

    st.divider()
    st.markdown("<div class='conductor-sidebar-label'>Settings</div>", unsafe_allow_html=True)
    _scn = _fetch_scenarios()
    _scenario_note = st.session_state.pop("_scenario_note", None)
    if _scenario_note:
        st.success(_scenario_note)
    with st.expander(f"Scenario · {(_scn or {}).get('label', 'unknown')}", icon=":material/dataset:",
                     key="sb_scenario"):
        _render_scenario_picker(key="sidebar", immediate=False)

    # Model settings stay reachable mid-session — switching backend or adjusting
    # a local model's context should not require a restart or a .env edit.
    _active = active_provider_name()
    # The header names the model in use, so it is visible without opening.
    with st.expander(f"Model · {_active_model_label()}", icon=":material/smart_toy:",
                     key="sb_model"):
        st.caption(_PROVIDER_LABELS.get(_active, _active))
        _backends = list(available_providers())
        _switch_to = st.radio(
            "Backend",
            _backends,
            index=_backends.index(_active) if _active in _backends else 0,
            format_func=lambda p: _PROVIDER_LABELS.get(p, p),
            horizontal=True,
            key="sidebar_provider",
        )
        st.divider()
        _saved = _SETTINGS_RENDERERS.get(_switch_to, _render_ollama_settings)(
            key_prefix="sidebar"
        )
        if _saved:
            st.success("Saved — applies to your next message.")
            st.rerun()

    # Answer checks. Every answer is checked by deterministic rules — each
    # figure traced to the tool result that produced it, described as its own
    # record describes it, and the question's quantity present — and the audit
    # panel under the answer always shows what they found. The one choice an
    # operator makes is whether the agent sees the findings before answering
    # and may revise once. ("Recorded silently", which measures what would
    # have fired without changing any answer, is a research setting:
    # CONDUCTOR_REFLECTION=warn.) Off by default — the configuration every
    # reported result was produced under.
    _research_mode = _env_reflection() == "warn"
    _review_on = bool(st.session_state.get("answer_review", _env_reflection() == "inform"))
    with st.expander(f"Answer checks · {'Agent reviews' if _review_on else 'Report only'}",
                     icon=":material/fact_check:", key="sb_checks"):
        st.toggle(
            "Let the agent review its answer",
            value=_review_on, key="answer_review",
            help=(
                "Every answer is checked: each figure is traced back to the tool result "
                "that produced it, compared against the fields beside it (so an "
                "undeliverable movement is not reported as an action), and the answer is "
                "checked for the kind of quantity the question asked for. What they find "
                "is always shown under the answer.\n\n"
                "**On:** the agent sees the findings before the answer is delivered and may "
                "keep it, revise it, or call another tool. It costs one more model call "
                "when the checks find something."
            ),
        )
        if _research_mode and not st.session_state.get("answer_review"):
            st.caption("Research setting active: findings are recorded silently "
                       "(CONDUCTOR_REFLECTION=warn).")

    # AI Agent Suggestion. On demand by default: suggestions nobody asked for
    # are more to read, which is the opposite of what they are for.
    _suggest_now = "Auto" if st.session_state.get("auto_suggest") else "On click"
    with st.expander(f"Advisor suggestions · {_suggest_now}", icon=":material/auto_awesome:",
                     key="sb_suggest"):
        st.toggle(
            "Suggest after every answer",
            key="auto_suggest",
            value=False,
            help=(
                "Off: the advisor suggests next steps only when you click ✨ Suggest "
                "next steps in its panel. On: suggestions "
                "are generated automatically after every answer, at the cost of "
                "one extra model call each time."
            ),
        )

    with st.expander("Manage uploaded data", icon=":material/folder_open:", key="sb_uploads"):
        st.caption("Choose exactly what to remove from data_files and systems.")
        cleanup_groups = _collect_uploaded_cleanup_groups()
        if not cleanup_groups:
            st.info("No removable uploaded artifacts found.")
        else:
            selected_paths: list[pathlib.Path] = []
            for label, paths in cleanup_groups.items():
                key_slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                checked = st.checkbox(f"{label} ({len(paths)})", key=f"cleanup_group_{key_slug}")
                if checked:
                    selected_paths.extend(paths)

            if selected_paths:
                reset_active_now = st.checkbox(
                    "Also reset active backend to default profile now",
                    key="cleanup_reset_active_now",
                )
                st.caption("Files to remove:")
                for p in sorted(set(selected_paths)):
                    rel = p.relative_to(_REPO_ROOT) if _path_within(p, _REPO_ROOT) else p
                    st.markdown(f"- {rel}")

                if st.button("Delete selected", type="secondary", width="stretch"):
                    deleted, errs = _delete_uploaded_artifacts(selected_paths)
                    if deleted:
                        st.success(f"Removed {deleted} file(s).")
                    if errs:
                        for err in errs:
                            st.warning(err)
                    if reset_active_now:
                        ok, msg = _reset_active_backend_profile("pglib_case14")
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(f"Could not reset active backend: {msg}")
                    if not deleted and not errs:
                        st.info("Nothing was removed.")
                    st.rerun()
            else:
                st.caption("Select at least one group to enable deletion.")

    st.divider()
    st.markdown("<div class='conductor-sidebar-label'>Prompt Starters</div>", unsafe_allow_html=True)

    # ── Example queries ──────────────────────────────────────────────────
    with st.expander("Example queries", expanded=True, icon=":material/lightbulb:"):
        st.caption("Click to drop into CONDUCTOR's chat box — edit if you like, then press Run.")
        examples = st.container(key="examples_list")
        _for_scenario = SCENARIO_QUERIES.get((_scn or {}).get("current") or "", [])
        if _for_scenario:
            examples.markdown("<div class='conductor-eq-group'>For this scenario</div>", unsafe_allow_html=True)
            for q_i, (query, q_icon) in enumerate(_for_scenario):
                # "eq_" keys: the list's own styling (left-aligned, compact) applies.
                if examples.button(query, icon=q_icon, type="tertiary", width="stretch", key=f"eq_s{q_i}"):
                    st.session_state._box_pending = query
            examples.markdown("<div class='conductor-eq-group'>Any network</div>", unsafe_allow_html=True)
        _shown = {q for q, _ in _for_scenario}
        for q_i, (query, q_icon) in enumerate(EXAMPLE_QUERIES):
            if query in _shown:
                continue
            if examples.button(query, icon=q_icon, type="tertiary", width="stretch",
                               key=f"eq_{q_i}"):
                # Copy into the input box (applied before the widget renders below);
                # do NOT auto-run — the user sends it themselves.
                st.session_state._box_pending = query

# ---------------------------------------------------------------------------
# System view
#
# The network as a picture, beside the conversation rather than inside it.
#
# Two costs shape this. Streamlit tabs are **not** lazy — both panes render on
# every script run — and computing a layout takes from a tenth of a second to
# several seconds depending on the size of the network. So the geometry is
# cached against the network's fingerprint and recomputed only when the
# network itself changes; joining a new operating point onto cached positions
# is cheap.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=900)
def _fetch_topology(fingerprint: str, network: str = "base", edits_hash: str = "") -> dict | None:
    """The network's shape — the base, or a proposal with what its edits added.

    Keyed on the fingerprint (and the proposal's edits), so an upload or an
    edit refetches. The parameter used to be `_fingerprint`, which Streamlit
    leaves out of the cache key: the old topology was served after an upload.
    """
    try:
        params = {} if network == "base" else {"network": network}
        response = httpx.get(f"{BASE_URL}/api/network/topology", params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False, ttl=900)
def _layout_for(_fingerprint: str, payload: dict):
    """Positions and the classified map. The expensive half, cached per network."""
    network = _network_map.from_payload(payload)
    positions = _network_map.compute_layout(network)
    return network, positions


def _assumptions_note(assumptions: dict) -> str:
    """For the map's subtitle: the assumptions it was computed under, when they
    are not the defaults."""
    notes = []
    pf = assumptions.get("load_power_factor") or {}
    if pf.get("source") not in (None, "default"):
        notes.append(f"loads pf {pf['value']:.2f}")
    n = len(assumptions.get("generator_model") or {})
    if n:
        notes.append(f"{n} unit{'s' if n != 1 else ''} changed")
    return "".join(f" · {x}" for x in notes)


def _state_fingerprint() -> str:
    """The network and the model assumptions together: a change of either
    (load power factor, generator models) redraws the map instead of serving
    the cached one."""
    return (f"{st.session_state.get('_network_fingerprint') or ''}|"
            f"{st.session_state.get('_assumptions_key') or ''}")


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_state(timestamp: str, vm_lower: float, vm_upper: float,
                 fingerprint: str = "", network: str = "base",
                 edits_hash: str = "") -> tuple[dict, dict]:
    """RSA and snapshot at one moment, cached per (moment, limits, network).

    Uncached, every rerun — any click anywhere in the app, every suggestion,
    every finished turn — ran a full assessment because Streamlit tabs are not
    lazy. `fingerprint` is part of the key only so a network swap is never
    answered from the previous network's cache; `edits_hash` likewise for a
    proposal whose edits changed.
    """
    body = {"data_source": "measurements", "timestamp": timestamp,
            "vm_lower_pu": vm_lower, "vm_upper_pu": vm_upper, "network": network}
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        rsa = client.post(f"{BASE_URL}/api/grid/rsa", json=body)
        rsa.raise_for_status()
        snapshot = client.post(f"{BASE_URL}/api/network/snapshot",
                               json={"data_source": "measurements", "timestamp": timestamp,
                                     "network": network})
        snapshot.raise_for_status()
    return rsa.json(), snapshot.json()


def _system_operating_point() -> str:
    """The moment the system view describes.

    The simulation clock by default, and whatever the conversation last
    settled on when a turn ran at an explicit timestamp — so the picture
    follows the answer on screen instead of drifting away from it.
    """
    return st.session_state.get("system_ts") or st.session_state.current_ts


def _diff_figure(fingerprint, payload, positions, proposal_map, proposal_rsa, timestamp,
                 limits, chosen, problems):
    """The proposal against the base at the same moment and limits, drawn as
    a difference; None (with the reason in `problems`) if the base cannot be
    read, so the caller falls back to the proposal's own map."""
    # The base as the base: its own topology, drawn at the proposal's
    # positions (indices are shared), so what the proposal added shows as added.
    base_payload = _fetch_topology(fingerprint)
    if not base_payload:
        problems.append("The base topology could not be read for the difference view.")
        return None
    base_map = _network_map.from_payload(base_payload)
    try:
        base_rsa, base_snapshot = _fetch_state(timestamp, limits.vm_lower, limits.vm_upper,
                                               _state_fingerprint())
    except Exception as exc:  # noqa: BLE001
        problems.append(f"The base could not be assessed for the difference view: {exc}")
        return None
    problems += _network_map.apply_assessment(base_map, base_rsa)
    problems += _network_map.apply_conditions(base_map, base_snapshot)
    new = _network_map.violated_elements(proposal_rsa) - _network_map.violated_elements(base_rsa)
    gone = _network_map.violated_elements(base_rsa) - _network_map.violated_elements(proposal_rsa)
    subtitle = (f"Proposal {chosen} − base · {proposal_rsa.get('timestamp', timestamp)} · "
                f"{len(new)} new violation(s), {len(gone)} resolved")
    return render_network_diff(base_map, proposal_map, positions, new_violations=new,
                               subtitle=subtitle)


def _render_system_view() -> None:
    status = last_grid_constants_status()
    if status is not None and not status.live:
        # A drawing of the bundled example while the backend is down would be
        # the most confidently wrong thing in the app.
        st.error(
            "**The loaded network cannot be identified**, so it is not drawn. "
            f"The backend at `{BASE_URL}` is not responding."
        )
        return

    fingerprint = st.session_state.get("_network_fingerprint") or "unknown"
    assumptions = _fetch_assumptions() or {}
    st.session_state._assumptions_key = assumptions.get("key", "")

    # Which grid the picture is of: the base, or one of the proposals.
    proposals = {p["id"]: p for p in (_list_proposals() or [])}
    if st.session_state.get("system_network") not in proposals:
        st.session_state.system_network = "base"
    chosen = st.session_state.get("system_network") or "base"
    proposal = proposals.get(chosen)
    edits_hash = (proposal or {}).get("edits_hash", "")

    payload = _fetch_topology(fingerprint, chosen, edits_hash)
    if not payload:
        st.error(f"Could not read the network topology from `{BASE_URL}`.")
        return

    _render_system_header(payload, assumptions)
    st.session_state.setdefault("system_pane", "map")
    pane = st.segmented_control(
        "System view", ["map", "assumptions"], key="system_pane", label_visibility="collapsed",
        format_func={"map": "🗺️ Network map", "assumptions": "⚙️ Model assumptions"}.get,
        selection_mode="single") or "map"
    if pane == "assumptions":
        _render_model_assumptions(assumptions)
        return

    if proposals:
        st.segmented_control(
            "Network", ["base", *proposals], key="system_network",
            format_func=lambda n: "Base (loaded)" if n == "base" else f"Proposal · {n}",
            selection_mode="single")
    diff = False
    if proposal:
        diff = st.segmented_control(
            "View", ["proposal", "diff"], key="system_view", default="proposal",
            format_func={"proposal": "Proposal", "diff": "Difference from the base"}.get,
            selection_mode="single") == "diff"
        st.info("**Proposal " + chosen + "** — the loaded network plus: "
                + "; ".join((proposal.get("summary") or {}).get("edits") or []) + ".",
                icon=":material/edit_note:")

    network, positions = _layout_for(f"{fingerprint}|{chosen}|{edits_hash}", payload)
    # The cached objects are shared between reruns, so state is joined onto a
    # copy: mutating the cache would leave last hour's voltages on the map.
    network = copy.deepcopy(network)

    timestamp = _system_operating_point()
    following = st.session_state.get("system_ts") is None
    limits = grid_limits()
    problems: list[str] = []
    violations: set[str] = set()
    rsa = None

    try:
        rsa, snapshot = _fetch_state(timestamp, limits.vm_lower, limits.vm_upper,
                                     _state_fingerprint(),
                                     chosen, (proposal or {}).get("edits_hash", ""))
        problems += _network_map.apply_assessment(network, rsa)
        problems += _network_map.apply_conditions(network, snapshot)
        violations = _network_map.violated_elements(rsa)
        subtitle = ((f"Proposal {chosen} · " if proposal else "")
                    + f"{rsa.get('timestamp', timestamp)} · "
                    f"{rsa.get('total_violations', 0)} violation(s) · "
                    f"band {limits.vm_lower}–{limits.vm_upper} pu"
                    + _assumptions_note(assumptions))
    except Exception as exc:  # noqa: BLE001
        subtitle = "topology only — the operating point could not be read"
        problems.append(str(exc))

    figure = None
    if diff and rsa is not None:
        figure = _diff_figure(fingerprint, payload, positions, network, rsa, timestamp, limits,
                              chosen, problems)
    st.plotly_chart(
        figure or render_network_map(network, positions, vm_lower=limits.vm_lower,
                                     vm_upper=limits.vm_upper, highlight=violations,
                                     subtitle=subtitle, max_loading=limits.max_loading),
        width="stretch", theme=None,
    )

    st.caption(
        f"Showing **{timestamp}** — "
        + ("following the simulation clock." if following
           else "the operating point the conversation last used.")
    )
    if not following and st.button("Follow the simulation clock", key="sys_follow"):
        st.session_state.system_ts = None
        st.rerun()

    for note in network.notes:
        st.caption(f"· {note}")
    for problem in problems:
        # Reported rather than swallowed: an element whose result did not join
        # renders grey, and grey must not be mistaken for healthy.
        st.warning(problem)


def _generators_request(method: str, body: dict | None = None) -> dict | None:
    """GET or POST /api/generators; an error is kept for the panel to show."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = (client.get(f"{BASE_URL}/api/generators") if method == "get"
                 else client.post(f"{BASE_URL}/api/generators", json=body or {}))
        if r.status_code >= 400:
            st.session_state._gen_error = r.json().get("detail", r.text)
            return None
        return r.json()
    except Exception as exc:  # noqa: BLE001
        st.session_state._gen_error = str(exc)
        return None


_GENERATORS_INTRO = (
    "Every unit that can produce power, grouped by how it is modelled. **Edit** a unit to change it; "
    "*What the columns mean* explains each value.")
_GENERATORS_WHEN = (
    "Applies to the base network and every study from the next one on — nothing is regenerated "
    "(the data carry active power only). Proposals are re-checked; one that no longer fits is "
    "dropped and named. Earlier results are not recomputed; every new result lists the units that "
    "differ from the network as loaded. Kept for this network across restarts; CONDUCTOR can read "
    "and change it too, when you ask.")
_GENERATORS_EXPLAINED = """
| Value | What it means | Changing it affects |
|---|---|---|
| **The table a unit is in** (its control) | *Voltage control*: the unit holds its bus voltage, producing or absorbing whatever reactive power that takes. *Fixed power factor*: it injects its active power with reactive power in a fixed ratio; its bus voltage is free to move. *Fixed Q*: it injects or absorbs a set reactive power whatever its active power. Large synchronous machines usually control voltage; inverter-based units (solar, wind, batteries) usually run at a fixed power factor. | Voltages near the unit, reactive flows, losses and the reactive power imported from the grid — in every study. Change it in the unit's **Edit** form; the unit moves to the other table. |
| **V set-point** | The voltage the unit holds (voltage control, condensers). | Higher V raises the voltages around it and its reactive output. |
| **Reactive output** | A fixed-Q unit's set reactive power: positive injects (raises nearby voltage), negative absorbs. | Reactive power and voltages around the unit, whatever its active output. |
| **Power factor** | The fixed ratio of reactive to active power, Q = P · tan(acos pf), injecting or absorbing (absorbing lowers nearby voltage). 1.00 = none. | Reactive power and voltages around the unit. |
| **Reactive range** | The unit's reactive capability (lowest to highest output). | **OPF:** the unit stays within it. **Power flows** (security assessment, N-1) do not enforce it — they report each unit's Q against it. |
| **Dispatchable** | Whether the OPF and flexibility studies may change the unit's active power. | No: it keeps its measured output in every optimisation (fewer corrective options). Assessments are not affected. |
| **Pmax**, **Bus** | Capacity and location, from the network file. | — (read-only) |
| **changed** | The unit differs from the network as loaded. | — |

**Condensers** have no active power, so they are always on voltage control and never dispatched;
only their set-point and reactive range can change. A generator at a fixed power factor keeps the
set-point and range it had, so switching it back to voltage control restores them.
"""


def _render_unit_table(group: str, group_rows: list[dict], editable: bool) -> None:
    """One kind of unit as a plain-text table (crisp at any zoom), each row
    with its Edit button."""
    spec = _gen_table.GROUPS[group]
    st.markdown(f"**{spec['title']}** · {len(group_rows)}")
    st.caption(spec["caption"])
    widths = [1.0] * len(spec["columns"]) + [0.45]
    with st.container(border=True, key=f"gentable_{group}", gap="small"):
        head = st.columns(widths, vertical_alignment="center")
        for col, name in zip(head, spec["columns"]):
            col.markdown(f"<span class='gen-head'>{html.escape(name)}</span>", unsafe_allow_html=True)
        for n, row in enumerate(group_rows):
            with st.container(key=f"genrow_{group}_{n}"):
                cols = st.columns(widths, vertical_alignment="center")
                for k, (col, text) in enumerate(zip(cols, row["cells"])):
                    mark = " <span class='gen-changed'>changed</span>" if k == 0 and row["changed"] else ""
                    col.markdown(f"<span class='gen-cell'>{html.escape(text)}</span>{mark}",
                                 unsafe_allow_html=True)
                cols[-1].button("Edit", key=f"gen_edit_{row['unit']}", type="tertiary", disabled=not editable,
                                icon=":material/edit:",
                                on_click=lambda name=row["unit"]: st.session_state.update(_gen_editing=name))


@st.dialog("Edit unit", width="medium")
def _edit_unit_dialog(unit: dict) -> None:
    """One unit's model, with only the fields that apply to it."""
    st.markdown(f"**{unit['unit']}** · {unit.get('bus_name') or unit['bus']} · "
                f"Pmax {unit['p_max_mw']:g} MW")
    values: dict = {}
    if unit.get("condenser"):
        st.caption("A synchronous condenser: no active power, so it always controls voltage and is never "
                   "dispatched. Its set-point and reactive range can change.")
        control = "voltage"
    else:
        control = st.radio("Control", list(_gen_table.CONTROL_LABELS), horizontal=True,
                           index=list(_gen_table.CONTROL_LABELS).index(unit["control"]),
                           format_func=_gen_table.CONTROL_LABELS.get, key=f"dlg_control_{unit['unit']}")
        values["control"] = control
        st.caption({"voltage": "Holds its bus voltage at a set-point; its reactive power is whatever that takes.",
                    "power_factor": "Injects its active power with reactive power in a fixed ratio; its bus "
                                    "voltage is free.",
                    "fixed_q": "Injects or absorbs a set reactive power, whatever its active power; its bus "
                               "voltage is free."}[control])
    fields = _gen_table.form_fields(unit, control)
    if "vm_pu" in fields:
        values["vm_pu"] = st.number_input("Voltage set-point (p.u.)", min_value=0.8, max_value=1.2,
                                          value=float(unit["vm_pu"] or 1.0), step=0.005, format="%.3f")
    if "q_min_mvar" in fields:
        c_lo, c_hi = st.columns(2)
        values["q_min_mvar"] = c_lo.number_input("Lowest reactive output (Mvar)", value=unit["q_min_mvar"],
                                                 step=1.0, placeholder="none on record")
        values["q_max_mvar"] = c_hi.number_input("Highest reactive output (Mvar)", value=unit["q_max_mvar"],
                                                 step=1.0, placeholder="none on record")
    if "q_mvar" in fields:
        values["q_mvar"] = st.number_input("Reactive output (Mvar; positive injects, negative absorbs)",
                                           value=float(unit.get("q_mvar") or 0.0), step=1.0, format="%.2f")
    if "pf" in fields:
        values["pf"] = st.number_input("Power factor (1.00 = no reactive power)", min_value=0.5, max_value=1.0,
                                       value=float(unit["pf"] or 1.0), step=0.01, format="%.2f")
    if "absorbing" in fields:
        values["absorbing"] = st.toggle("Absorbs reactive power (lowers nearby voltage) instead of injecting it",
                                        value=bool(unit["absorbing"]))
    if "dispatchable" in fields:
        values["dispatchable"] = st.toggle("Dispatchable — the OPF may change its active power",
                                           value=bool(unit["dispatchable"]))
    change = _gen_table.form_changes(unit, values)
    error = st.session_state.pop("_gen_error", None)
    if error:
        st.warning(error)
    c_apply, c_cancel = st.columns(2)
    if c_apply.button("Apply", type="primary", disabled=change is None, width="stretch", key="gen_apply"):
        result = _generators_request("post", {"changes": [change]})
        if result is not None:
            if result.get("proposals_dropped"):
                st.session_state._gen_error = ("Applied. These proposals no longer apply and were dropped: "
                                               + "; ".join(result["proposals_dropped"]))
            st.rerun()
        st.rerun(scope="fragment")              # keep the dialog open to show why
    if c_cancel.button("Cancel", width="stretch", key="gen_cancel"):
        st.rerun()


def _render_model_assumptions(assumptions: dict) -> None:
    """The assumptions every study is built on: how loads draw reactive power
    and how each unit is modelled."""
    st.caption("What every study assumes beyond the data. Changes take effect from the next study; "
               "nothing needs to be regenerated.")
    _render_generator_models()
    st.divider()
    _render_load_pf(assumptions)


def _open_assumptions() -> None:
    st.session_state.system_pane = "assumptions"


def _toggle_nominal() -> None:
    st.session_state.op_nominal = not st.session_state.get("op_nominal", False)


def _render_system_header(payload: dict, assumptions: dict) -> None:
    """One line for the network, then the assumptions every study uses, always
    in view — each chip opens them (the nominal chip switches the point)."""
    counts = payload.get("counts") or {}
    parts = [f"{counts.get(k, '—')} {label}" for k, label in (
        ("buses", "buses"), ("lines", "lines"), ("trafos", "transformers"),
        ("loads", "loads"), ("generators", "units"))]
    levels = payload.get("voltage_levels") or []
    if len(levels) == 1 and (counts.get("buses") or 0) > 1:
        volts = ("per-unit case (every bus at 1 kV)" if abs(levels[0] - 1.0) < 1e-9
                 else f"every bus at {levels[0]:g} kV")
    else:
        volts = ", ".join(f"{v:g} kV" for v in levels) or "unknown voltage levels"
    st.markdown(f"**{html.escape(str(payload.get('name', 'Network')))}** · " + " · ".join(parts))
    st.caption(volts)

    pf = assumptions.get("load_power_factor") or {}
    pf_changed = pf.get("source") not in (None, "default")
    n_changed = len(assumptions.get("generator_model") or {})
    timeline = _tl.Timeline.from_payload(_fetch_timeline())
    nominal_on = bool(st.session_state.get("op_nominal")) and timeline.nominal_available
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="sys_chips"):
        st.caption("Assumptions", width="content")
        q_measured = ((assumptions.get("reactive_measured") or {}).get("measurements") or {}).get("consumption")
        if pf and q_measured:
            st.button(f"Loads Q measured · pf {pf['value']:.2f} for gaps", key="chip_pf",
                      type="primary" if pf_changed else "secondary",
                      help="The data carry reactive demand; the power factor fills only missing values — "
                           "open the model assumptions", on_click=_open_assumptions)
        elif pf:
            st.button(f"Loads pf {pf['value']:.2f}" + (" · changed" if pf_changed else ""),
                      key="chip_pf", type="primary" if pf_changed else "secondary",
                      help="The power factor every study gives measured loads — open the model assumptions",
                      on_click=_open_assumptions)
        st.button("Generators · " + (f"{n_changed} changed" if n_changed else "as loaded"),
                  key="chip_gen", type="primary" if n_changed else "secondary",
                  help="How each unit is modelled — open the model assumptions", on_click=_open_assumptions)
        st.button("Nominal point · " + ("on" if nominal_on else "off"), key="chip_nominal",
                  type="primary" if nominal_on else "secondary", disabled=not timeline.nominal_available,
                  help=("Studies use the network's own operating point instead of a moment of the data — "
                        "click to switch (also in the sidebar)." if timeline.nominal_available
                        else f"Not available for this network. {timeline.nominal_reason}"),
                  on_click=_toggle_nominal)


def _render_generator_models() -> None:
    """How each unit of the loaded network is modelled — the operator's to
    change at any time (CONDUCTOR can too, when asked)."""
    data = _generators_request("get")
    error = st.session_state.pop("_gen_error", None)
    n_changed = len((data or {}).get("changed") or {})
    st.markdown("##### Generators — how each unit is modelled"
                + (f" · {n_changed} changed" if n_changed else ""))
    with st.container(key="sys_generators"):
        if error:
            st.warning(error)
        if data is None:
            st.caption("Could not read the generators from the backend.")
            return
        if not data.get("editable"):
            st.caption(data.get("reason") or "Unit models cannot be changed for this network.")
        changed = data.get("changed") or {}
        st.markdown(_GENERATORS_INTRO)
        t_units, t_help = st.tabs(["Units", "What the columns mean"])
        with t_help:
            st.markdown(_GENERATORS_EXPLAINED)
        with t_units:
            units = {u["unit"]: u for u in data.get("units") or []}
            groups = _gen_table.rows(list(units.values()), changed)
            if not groups:
                st.caption("This network has no generating units besides the external grid.")
                return
            editable = bool(data.get("editable"))
            for group, group_rows in groups.items():
                _render_unit_table(group, group_rows, editable)
            if st.button("Reset all to as loaded", key="gen_reset", disabled=not changed):
                _generators_request("post", {"reset": True})
                st.rerun()
            st.caption(_GENERATORS_WHEN)
            editing = st.session_state.pop("_gen_editing", None)
            if editing in units:
                _edit_unit_dialog(units[editing])


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
_render_main_hero(compact=bool(st.session_state.messages))

# Conversation history renders in the MAIN script run, so it stays solid (not
# greyed) while the input fragment below runs the slow agent call. That fragment
# isolation is what stops the previous answer from "ghosting" grey during the wait.
# The input box stays outside the tabs on purpose: `st.chat_input` pins itself
# to the bottom of the viewport only at the top level, and inside a tab it
# would render inline and appear on one pane only. Out here it stays pinned and
# One label per tool the agent can call — shown in the live status box, the
# audit panel and chart captions. Kept complete by
# `tests/test_app_labels.py`: four entries here once named tools that do not
# exist, and the deterministic OPF was labelled "Robust OPF", telling the
# operator a dispatch carried a robustness guarantee it did not.
_TOOL_LABELS = {
    "locate_network_element": "Element lookup",
    "get_current_timestamp": "Reading timestamp",
    "advance_timestamp": "Advancing timestamp",
    "get_current_conditions": "Current conditions",
    "get_element_timeseries": "Element time series",
    "run_rsa": "Security assessment (RSA)",
    "simulate_contingency": "Single N-1 contingency",
    "simulate_all_contingencies": "N-1 contingency screen",
    "find_worst_case_timestamp": "Worst-case scan",
    "scan_rsa_over_time": "Security scan over time",
    "compute_violation_attribution": "Violation attribution",
    "compare_results": "Result comparison",
    "edit_proposal": "Network proposal",
    "set_load_power_factor": "Load power factor setting",
    "get_data_quality": "Data quality report",
    "set_generator_model": "Generator model",
    "compare_networks": "Base vs proposal",
    "run_probabilistic_rsa": "Probabilistic security assessment",
    "scan_scenarios": "Renewable scenario scan",
    "compute_historical_risk": "Historical risk",
    "optimize_contingency": "Post-contingency OPF",
    "optimize_flexibility": "Flexibility OPF (deterministic)",
    "optimize_robust_flexibility": "Robust flexibility OPF",
    "compute_flexibility_envelope": "Flexibility envelope",
    "compute_hosting_capacity": "Hosting capacity",
    "evaluate_kpis": "KPI evaluation",
    "forecast_kpis": "KPI forecast",
}


def _render_trace(trace, key: str) -> None:
    """The audit panel under an answer: what ran, where the numbers came from,
    and what was not checked.

    Collapsed by default — an operator who trusts the answer never opens it —
    but the label always carries the concern count, because a panel nobody
    opens cannot warn anybody.

    Deliberately describes rather than certifies. "12 figures traced" is a
    statement about where numbers came from; it is not a claim that the answer
    is correct, and the limits block at the bottom says so explicitly.
    """
    # `concerns` is in the guard deliberately. A turn can answer from memory —
    # no tools, no figures — and still raise a completeness gap or a
    # characterisation conflict, and that is precisely the answer that must not
    # render without its warning.
    if trace is None or not (trace.steps or trace.figures or trace.concerns):
        return

    icon = "⚠️" if trace.has_concerns else "🔍"
    with st.expander(f"{icon} How this answer was produced — {trace.headline()}"):

        # Concerns first: anything worth acting on should not be below a table
        # the operator has to scroll past.
        if trace.concerns:
            st.markdown("**Worth checking**")
            for item in trace.concerns:
                st.markdown(f"- {item}")
            st.divider()

        if trace.steps:
            st.markdown("**What ran**")
            for step in trace.steps:
                label = _TOOL_LABELS.get(step.name, step.name.replace("_", " ").title())
                mark = "" if step.ok else " ❌"
                st.markdown(f"`{step.order}.` **{label}**{mark}")
                if step.args:
                    # Each argument in its own code span. A bare "·" separator
                    # sat flush against the preceding value — `…pct=100· data…`
                    # reads as a decimal point, which is the one typo an
                    # operator scanning limits must not have to second-guess.
                    # The monospace box also gives every value a visible
                    # boundary, so a trailing digit cannot merge into the next
                    # key.
                    st.markdown(
                        "&nbsp; ".join(f"`{k}={v}`" for k, v in step.args.items()),
                        unsafe_allow_html=True,
                        help="Arguments this tool actually ran with.",
                    )
                if step.inherited:
                    # The only place the repair guard becomes visible to a
                    # human: a value the operator set earlier, carried forward
                    # so this study matched the one before it. Same code-span
                    # treatment as the arguments above — this line states a
                    # limit, and a limit an operator has to squint at is worse
                    # than no line.
                    carried = "&nbsp; ".join(
                        f"`{field}={detail.get('value')}`"
                        f" _(from {detail.get('from')})_"
                        if isinstance(detail, dict) else f"`{field}={detail}`"
                        for field, detail in sorted(
                            step.inherited.items(), key=lambda kv: str(kv[0]))
                    )
                    st.markdown(f"↳ carried forward: {carried}",
                                unsafe_allow_html=True)
                if step.error:
                    st.caption(f"↳ {step.error}")
            st.divider()

        if trace.figures:
            st.markdown("**Where the numbers came from**")
            st.dataframe(
                [
                    {
                        "Figure": f"{f.value:g}" + (f" {f.unit}" if f.unit else ""),
                        "Status": f.detail,
                        "Source": f.source or "—",
                    }
                    for f in trace.figures
                ],
                hide_index=True, width="stretch",
                key=f"trace_figs_{key}",
            )
            st.divider()

        st.markdown("**What this does not tell you**")
        for limit in trace.limits:
            st.caption(f"· {limit}")


# ---------------------------------------------------------------------------
# AI Agent Suggestion
#
# A second agent, run on demand, that reads the conversation and proposes how
# to continue. Its output is model reasoning, not tool output, and is shown in
# its own panel with that said plainly. A suggestion is never sent on the
# operator's behalf: choosing one places its prompt in the chat box.
# ---------------------------------------------------------------------------

_AI_SUGGESTION_CAPTION = ("Run in CONDUCTOR puts a suggestion in CONDUCTOR's chat box — edit it if "
                          "you like, then press Run.")


def _suggestion_context() -> dict:
    """Session facts the suggestion agent needs to write runnable prompts."""
    status = last_grid_constants_status()
    gc = dict(status.values) if status is not None else {}
    names = list(gc.get("substation_names") or [])
    view = st.session_state.get("op_view") or {}
    mode = view.get("mode", _tl.MEASURED)
    ctx = {
        "network": gc.get("name"),
        "simulation_clock": st.session_state.get("current_ts"),
        "data_source": {_tl.FORECAST: "forecasts",
                        _tl.COMPARE: "measurements and forecasts",
                        _tl.NOMINAL: "nominal (the network's own operating point, no time)",
                        }.get(mode, "measurements"),
        "viewing_timestamp": view.get("ts") if mode not in (_tl.MEASURED, _tl.NOMINAL) else None,
        "default_voltage_band_pu": [gc.get("vm_lower"), gc.get("vm_upper")],
        "max_loading_pct": gc.get("max_loading_pct"),
        "substations": names[:40] + ([f"… and {len(names) - 40} more"] if len(names) > 40 else []),
        # Network proposals that exist now — read fresh: the grid constants
        # were fetched before the turn that may have created one.
        "proposals": [{"id": p["id"], "edits": (p.get("summary") or {}).get("edits") or []}
                      for p in (_list_proposals() or [])] or None,
    }
    return {k: v for k, v in ctx.items() if v not in (None, [], [None, None])}


def _shown_suggestions() -> list[dict]:
    """Every suggestion displayed in this conversation, so none is repeated."""
    out: list[dict] = []
    for m in st.session_state.messages:
        for rnd in m.get("suggestion_rounds") or []:
            out.extend(rnd.get("suggestions") or [])
    return out


def _run_ai_suggestion(idx: int, trigger: str) -> None:
    """Call the suggestion agent for message `idx`, store and log the round."""
    msg = st.session_state.messages[idx]
    rounds = msg.setdefault("suggestion_rounds", [])
    upto = st.session_state.messages[: idx + 1]
    visible = [{"role": m["role"], "text": m.get("text", "")} for m in upto]
    records = [m["record"] for m in upto if m.get("record")]

    started = time.perf_counter()
    with st.spinner("The advisor is considering how to continue…"):
        try:
            result = _recommender.suggest(
                visible, msg.get("record"), records, _suggestion_context(),
                shown=_shown_suggestions(),
            )
        except Exception as exc:  # noqa: BLE001
            # Provider failures are already returned as results; this catches
            # anything else, so a suggestion can never take the page down.
            logging.getLogger(__name__).warning("AI Agent Suggestion failed.", exc_info=True)
            result = _recommender.SuggestionResult((), error=str(exc))

    round_id = f"{st.session_state.ui_conversation_id}-{idx}-{len(rounds) + 1}"
    entry = {
        "id": round_id,
        "comment": result.comment,
        "comment_unverified": list(result.comment_unverified),
        "suggestions": [s.as_record() for s in result.suggestions],
        "note": result.note,
        "error": result.error,
    }
    rounds.append(entry)

    try:
        provider_info = describe_provider(get_provider())
    except Exception:  # noqa: BLE001
        provider_info = {}
    _loop_module.log_event({
        "kind": "suggestion",
        "round_id": round_id,
        "conversation": st.session_state.ui_conversation_id,
        "after_turn_key": (msg.get("record") or {}).get("turn_key"),
        "trigger": trigger,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_s": round(time.perf_counter() - started, 3),
        **provider_info,
        "attention": msg.get("attention") or [],
        "comment": result.comment,
        "comment_unverified": entry["comment_unverified"],
        "suggestions": entry["suggestions"],
        "note": result.note,
        "error": result.error,
        # Kept only when parsing failed, to diagnose what the model returned.
        "raw": result.raw if result.error else "",
    })


def _log_suggestion_outcome(sent: str) -> None:
    """Record what the operator did after suggestions were shown.

    `used` — sent a suggestion unchanged; `edited` — chose one and changed it
    (similarity kept, so the threshold is the analyst's choice, not ours);
    `ignored` — suggestions were on screen and the operator asked something
    else.
    """
    used = st.session_state.pop("_suggestion_used", None)
    last = st.session_state.messages[-1] if st.session_state.messages else {}
    rounds = last.get("suggestion_rounds") or []
    if used is None and not rounds:
        return
    record = {
        "kind": "suggestion_outcome",
        "conversation": st.session_state.ui_conversation_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sent": sent,
    }
    if used is not None:
        similarity = difflib.SequenceMatcher(None, used["prompt"].strip(), sent.strip()).ratio()
        record.update({
            "outcome": "used" if sent.strip() == used["prompt"].strip() else "edited",
            "round_id": used["round_id"],
            "index": used["index"],
            "similarity": round(similarity, 3),
        })
    else:
        record.update({"outcome": "ignored", "round_id": rounds[-1]["id"]})
    _loop_module.log_event(record)


def _ai_box(label: str, text: str, key: str) -> None:
    """The AI agent's prose in its purple box, rendered as Markdown (bold,
    lists, code). Drawn as a keyed container styled by CSS rather than an
    HTML block: Markdown inside raw HTML is not rendered, and escaping it
    showed the asterisks. Raw HTML from the model stays escaped."""
    with st.container(key=f"aibox_{key}"):
        st.markdown(f"<div class='conductor-ai-comment-label'>{html.escape(label)}</div>",
                    unsafe_allow_html=True)
        st.markdown((text or "").replace("$", "\\$"))


def _render_suggestion_round(rnd: dict, idx: int, r_i: int, expanded: bool = True,
                             framed: bool = True) -> None:
    # Collapsible, so suggestions the operator has read or does not want can
    # be folded away. Unframed inside "Earlier in this conversation", where an
    # expander of its own could not be nested.
    frame = (st.expander(":violet[**✨ Suggestions**]", expanded=expanded) if framed
             else st.container())
    with frame:
        st.caption(_AI_SUGGESTION_CAPTION)
        if rnd.get("error"):
            st.caption("The advisor could not produce suggestions this time. "
                       "Try again in a moment.")
            return
        if rnd.get("comment"):
            # Its reading of what happened, above the ways to continue.
            _ai_box("Advisor · comment", rnd["comment"], key=f"c_{idx}_{r_i}")
            if rnd.get("comment_unverified"):
                st.caption("⚠️ Not found in this conversation's results: "
                           + ", ".join(rnd["comment_unverified"]))
        if not rnd.get("suggestions"):
            st.caption(rnd.get("note") or "The advisor had nothing new to suggest.")
            return
        for s_i, sug in enumerate(rnd["suggestions"]):
            with st.container(key=f"card_s_{idx}_{r_i}_{s_i}", gap="small"):
                _suggestion_card(rnd, sug, idx, r_i, s_i)


def _suggestion_card(rnd: dict, sug: dict, idx: int, r_i: int, s_i: int) -> None:
    def _use(prompt=sug["prompt"], index=s_i) -> None:
        # Same path as the example queries: into the box, not sent.
        st.session_state._box_pending = prompt
        st.session_state._suggestion_used = {
            "round_id": rnd["id"], "index": index, "prompt": prompt,
        }

    # Title row sized by its content, like a study card; why below it.
    with st.container(horizontal=True, vertical_alignment="center", gap="small",
                      key=f"row_s_{idx}_{r_i}_{s_i}"):
        label = _recommender.ANGLE_LABELS.get(sug["angle"], sug["angle"])
        st.markdown(
            f"<span class='conductor-ai-badge'>{html.escape(label)}</span>"
            f"<strong>{html.escape(sug['title'])}</strong>",
            unsafe_allow_html=True, width="stretch",
        )
        # The full request on hover: the row stays one line.
        st.button(_RUN_IN_CONDUCTOR, key=f"ais_use_{idx}_{r_i}_{s_i}", disabled=_ui_busy(),
                  help=_handoff_help(sug["prompt"]), on_click=_use, width="content")
    if sug.get("why"):
        # `$` would otherwise be read as LaTeX.
        st.caption(sug["why"].replace("$", "\\$"))
    if sug.get("unverified"):
        st.caption("⚠️ Not found in this conversation's results: " + ", ".join(sug["unverified"]))


# ---------------------------------------------------------------------------
# Discussion with the AI agent — a side thread under an answer
#
# The same AI agent as the suggestions, in its discussion mode: the operator
# thinks a result through with it. It runs nothing. Studies it proposes go
# to the chat box through "Use", exactly like suggestions, and the operator
# sends them; their answers then reach the thread as evidence. Closing the
# thread records a decision note in the session log.
# ---------------------------------------------------------------------------
_DISCUSS_CAPTION = (
    "Think a result through with the advisor — e.g. *is this result reliable?*, *what should I "
    "check before deciding?* Studies it proposes go to CONDUCTOR with Run in CONDUCTOR, and you "
    "decide whether to send them."
)
# Open questions, not fixed ones: the box takes anything, and its placeholder
# shows the kind of thing worth asking when the discussion is empty.
_DISCUSS_PLACEHOLDER = ("Ask anything about this result — e.g. why might it be wrong, what to "
                        "check before deciding, what the concerns mean…")
# Before the first question there is no result yet: the discussion helps plan.
_PLAN_CAPTION = (
    "Talk a question through before asking CONDUCTOR — what to study, at which moment, against "
    "which limits. Studies the advisor proposes go to CONDUCTOR with Run in CONDUCTOR."
)
_PLAN_PLACEHOLDER = ("e.g. we are asked to connect a 30 MW data centre — how should we assess "
                     "it?")


def _discussions() -> list[dict]:
    """This conversation's discussions with the AI agent, oldest first:
    `{"thread", "note", "started_at", "closed_at"}` (message indices)."""
    return st.session_state.setdefault("discussions", [])


def _active_discussion() -> dict | None:
    """The open discussion. There is at most one, and it follows the
    conversation: it is drawn under the latest answer, so a study sent from
    it is answered just above where the discussion continues."""
    found = _discussions()
    return found[-1] if found and found[-1].get("note") is None else None


def _answered_turn(prompt: str, after: int) -> int | None:
    """The turn (1-based, counting questions) that answered a study sent from
    a discussion — the first question after `after` that matches it."""
    turn = 0
    for i, m in enumerate(st.session_state.messages):
        if m.get("role") != "user":
            continue
        turn += 1
        if i > after and difflib.SequenceMatcher(
                None, (m.get("text") or "").strip(), prompt.strip()).ratio() > 0.9:
            answered = i + 1 < len(st.session_state.messages) and \
                st.session_state.messages[i + 1].get("role") == "assistant"
            return turn if answered else None
    return None


_RUN_IN_CONDUCTOR = "Run in CONDUCTOR →"


def _handoff_help(prompt: str, why: str | None = None) -> str:
    """The hand-off's tooltip: why (for a study), the exact request, and what
    the click does."""
    parts = [f"**Why:** {why}"] if why else []
    parts += [f"**Request:** {prompt}", "Puts it in CONDUCTOR's chat box — you send it with **Run**."]
    return "\n\n".join(parts).replace("$", "\\$")


def _render_study_card(study: dict, key: str, disc: dict | None) -> None:
    """A study the advisor proposed; "Run in CONDUCTOR" only while its
    discussion is open — only CONDUCTOR runs studies."""
    with st.container(key=f"card_{key}", gap="small"):
        _study_card_body(study, key, disc)


def _study_card_body(study: dict, key: str, disc: dict | None) -> None:
    # One row sized by its content: the title takes what is left (and wraps),
    # the ⓘ and the hand-off only their own width.
    with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"row_{key}"):
        st.markdown("<span class='conductor-ai-badge'>Study</span>"
                    f"<span class='conductor-study-title'>{html.escape(study['title'])}</span>",
                    unsafe_allow_html=True, width="stretch")
        if disc is None:
            # A closed discussion: the request stays readable, nothing to send.
            with st.popover("", icon=":material/info:", type="tertiary", help="Why, and the full request"):
                if study.get("why"):
                    st.markdown(f"**Why:** {study['why']}".replace("$", "\\$"))
                st.caption(study["prompt"].replace("$", "\\$"))
        else:
            def _use() -> None:
                # A callback: it runs before the redraw, so the "sent" line
                # shows at once (set while drawing, it appeared a click later).
                st.session_state._box_pending = study["prompt"]
                disc["thread"].append({"role": "sent", "text": study["title"], "prompt": study["prompt"]})
                _loop_module.log_event({
                    "kind": "advisor_study_sent",
                    "conversation": st.session_state.ui_conversation_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "study": study,
                })

            # Why it is proposed and the exact request on hover: the row stays one line.
            st.button(_RUN_IN_CONDUCTOR, key=key, disabled=_ui_busy(),
                      help=_handoff_help(study["prompt"], study.get("why")), on_click=_use,
                      width="content")
    if study.get("unverified"):
        st.caption("⚠️ Not found in this conversation's results: " + ", ".join(study["unverified"]))


def _render_decision_note(note: dict) -> None:
    with st.container(border=True):
        st.markdown(":violet[**📝 Decision note**]")
        st.markdown(f"**Decision:** {note['decision']}".replace("$", "\\$"))
        for label, key in (("Evidence", "evidence"), ("Open risks", "open_risks"),
                           ("Studies run", "studies_run")):
            if note.get(key):
                st.markdown(f"**{label}:**\n" + "\n".join(f"- {x}" for x in note[key]).replace("$", "\\$"))
        if note.get("unverified"):
            st.caption("⚠️ Not found in this conversation's results: " + ", ".join(note["unverified"]))
        st.caption("Written by the advisor from the discussion; recorded in the session log.")


def _render_operator_bubble(text: str, key: str) -> None:
    """The operator's message: a small right-aligned bubble — one element, so
    Streamlit's spacing between elements cannot push the text out of it."""
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(f"<div class='conductor-you-row'><div class='conductor-you'>{body}</div></div>",
                unsafe_allow_html=True)


def _render_thread_item(disc: dict, k: int, t_i: int, item: dict, interactive: bool) -> None:
    role = item.get("role")
    if role == "operator":
        _render_operator_bubble(item["text"], key=f"{k}_{t_i}")
    elif role == "advisor":
        if item.get("error"):
            st.caption(f"The advisor could not reply: {item['error']}")
            return
        _ai_box("Advisor", item.get("text") or "", key=f"t_{k}_{t_i}")
        if item.get("unverified"):
            st.caption("⚠️ Not found in this conversation's results: " + ", ".join(item["unverified"]))
        for s_i, study in enumerate(item.get("studies") or []):
            _render_study_card(study, key=f"adv_use_{k}_{t_i}_{s_i}", disc=disc if interactive else None)
    elif role == "sent":
        turn = _answered_turn(item.get("prompt", ""), disc.get("started_at", -1))
        state = f"answered in turn {turn}" if turn else "not sent yet"
        st.markdown(f"<div class='conductor-sent'>↪ Sent: {html.escape(item['text'])} · {state}</div>",
                    unsafe_allow_html=True)


def _render_thread(disc: dict, k: int, interactive: bool) -> None:
    """The discussion as a chat. In the open discussion only the latest
    exchange — the operator's last message and what followed — is shown in
    full; earlier ones fold away, so a long discussion stays readable."""
    thread = disc.get("thread") or []
    last_ask = max((i for i, it in enumerate(thread) if it.get("role") == "operator"), default=0)
    if interactive and last_ask > 0:
        n_q = sum(1 for it in thread[:last_ask] if it.get("role") == "operator")
        with st.expander(f"{n_q} earlier question{'s' if n_q != 1 else ''} and the replies", expanded=False):
            for t_i in range(last_ask):
                _render_thread_item(disc, k, t_i, thread[t_i], interactive)
        start = last_ask
    else:
        start = 0
    for t_i in range(start, len(thread)):
        _render_thread_item(disc, k, t_i, thread[t_i], interactive)


def _render_discussion_input(disc: dict | None, k: int, idx: int) -> None:
    """The question box and the close button — or, while the AI agent is
    answering, the operator's message and the thinking indicator in their place."""
    thread = (disc or {}).get("thread") or []
    pending = st.session_state.get("_advisor_request")
    if pending:
        kind, text, _ = pending
        if kind == "ask":
            _render_operator_bubble(text, key=f"{k}_pending")
        _AGENT_SLOT["slot"] = st.empty()      # the reply's place, while it is written
        return
    if _agent_busy():
        return

    def _ask() -> None:
        question = (st.session_state.get(f"adv_q_{k}") or "").strip()
        if question:
            st.session_state._advisor_request = ("ask", question, idx)

    with st.container(key="adv_input"):
        with st.form(key=f"adv_form_{k}", clear_on_submit=True, border=False):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.text_input("Discuss with the advisor", key=f"adv_q_{k}",
                              placeholder=(_PLAN_PLACEHOLDER if idx < 0 else _DISCUSS_PLACEHOLDER)
                              if not thread else "Ask a follow-up…", label_visibility="collapsed",
                              width="stretch")
                st.form_submit_button("Discuss", width="content", on_click=_ask)
        # A decision needs evidence: not before the first answer.
        if idx >= 0 and any(i.get("role") == "advisor" and not i.get("error") for i in thread):
            st.button("📝 Close and record a decision", key=f"adv_close_{k}", type="tertiary",
                      on_click=lambda: st.session_state.update(_advisor_request=("note", "", idx)))


def _run_advisor(kind: str, text: str, idx: int) -> None:
    """One call of the AI agent's discussion mode, on the open discussion
    (started here when there is none)."""
    disc = _active_discussion()
    if disc is None:
        disc = {"thread": [], "note": None, "started_at": idx, "closed_at": None}
        _discussions().append(disc)
    thread = disc["thread"]
    for item in thread:
        if item.get("role") == "sent":
            item["answered_turn"] = _answered_turn(item.get("prompt", ""), disc["started_at"])
    visible = [{"role": m["role"], "text": m.get("text", "")} for m in st.session_state.messages]
    records = [m["record"] for m in st.session_state.messages if m.get("record")]
    # The newest results are what the discussion is about now — including the
    # answers to studies sent from the thread.
    latest = records[-1] if records else None
    started = time.perf_counter()
    with st.spinner("The advisor is thinking…"):
        try:
            if kind == "note":
                result = _recommender.decision_note(thread, visible, latest, records, _suggestion_context())
            else:
                result = _recommender.discuss(text, thread, visible, latest, records, _suggestion_context())
        except Exception as exc:  # noqa: BLE001 — never takes the page down
            logging.getLogger(__name__).warning("AI agent discussion failed.", exc_info=True)
            result = (_recommender.DecisionNote("", error=str(exc)) if kind == "note"
                      else _recommender.AdvisorReply("", error=str(exc)))
    try:
        provider_info = describe_provider(get_provider())
    except Exception:  # noqa: BLE001
        provider_info = {}
    anchor = (st.session_state.messages[idx].get("record") or {}) if idx >= 0 else {}
    base = {"conversation": st.session_state.ui_conversation_id,
            "after_turn_key": anchor.get("turn_key"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "duration_s": round(time.perf_counter() - started, 3), **provider_info}
    if kind == "note":
        if result.ok:
            disc["note"] = result.as_record()
            disc["closed_at"] = idx
        else:
            thread.append({"role": "advisor", "text": "", "error": result.error})
        _loop_module.log_event({"kind": "decision_note", **base,
                                **(result.as_record() if result.ok else {"error": result.error})})
        return
    thread.append({"role": "operator", "text": text})
    thread.append({"role": "advisor", "text": result.text, "error": result.error,
                   "studies": [s.as_record() for s in result.studies],
                   "unverified": list(result.unverified)})
    _loop_module.log_event({"kind": "advisor_message", **base, "question": text,
                            "reply": result.text, "studies": [s.as_record() for s in result.studies],
                            "unverified": list(result.unverified), "error": result.error})


def _attention_hint(attention: list[str]) -> str:
    # A proposal's gap and an answer check keep their subject; other reasons
    # drop the tool prefix.
    first = attention[0]
    return first if first.startswith(("proposal ", "answer check:")) else first.split(": ", 1)[-1]


def _turn_of(idx: int) -> int:
    """The turn number (questions asked) a message index belongs to."""
    return sum(1 for m in st.session_state.messages[: idx + 1] if m.get("role") == "user")


def _turn_label(idx: int, width: int = 70) -> str:
    """'turn 2: "What if we connect a 45 MW data centre…"' — the number alone
    did not say which answer was meant."""
    question = next((m.get("text") or "" for m in reversed(st.session_state.messages[: idx + 1])
                     if m.get("role") == "user"), "")
    question = " ".join(question.split())
    if len(question) > width:
        question = question[: width - 1].rstrip() + "…"
    return f"turn {_turn_of(idx)}" + (f": “{question}”" if question else "")


def _render_answer_footer(msg: dict, idx: int) -> None:
    """Under an answer in the conversation: one line when the AI agent has
    something to flag about the latest answer, and the decision notes
    recorded here. Everything else of the AI agent lives in its panel."""
    is_last = idx == len(st.session_state.messages) - 1
    panel_open = st.session_state.get("agent_panel_open")
    _render_staleness(msg, idx)
    if is_last and msg.get("attention"):
        where = "the advisor (right) can help" if panel_open else "open the advisor (✨ on the right) to"
        st.caption(f"⚠️ {_attention_hint(msg['attention'])} — {where} decide what to check next.")
    if is_last and not (panel_open and st.session_state.get("agent_mode") == "discuss"):
        # The way from an answer to the advisor, about that answer.
        st.button("💬 Discuss with the advisor", key=f"discuss_{idx}", type="tertiary",
                  disabled=_ui_busy(),
                  help="Open the advisor to think this answer through. It runs no studies.",
                  on_click=lambda: st.session_state.update(agent_panel_open=True, agent_mode="discuss"))
    for disc in _discussions():
        if disc.get("note") is not None and disc.get("closed_at") == idx:
            _render_decision_note(disc["note"])


# Where a pending AI agent call draws its "thinking" indicator: a slot in the
# panel, exactly where its result will appear. The call itself runs at the end
# of the script, after the whole page — the chat box included — has been drawn
# in its waiting state; a request is set by a button's callback, so the run
# that handles it knows from the start that it is busy.
_AGENT_SLOT: dict = {}


def _agent_busy() -> bool:
    """Whether an AI agent call is about to run in this script run."""
    return bool(st.session_state.get("_suggestion_request") or st.session_state.get("_advisor_request"))


def _ui_busy() -> bool:
    """Whether anything is running — a turn or an AI agent call. The panel's
    controls are disabled meanwhile: a click during a run stops it (Streamlit
    interrupts the script), and the work it was doing was lost."""
    return _agent_busy() or st.session_state.get("_pending_turn") is not None


# A request interrupted all the same (a click elsewhere) is kept and resumed on
# the next run, at most this many times, so it can never loop.
_MAX_ATTEMPTS = 2


def _attempt(key: str) -> bool:
    """Count one more attempt at the pending work under `key`; False once it
    has been tried too often (it is then dropped)."""
    n = st.session_state.get(f"_attempts{key}", 0) + 1
    st.session_state[f"_attempts{key}"] = n
    return n <= _MAX_ATTEMPTS


def _panel_suggestions(msg: dict, idx: int) -> None:
    rounds = msg.get("suggestion_rounds") or []
    attention = msg.get("attention") or []
    if st.session_state.get("auto_suggest") and not rounds and not msg.get("auto_suggested"):
        msg["auto_suggested"] = True
        st.session_state._suggestion_request = (idx, "auto")
    pending = st.session_state.get("_suggestion_request")
    if attention and not rounds:
        st.caption(f"⚠️ {_attention_hint(attention)}")
    label = "✨ More ideas" if rounds else "✨ Suggest next steps"
    st.button(label, key=f"ais_{idx}_{len(rounds)}", width="stretch",
              type="primary" if attention and not rounds else "secondary",
              disabled=_ui_busy(),
              help="Ask the advisor how to continue from here. It reasons over the "
                   "conversation; it runs no analysis.",
              on_click=lambda: st.session_state.update(
                  _suggestion_request=(idx, "more" if rounds else "button")))
    if pending:
        _AGENT_SLOT["slot"] = st.empty()      # the new round's place, while it is written
    for r_i, rnd in enumerate(rounds):
        _render_suggestion_round(rnd, idx, r_i, expanded=r_i == len(rounds) - 1)


def _panel_discussion(idx: int) -> None:
    active = _active_discussion()
    k = len(_discussions()) - 1 if active is not None else len(_discussions())
    if active is None or not active.get("thread"):
        st.caption(_PLAN_CAPTION if idx < 0 else _DISCUSS_CAPTION)
    if active is not None:
        _render_thread(active, k, interactive=True)
    _render_discussion_input(active, k, idx)


def _panel_history(last_idx: int) -> None:
    """Suggestions under earlier answers and closed discussions, folded."""
    messages = st.session_state.messages
    earlier = [(i, m) for i, m in enumerate(messages) if i != last_idx and m.get("suggestion_rounds")]
    closed = [(k, d) for k, d in enumerate(_discussions()) if d.get("note") is not None]
    if not (earlier or closed):
        return
    with st.expander("Earlier in this conversation", expanded=False):
        for i, m in earlier:
            st.markdown(f"**Suggestions after {_turn_label(i, 50)}**".replace("$", "\\$"))
            for r_i, rnd in enumerate(m["suggestion_rounds"]):
                _render_suggestion_round(rnd, i, r_i, framed=False)
        for k, d in closed:
            st.markdown(f"**Discussion · decision recorded at {_turn_label(d['closed_at'], 50)}**"
                        .replace("$", "\\$"))
            _render_thread(d, k, interactive=False)
            _render_decision_note(d["note"])


def _render_agent_panel() -> None:
    """The AI agent, beside the conversation: suggestions and discussion about
    the latest answer, with earlier ones folded below. The conversation on the
    left stays the record of questions, studies and answers."""
    messages = st.session_state.messages
    answers = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    c_title, c_hide = st.columns([6, 1], vertical_alignment="center")
    c_title.markdown(":violet[**✨ Advisor**]")
    with c_hide, st.container(horizontal=True, horizontal_alignment="right"):
        st.button("", key="agent_hide", icon=":material/keyboard_double_arrow_right:",
                  disabled=_ui_busy(),
                  help="Hide the advisor — reopen it with ✨ in the same place.",
                  on_click=lambda: st.session_state.update(agent_panel_open=False))
    st.caption("Thinks results through with you and suggests what to run next. Reasoning by an "
               "AI model, not a solver result — it runs no studies; CONDUCTOR does.")
    if not answers:
        # Before the first answer: a discussion only — there is no result to
        # suggest next steps from. The same discussion carries on afterwards.
        st.markdown("**Before you ask** — plan what to study")
        _panel_discussion(-1)
        return
    idx = answers[-1]
    st.markdown(f"**About the latest answer** — {_turn_label(idx)}".replace("$", "\\$"))
    st.session_state.setdefault("agent_mode", "discuss" if _active_discussion() else "suggest")
    mode = st.segmented_control(
        "Advisor", ["suggest", "discuss"], key="agent_mode", label_visibility="collapsed",
        format_func={"suggest": "✨ Suggestions", "discuss": "💬 Discuss"}.get,
        selection_mode="single", disabled=_ui_busy()) or "suggest"
    if mode == "discuss":
        _panel_discussion(idx)
    else:
        _panel_suggestions(messages[idx], idx)
    _panel_history(idx)


# ---------------------------------------------------------------------------
# Data view — the loaded measurements and forecasts, as they are
# (logic in agent/data_view.py; data from /api/data/series)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_series(signature: str, data_source: str, substations: tuple) -> dict | None:
    """One dataset's series. `signature` changes when the loaded data does."""
    try:
        r = httpx.post(f"{BASE_URL}/api/data/series", timeout=HTTP_TIMEOUT,
                       json={"data_source": data_source, "substations": list(substations)})
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_quality(signature: str) -> dict | None:
    """The loaded datasets' quality reports. `signature` changes with the data."""
    try:
        r = httpx.get(f"{BASE_URL}/api/data/quality", timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        return None


_QUALITY_ICONS = {"error": "⛔", "warning": "⚠️", "info": "ℹ️"}


def _follow_measured(subs: tuple) -> None:
    """Switch the voltage-controlled units at `subs` to fixed Q (they then
    follow the measured reactive generation)."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            r = client.post(f"{BASE_URL}/api/generators/follow_measured", json={"substations": list(subs)})
        st.session_state._dq_note = (("Switched to fixed Q: " + ", ".join(r.json().get("switched") or []))
                                     if r.status_code < 400 else r.json().get("detail", r.text))
    except Exception as exc:  # noqa: BLE001
        st.session_state._dq_note = str(exc)
    _fetch_quality.clear()


def _render_data_quality(signature: str) -> None:
    """What the loaded data look like — reactive columns, coverage, anything
    implausible. Nothing here changes the data."""
    reports = _fetch_quality(signature) or {}
    shown = [(label, reports.get(key)) for label, key in (("Measured", "measurements"), ("Forecast", "forecasts"))
             if reports.get(key)]
    if not shown:
        return
    worst = min((f["level"] for _, r in shown for f in r.get("findings") or []),
                key=["error", "warning", "info"].index, default="info")
    title = "Data quality" + {"error": " · problems found", "warning": " · check"}.get(worst, "")
    note = st.session_state.pop("_dq_note", None)
    with st.expander(title, icon=":material/fact_check:", expanded=worst == "error" or bool(note),
                     key="dv_quality"):
        if note:
            st.info(note)
        for label, report in shown:
            st.markdown(f"**{label}** — {report.get('summary', '')}")
            for f in report.get("findings") or []:
                st.markdown(f"{_QUALITY_ICONS.get(f['level'], '')} {f['message']}".replace("$", "\\$"))
                if f["code"] == "reactive_at_voltage_units" and label == "Measured":
                    st.button("Switch these units to follow the data (fixed Q)", key="dq_follow",
                              icon=":material/sync_alt:",
                              help="They then take the measured reactive generation at every tick that has it; "
                                   "the substation's median applies where it is missing. Undo in the System tab.",
                              on_click=_follow_measured, args=(tuple(f.get("substations") or ()),))


def _render_data_view() -> None:
    timeline = _tl.Timeline.from_payload(_fetch_timeline())
    if not timeline.modes():
        st.warning("No data to show — the backend is not responding.")
        return
    signature = "|".join(str(x) for x in (
        st.session_state.get("_network_fingerprint"), timeline.measured_source, timeline.forecast_source,
        len(timeline.measured), timeline.measured[:1], timeline.measured[-1:],
        len(timeline.forecast), timeline.forecast[:1], timeline.forecast[-1:]))

    datasets = [d for d, ticks in (("Measured", timeline.measured), ("Forecast", timeline.forecast)) if ticks]
    if len(datasets) == 2:
        datasets.append("Both")
    base = _fetch_series(signature, "measurements" if timeline.measured else "forecasts", ())
    names = (base or {}).get("available_substations") or []

    c_ds, c_items, c_qty = st.columns([2, 4, 3])
    choice = c_ds.radio("Dataset", datasets, index=len(datasets) - 1, horizontal=True, key="dv_dataset")
    picked = c_items.multiselect("Substations", names, key="dv_items", placeholder="System total",
                                 help="Leave empty for the system total.")
    offered = _dv.quantities(base)
    if st.session_state.get("dv_quantity") not in offered:
        st.session_state.dv_quantity = "consumption"
    if len(offered) > len(_dv.QUANTITIES):
        # With reactive power there are five: a dropdown fits where buttons overflow.
        quantity = c_qty.selectbox("Quantity", list(offered), format_func=offered.get, key="dv_quantity")
    else:
        quantity = c_qty.segmented_control("Quantity", list(offered), format_func=offered.get,
                                           default="consumption", required=True, key="dv_quantity")
    items = list(picked) or [_dv.TOTAL]

    subs = tuple(sorted(picked))
    series = {
        "measured": (_fetch_series(signature, "measurements", subs)
                     if choice in ("Measured", "Both") and timeline.measured else None),
        "forecast": (_fetch_series(signature, "forecasts", subs)
                     if choice in ("Forecast", "Both") and timeline.forecast else None),
    }
    view = st.session_state.get("op_view") or {}
    st.plotly_chart(_polish(_dv.figure(series, items, quantity, marker=view.get("ts"))),
                    width="stretch", theme=None, key="dv_chart")

    facts = []
    for label, payload, ticks in (("Measured", series["measured"], timeline.measured),
                                  ("Forecast", series["forecast"], timeline.forecast)):
        if payload:
            res = _tl.resolution(ticks) or "—"
            facts.append(f"**{label}** {ticks[0][:16]} → {ticks[-1][:16]} · {len(ticks)} ticks · "
                         f"{res} · {payload.get('source') or '—'}"
                         + (f" · shown every {payload['stride']}th tick" if payload.get("stride", 1) > 1 else ""))
    st.caption("  \n".join(facts))

    if choice == "Both":
        for item in items:
            err = _dv.forecast_error(series["measured"], series["forecast"], item, quantity)
            if err:
                st.caption(
                    f"Forecast error, {item}, {_dv.QUANTITIES[quantity].lower()}, over the {err['n']} "
                    f"shared ticks ({err['first'][5:16]} → {err['last'][5:16]}): "
                    f"bias {err['bias']:+.2f} MW · MAE {err['mae']:.2f} MW"
                    + (f" · MAPE {err['mape_pct']:.1f} %" if err["mape_pct"] is not None else ""))

    _render_data_quality(signature)

    with st.expander("Table", icon=":material/table:"):
        columns = {}
        for dataset, payload in series.items():
            for item in items:
                ys = _dv.values(payload, item, quantity)
                if ys:
                    columns[f"{item} — {dataset} (MW)"] = pd.Series(ys, index=payload["timestamps"])
        if columns:
            frame = pd.DataFrame(columns)
            frame.index.name = "timestamp"
            st.dataframe(frame, width="stretch", height=360)
        else:
            st.caption("Nothing to show for this selection.")


# a question can be asked while looking at the diagram.
_chat_tab, _system_tab, _data_tab = st.tabs(["Chat", "System", "Data"])

with _chat_tab:
    # The AI agent starts closed — the conversation is the main work, and the
    # agent is something the operator opens (its launcher is flagged when an
    # answer has concerns). Its column is there from the start, so it can be
    # opened to plan before the first question.
    _panel_open = st.session_state.get("agent_panel_open", False)
    with st.container(key="chat_layout"):
        _col_chat, _col_agent = st.columns([0.63, 0.37] if _panel_open else [0.955, 0.045],
                                           gap="large" if _panel_open else "small")
    if not st.session_state.messages:
        with _col_chat, st.chat_message("assistant", avatar=_AVATARS["assistant"]):
            st.markdown(
                f"Hello! I'm **{_BRAND_NAME}**. I run validated power-system studies on the loaded network — "
                "security, N-1, probabilistic risk, flexibility, hosting capacity, KPIs — and can test changes "
                "to the grid as proposals without touching it.\n\n"
                "- **Ask me** (the box below, **Run**) to run or explain a study — e.g. *Run an N-1 "
                "screen at 14:45*, *What if we add 30 MW at Bus 9?*\n"
                "- **Open the advisor** (✨ on the right) to think things through — e.g. *Is this result "
                "reliable?*, *What should I check before deciding?* It runs nothing itself; studies "
                "it proposes come to me.\n\n"
                "Try one of the **Example queries** in the sidebar, or ask me anything about the active power-system model."
            )
    for _msg_idx, msg in enumerate(st.session_state.messages):
        with _col_chat, st.chat_message(msg["role"], avatar=_AVATARS.get(msg["role"])):
            st.markdown(msg["text"])
            if msg.get("charts"):
                render_charts(msg["charts"])
            if msg.get("trace") is not None:
                # History re-renders on every rerun, so an exception here would
                # not break one panel — it would permanently break the whole
                # conversation view. The audit is worth less than the record of
                # what was said.
                try:
                    _render_trace(msg["trace"], key=f"hist_{_msg_idx}")
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "Could not render the audit panel for message %s.",
                        _msg_idx, exc_info=True)
                    st.caption("⚠️ The audit panel could not be rendered for this answer.")
            if msg["role"] == "assistant":
                # Same rule as the audit panel: a failure here must not take
                # the conversation view down with it.
                try:
                    _render_answer_footer(msg, _msg_idx)
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "Could not render the AI agent notes for message %s.", _msg_idx, exc_info=True)
    if not _panel_open:
        # Collapsed: only the open button, where the hide button was, and in
        # view while the conversation scrolls. Highlighted when the latest
        # answer has something the AI agent would flag.
        _flag = bool((st.session_state.messages or [{}])[-1].get("attention"))
        with _col_agent, st.container(key="ai_panel_closed"):
            # A round launcher in the agent's colour (styled in CSS by its
            # key); filled, with a dot, when there is something to flag.
            st.button("✨",
                      key="agent_open_flag" if _flag else "agent_open", disabled=_ui_busy(),
                      help="Open the advisor" + (" — it has something to flag about the latest answer"
                                                  if _flag else ""),
                      on_click=lambda: st.session_state.update(agent_panel_open=True))
    else:
        with _col_agent, st.container(key="ai_panel"):
            try:
                _render_agent_panel()
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).warning("Could not render the AI agent panel.", exc_info=True)
                st.caption("⚠️ The advisor panel could not be rendered.")


with _system_tab:
    _render_system_view()

with _data_tab:
    try:
        _render_data_view()
    except Exception:  # noqa: BLE001 — the viewer must never take the page down
        logging.getLogger(__name__).warning("Data view failed.", exc_info=True)
        st.caption("⚠️ The data view could not be drawn.")



def _safe_attention(record: dict | None) -> list[str]:
    """`needs_attention`, never allowed to cost the answer it describes."""
    try:
        return _recommender.needs_attention(record)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning("Could not compute the attention reasons.", exc_info=True)
        return []


def _queue_turn() -> None:
    """Send's callback: record the question. The fragment's rerun then draws a
    waiting line instead of the box before the turn starts — a box that stays
    on screen during a turn looks ready to take the next question, and is not."""
    typed = (st.session_state.get(f"chat_box_{st.session_state.get('_box_n', 0)}") or "").strip()
    if typed:
        st.session_state._pending_turn = typed
        st.session_state._turn_redraw = True
        st.session_state.pop("_attempts_turn", None)


def _busy_bar(text: str, who: str = "conductor") -> None:
    """Where the chat box is, while CONDUCTOR (blue) or the AI agent (purple)
    is working: a spinner and one line, centred in a box the chat box's size."""
    with st.container(border=True, key="chat_busy"):
        st.markdown(f"<div class='conductor-busy conductor-busy-{who}'><span class='conductor-spin'></span>"
                    f"<span>{html.escape(text)}</span></div>", unsafe_allow_html=True)


@st.fragment
def _chat_input_fragment():
    """Input box + turn processing, isolated in a fragment.

    Submitting the form triggers a *fragment-scoped* rerun, so the slow agent
    call does not put the conversation history above into rerun-limbo — which is
    what greyed-out / "ghosted" the previous answer for the whole wait. Once the
    reply is ready we fold it into history with a fast full rerun.
    """
    # The in-progress turn renders here, just above the form.
    work = st.container()

    pending_turn = st.session_state.get("_pending_turn")
    if pending_turn is not None:
        if st.session_state.pop("_turn_redraw", False):
            # Sent from this fragment: redraw the whole page first, so the
            # panel's controls are disabled before the turn starts.
            st.rerun(scope="app")
        if not _attempt("_turn"):
            st.session_state.pop("_pending_turn", None)
            st.session_state.pop("_attempts_turn", None)
            st.warning("The question was interrupted twice and was not answered — please send it again.")
        else:
            _busy_bar("CONDUCTOR is working on your question — you can ask the next one when it finishes.")
            _process_turn(pending_turn, work)
            return
    if _agent_busy():
        _busy_bar("The advisor is thinking — CONDUCTOR's chat box is back in a moment.", who="agent")
        return

    # Apply a pending prefill (example query, AI Agent Suggestion) BEFORE the
    # widget renders — as a new widget whose initial value is the text.
    #
    # Assigning the text to the widget's key instead looks equivalent and is
    # not: once the form has been submitted once, `clear_on_submit` leaves the
    # browser holding an empty value, the assignment never reaches the box, and
    # Send submits nothing — the operator had to type something before Send
    # worked. A fresh key per prefill sidesteps that; verified in a real
    # browser, which the headless app tests cannot do.
    st.session_state.setdefault("_box_n", 0)
    st.session_state.setdefault("_box_value", "")
    if st.session_state.get("_box_pending") is not None:
        st.session_state._box_value = st.session_state._box_pending
        st.session_state._box_pending = None
        st.session_state._box_n += 1

    with st.form("chat_form", clear_on_submit=True):
        # Labelled as CONDUCTOR's: the advisor has its own box in its panel,
        # and only this one runs studies.
        # One row sized by its content: the label only as wide as itself.
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown("<span class='conductor-who'>⚡ CONDUCTOR</span>", unsafe_allow_html=True,
                        width="content")
            typed = st.text_input(
                "Ask CONDUCTOR",
                key=f"chat_box_{st.session_state._box_n}",
                value=st.session_state._box_value,
                placeholder="Ask CONDUCTOR to run, compare or explain a study…",
                label_visibility="collapsed",
                width="stretch",
            )
            st.form_submit_button("Run", width=96, type="primary", on_click=_queue_turn,
                                  help="CONDUCTOR runs the studies and answers with solver results.")
    del typed


def _process_turn(effective_input: str, work) -> None:
    """One orchestrator turn for a question the operator sent."""
    # Sent: the next render starts from a fresh, empty box rather than
    # `clear_on_submit` restoring the prefilled text as the widget's default.
    st.session_state._box_value = ""
    st.session_state._box_n += 1

    # Networks are uploaded by client-side JS straight to the backend, so no
    # Python code runs when one is swapped and nothing clears the conversation.
    # Detect it here — the last moment before stale history is sent to the model
    # — rather than trusting any particular upload path to announce itself.
    _swapped = _reset_if_network_changed()

    # On a forecast, or comparing, the simulation clock is not the moment of
    # interest, so the agent is told which data and which time (agent/timeline.py).
    _view = st.session_state.get("op_view") or {}
    _note = _tl.agent_note(_view.get("mode", _tl.MEASURED), _view.get("ts"))
    llm_input = f"{_note}\n\n{effective_input}" if _note else effective_input

    # Before the new message is appended: the outcome refers to suggestions
    # shown under the previous answer.
    try:
        _log_suggestion_outcome(effective_input)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning("Could not log the suggestion outcome.", exc_info=True)

    with work:
        if _swapped:
            st.info(
                f"**Network changed — started a new conversation.**\n\n"
                f"Now analysing **{_swapped}**. Earlier messages referred to a "
                f"different network, so they were cleared rather than mixed into "
                f"this one."
            )

        with st.chat_message("user"):
            st.markdown(effective_input)

        with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
            status_box = st.status("Thinking…", expanded=True)

            def _on_event(event: str, data: dict) -> None:
                with status_box:
                    if event == "llm_call":
                        turn = data.get("turn", 1)
                        st.write("⚡ Calling your LLM Orchestrator…" if turn == 1 else f"⚡ LLM Orchestrator reasoning (turn {turn})…")
                    elif event == "tool_start":
                        label = _TOOL_LABELS.get(data["name"], data["name"].replace("_", " ").title())
                        st.write(f"⚙️ Running **{label}**…")
                    elif event == "tool_done":
                        label = _TOOL_LABELS.get(data["name"], data["name"].replace("_", " ").title())
                        has_error = isinstance(data.get("result"), dict) and "error" in data["result"]
                        st.write(f"⚠️ **{label}** returned an error" if has_error else f"✅ **{label}** complete")
                    elif event == "tool_error":
                        label = _TOOL_LABELS.get(data["name"], data["name"].replace("_", " ").title())
                        st.write(f"❌ **{label}** failed: {data.get('error', '')}")
                    elif event == "tool_rejected":
                        label = _TOOL_LABELS.get(data["name"], data["name"].replace("_", " ").title())
                        st.write(
                            f"🛑 **{label}** not run — invalid parameters: "
                            + "; ".join(data.get("problems", []))
                        )
                    elif event == "integrity_warning":
                        label = _TOOL_LABELS.get(data["name"], data["name"].replace("_", " ").title())
                        st.write(
                            f"⚠️ **{label}** returned an internally inconsistent result: "
                            + "; ".join(data.get("problems", []))
                        )
                    elif event == "reflection":
                        n_g = data.get("n_findings", 0)
                        n_c = data.get("n_conflicts", 0)
                        n_m = data.get("n_gaps", 0)
                        parts = []
                        if n_g:
                            parts.append(f"{n_g} figure(s) not traced to the evidence")
                        if n_c:
                            parts.append(f"{n_c} described against its own record")
                        if n_m:
                            parts.append(f"{n_m} thing(s) the question asked for and the answer omits")
                        if not data.get("triggered"):
                            st.write("🔎 Answer checks: every figure traced and consistent")
                        elif data.get("shown"):
                            st.write(f"🔎 Answer checks: {'; '.join(parts)} — returning to the agent")
                        else:
                            st.write(f"🔎 Answer checks: {'; '.join(parts)} (recorded only)")
                    elif event == "model_loading":
                        st.write(
                            f"⏳ Loading **{data.get('model', 'the model')}** into memory "
                            "and reprocessing the prompt — this can take a few minutes "
                            "when the model has been idle. Later questions are fast."
                        )
                    elif event == "retry":
                        reason = data.get("reason", "Transient error")
                        wait = data.get("wait_s", "?")
                        attempt = data.get("attempt", "?")
                        total = data.get("total", "?")
                        st.write(f"⚠️ {reason} — waiting {wait}s before retry {attempt}/{total}…")

            try:
                final_text, updated_history = run_agent_turn(
                    user_message=llm_input,
                    history=st.session_state.history,
                    # One id per conversation, so its turns group in the log;
                    # without it every turn got a fresh `id(history)`.
                    conversation_id=st.session_state.ui_conversation_id,
                    on_event=_on_event,
                    reflection=_reflection_mode(),
                )
                # Persist history
                st.session_state.history = updated_history

                # Update sidebar timestamp if it changed
                if _tools_module._last_tool_results:
                    for tname, tres in _tools_module._last_tool_results:
                        if tname in ("get_current_timestamp", "advance_timestamp"):
                            new_ts = tres.get("current_timestamp", tres.get("timestamp"))
                            if new_ts:
                                st.session_state.current_ts = new_ts
                                break

                # Point the system view at whatever moment this turn analysed, so
                # the picture describes the answer on screen. A turn that spanned
                # two operating points names none, and the view is left alone.
                turn_point = _network_map.operating_point_of(_tools_module._last_tool_results)
                if turn_point:
                    st.session_state.system_ts = turn_point

                # Capture chart data and store message.
                #
                # A turn can call the same tool twice with the same arguments — the
                # agent re-reads conditions to derive a figure, say — and the second
                # call returns the identical payload. Rendering it again shows the
                # operator the same table twice and reads as two measurements. Only
                # byte-identical repeats are dropped: two calls that differ in
                # arguments or results are genuinely different analyses and both
                # belong on screen.
                charts_this_turn = _dedupe_tool_results(_tools_module._last_tool_results)
                assistant_text = final_text if final_text else "_No text response from agent._"

                # The audit panel is built from the turn record the loop just
                # wrote — the same input the offline graders read, so the panel and
                # the evaluation cannot disagree about a turn. Never allowed to
                # break the answer: a missing panel costs an audit, a raised
                # exception costs the reply.
                turn_trace = None
                turn_record = None
                try:
                    record = _loop_module.session_log[-1] if _loop_module.session_log else None
                    # Verify the record belongs to *this* question. A turn that
                    # aborts before logging leaves the previous turn's record at
                    # the end of the list, and attributing one answer's evidence to
                    # a different question is the worst thing this panel could do —
                    # it would assert provenance for numbers never produced. No
                    # panel is strictly better than a wrong one.
                    if record is not None and record.get("user") == llm_input:
                        turn_record = record
                        turn_trace = _trace.build(record)
                    elif record is not None:
                        logging.getLogger(__name__).warning(
                            "Skipping the audit panel: the newest log record is for "
                            "a different prompt.")
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "Could not build the answer trace.", exc_info=True)

                # The question and its answer are saved together, before
                # anything is drawn: the answer used to be saved after the
                # status box was updated, and a page interrupted at that
                # instant kept the question and lost the answer — the
                # operator saw nothing happen and sent it again.
                st.session_state.messages.append({"role": "user", "text": effective_input, "charts": None})
                st.session_state.pop("_pending_turn", None)
                st.session_state.pop("_attempts_turn", None)
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": assistant_text,
                    "charts": charts_this_turn,
                    "trace": turn_trace,
                    # Kept for the AI Agent Suggestion: the record is what it reads
                    # and what its figures are grounded against.
                    "record": turn_record,
                    "attention": _safe_attention(turn_record),
                    # What its studies ran under, to tell later whether the
                    # settings have changed since (agent/staleness.py).
                    "context": _staleness.stamp(charts_this_turn, _fetch_assumptions()),
                })
                status_box.update(label="Analysis complete", state="complete", expanded=False)
                # The turn just refetched grid constants; if that failed, the
                # answer above was produced without knowing the real network.
                _warn_if_grid_identity_unverified()
            except RuntimeError as exc:
                st.session_state.pop("_pending_turn", None)
                st.session_state.pop("_attempts_turn", None)
                status_box.update(label="Error", state="error", expanded=True)
                with status_box:
                    st.error(str(exc))
                st.stop()

            # Render reply inside the same chat bubble
            st.markdown(assistant_text)
            render_charts(charts_this_turn)
            try:
                _render_trace(turn_trace, key=f"live_{len(st.session_state.messages)}")
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "Could not render the audit panel.", exc_info=True)
                st.caption("⚠️ The audit panel could not be rendered for this answer.")

    # Fold the finished turn into the main conversation history with a full app
    # rerun. It's fast now — the answer is already computed, so nothing blocks it.
    st.rerun(scope="app")


_chat_input_fragment()

# A requested AI agent call runs last, once the page — the panel's thinking
# indicator and the chat box's waiting line included — is on screen; then a
# rerun shows its result in the panel. Outside every guard: `st.rerun` works
# by raising, and a broad `except` would swallow it.
# The request stays until the call has finished: a click during the call stops
# the script, and the next run resumes it (at most _MAX_ATTEMPTS times).
_pending_suggestion = st.session_state.get("_suggestion_request")
_pending_advice = st.session_state.get("_advisor_request")
if _pending_suggestion is not None or _pending_advice is not None:
    if _attempt("_agent"):
        with _AGENT_SLOT.get("slot") or st.container():
            if _pending_suggestion is not None:
                _run_ai_suggestion(*_pending_suggestion)
            else:
                _run_advisor(*_pending_advice)
    st.session_state.pop("_suggestion_request", None)
    st.session_state.pop("_advisor_request", None)
    st.session_state.pop("_attempts_agent", None)
    _AGENT_SLOT.clear()
    st.rerun()
