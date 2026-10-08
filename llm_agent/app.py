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
from agent import timeline as _tl
from agent import data_view as _dv
from agent import trace as _trace
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
from agent.renderers import RENDERER_MAP, render_network_map
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
_BRAND_TAGLINE = "An LLM-Orchestrated Digital Twin for Uncertainty-Aware Distribution Grid Operations"
_BRAND_CAPTION = (
    "Natural-language access to deterministic and uncertainty-aware grid studies, "
    "including probabilistic security assessment, robust corrective dispatch, "
    "flexibility envelopes, and hosting-capacity analysis."
)
_SIDEBAR_BRIEF = (
    "Deterministic RSA, probabilistic risk, robust dispatch, flexibility envelopes, "
    "and hosting-capacity studies."
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
    .conductor-hero-compact .conductor-brand img { width: 44px !important; }
    .conductor-hero-compact .conductor-brand-title { font-size: 1.35rem; }
    .conductor-hero-compact .conductor-brand-tagline,
    .conductor-hero-compact .conductor-brand-support { display: none; }
    .conductor-hero-compact { padding: 0; margin-bottom: 0; }
    /* AI Agent Suggestion: a different accent from tool output, because this
       is model reasoning, not a solver result. */
    .conductor-ai-comment {
        background: rgba(107, 79, 216, 0.06);
        border-left: 3px solid #8E5CD9;
        border-radius: 6px;
        padding: 0.55rem 0.85rem;
        margin: 0.1rem 0 0.7rem 0;
        color: var(--cd-ink);
        line-height: 1.55;
    }
    .conductor-ai-comment-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #5b3fc4;
        margin-bottom: 0.2rem;
    }
    .conductor-ai-badge {
        display: inline-block;
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
                Ask about deterministic security, N-1 contingencies, probabilistic risk, robust dispatch,
                flexibility envelopes, hosting capacity, KPIs, and more. Charts appear automatically after each response.
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
                    width: 30px;
                    height: auto;
                    display: block;
                    filter: saturate(1.05);
                }
                #${brandId} .cb-name {
                    display: block;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    font-size: 1rem;
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
                    height: 1.1rem;
                    background: rgba(46,49,64,0.18);
                    margin: 0 0.2rem;
                    flex-shrink: 0;
                }
                .cb-upload-btn {
                    background: none;
                    border: none;
                    cursor: pointer;
                    font-size: 0.8rem;
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
                parentDoc.querySelectorAll('.cb-panel').forEach(p => {
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
            ['cb-net-panel', 'cb-data-panel', 'cb-info-panel', 'cb-adm-overlay', 'cb-adm-drawer'].forEach(id => {
                const el = parentDoc.getElementById(id);
                if (el) el.remove();
            });
            if (typeof parentWindow.__conductorCloseAll === 'function') {
                parentDoc.removeEventListener('click', parentWindow.__conductorCloseAll);
            }

            const netPanel = parentDoc.createElement('div');
            netPanel.id = 'cb-net-panel';
            netPanel.className = 'cb-panel';
            netPanel.innerHTML = `
                <div class="cb-panel-title">Upload Network</div>
                <input type="file" id="cb-net-file" accept=".m,.json,.xlsx,.uct" />
                <label class="cb-check">
                    <input type="checkbox" id="cb-net-convert" checked />
                    Convert generators to controllable sgen (enables OPF &amp; flexibility tools)
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
            dataPanel.className = 'cb-panel';
            dataPanel.innerHTML = `
                <div class="cb-panel-title">Upload Time-series Data (.csv)</div>
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
                <div class="cb-fmt-guide">
                    <div class="cb-fmt-desc" style="margin-bottom:0.4rem">Both datasets use the <strong>same CSV format</strong>. Measurements replace what the clock runs on now; forecasts are read-only look-ahead data for planning tools.</div>
                    <div class="cb-fmt-title">Required columns</div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">timestamp</span><span class="cb-fmt-desc">Datetime string parseable by pandas — e.g. <code>2024-01-01 00:15:00</code>. Any regular interval; 15 min recommended.</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">substation_name</span><span class="cb-fmt-desc">Bus name from the loaded network. Fuzzy-matched — partial or prefix names are fine.</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">production_mw</span><span class="cb-fmt-desc">Total generator output at that bus in MW. Use <code>0.0</code> for load-only buses.</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">consumption_mw</span><span class="cb-fmt-desc">Total load at that bus in MW. Use <code>0.0</code> for generation-only buses.</span></div>
                    <div class="cb-fmt-title" style="margin-top:0.45rem">Example</div>
                    <pre class="cb-csv-example">timestamp,substation_name,production_mw,consumption_mw
2024-01-01 00:00:00,Bus 1,2.5,1.2
2024-01-01 00:00:00,Bus 2,0.0,3.4
2024-01-01 00:15:00,Bus 1,2.3,1.4
2024-01-01 00:15:00,Bus 2,0.0,3.6</pre>
                    <div class="cb-fmt-desc" style="margin-top:0.3rem">Long format — one row per timestamp × bus. Comma-separated UTF-8; Excel BOM exports accepted.</div>
                </div>
            `;
            parentDoc.body.appendChild(dataPanel);

            const infoPanel = parentDoc.createElement('div');
            infoPanel.id = 'cb-info-panel';
            infoPanel.className = 'cb-panel';
            infoPanel.innerHTML = `
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

            const positionUnder = (panel, btn) => {
                const r = btn.getBoundingClientRect();
                panel.style.left = r.left + 'px';
            };

            const closeAll = () => {
                netPanel.classList.remove('is-open');
                dataPanel.classList.remove('is-open');
                infoPanel.classList.remove('is-open');
            };
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
    st.markdown("### Choose how to run the assistant")

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
        st.radio(
            "Backend",
            _labels,
            index=_index,
            horizontal=True,
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
]


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


def _render_operating_point() -> None:
    ss = st.session_state
    timeline = _tl.Timeline.from_payload(_fetch_timeline())
    modes = timeline.modes()
    if not modes:
        st.warning("Timeline unavailable — the backend is not responding.")
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


with st.sidebar:
    # The brand is already in the page header and the hero; a third copy here
    # pushed the controls an operator uses — the clock first — below the fold.
    # The most used action first.
    if st.button("New conversation", icon=":material/add_comment:", width="stretch",
                 type="secondary", key="new_conversation"):
        _reset_conversation_state()
        st.rerun()

    st.markdown("<div class='conductor-sidebar-label'>Operating point</div>", unsafe_allow_html=True)
    with st.container(border=True, key="op_card"):
        _render_operating_point()

    st.divider()
    st.markdown("<div class='conductor-sidebar-label'>Settings</div>", unsafe_allow_html=True)

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

    # Answer checks. One control, not one per check: an operator choosing how
    # much the agent second-guesses itself is making a single decision, and a
    # switch per check would turn that into a configuration exercise. Which
    # checks run is ours to decide; the session log still records which one
    # produced each finding.
    #
    # Off by default — it is the configuration every reported result was
    # produced under, and the other two levels are experiments.
    _REFLECT_LABELS = {
        "off": "Off",
        "warn": "Check silently",
        "inform": "Show the agent",
    }
    _reflect_now = _REFLECT_LABELS.get(st.session_state.get("reflection_mode", "off"), "Off")
    with st.expander(f"Answer checks · {_reflect_now}", icon=":material/fact_check:",
                     key="sb_checks"):
        st.radio(
            "Before answering",
            list(_REFLECT_LABELS),
            format_func=lambda m: _REFLECT_LABELS[m],
            key="reflection_mode",
            help=(
                "Two checks run over the draft answer: every figure is traced "
                "back to the tool result that produced it, and each figure is "
                "compared against the fields beside it in that result — so a "
                "movement the payload marks undeliverable is not reported as an "
                "action. A third checks the answer still contains what the "
                "question asked for.\n\n"
                "**Off** grades afterwards only. **Check silently** records what "
                "the checks found without changing the answer. **Show the agent** "
                "hands the findings back before the answer is delivered — the "
                "agent may keep it, revise it, or call another tool."
            ),
        )

    # AI Agent Suggestion. On demand by default: suggestions nobody asked for
    # are more to read, which is the opposite of what they are for.
    _suggest_now = "Auto" if st.session_state.get("auto_suggest") else "On click"
    with st.expander(f"AI Agent Suggestion · {_suggest_now}", icon=":material/auto_awesome:",
                     key="sb_suggest"):
        st.toggle(
            "Suggest after every answer",
            key="auto_suggest",
            value=False,
            help=(
                "Off: a ✨ AI Agent Suggestion button appears under the latest "
                "answer, and nothing runs until you click it. On: suggestions "
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
        st.caption("Click to drop into the chat box — edit if you like, then send.")
        examples = st.container(key="examples_list")
        for q_i, (query, q_icon) in enumerate(EXAMPLE_QUERIES):
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
def _fetch_topology(_fingerprint: str) -> dict | None:
    """The live network's shape. Keyed on the fingerprint, so an upload refetches."""
    try:
        response = httpx.get(f"{BASE_URL}/api/network/topology", timeout=HTTP_TIMEOUT)
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


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_state(timestamp: str, vm_lower: float, vm_upper: float,
                 fingerprint: str = "") -> tuple[dict, dict]:
    """RSA and snapshot at one moment, cached per (moment, limits, network).

    Uncached, every rerun — any click anywhere in the app, every suggestion,
    every finished turn — ran a full assessment because Streamlit tabs are not
    lazy. `fingerprint` is part of the key only so a network swap is never
    answered from the previous network's cache.
    """
    body = {"data_source": "measurements", "timestamp": timestamp,
            "vm_lower_pu": vm_lower, "vm_upper_pu": vm_upper}
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        rsa = client.post(f"{BASE_URL}/api/grid/rsa", json=body)
        rsa.raise_for_status()
        snapshot = client.post(f"{BASE_URL}/api/network/snapshot",
                               json={"data_source": "measurements", "timestamp": timestamp})
        snapshot.raise_for_status()
    return rsa.json(), snapshot.json()


def _system_operating_point() -> str:
    """The moment the system view describes.

    The simulation clock by default, and whatever the conversation last
    settled on when a turn ran at an explicit timestamp — so the picture
    follows the answer on screen instead of drifting away from it.
    """
    return st.session_state.get("system_ts") or st.session_state.current_ts


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
    payload = _fetch_topology(fingerprint)
    if not payload:
        st.error(f"Could not read the network topology from `{BASE_URL}`.")
        return

    counts = payload.get("counts") or {}
    columns = st.columns(5)
    for column, (label, value) in zip(columns, (
        ("Buses", counts.get("buses", "—")),
        ("Lines", counts.get("lines", "—")),
        ("Transformers", counts.get("trafos", "—")),
        ("Loads", counts.get("loads", "—")),
        ("Generators", counts.get("generators", "—")),
    )):
        column.metric(label, value)

    levels = ", ".join(f"{v:g} kV" for v in payload.get("voltage_levels") or [])
    st.caption(f"**{payload.get('name', 'network')}** · {levels or 'unknown voltage levels'}")

    network, positions = _layout_for(fingerprint, payload)
    # The cached objects are shared between reruns, so state is joined onto a
    # copy: mutating the cache would leave last hour's voltages on the map.
    network = copy.deepcopy(network)

    timestamp = _system_operating_point()
    following = st.session_state.get("system_ts") is None
    limits = grid_limits()
    problems: list[str] = []
    violations: set[str] = set()

    try:
        rsa, snapshot = _fetch_state(timestamp, limits.vm_lower, limits.vm_upper,
                                     st.session_state.get("_network_fingerprint") or "")
        problems += _network_map.apply_assessment(network, rsa)
        problems += _network_map.apply_conditions(network, snapshot)
        violations = _network_map.violated_elements(rsa)
        subtitle = (f"{rsa.get('timestamp', timestamp)} · "
                    f"{rsa.get('total_violations', 0)} violation(s) · "
                    f"band {limits.vm_lower}–{limits.vm_upper} pu")
    except Exception as exc:  # noqa: BLE001
        subtitle = "topology only — the operating point could not be read"
        problems.append(str(exc))

    st.plotly_chart(
        render_network_map(network, positions, vm_lower=limits.vm_lower,
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

_AI_SUGGESTION_CAPTION = (
    "Reasoning by an AI model — not a solver result. Choose one to place it in "
    "the chat box, edit it if you like, then send."
)


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
                        _tl.COMPARE: "measurements and forecasts"}.get(mode, "measurements"),
        "viewing_timestamp": view.get("ts") if mode != _tl.MEASURED else None,
        "default_voltage_band_pu": [gc.get("vm_lower"), gc.get("vm_upper")],
        "max_loading_pct": gc.get("max_loading_pct"),
        "substations": names[:40] + ([f"… and {len(names) - 40} more"] if len(names) > 40 else []),
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
    with st.spinner("The AI agent is considering how to continue…"):
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


def _render_suggestion_round(rnd: dict, idx: int, r_i: int, expanded: bool = True) -> None:
    # Collapsible, so suggestions the operator has read or does not want can
    # be folded away; open under the latest answer, folded under older ones.
    with st.expander(":violet[**✨ AI Agent Suggestion**]", expanded=expanded):
        st.caption(_AI_SUGGESTION_CAPTION)
        if rnd.get("error"):
            st.caption("The AI agent could not produce suggestions this time. "
                       "Try again in a moment.")
            return
        if rnd.get("comment"):
            # Its reading of what happened, above the ways to continue.
            text = html.escape(rnd["comment"]).replace("$", "&#36;")
            st.markdown(
                "<div class='conductor-ai-comment'>"
                "<div class='conductor-ai-comment-label'>AI Agent Comment</div>"
                f"{text}</div>",
                unsafe_allow_html=True,
            )
            if rnd.get("comment_unverified"):
                st.caption("⚠️ Not found in this conversation's results: "
                           + ", ".join(rnd["comment_unverified"]))
        if not rnd.get("suggestions"):
            st.caption(rnd.get("note") or "The AI agent had nothing new to suggest.")
            return
        for s_i, sug in enumerate(rnd["suggestions"]):
            c_text, c_btn = st.columns([6, 2])
            with c_text:
                label = _recommender.ANGLE_LABELS.get(sug["angle"], sug["angle"])
                st.markdown(
                    f"<span class='conductor-ai-badge'>{html.escape(label)}</span>"
                    f"<strong>{html.escape(sug['title'])}</strong>",
                    unsafe_allow_html=True,
                )
                if sug.get("why"):
                    # `$` would otherwise be read as LaTeX.
                    st.caption(sug["why"].replace("$", "\\$"))
                if sug.get("unverified"):
                    st.caption("⚠️ Not found in this conversation's results: "
                               + ", ".join(sug["unverified"]))
            # The full prompt sits behind an ⓘ next to "Use" — a popover, not
            # an expander, because expanders cannot be nested.
            with c_btn, st.container(horizontal=True, horizontal_alignment="right",
                                     vertical_alignment="center"):
                with st.popover("", icon=":material/info:", type="tertiary",
                                help="Show the full prompt"):
                    st.caption(sug["prompt"].replace("$", "\\$"))
                if st.button("Use", key=f"ais_use_{idx}_{r_i}_{s_i}",
                             help="Place this prompt in the chat box"):
                    # Same path as the example queries: into the box, not sent.
                    st.session_state._box_pending = sug["prompt"]
                    st.session_state._suggestion_used = {
                        "round_id": rnd["id"], "index": s_i, "prompt": sug["prompt"],
                    }


def _render_ai_suggestion(msg: dict, idx: int) -> str | None:
    """The suggestion button and panels under an assistant message.

    The button appears only under the latest answer; earlier rounds stay
    visible under the answer they followed. Returns how a new round was
    requested ("button", "auto", "more"), or None — the caller runs it, outside
    the error guard that protects the conversation view.
    """
    rounds = msg.get("suggestion_rounds") or []
    is_last = idx == len(st.session_state.messages) - 1
    trigger = None

    if is_last and not rounds:
        if st.session_state.get("auto_suggest") and not msg.get("auto_suggested"):
            msg["auto_suggested"] = True
            trigger = "auto"
        else:
            attention = msg.get("attention") or []
            if attention:
                # Decided in code from the result's own verdict fields — costs
                # nothing until clicked.
                first = attention[0].split(": ", 1)[-1]
                st.caption(f"⚠️ This result has issues ({first}) — an AI Agent "
                           "Suggestion can help decide what to check next.")
            if st.button("✨ AI Agent Suggestion", key=f"ais_{idx}",
                         type="primary" if attention else "secondary",
                         help="Ask a second AI agent how to continue from here. "
                              "It reasons over the conversation; it runs no analysis."):
                trigger = "button"

    for r_i, rnd in enumerate(rounds):
        _render_suggestion_round(rnd, idx, r_i, expanded=is_last)

    if is_last and rounds and trigger is None:
        if st.button("✨ More ideas", key=f"ais_more_{idx}"):
            trigger = "more"

    return trigger


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
    quantity = c_qty.segmented_control("Quantity", list(_dv.QUANTITIES), format_func=_dv.QUANTITIES.get,
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
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
            st.markdown(
                f"Hello! I'm **{_BRAND_NAME}**, an LLM-orchestrated digital twin for uncertainty-aware distribution grid operations. "
                "I can help you analyze deterministic security, run N-1 contingency studies, quantify probabilistic risk, "
                "optimize robust corrective dispatch, and evaluate flexibility, hosting-capacity, and KPI studies.\n\n"
                "Try one of the **Example queries** in the sidebar, or ask me anything about the active power-system model."
            )
    _suggestion_request: tuple[int, str] | None = None
    for _msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=_AVATARS.get(msg["role"])):
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
                    _trigger = _render_ai_suggestion(msg, _msg_idx)
                    if _trigger:
                        _suggestion_request = (_msg_idx, _trigger)
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "Could not render the AI Agent Suggestion for message %s.",
                        _msg_idx, exc_info=True)

    # Run a requested suggestion round after the history is drawn, then rerun
    # so it appears under its answer. Outside the guard above: `st.rerun`
    # works by raising, and a broad `except` would swallow it.
    if _suggestion_request is not None:
        _run_ai_suggestion(*_suggestion_request)
        st.rerun()

with _system_tab:
    _render_system_view()

with _data_tab:
    try:
        _render_data_view()
    except Exception:  # noqa: BLE001 — the viewer must never take the page down
        logging.getLogger(__name__).warning("Data view failed.", exc_info=True)
        st.caption("⚠️ The data view could not be drawn.")



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
        fc_in, fc_send = st.columns([8, 1])
        with fc_in:
            typed = st.text_input(
                "Ask about the grid…",
                key=f"chat_box_{st.session_state._box_n}",
                value=st.session_state._box_value,
                placeholder="Ask about the grid…",
                label_visibility="collapsed",
            )
        with fc_send:
            send = st.form_submit_button("Send", width="stretch", type="primary")

    if not (send and typed and typed.strip()):
        return

    effective_input = typed.strip()
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

    st.session_state.messages.append({"role": "user", "text": effective_input, "charts": None})

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
                    reflection=st.session_state.get("reflection_mode", "off"),
                )
                status_box.update(label="Analysis complete", state="complete", expanded=False)
                # The turn just refetched grid constants; if that failed, the
                # answer above was produced without knowing the real network.
                _warn_if_grid_identity_unverified()
            except RuntimeError as exc:
                status_box.update(label="Error", state="error", expanded=True)
                with status_box:
                    st.error(str(exc))
                st.stop()

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

            st.session_state.messages.append({
                "role": "assistant",
                "text": assistant_text,
                "charts": charts_this_turn,
                "trace": turn_trace,
                # Kept for the AI Agent Suggestion: the record is what it reads
                # and what its figures are grounded against.
                "record": turn_record,
                "attention": _recommender.needs_attention(turn_record),
            })

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
