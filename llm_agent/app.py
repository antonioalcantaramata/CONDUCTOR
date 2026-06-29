"""
app.py — Streamlit chat interface for CONDUCTOR.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from agent.config import BASE_URL, GEMINI_MODEL, HTTP_TIMEOUT
from agent.loop import run_agent_turn
from agent.renderers import RENDERER_MAP
from agent import tools as _tools_module
from agent.tools import get_current_timestamp

_APP_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _APP_DIR.parent
_DATA_FILES_DIR = _REPO_ROOT / "data_files"
_SYSTEMS_DIR = _REPO_ROOT / "systems"
_LOGO_PATH = _APP_DIR / "assets" / "conductor_logo.png"
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
    page_icon=str(_LOGO_PATH),
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


def _render_main_hero() -> None:
    logo_uri = _image_as_data_uri(_LOGO_PATH)
    st.markdown(
        f"""
        <div id="conductor-hero" class="conductor-hero-shell">
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
    components.html(
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
                    background: #2a72e8;
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
                .cb-fmt-tag { font-family: monospace; font-size: 0.73rem; font-weight: 700; color: #2a72e8; background: rgba(42,114,232,0.09); padding: 0.05rem 0.32rem; border-radius: 4px; flex-shrink: 0; margin-top: 0.08rem; }
                .cb-fmt-desc { font-size: 0.74rem; color: #5c6272; line-height: 1.4; }
                .cb-fmt-desc code { font-family: monospace; font-size: 0.7rem; background: rgba(46,49,64,0.07); padding: 0.02rem 0.25rem; border-radius: 3px; }
                .cb-csv-example { font-family: monospace; font-size: 0.67rem; background: rgba(46,49,64,0.06); border-radius: 4px; padding: 0.4rem 0.5rem; margin: 0.1rem 0 0; color: #2e3140; overflow-x: auto; line-height: 1.5; white-space: pre; }
                .cb-warn-box { background: #fff8e1; border: 1px solid #f0c040; border-radius: 5px; padding: 0.45rem 0.55rem; font-size: 0.75rem; color: #7a5a00; line-height: 1.5; }
                .cb-warn-box strong { color: #5a3e00; }
                .cb-reload-btn { margin-top: 0.45rem; width: 100%; padding: 0.35rem 0; font-size: 0.78rem; font-weight: 600; background: #2a72e8; color: #fff; border: none; border-radius: 5px; cursor: pointer; }
                .cb-reload-btn:hover { background: #1d5fd4; }
                .cb-dl-btn { background: #fff; color: #2a72e8; border: 1.5px solid #2a72e8; }
                .cb-dl-btn:hover { background: #f0f7ff; }
                .cb-bus-chips { display: flex; flex-wrap: wrap; gap: 0.25rem; max-height: 7rem; overflow-y: auto; padding: 0.1rem 0; }
                .cb-bus-chip { font-family: monospace; font-size: 0.67rem; background: rgba(42,114,232,0.09); color: #2a72e8; border-radius: 3px; padding: 0.05rem 0.28rem; white-space: nowrap; }
                .cb-success-box { background: #f0f7ff; border: 1px solid #9dc8f5; border-radius: 5px; padding: 0.45rem 0.55rem; font-size: 0.75rem; color: #1a3a5c; line-height: 1.5; }
                .cb-ds-toggle { display: flex; gap: 0.25rem; background: rgba(46,49,64,0.07); border-radius: 7px; padding: 0.2rem; margin-bottom: 0.55rem; }
                .cb-ds-tab { flex: 1; border: none; background: transparent; color: #5a5f6c; font-size: 0.78rem; font-weight: 600; padding: 0.32rem 0; border-radius: 5px; cursor: pointer; transition: background 0.12s, color 0.12s; }
                .cb-ds-tab:hover { color: #2e3140; }
                .cb-ds-tab-active { background: #fff; color: #2a72e8; box-shadow: 0 1px 2px rgba(46,49,64,0.15); }
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
                brand.setAttribute('aria-hidden', 'true');
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
                    <span class="cb-info-value">GEMINI_MODEL_PLACEHOLDER</span>
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
        """.replace("DATA_URI_PLACEHOLDER", logo_uri).replace("BRAND_NAME_PLACEHOLDER", _BRAND_NAME).replace("BASE_URL_PLACEHOLDER", BASE_URL).replace("GEMINI_MODEL_PLACEHOLDER", GEMINI_MODEL),
        height=0,
    )


def _has_api_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


if not _has_api_key():
    _render_brand_block(f"{_BRAND_NAME} Setup", _BRAND_TAGLINE, image_width=120)
    st.caption(_BRAND_CAPTION)
    st.markdown(
        """
        ### A Gemini API key is required to run the assistant.

        The **free tier** at Google AI Studio is sufficient — no payment needed.

        **How to get a key (takes ~2 minutes):**
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
        submitted = st.form_submit_button("Save & Start", type="primary", use_container_width=True)

    if submitted:
        key = key_input.strip()
        if not key:
            st.error("Please paste a valid API key before continuing.")
        else:
            # Write / update the .env file
            if _ENV_PATH.exists():
                lines = _ENV_PATH.read_text(encoding="utf-8").splitlines()
                new_lines, updated = [], False
                for line in lines:
                    if line.startswith("GEMINI_API_KEY="):
                        new_lines.append(f"GEMINI_API_KEY={key}")
                        updated = True
                    else:
                        new_lines.append(line)
                if not updated:
                    new_lines.append(f"GEMINI_API_KEY={key}")
                _ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            else:
                _ENV_PATH.write_text(f"GEMINI_API_KEY={key}\n", encoding="utf-8")

            # Inject into the current process so the lazy client picks it up immediately
            os.environ["GEMINI_API_KEY"] = key
            st.success("API key saved! Starting the assistant…")
            st.rerun()

    st.stop()  # Do not render the rest of the app until setup is complete

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "current_ts" not in st.session_state:
    ts_result = get_current_timestamp()
    st.session_state.current_ts = ts_result.get(
        "current_timestamp", ts_result.get("timestamp", "—")
    )

# Display messages: [{"role": "user"|"assistant", "text": str, "charts": list|None}]
# Separate from `history` (which is the Gemini proto history).
# Charts are stored here so they survive re-runs.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES: list[str] = [
    "Is the grid secure at the current timestamp?",
    "Find the worst-case timestamp in this week's measurements.",
    "Will the grid stay secure across the forecast horizon?",
    "Which single N-1 outage is the most critical right now?",
    "Run a probabilistic risk assessment under load and generation uncertainty.",
    "Find a robust corrective dispatch to clear any violations.",
    "What is the hosting capacity at the most heavily loaded bus?",
    "Evaluate the flexibility KPIs for the current operating point.",
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


def _reset_conversation_state() -> None:
    """Clear chat memory so the next prompt starts a fresh LLM conversation."""
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.pop("chat_box", None)
    st.session_state.pop("_box_pending", None)
    st.session_state.pop("viewing_forecast_ts", None)
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


def render_charts(tool_results: list) -> None:
    """
    Render Plotly charts for a list of (tool_name, result) tuples.
    Works both for live results and stored results replayed from session state.
    """
    for chart_idx, (tool_name, result) in enumerate(tool_results):
        renderer = RENDERER_MAP.get(tool_name)
        if renderer is None:
            continue
        if "error" in result:
            continue

        figs = renderer(result)

        if isinstance(figs, list) and figs and isinstance(figs[0], list):
            # Multi-row layout: each inner list is one row of figures
            for row_idx, row_figs in enumerate(figs):
                cols = st.columns(len(row_figs))
                for fig_idx, (col, fig) in enumerate(zip(cols, row_figs)):
                    with col:
                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"chart_{id(tool_results)}_{chart_idx}_{row_idx}_{fig_idx}")
        elif isinstance(figs, list):
            # Single-row layout: all figures in one row of columns
            cols = st.columns(len(figs))
            for fig_idx, (col, fig) in enumerate(zip(cols, figs)):
                with col:
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"chart_{id(tool_results)}_{chart_idx}_{fig_idx}")
        else:
            st.plotly_chart(figs, use_container_width=True,
                            key=f"chart_{id(tool_results)}_{chart_idx}")


# ---------------------------------------------------------------------------
# Chart injection helper
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(str(_LOGO_PATH), use_container_width=True)
    st.markdown(f"### {_BRAND_NAME}")
    st.caption(_BRAND_TAGLINE)
    st.markdown(f'<div class="conductor-sidebar-copy">{_SIDEBAR_BRIEF}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    if st.button("Start New Conversation", use_container_width=True, type="secondary"):
        _reset_conversation_state()
        st.rerun()

    if "show_manage_uploads" not in st.session_state:
        st.session_state.show_manage_uploads = False
    if st.button("Manage uploaded data", use_container_width=True):
        st.session_state.show_manage_uploads = not st.session_state.show_manage_uploads

    if st.session_state.show_manage_uploads:
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

                if st.button("Delete selected", type="secondary", use_container_width=True):
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
    st.markdown("<div class='conductor-sidebar-label'>Workspace</div>", unsafe_allow_html=True)

    # ── Simulation clock — single timeline spanning measurements + forecast ─
    _timeline = _fetch_timeline()
    _meas = _timeline.get("timestamps") or []
    _fcts = _timeline.get("forecast_timestamps") or []
    _meas_src = _timeline.get("measurements_source", "synthetic")

    with st.expander("🕐 Simulation clock", expanded=True):
        if not _meas:
            st.info(st.session_state.current_ts)
        else:
            # One timeline = the time-sorted union of measurement + forecast ticks.
            # Sorting (rather than meas+fcts) keeps the order correct even if the two
            # series are NOT contiguous — a gap, or a forecast period that doesn't sit
            # right after the measurements. Duplicates (a timestamp present in both)
            # resolve to "measurement" since that is the actual/live data.
            _meas_set = set(_meas)
            _combined = sorted(_meas_set | set(_fcts))
            backend_ts = _timeline.get("current_timestamp") or st.session_state.current_ts

            # Infer the data's time resolution from consecutive ticks, so the step
            # buttons are labelled with the real interval (15 min, 1 h, …).
            def _infer_step_minutes(seq: list) -> int:
                if len(seq) < 2:
                    return 15
                try:
                    a = datetime.strptime(seq[0], "%Y-%m-%d %H:%M:%S")
                    b = datetime.strptime(seq[1], "%Y-%m-%d %H:%M:%S")
                    return max(1, int((b - a).total_seconds() // 60))
                except Exception:
                    return 15

            _step_min = _infer_step_minutes(_meas)
            if _step_min % 1440 == 0:
                _step_label = f"{_step_min // 1440} day"
            elif _step_min % 60 == 0:
                _step_label = f"{_step_min // 60} h"
            else:
                _step_label = f"{_step_min} min"

            # Sync rules: follow a tool-driven clock move only while the cursor is
            # tracking the clock (in the measurement half); otherwise leave it where
            # the user dragged it.
            cur = st.session_state.get("ts_scrubber")
            if (cur is None) or (cur not in _combined):
                st.session_state.ts_scrubber = backend_ts if backend_ts in _combined else _combined[0]
            elif (backend_ts != st.session_state.get("_last_backend_ts")
                  and cur in _meas_set and backend_ts in _combined):
                st.session_state.ts_scrubber = backend_ts
            st.session_state._last_backend_ts = backend_ts

            def _shift_clock(n: int):
                i = _combined.index(st.session_state.ts_scrubber)
                st.session_state.ts_scrubber = _combined[max(0, min(len(_combined) - 1, i + n))]

            pos_ts = st.session_state.ts_scrubber
            ci = _combined.index(pos_ts)
            is_meas = pos_ts in _meas_set
            # When the cursor sits on a forecast tick the clock can't follow it
            # (it's the future / query-only). Remember it so the chat handler can
            # tell the LLM to analyse that forecast timestamp instead of the clock.
            st.session_state.viewing_forecast_ts = None if is_meas else pos_ts
            try:
                _wd = datetime.strptime(pos_ts, "%Y-%m-%d %H:%M:%S").strftime("%a")
            except ValueError:
                _wd = ""
            if is_meas:
                _mi = _meas.index(pos_ts) + 1
                _sub = f"clock · tick {_mi} / {len(_meas)} · measurements ({_meas_src})"
            else:
                _fi = _fcts.index(pos_ts) + 1 if pos_ts in _fcts else ci + 1
                _sub = (f"forecast · point {_fi} / {len(_fcts)} · query-only"
                        f" · clock at {backend_ts[5:16]}")
            st.markdown(
                f"<div style='font-size:0.95rem;font-weight:600;color:#2e3140'>{_wd} {pos_ts}</div>"
                f"<div style='font-size:0.76rem;color:#7a7f8c;margin-bottom:0.3rem'>{_sub}</div>",
                unsafe_allow_html=True,
            )

            # Tint the slider to match the region: blue on measurements, purple on
            # forecast (same colours as the legend dots). Streamlit has no native
            # per-state slider colour, so this targets the BaseWeb slider internals
            # — the thumb + value label are reliable; the track fill is best-effort.
            _bar = "#2a72e8" if is_meas else "#b07cd6"
            st.markdown(
                "<style>"
                f"section[data-testid='stSidebar'] div[data-baseweb='slider'] [role='slider']"
                f"{{background-color:{_bar} !important;border-color:{_bar} !important}}"
                f"section[data-testid='stSidebar'] div[data-baseweb='slider'] [data-testid='stSliderThumbValue']"
                f"{{color:{_bar} !important}}"
                f"section[data-testid='stSidebar'] div[data-baseweb='slider'] > div > div > div:first-child"
                f"{{background:{_bar} !important}}"
                "</style>",
                unsafe_allow_html=True,
            )
            st.select_slider(
                "Timeline",
                options=_combined,
                key="ts_scrubber",
                format_func=lambda t: t[5:16],  # MM-DD HH:MM
                label_visibility="collapsed",
            )
            if _fcts:
                st.markdown(
                    "<div style='display:flex;justify-content:space-between;font-size:0.7rem;"
                    "color:#7a7f8c;margin:-0.2rem 0 0.5rem'>"
                    "<span><span style='color:#2a72e8'>●</span> measurements "
                    f"{_meas[0][5:10]}–{_meas[-1][5:10]}</span>"
                    "<span><span style='color:#b07cd6'>●</span> forecast "
                    f"{_fcts[0][5:10]}–{_fcts[-1][5:10]}</span></div>",
                    unsafe_allow_html=True,
                )

            bcol = st.columns(2)
            bcol[0].button(f"− {_step_label}", on_click=_shift_clock, args=(-1,),
                           use_container_width=True, key="b_prev")
            bcol[1].button(f"+ {_step_label}", on_click=_shift_clock, args=(1,),
                           use_container_width=True, key="b_next")

            # Push to the backend clock only when the cursor is in the measurement
            # half — the clock cannot occupy a forecast (future) timestamp.
            if pos_ts in _meas_set and pos_ts != backend_ts:
                _jump_to_timestamp(pos_ts)
                st.session_state.current_ts = pos_ts
                # Record that *we* moved the backend, so next run's follow-logic
                # doesn't mistake our own push for a tool-driven move and snap back.
                st.session_state._last_backend_ts = pos_ts

    st.divider()
    st.markdown("<div class='conductor-sidebar-label'>Prompt Starters</div>", unsafe_allow_html=True)

    # ── Example queries ──────────────────────────────────────────────────
    with st.expander("💡 Example queries", expanded=True):
        st.caption("Click to drop into the chat box — edit if you like, then send.")
        for query in EXAMPLE_QUERIES:
            if st.button(query, use_container_width=True, key=f"eq_{hash(query)}"):
                # Copy into the input box (applied before the widget renders below);
                # do NOT auto-run — the user sends it themselves.
                st.session_state._box_pending = query

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
_render_main_hero()

# Conversation history renders in the MAIN script run, so it stays solid (not
# greyed) while the input fragment below runs the slow agent call. That fragment
# isolation is what stops the previous answer from "ghosting" grey during the wait.
with st.container():
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                f"Hello! I'm **{_BRAND_NAME}**, an LLM-orchestrated digital twin for uncertainty-aware distribution grid operations. "
                "I can help you analyze deterministic security, run N-1 contingency studies, quantify probabilistic risk, "
                "optimize robust corrective dispatch, and evaluate flexibility, hosting-capacity, and KPI studies.\n\n"
                "Try one of the **Example queries** in the sidebar, or ask me anything about the active power-system model."
            )
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("charts"):
                render_charts(msg["charts"])

_TOOL_LABELS = {
    "run_rsa": "Security Assessment (RSA)",
    "run_n1_contingency": "N-1 Contingency Analysis",
    "run_probabilistic_risk": "Probabilistic Risk Assessment",
    "optimize_flexibility": "Robust OPF / Flexibility",
    "evaluate_kpis": "KPI Evaluation",
    "scan_rsa_over_time": "Time-series Scan",
    "get_current_timestamp": "Reading timestamp",
    "advance_timestamp": "Advancing timestamp",
    "get_network_summary": "Loading network summary",
    "get_load_generation_profile": "Loading generation profile",
}


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

    # Apply a pending example prefill BEFORE the widget renders.
    if st.session_state.get("_box_pending") is not None:
        st.session_state.chat_box = st.session_state._box_pending
        st.session_state._box_pending = None

    with st.form("chat_form", clear_on_submit=True):
        fc_in, fc_send = st.columns([8, 1])
        with fc_in:
            typed = st.text_input(
                "Ask about the grid…",
                key="chat_box",
                placeholder="Ask about the grid…",
                label_visibility="collapsed",
            )
        with fc_send:
            send = st.form_submit_button("Send", use_container_width=True, type="primary")

    if not (send and typed and typed.strip()):
        return

    effective_input = typed.strip()
    # If the timeline scrubber is parked on a forecast tick, the simulation clock
    # can't be there (forecasts are query-only), so steer the LLM to analyse that
    # forecast timestamp explicitly.
    _vft = st.session_state.get("viewing_forecast_ts")
    if _vft:
        llm_input = (
            f"[The user is viewing the FORECAST timestamp {_vft} on the timeline. "
            f"Treat this as the timestamp of interest: use data_source=\"forecasts\" and "
            f"timestamp=\"{_vft}\" for time-specific tools unless they ask otherwise.]\n\n"
            + effective_input
        )
    else:
        llm_input = effective_input

    st.session_state.messages.append({"role": "user", "text": effective_input, "charts": None})

    with work:
        with st.chat_message("user"):
            st.markdown(effective_input)

        with st.chat_message("assistant"):
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
                    on_event=_on_event,
                )
                status_box.update(label="Analysis complete", state="complete", expanded=False)
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

            # Capture chart data and store message
            charts_this_turn = list(_tools_module._last_tool_results)
            assistant_text = final_text if final_text else "_No text response from agent._"
            st.session_state.messages.append({
                "role": "assistant",
                "text": assistant_text,
                "charts": charts_this_turn,
            })

            # Render reply inside the same chat bubble
            st.markdown(assistant_text)
            render_charts(charts_this_turn)

    # Fold the finished turn into the main conversation history with a full app
    # rerun. It's fast now — the answer is already computed, so nothing blocks it.
    st.rerun(scope="app")


_chat_input_fragment()
