"""
app.py — Streamlit chat interface for CONDUCTOR.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import os
import pathlib
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from agent.config import BASE_URL, GEMINI_MODEL
from agent.loop import run_agent_turn
from agent.renderers import RENDERER_MAP
from agent import tools as _tools_module
from agent.tools import get_current_timestamp

_APP_DIR = pathlib.Path(__file__).parent
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
            ['cb-net-panel', 'cb-data-panel', 'cb-info-panel'].forEach(id => {
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
                <button id="cb-net-submit" class="cb-submit">Upload &amp; Load</button>
                <div id="cb-net-status" class="cb-status"></div>
                <div class="cb-fmt-guide">
                    <div class="cb-fmt-title">Supported formats</div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.m</span><span class="cb-fmt-desc">MATPOWER case file — IEEE cases, pglib-opf, or any compatible model</span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.json</span><span class="cb-fmt-desc">pandapower JSON — export with <code>pp.to_json(net, "file.json")</code></span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.xlsx</span><span class="cb-fmt-desc">pandapower Excel — export with <code>pp.to_excel(net, "file.xlsx")</code></span></div>
                    <div class="cb-fmt-row"><span class="cb-fmt-tag">.uct</span><span class="cb-fmt-desc">UCTE/CGMES exchange — ENTSO-E standard for European TSO networks</span></div>
                </div>
            `;
            parentDoc.body.appendChild(netPanel);

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

            parentDoc.getElementById('cb-net-submit').addEventListener('click', async () => {
                const file = parentDoc.getElementById('cb-net-file').files[0];
                const convert = parentDoc.getElementById('cb-net-convert').checked;
                const status = parentDoc.getElementById('cb-net-status');
                const btn = parentDoc.getElementById('cb-net-submit');
                if (!file) { status.textContent = 'Select a file first (.m, .json, .xlsx, .uct).'; status.className = 'cb-status err'; return; }
                status.textContent = 'Uploading…'; status.className = 'cb-status'; btn.disabled = true;
                const fd = new FormData();
                fd.append('file', file, file.name);
                fd.append('convert_gen_to_sgen', convert ? 'true' : 'false');
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

    with st.expander("About CONDUCTOR", expanded=False):
        st.caption(_BRAND_CAPTION)

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
