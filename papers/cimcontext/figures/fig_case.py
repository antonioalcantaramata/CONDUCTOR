"""
Fig. 2 of the GM 2027 paper: the CIGRE MV network as the semantic layer describes it.

Everything drawn comes from the layer (`semantics.build` / `semantics.around`) on the network the demo loads: which
buses Line 2-3 cuts off, the open tie that could restore them, every switch's normal state, and each unit's energy
source with its basis. Only the positions of the buses are fixed by hand. No model is called.

    python papers/cimcontext/figures/fig_case.py [OUT.tex]

writes a TikZ picture (default: ../GM_paper/final_paper/fig_case.tex next to the repository) and defines
\\figcasefacts, the layer's answer for the caption (what the outage cuts off and the tie that could restore it).
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

import pandapower.networks as pn  # noqa: E402

import semantics as sm  # noqa: E402

DEFAULT_OUT = ROOT.parent / "GM_paper" / "final_paper" / "fig_case.tex"
OUTAGE_LINE = 1                       # Line 2-3

# Bus positions (x, y), by hand: feeder 1 on the top two rows, feeder 2 on the left of the second row.
POS = {0: (-1, 2), 1: (0, 2), 2: (1, 2), 3: (2, 2), 4: (3, 2), 5: (4, 2), 6: (5, 2),
       12: (-1, 1), 13: (0, 1), 14: (1, 1), 8: (2, 1), 9: (3, 1), 10: (4, 1), 11: (5, 1), 7: (6.1, 1.5)}
# Branches drawn with a bend so that nothing crosses.
ROUTES = {"Line 7-8": "(b7) |- (2,0.42) -- (b8)"}
TAG = {"solar": "S", "wind": "W", "thermal": "T", "storage": "B"}


def _bus(name: str) -> int:
    return int(re.search(r"\d+", name).group())


def build_facts():
    net = pn.create_cigre_network_mv(with_der="all")
    sem = sm.build(net)
    out = sm.around(net, sem, "line", OUTAGE_LINE)["without_it"]
    facts = sm.compact(sem)
    return net, sem, out, facts


def render(net, sem, out, facts) -> str:
    cut = {_bus(b) for b in out["buses_cut_off"]}
    ties = out["open_ties_that_could_restore"]
    tie_lines = {t.split(" | ")[0] for t in ties}
    open_switch = {}                                  # line name -> switch name, for normally open switches
    for sw in sem["switches"]:
        if sw["normal_open"]["value"] and sw["where"]["kind"] == "line":
            open_switch[sw["where"]["element"].split(" | ")[0]] = (sw["name"], _bus(sw["where"]["at"]))
    units: dict[int, list[str]] = {}
    for name, u in facts["units"].items():
        assert u["basis"] == "stated", (name, u)      # the figure says every source is stated; keep it true
        units.setdefault(_bus(u["bus"]), []).append(TAG[u["source"]])
    outage = facts["branches"]
    outage_name = next(k.split(" | ")[0] for k in outage if k.endswith(f"[L{OUTAGE_LINE}]"))

    lines = [r"\begin{tikzpicture}[x=10.2mm, y=11mm, font=\scriptsize,",
             r"  bus/.style={circle, draw, inner sep=0pt, minimum size=3.6mm, fill=white},",
             r"  cut/.style={fill=orange!35},",
             r"  br/.style={thick}, nop/.style={densely dashed, gray},",
             r"  tag/.style={inner sep=0.6pt, font=\scriptsize\bfseries},",
             r"  ren/.style={tag, text=green!45!black}, oth/.style={tag, text=gray}]"]
    for b, (x, y) in POS.items():
        style = "bus, cut" if b in cut else "bus"
        lines.append(rf"\node[{style}] (b{b}) at ({x},{y}) {{{b}}};")
    lines.append(r"\node[left=0.5mm of b0, align=right] {110\,kV\\grid};")
    # Branches.
    for key, br in facts["branches"].items():
        name = key.split(" | ")[0]
        a, b = (_bus(e) for e in br["ends"])
        if name == outage_name:
            lines.append(rf"\draw[br] (b{a}) -- node[red, font=\normalsize\bfseries, inner sep=0pt] {{$\times$}} (b{b});")
            continue
        style = "br"
        if name in open_switch:
            style = "nop, very thick, green!55!black" if name in tie_lines else "nop"
        path = ROUTES.get(name, rf"(b{a}) -- (b{b})")
        lines.append(rf"\draw[{style}] {path};")
        if name in open_switch:
            sw, at = open_switch[name]
            colour = "green!45!black" if name in tie_lines else "gray"
            lines.append(rf"\path (b{a}) -- node[{colour}, fill=white, inner sep=0.5pt, font=\scriptsize] {{{sw}}} (b{b});")
        if br["type"] == "trafo":
            lines.append(rf"\path (b{a}) -- node[fill=white, inner sep=0.5pt, font=\scriptsize] {{T}} (b{b});")
    # Units, by source: wind and solar (what a change to wind and solar scales) in green, the rest in grey.
    for b, tags in units.items():
        x, y = POS[b]
        above = y >= 2 or b == 7
        anchor = "above" if above else ("below right" if b == 8 else "below")
        text = " ".join(rf"\textcolor{{{'green!45!black' if t in 'SW' else 'gray'}}}{{{t}}}" for t in sorted(tags))
        lines.append(rf"\node[tag, {anchor}=0.4mm of b{b}] {{{text}}};")
    # What the layer says goes into the caption, as a macro the paper's caption uses.
    n_loads = len(out["loads_cut_off"])
    tie_text = ", ".join(f"{open_switch[t][0]} on {t.replace('Line ', 'Line~')}" for t in sorted(tie_lines))
    facts_text = (f"Buses {min(cut)}--{max(cut)} (shaded; {n_loads} loads, {out['load_cut_off_mw']:.1f}\\,MW nominal) "
                  f"lose supply, and the normally open tie {tie_text} could restore them")
    lines.append(r"\end{tikzpicture}")
    return ("% Generated by papers/cimcontext/figures/fig_case.py from the semantic layer; do not edit by hand.\n"
            + "\\gdef\\figcasefacts{" + facts_text + "}\n"
            + "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_path = pathlib.Path(argv[0]) if argv else DEFAULT_OUT
    tex = render(*build_facts())
    out_path.write_text(tex, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
