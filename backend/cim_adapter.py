"""
cim_adapter.py — the CIM model projected to a computation tool, and the tool's results back by mRID.

The CIM model is the network's description; a tool computes on its own copy. An adapter makes that copy and keeps
the link explicit: every element of the copy maps to the mRID of the model's object it stands for (`ElementMap`),
and the tool's results come back keyed by mRID, so they read the same whatever tool computed them.

`PandapowerAdapter` reads a CGMES model with pandapower's own converter (network_loader.read_cgmes, which also adds
the switch classes the converter leaves out), then puts back by mRID what CGMES has no place for but the computation
needs (`conductor_extras.json`, written by cgmes_writer.py: costs, OPF limits, how loads follow the voltage, scaling,
storage, the network's frequency). Another tool (PowSyBl, PowerFactory) gets an adapter of its own, reading the same
model.

`round_trip` checks the whole path for a network: pandapower → CGMES → pandapower, the same power flow element by
element.
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import tempfile
import zipfile

import pandas as pd

# The tables whose rows stand for an object of the model, and the results each one reports.
_RESULTS = {"bus": ("vm_pu", "va_degree"), "line": ("loading_percent", "p_from_mw", "q_from_mvar"),
            "trafo": ("loading_percent", "p_hv_mw", "q_hv_mvar"), "gen": ("p_mw", "q_mvar"),
            "sgen": ("p_mw", "q_mvar"), "load": ("p_mw", "q_mvar"), "ext_grid": ("p_mw", "q_mvar"),
            "shunt": ("q_mvar",), "storage": ("p_mw", "q_mvar"), "trafo3w": ("loading_percent", "p_hv_mw", "q_hv_mvar"),
            "ward": ("p_mw", "q_mvar"), "xward": ("p_mw", "q_mvar"), "impedance": ("p_from_mw", "q_from_mvar"),
            "motor": ("p_mw", "q_mvar"), "svc": ("q_mvar", "vm_pu"),
            "dcline": ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"),
            "switch": ()}


def _norm(identifier) -> str:
    s = str(identifier or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if s.lower().startswith("urn:uuid:"):
        s = s[9:]
    return s.lstrip("_")


class ElementMap:
    """mRID ↔ (table, index) for a tool's copy of the model."""

    def __init__(self, by_mrid: dict[str, tuple[str, int]]):
        self.by_mrid = dict(by_mrid)
        self.by_element = {v: k for k, v in self.by_mrid.items()}

    @classmethod
    def of(cls, net, network_id: str = "local") -> "ElementMap":
        """The map of a pandapower network: each row's mRID as the CIM layer gives it (its model's own, kept by the
        converter as `origin_id`, else one derived from its table and index)."""
        import semantics as sm
        out = {}
        for table in _RESULTS:
            df = net[table] if table in net else None
            if isinstance(df, pd.DataFrame):
                for i, m in sm.row_mrids(net, table, network_id).items():
                    out[m] = (table, i)
        return cls(out)

    def element(self, mrid) -> tuple[str, int] | None:
        return self.by_mrid.get(_norm(mrid))

    def mrid(self, table: str, index) -> str | None:
        return self.by_element.get((table, int(index)))

    def __len__(self) -> int:
        return len(self.by_mrid)


def _extras(source) -> dict:
    with zipfile.ZipFile(source) as z:
        names = [n for n in z.namelist() if n.endswith("conductor_extras.json")]
        return json.loads(z.read(names[0])) if names else {}


def apply_extras(net, extras: dict, emap: ElementMap) -> list[str]:
    """Put back on a copy read from CGMES what the model's profiles have no place for (in place). Returns notes."""
    import pandapower as pp
    notes = []
    f_hz = float(extras.get("f_hz") or 50.0)
    if abs(f_hz - 50.0) > 1e-9:
        # pandapower's converter reads a line's charging susceptance back at 50 Hz.
        net.line["c_nf_per_km"] = net.line["c_nf_per_km"] * 50.0 / f_hz
    net.f_hz = f_hz
    if extras.get("sn_mva"):
        net.sn_mva = float(extras["sn_mva"])
    if extras.get("name"):
        net.name = extras["name"]
    moved = {}
    for mrid, entry in (extras.get("elements") or {}).items():
        hit = emap.element(mrid)
        if entry.get("table") == "dcline":
            # pandapower's converter reads a DC line from its segment and converters (its losses its own way): the
            # line's own values go on that row — or, read by none, it is made again between its buses.
            fields = dict(entry.get("fields") or {})
            row = dict(fields.get("dcline") or {})
            if hit is None:
                ends = [emap.element(fields.get(k)) for k in ("from_node", "to_node")]
                if not all(e and e[0] == "bus" for e in ends):
                    notes.append(f"a DC line ({mrid}) whose ends are not in the model is not put back")
                    continue
                new = pp.create_dcline(net, from_bus=ends[0][1], to_bus=ends[1][1], p_mw=0.0, loss_percent=0.0,
                                       loss_mw=0.0, vm_from_pu=1.0, vm_to_pu=1.0)
                if "origin_id" not in net.dcline.columns:
                    net.dcline["origin_id"] = None
                net.dcline.at[new, "origin_id"] = mrid
                moved[mrid] = hit = ("dcline", int(new))
            for c, v in row.items():
                net.dcline.at[hit[1], c] = v
            continue
        if hit is None:
            continue
        table, idx = hit
        fields = dict(entry.get("fields") or {})
        if entry.get("table") == "storage" and table == "sgen":
            # Written as an energy source (CGMES 2.4.15 has no battery): back to a storage unit.
            row = net.sgen.loc[idx]
            info = fields.pop("storage", {}) or {}
            new = pp.create_storage(net, bus=int(row.bus), p_mw=float(fields.get("p_mw", -row.p_mw)),
                                    max_e_mwh=float(info.get("max_e_mwh") or 0.0),
                                    q_mvar=float(fields.get("q_mvar", -row.q_mvar)), name=row.get("name"),
                                    in_service=bool(row.in_service))
            for c in ("min_e_mwh", "soc_percent", "sn_mva"):
                if info.get(c) is not None:
                    net.storage.at[new, c] = info[c]
            if "origin_id" in net.sgen.columns:
                net.storage.at[new, "origin_id"] = net.sgen.at[idx, "origin_id"]
            net.sgen = net.sgen.drop(index=idx)
            moved[mrid] = ("storage", int(new))
            table, idx = "storage", int(new)
            fields = {k: v for k, v in fields.items() if k in ("scaling",)}
        if entry.get("table") == "svc" and table == "shunt":
            # Read as a fixed shunt by pandapower's converter: back to a static var compensator.
            row = net.shunt.loc[idx]
            params = dict(fields.pop("svc", {}) or {})
            new = pp.create_svc(net, bus=int(row.bus), x_l_ohm=float(params.get("x_l_ohm", 1.0)),
                                x_cvar_ohm=float(params.get("x_cvar_ohm", -1.0)),
                                set_vm_pu=float(params.get("set_vm_pu", 1.0)),
                                thyristor_firing_angle_degree=float(params.get("thyristor_firing_angle_degree", 90.0)),
                                name=row.get("name"), in_service=bool(row.in_service))
            for c, v in params.items():
                if c in net.svc.columns and c not in ("x_l_ohm", "x_cvar_ohm", "set_vm_pu",
                                                      "thyristor_firing_angle_degree"):
                    net.svc.at[new, c] = v
            if "origin_id" in net.shunt.columns:
                net.svc.at[new, "origin_id"] = net.shunt.at[idx, "origin_id"]
            net.shunt = net.shunt.drop(index=idx)
            moved[mrid] = ("svc", int(new))
            continue
        fields.pop("row", None)                        # the row it had (project_onto places it there)
        for col in fields.pop("unset", None) or []:
            if col in net[table].columns:              # left empty by the network; the model states a stand-in
                net[table].at[idx, col] = float("nan")
        for col, value in fields.items():
            if col == "in_service" and table == "bus":
                net.bus.at[idx, "in_service"] = bool(value)
                continue
            if col not in net[table].columns:
                net[table][col] = pd.Series(dtype=object if isinstance(value, (str, bool)) else float)
            net[table].at[idx, col] = value
    emap.by_mrid.update(moved)
    emap.by_element = {v: k for k, v in emap.by_mrid.items()}
    for cost in extras.get("costs") or []:
        hit = emap.element(cost["element"])
        if hit is None:
            notes.append(f"a cost for an element not in the model ({cost['element']}) is not put back")
            continue
        table, idx = hit
        values = {k: v for k, v in cost.items() if k not in ("table", "element") and v is not None}
        if cost["table"] == "poly_cost":
            pp.create_poly_cost(net, idx, table, **{k: v for k, v in values.items()
                                                    if k.startswith(("cp", "cq"))})
        else:
            points = values.get("points")
            if points is not None:
                pp.create_pwl_cost(net, idx, table, points, power_type=values.get("power_type", "p"))
    return notes


class PandapowerAdapter:
    """The model as a pandapower network, and pandapower's results back by mRID."""

    name = "pandapower"

    def project(self, source, network_id: str = "local", extras: bool = True,
                partial: bool = False) -> tuple[object, ElementMap, list[str]]:
        """(network, element map, notes) of a CGMES zip (a path, or its bytes). `extras`: put back what the profiles
        have no place for (False: the network as any CGMES reader sees it). `partial`: a part of a model (what a
        change touches), which need not describe a network of its own."""
        import network_loader as nl
        data = source if isinstance(source, (bytes, bytearray)) else None
        with tempfile.TemporaryDirectory() as scratch:
            path = source
            if data is not None:
                path = os.path.join(scratch, "model.zip")
                with open(path, "wb") as fh:
                    fh.write(data)
            net = nl.read_cgmes(path, partial=partial)
            kept = _extras(path) if extras else {}
        emap = ElementMap.of(net, network_id)
        notes = list(net.get("_import_notes") or [])
        if kept:
            notes += apply_extras(net, kept, emap)
        return net, emap, notes

    def results(self, net, emap: ElementMap) -> dict[str, dict]:
        """{mRID: {result: value}} of a solved network."""
        out = {}
        for mrid, (table, idx) in emap.by_mrid.items():
            res = f"res_{table}"
            if res not in net or idx not in net[res].index:
                continue
            row = {c: float(net[res].at[idx, c]) for c in _RESULTS[table] if c in net[res].columns
                   and pd.notna(net[res].at[idx, c])}
            if row:
                out[mrid] = row
        return out


ROUND_TRIP_TOLERANCE = {"vm_pu": 1e-6, "va_degree": 1e-4, "loading_percent": 1e-3, "mw": 1e-4}


def round_trip(net, network_id: str = "local", extras: bool = True) -> dict:
    """pandapower → CGMES (cgmes_writer) → pandapower (PandapowerAdapter): both power flows, compared element by
    element through the mRIDs. {"matches", "compared", "max_diff", "worst", "missing", "losses", "notes"}.
    `extras=False`: the profiles alone, as a tool other than CONDUCTOR reads them."""
    import pandapower as pp

    import cgmes_writer as cw
    original = copy.deepcopy(net)
    package = cw.package(original, network_id)
    adapter = PandapowerAdapter()
    copy_net, emap, notes = adapter.project(package, network_id, extras=extras)
    pp.runpp(original, numba=False)
    with zipfile.ZipFile(io.BytesIO(package)) as z:
        losses = json.loads(z.read("conductor_extras.json")).get("losses") or []
    try:
        pp.runpp(copy_net, numba=False)
    except Exception as exc:  # noqa: BLE001 — the copy as read does not solve: a difference, reported
        return {"matches": False, "compared": 0, "max_diff": {}, "worst": [], "missing": [], "n_missing": 0,
                "losses": losses, "notes": notes + [f"The network read back from CGMES does not solve ({exc})."],
                "tolerance": ROUND_TRIP_TOLERANCE, "elements": len(emap)}
    original_map = ElementMap.of(original, network_id)
    before = adapter.results(original, original_map)
    after = adapter.results(copy_net, emap)
    # Read without the extras, a storage unit is an energy source: pandapower reports it with the generator's sign.
    for mrid, (table, _idx) in original_map.by_mrid.items():
        hit = emap.element(mrid)
        if table == "storage" and hit and hit[0] == "sgen" and mrid in after:
            after[mrid] = {k: -v for k, v in after[mrid].items()}
    diff, worst, missing = {}, [], []
    for mrid, row in before.items():
        other = after.get(mrid)
        if other is None:
            missing.append(mrid)
            continue
        for key, value in row.items():
            if key not in other:
                continue
            d = abs(value - other[key])
            kind = key if key in ("vm_pu", "va_degree", "loading_percent") else "mw"
            if d > diff.get(kind, 0.0):
                diff[kind] = d
            worst.append((d / ROUND_TRIP_TOLERANCE[kind], mrid, key, value, other[key]))
    worst.sort(reverse=True)
    matches = not missing and all(diff.get(k, 0.0) <= tol for k, tol in ROUND_TRIP_TOLERANCE.items())
    return {"matches": matches, "compared": len(before) - len(missing), "max_diff": diff,
            "worst": [{"mrid": m, "result": k, "before": round(a, 8), "after": round(b, 8)}
                      for _r, m, k, a, b in worst[:5]],
            "missing": missing[:10], "n_missing": len(missing), "losses": losses, "notes": notes,
            "tolerance": ROUND_TRIP_TOLERANCE, "elements": len(emap)}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(float(v))
