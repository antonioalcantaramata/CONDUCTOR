# What the network is

A study reads tables: buses, lines, generators. What an operator means by *the
renewable units*, *the 20 kV feeder* or *the lines to the external grid* is not
a table, and reading it off one goes wrong. Three examples, all observed:

- every static generator taken for a renewable unit, when in a converted
  MATPOWER case the conventional units are static generators too;
- the 1 kV of a converted per-unit case taken for a voltage level;
- the external grid's injection taken for the lines that carry it.

CONDUCTOR therefore keeps a **semantic layer** over whatever network is loaded
(`backend/semantics.py`). It states what each element is, in the classes of the
Common Information Model (IEC 61970), and how that is known. The agent answers
from it, every answer is checked against it, and the operator sees and corrects
it under **System → What it is**.

## Every fact has a basis

| Basis | Meaning | Example |
| --- | --- | --- |
| **stated** | The network says so: a type column, a class in a CIM import, a proposal's edit, the network's profile — or the operator declared it. | CIGRE MV's `PV 3`, type `PV`: solar. |
| **named** | Read from the element's name: a likely reading, said as such. | `Wind Park` with a generic type: wind. |
| **inferred** | By a rule named with the fact. | A unit with no active power is a synchronous condenser; buses joined by transformers or bus couplers form a substation. |
| **unknown** | The network does not say. | Every unit of PGLib's IEEE 30: MATPOWER does not record fuel. |

An unknown fact is never filled in with a guess. A study that needs it says it
is unknown. For example, a renewable scaling applies only to units stated or
named wind or solar, and says when there are none.

A type is matched on its whole value (`PV`, `WP`, `CHP diesel`) and a name on
its words. A generic type (`SGEN`, `wye`) and a benchmark code (CIGRE's
`Load R1`) are no statement.

## What is described

| Element | CIM class | Facts |
| --- | --- | --- |
| Bus | TopologicalNode | nominal voltage (unknown in a per-unit case, where every bus is at 1 kV), substation |
| Line | ACLineSegment | ends, voltage, rating in MVA (from its current rating at its voltage) |
| Transformer | PowerTransformer | high- and low-voltage ends, rated power |
| Unit | as distribution CIM (CIM100) has it: a **PowerElectronicsConnection** with its PhotovoltaicUnit, PowerElectronicsWindUnit, BatteryUnit or PowerElectronicsUnit (an inverter: PV, a wind turbine modelled as a static generator, a battery, a fuel cell), or a **SynchronousMachine** with its Thermal-, Hydro-, Nuclear-, Wind- or SolarGeneratingUnit (a rotating machine); the abstract GeneratingUnit when the source is unknown | energy source, control as a RegulatingControl mode (voltage, powerFactor, reactivePower), bus |
| Load | EnergyConsumer | category (residential, commercial, industrial, rural, agricultural, EV charging, data centre, heat pump, a downstream network), bus, demand |
| External grid | ExternalNetworkInjection | its bus and substation, and **the branches that carry its power**: the lines leaving its substation and the transformers inside it |
| Switch | Breaker, LoadBreakSwitch, Disconnector (from the network's type: CB, LBS, DS), else Switch | its **normal state** (as the network was loaded) and its **present state**, what it switches, its role |
| Shunt | LinearShuntCompensator | capacitor or reactor (from the sign of its reactive power), its size at 1 p.u. |
| Substation, voltage level | Substation, VoltageLevel | the buses at each level |
| Feeder | Feeder | its head line, the buses, units and loads it supplies, its ties |

A substation is stated when the network names it (a CIM import). Otherwise it is
inferred: buses joined by transformers or closed bus couplers, named after the
highest-voltage bus among them.

Every element has an **mRID**. A model imported from CIM/CGMES keeps its own
(stated); otherwise it is derived from the network's identity and the
element's key as loaded, so it survives renames and restarts and differs
between networks. An element can be looked up by its mRID in every lookup.

## Switches and feeders

CIM keeps a switch's **normal** state (`normalOpen`) apart from its **present**
one, and so does the layer: the normal state is the one the network was loaded
with, the present one is what outages and proposals left. A normally open
switch on a line has a role, inferred from the feeders on either side:

- a **tie between feeders** — closing it restores supply from the other
  feeder (CIGRE MV's S1, on Line 14-8);
- a **normally open point** closing a loop within one feeder (S2, S3).

Closed line switches are **sectionalising** switches; switches between buses
are **bus couplers**. A **feeder** is a container named after its head line —
the first line after the last transformer on the route from the external
grid — holding the buses, units and loads it supplies and the switches that
tie it to others. Feeders exist only where the voltage level is radial: in a
meshed one power comes over several routes, and each bus says so instead of
naming a feeder. They are inferred from the topology as switched now.

## Where each value comes from

Facts have a basis; so do the values of the present operating point (the
simulation clock's moment). For each load and unit, P and Q are one of:

| Kind | Meaning |
| --- | --- |
| **measured** (or forecast) | the data give it for this element alone |
| **allocated** | the data give its bus's total, split across the elements there (loads by their demand as loaded, units by capacity): the total is measured, each share is an estimate |
| **derived** | from an assumption: a load's Q from the load power factor, a unit's Q from its fixed power factor |
| **solved** | the power flow sets it: the Q of a unit holding a voltage |
| **set** | the element's own set-point (storage, a fixed-Q unit) |
| **nominal** | the data do not cover it: the case's own value |
| **proposal** | as the proposal that added it sets it |

`describe_network(about="data")` says it per element, with the moment, whether
the data are synthetic, and the power factor in force. On CIGRE MV, for
example, the PV unit at Bus 9 shares its bus's measurement with the CHP unit
and a fuel cell: its output is *allocated*, so at night it carries a share of
their output. That is a limitation of per-bus data, and the layer now says
so instead of hiding it.

## As CIM, for other tools

**System → What it is → Export as CIM** (or `GET /api/network/cim`) writes the
layer as CIM RDF/XML or JSON-LD in the distribution namespace (CIM100): base
voltages, substations, voltage levels, topological nodes with a terminal per
piece of equipment, the equipment above with its units, controls and switches
(`normalOpen`, `open`), and the feeders. What standard CIM has no place for —
each fact's basis and origin, a unit's energy source, a load's category, where
each value comes from — goes beside it in a namespace of CONDUCTOR's own
(`cdr:`), so a reader that knows only CIM still reads a valid model. Units
follow CGMES (kV, MW, km, ohm). The model is bus-branch: a switch is placed
at the end of the branch it opens. The tests read the export back with rdflib
and query it with SPARQL.

A CGMES model can also be **uploaded** (a `.zip`; see
[Network, data and the operating point](network-and-data.md)): its
identifiers, substations, unit types and machine controls are then stated by
the model, not inferred. A model that leaves lines without a node at one end
(its boundary set missing) is refused with what to add; one the layer cannot
fully connect lists the branches it could not place (`unresolved_branches`)
instead of failing.

## Around an element

Facts are linked: a unit or a load to its bus, a bus to its substation and
voltage level, a branch to its two ends, the external grid to the branches that
carry it. The agent follows those links in one call. `describe_network` with
`about="around"` and an element returns:

- **the element** and its facts;
- **its substation** (each end's, for a line): voltage levels, buses, the units
  and loads there with their sources and categories, the branches leaving it
  and where they lead, and whether the external grid connects there;
- **its supply**: the route from the external grid, as the network is switched
  now, and its **feeder**, meaning the first line after the last transformer on
  that route. Both are *inferred* from the topology. A feeder is a radial
  idea, so it is given only where the element's own voltage level is radial.
  Where the network above is meshed, the route shown is the shortest and the
  result says power may take others;
- **without it** (a bus, branch or substation): the buses, units and loads that
  would lose every connection to an external grid, with their demand, and the
  **normally open ties** that could restore them. Topology only: no power
  flow is run, and closing a tie is not checked against flows.

## The operator's declarations

What the network does not state, the operator can: a unit's energy source or a
load's category, in the chat (*"Gen_bus1 is a wind farm"* — the agent records it
with `declare_network_facts`) or in **System → What it is** (**Edit**). A
declaration is stated, overrides a name, and is kept for that network across
restarts, like a rename. **Forget** goes back to what the network says. A
network's profile can state the same (`unit_sources`, `load_categories`), and
also its loads' own power factor (`load_power_factor`).

## Checked in every answer

Each turn's record keeps the facts of the elements its answer names
(`network_facts`). The answer is then checked against them, as its figures are
checked against the studies (`llm_agent/agent/entities.py`):

| Claim | Unsupported (warning) | Contradicted (failure) |
| --- | --- | --- |
| A unit's source (*"Gen_bus7, a wind farm"*) | the network does not say what drives it | it says otherwise |
| A load's category | the network does not say | it says otherwise |
| A group (*"the solar generation"*) | no unit is stated or named of that kind | — |
| A voltage (*"the 132 kV substation"*) | the network has no such level (none at all in a per-unit case) | the bus named beside it is at another |
| A branch's kind or ends | — | a line called a transformer (or the reverse), a branch placed between other buses |
| What carries the external grid | — | a branch that does not |
| An element (*"Load R9"*) | — | named like the network's own but not among them, in every network the turn studied |
| A value called measured (*"Load 5's measured reactive demand"*) | it is a share of its bus's measurement | it is derived, solved, set or the case's own |

A source or category counts only when it is attached to the element's name
(*"WKA 7, a wind turbine"*, *"the solar unit PV 3"*, *"Load 5 is a data
centre"*): a word that is merely near it, in a list of units, says nothing about
it. A sentence that holds a claim back (*if*, *would*, *not stated*) or denies it
(*no wind units*, *there is no Load R9*) is not a claim. A figure or a name the
operator gave in the question is theirs to use.

An element name is recognised by the network's own naming: a shape shared by at
least three names and containing a word (*Load R‹n›*, *PV ‹n›*, *Line ‹n›-‹n›*).
Observed live: asked what Line 2-3's outage cuts off, the agent listed a *Load R9*
among the loads; CIGRE MV has none.

## Adapting to the network

Nothing is written for one network. The same code gives, for example:

- **PGLib IEEE 30:** every unit and load unknown, one stated level of 135 kV,
  thirty one-bus substations, a meshed network with no feeders;
- **CIGRE MV:** every unit stated by its type, 110 and 20 kV, the 110/20 kV
  substation inferred from its two transformers, two radial feeders with their
  ties;
- **pandapower's multi-voltage example:** a wind park stated, a biogas plant
  named, loads named residential, industrial or rural, double busbars joined by
  couplers into substations, and feeders on the radial LV networks below a
  meshed HV grid.
