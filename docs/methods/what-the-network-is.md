# What the network is

A study reads tables: buses, lines, generators. What an operator means by *the
renewable units*, *the 20 kV feeder* or *the lines to the external grid* is not
a table, and reading it off one goes wrong. Three examples, all observed:

- every static generator taken for a renewable unit, when in a converted
  MATPOWER case the conventional units are static generators too;
- the 1 kV of a converted per-unit case taken for a voltage level;
- the external grid's injection taken for the lines that carry it.

CONDUCTOR therefore keeps a **CIM layer** over whatever network is loaded
(the *semantic layer* in the code, `backend/semantics.py`). It states what each element is, in the classes of the
Common Information Model (IEC 61970), and how that is known. The agent answers
from it, every answer is checked against it, and the operator sees and corrects
it under **System → CIM layer**.

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

**System → CIM layer → Export as CIM** (or `GET /api/network/cim`) writes the
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
  **normally open ties** that could restore them. What is cut off comes from
  the topology. Each tie (up to five) is then checked with a power flow at the
  clock's operating point, with the element out and the tie closed: how many of
  the cut-off buses it restores, the lowest voltage, the highest loading, and any
  of the network's limits broken.

## The operator's declarations

What the network does not state, the operator can: a unit's energy source or a
load's category, in the chat (*"Gen_bus1 is a wind farm"* — the agent records it
with `declare_network_facts`) or in **System → CIM layer** (**Edit**). A
declaration is stated, overrides a name, and is kept for that network across
restarts, like a rename. **Forget** goes back to what the network says. A
network's profile can state the same (`unit_sources`, `load_categories`), and
also its loads' own power factor (`load_power_factor`).

## What the studies use it for

The studies read the layer, not the names, for any network loaded:

| Study | What it takes from the layer |
| --- | --- |
| Renewable scaling (what-if, scenario scans) | the units that are wind or solar: only those are scaled |
| Probabilistic and robust studies, the look-ahead | the same units: only their output is uncertain |
| Optimisations (OPF, flexibility options, N-1 corrective and preventive plans) | wind and solar units are curtailable at 40 €/MWh, produce at 0 €/MWh, and are capped at what their resource gives at the moment studied |
| Outage studies | the normally open ties that could restore what an outage cuts off, each checked with a power flow |
| N-1 screen | a scope: the outages of one feeder, substation or voltage level |
| Synthetic data | each unit follows its source, each load its category (residential, commercial, industrial, a data centre…) |
| The reading check | a study that depends on which units are wind or solar is re-run with the unknown units counted as wind or solar, and the name-only ones not counted (`assume_unit_sources`, nothing kept) |
| Where values come from | which values are measured, allocated, derived, solved or set |

A unit or a load counts by what is **stated** (by the network, its profile or
you) or **named**; one that is **unknown** keeps the default treatment, and the
result says so. With the layer switched off in the studies (the CIM layer
paper's ablation, `CONDUCTOR_ONTOLOGY=none`), every study behaves as before it.

## Results linked to the layer

A study's result names elements (*Bus 14 at 0.912 p.u.*, *Line 12-13 cuts off
supply*) but not where they are. The elements a result names **as a problem** — a
violation, an outage that cuts supply, a binding limit, a curtailed unit, a
probability of violating — are linked to the layer by their mRIDs
(`llm_agent/agent/grid_context.py`; `POST /api/network/cim/links`). The fields are
found generically, by what they say (violations, outages, cut off, worst, binding…),
never the lists of every element (all voltages, every unit's dispatch). The element
the study is **about** is linked too, first, and marked *studied*: what its
arguments name — the outage (by type and index), the bus of a hosting-capacity
study, the units left out or held.

- **The logged result keeps only the links**: the layer's version and, for the
  first eight elements named, each one's mRID (`cim_links`). The facts stay in one
  place, the layer.
- **The agent's copy of the result** carries what the layer says each of those
  eight is, with its basis (`grid_context`): a bus's voltage, feeder and where on
  it (*bus 2 of 2 from the substation, its far end*), the units and loads there; a
  branch's ends and which buses and loads lose supply without it (topology only);
  a unit's source
  or a load's category; for each unit and load, how its P and Q are obtained from
  the data (*P allocated, Q derived*: a share of its bus's measurement, Q from the
  power factor), so an answer does not call measured what is not; and for each
  feeder, its normally open points — the switch, the branch it is on, and the
  feeder on the other side or a loop within the feeder (whether closing one
  restores supply is a study: an outage study checks each with a power flow).
  Every element the result names, however many, is placed by
  feeder and voltage level together (*feeder: 19 of 27 on Line 1-2, 4 on Line
  12-13*). A bus on no feeder (a meshed level, a busbar) gets the buses next to
  it, and why it has no feeder is said once.
- **MW only at the study's moment.** Demand belongs to the operating point, not to
  the layer. A study of one moment on the network's own demand — an assessment, an
  outage, an N-1 sweep or a violation attribution, with no load scaling, no
  proposal and no option applied — sends its moment (the result's timestamp and
  data source), and the layer adds the demand there: at a bus, and cut off by a
  branch (*0.258 MW at the study's moment*), read from the same data the study
  read; it equals the MW the study reports cut off. Any other study gets counts and
  names only, and its own result is where its MW comes from. The links keep the
  moment and the data's identity, so a later reading shows the same MW whatever the
  clock says — or, if other data have been loaded since, says so and shows none.
- **Whatever reads the links later resolves them** (`GET
  /api/network/cim/resolve`): the turn's record keeps them resolved for the answer
  checks (`cim_context`), so a figure the answer takes from them traces to the
  layer, and the answer card's **Where** section shows them. The layer's version
  (`semantics.layer_version`: a digest of what the layer says, not of the
  operating point) tells when the layer has changed since the study ran — after a
  declaration, say.

With the layer hidden from the agent (`CONDUCTOR_ONTOLOGY=off`, `none` or
`tables`) nothing is linked, so the arms of the CIM layer paper's ablation stay
comparable.

## Choosing elements by meaning

`semantics.select` reads a phrase as the layer, not names, would: *the solar
units*, *the wind units on feeder Line 1-2*, *industrial loads at 20 kV*, *units in
substation Bus 5*, *unknown units*, *transformers*. A qualifier is an energy source
or a group (renewable, variable, conventional, dispatchable) for units, a category
for loads; a place is a feeder, a substation or a voltage level. The agent reads
it with `describe_network(about="select")`, and the phrase itself is accepted
where a study takes units: the units left out of a flexibility study, cost and
generator-model changes, and declarations (*"the units on feeder Line 1-2 are
solar"*). Storage is left out where only generators apply.

## The layer checked against itself

`semantics.data_issues` (and `describe_network(about="issues")`) looks for what
in the network's own data is inconsistent, from the facts alone: a unit or a load
whose name says one thing and whose stated type another, a line between buses at
two voltage levels, a transformer whose rated voltage is not its bus's, a load
drawing negative power, loads with no path to a source, a switch away from its
normal state, loads partly categorised, names used twice.

## A proposal as a difference

Elements keep their mRIDs between the base and a proposal (they are derived from
the network as loaded), so `cim_export.difference` compares the two layers
element by element: added, removed, or the statements that changed. It is written
as a CIM difference model (`GET /api/network/cim/difference?network=<proposal>`;
IEC 61970-552: forward and reverse differences) — applied to the base's export, it
gives the proposal's — and summarised for the agent
(`describe_network(network=<proposal>, about="changes")`).

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
| A feeder (*"Bus 14, at the end of feeder Line 1-2"*) | the element is on no feeder (a busbar, a meshed level) | the layer infers another feeder for it, in every network the turn studied |
| A branch's kind or ends | — | a line called a transformer (or the reverse), a branch placed between other buses |
| What carries the external grid | — | a branch that does not |
| An element (*"Load R9"*) | — | named like the network's own but not among them, in every network the turn studied |
| A value called measured (*"Load 5's measured reactive demand"*) | it is a share of its bus's measurement | it is derived, solved, set or the case's own |
| A fact only a name gives (*"Solar Plant A is the solar unit"*, basis *named*) | said as a fact, not as read from the name | — |

A source or category counts where the answer attaches it to the element: beside
its name (*"WKA 7, a wind turbine"*, *"the solar unit PV 3"*, *"Load 5 is a data
centre"*, *"Solar Plant A at Bus 1 is the solar unit"*), in a list under a source
(*"The wind units are Wind Park West and WKA 7"*, or a bulleted list under
*"Solar units:"*), in a table row with a source or type column, or as the answer
to a question about a source (*"Where is the storage?"* — *"Battery Storage 3 is
at Bus 5"*, whose name says storage). A word that is merely near a name, in a
list of units, says nothing about it. A sentence that holds a claim back (*if*, *would*, *not stated*) or denies it
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
