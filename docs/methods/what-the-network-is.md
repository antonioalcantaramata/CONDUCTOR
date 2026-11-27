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
follow CGMES (kV, MW, km, ohm). Nodes are topological nodes, or, for a
node-breaker CIM model, its connectivity nodes; a bus coupler (and every switch
of a node-breaker model) is joined to the nodes on both sides, and any other
switch is placed at the end of the branch it opens. Voltage levels keep the
model's names. The tests read the export back with rdflib and query it with
SPARQL.

A CGMES model can also be **uploaded** (a `.zip`; see
[Network, data and the operating point](network-and-data.md)): its
identifiers, substations, unit types and machine controls are then stated by
the model, not inferred. A model that leaves lines without a node at one end
(its boundary set missing) is refused with what to add; one the layer cannot
fully connect lists the branches it could not place (`unresolved_branches`)
instead of failing.

## A CIM model kept whole

pandapower's converter turns a CGMES model into the tables a power flow needs
and keeps only what fits them. CONDUCTOR also keeps the **whole model**
(`backend/cim_model.py`): every object of every profile (EQ, SSH, TP, SV, DL,
GL and the boundary set), whatever its class, by its mRID, with its attributes
and its references, tied to the network it came with. The layer reads its facts
there first, then from pandapower, then infers:

| From the model | In the layer |
|---|---|
| ConnectivityNode / TopologicalNode | each bus's class |
| VoltageLevel, Bay, Substation, regions | voltage levels with their names, bays and busbars; substations with their names and mRIDs |
| BusbarSection, BusNameMarker | the busbar at a node; the bus name the source tool shows |
| OperationalLimitSet, CurrentLimit | every limit on a branch, permanent or temporary (held for a stated time) |
| RegulatingControl, TapChangerControl | each machine's, shunt's and tap changer's control: mode, target, whether enabled |
| PowerTransformerEnd, tap changers | rated voltage and power per end; tap steps, position and control |
| GeneratingUnit, LoadResponseCharacteristic | each unit's generating unit and limits; how each load follows the voltage |
| equipment pandapower has no table for | listed as *other equipment*, not in the computation |

A **loss report** says, class by class, how many objects the model holds, how
many reach the computation (pandapower's tables) and how many the layer
describes, with a status: *described*, *partly*, *read into* other entries
(limit sets into each branch's limits), *solved state* (the source tool's
power-flow solution) or *model only*. Nothing is out of reach:
`describe_network(about="around", element=<mRID>)` returns any object as the
model states it, a limit set or a control area as much as a line, and a busbar,
a bus name or a voltage level can be named as the model names it.

**What the studies ignore.** The studies run on pandapower's copy, so some of
what the model states they do not use: they check the permanent limit only;
they hold shunts and taps at fixed positions, not following their controls;
pandapower's converter makes every load constant power, whatever its load
response says. `describe_network(about="model")` and the data issues say so for
the network loaded, so the agent can say what a study did not take into account.

**The model's own solution.** When the model carries the source tool's power-flow
solution (its SV profile), CONDUCTOR's power flow on the network as loaded is
compared with it: every node's voltage and angle and every injection's power.
On ENTSO-E's SmallGrid (1,225 nodes, node-breaker and bus-branch) they agree
within 0.0002 pu and 0.01°: the conversion keeps the model electrically intact.
A difference beyond the tolerance (0.005 pu, 0.5°, 1 MW) is a data issue that
names the largest differences.

**On the map.** A node-breaker model has a node for every busbar section and every
bay; drawn bus by bus, most look unconnected (their switches are not branches).
The map draws the nodes joined by closed switches as one electrical node, named
after its busbar or bus name in the model, as the power flow sees them; results
reported for any of its buses land on it. When the model brings its own diagram
(the DL profile), the map is drawn as that diagram: ENTSO-E's SmallGrid, 1,225
nodes, becomes the 118 electrical nodes laid out as its source tool lays them out.

**In the System tab.** The *CIM layer* pane shows the model too: its objects in the
headline, a picture of what the layer does with each class (described, read into
other entries, the source tool's solution, kept in the model only), and a *CIM
model* tab with the solved-state check, what the studies do not use and the loss
report; an *Other equipment* tab lists what the layer has no kind of its own for.
Buses show their voltage level, busbar and name in the model; lines their limits
by duration; transformers their tap position.

**Exported losslessly.** Beside the layer's own export, a network loaded as CIM
can be downloaded as *your model + CONDUCTOR's additions*
(`GET /api/network/cim/package`): the uploaded zip, byte for byte, and one more
profile with what the layer adds about the model's objects (energy sources,
categories, feeders, switch roles, each with how it is known), referring to them
by the model's own identifiers, with the feeders it infers. Read with the
model's profiles, the additions sit on the model's objects; nothing of the model
is regenerated.

**Every network has a CIM model.** A network that did not come as CIM — a
MATPOWER case, a pandapower file, the bundled demos — is written as CGMES 3.0
(EQ, SSH, TP and, for transformers' vector groups, SC; 2.4.15 on request) from
its data when it loads (`backend/cgmes_writer.py`): PV units, wind turbines and
batteries as power-electronics connections with their units, each
element under the CIM layer's mRID. What CGMES has no place for but the
computation needs (generation costs, OPF limits, how loads follow the voltage,
scaling, storage, the frequency) is kept beside it by mRID; what neither can
carry is listed as a loss. The written model is *derived*: the layer never takes
it for a model's own statement, so the facts keep their basis (stated by the
network, named, inferred). It is what the network is exported as (**Export as
CIM → The network as CGMES**) and what other tools can read.

**The computation is a copy of the model.** An adapter (`backend/cim_adapter.py`)
makes a tool's copy of a CGMES model with an explicit map from each element to
the model's mRID, puts back what the profiles cannot carry, and returns the
tool's results keyed by mRID, so they read the same whatever tool computed them.
For pandapower, the copy is read by pandapower's own CIM converter. The round
trip, pandapower → CGMES → pandapower, gives the same power flow element by
element — within 10⁻⁷ MW and 10⁻¹⁰ pu on IEEE 30, CIGRE MV, MV Oberrhein, IEEE
118, PEGASE 1354 and 9241, and pandapower's multi-voltage example (three-winding
transformer, equivalents) — and the tests check it on every bundled network. The
CGMES profiles alone, as another tool reads them, carry the bundled networks
exactly too, and so do PEGASE's phase shifters (a shift that is not a multiple of
30° is written as the clock's multiple plus a linear phase tap changer; beside a
ratio tap, that phase tap changer goes on the other end, and the importer undoes
pandapower's converter copying each tap's step to the other). DC lines are
written as two voltage-source converters and a DC line segment between their DC
nodes, which pandapower's converter reads back as the same DC line. What it
reads back differently is listed, and CONDUCTOR's copy gets it from the extras:
the sign of a negative resistance or reactance (read as a magnitude), a motor's
consumption (read with the generator's sign), a static var compensator (read as
a fixed shunt) and a DC line's loss percentage (read as a fixed loss). Motors
are written as asynchronous machines, SVCs as static var compensators.
`describe_network(about="model")` reports the round trip for the network loaded.

**CGMES 3.0 through pandapower.** pandapower 3.5's reader cannot read a CGMES
3.0 model with lines at all (its 3.0 parser keeps no line in-service state, which
its converter then reads), and drops every power-electronics connection — PV,
wind, batteries — which it converts for the LTDS profile only. CONDUCTOR's
importer puts both back (ENTSO-E's 3.0 MicroGrid loads and solves), gives a
boundary node with no nominal voltage its neighbour's and a static var
compensator with no set point its bus's, and says each in the import notes. A
3.0 model sent without its ShortCircuit profile has no vector groups: said too.

**The network's state is its model and differences.** What CONDUCTOR studies
is the CIM model and the changes made to it, each kept as a difference against
the model (`backend/cim_changes.py`: objects added and removed, statements set
and unset, keyed by mRID; IEC 61970-552 when exported):

- *the operator's standing changes* — a unit's generator model (its control,
  set-points and limits) and the names given — against the model as loaded
  (`GET /api/network/cim/operator`);
- *each proposal* against the base (`GET /api/network/cim/proposal`).

The pandapower network a study runs on is projected from them
(`project_onto`): the network as loaded, with each difference applied element by
element by mRID — an element changed takes the values the adapter reads from
the part of the model the difference touches, one added is placed at its row,
one moved between tables (a voltage-controlled unit set to a fixed power factor)
keeps its mRID and takes its cost along, every other row is untouched. An
element the base has takes only the values the difference changes: the touched
part of the model is read without the difference and with it, and only what
differs between the two readings reaches the row. For a network loaded as CIM,
the differences name the model's own objects — its terminals, transformer ends,
tap changers, regulating controls, generating units and limits — so they apply
to its own files. Only the
touched part of the model is read, so a change costs what it touches: on PEGASE
9241, about half a second, against a few milliseconds to build it from the edits.
If a difference cannot be made or projected, the network is built as before, by
the edits or the generator models, and the log says so.

The edits and the operator's settings stay the language they are given in. Each
is translated into a difference: a line reinforced, a transformer upgraded, a
load set or scaled, a switch opened or closed (`set_switch`: its `Switch.open`),
a move within a voltage level change the model's
statements directly; any other edit — equipment added, a move to another
voltage level, a parameter such as a line's length — and the operator's changes
are applied to a scratch copy, and the elements they changed are written again
by the writer (`reemit`), which stays the one definition of how a pandapower row
is written as CIM. Each proposal records how its edits were translated.

**Every moment as the model's operating point.** At each tick the measurements
are allocated to the loads and units (as before). The result is a moment: every
measured element's active and reactive power, keyed by mRID
(`backend/operating_point.py`). It is computed once for the networks that share
their measured elements and what the allocation reads of them — the base and
the proposals that leave its measured elements alone — and applied to each by
mRID; a network whose measured elements differ (a unit's capacity changed) is
allocated on its own. `GET /api/network/cim/moment?timestamp=…` gives a moment
as a CGMES SSH profile against the network's model (or each element's values by
mRID with `format=json`).

**Solved states as the model's SV.** `GET /api/network/cim/solved?timestamp=…`
(or `source=nominal`) solves the network at a moment and writes the result as a
CGMES state-variables profile against its model: each node's voltage and angle,
the power at every terminal (branches at both ends), the tap positions — under
the model's own identifiers for a network loaded as CIM — so another tool can
read CONDUCTOR's solution or compare it with its own. On ENTSO-E's SmallGrid at
its own operating point, every node of the model is there and the voltages are
within 0.03 kV of the model's own SV.

**Elements named by their mRID.** Every study that takes an element (a line to
take out, a unit to hold) accepts its mRID as well as its name: the model's own
for a network loaded as CIM, the layer's for any other. A unit a CIM model
names keeps its name and its mRID when CONDUCTOR converts it to a static unit
(the second of two alike is numbered), and its source is what the model's
generating unit says (a `ThermalGeneratingUnit` is thermal, stated).

**Saved and exported as files.** The differences are saved with the proposals
and the operator's settings, with what they were made against: the network as
loaded, the operator's changes, the writer's version. After a restart, a
difference made against the same model is used as saved; otherwise it is made
again from its edits or settings. The CIM package (**Export as CIM**) carries
the model, the operator's changes, each proposal (`proposals/<id>.xml`) and a
manifest of what each file is. A difference file carries, in CONDUCTOR's own
namespace (which other tools ignore), the profile of each statement and what
CGMES has no place for, so it reads back as the same difference
(`cim_changes.from_xml`).

**Checked end to end.** `python -m evaluation.cim_central run` builds every
network these four ways and the way CONDUCTOR built it before, and compares them
element by element through the mRIDs — every column of every table, the costs,
the row indices and the power flow:

- proposals: every kind of edit;
- operator changes: generator models and names;
- moments: the tick allocated on each network, or the base's moment applied;
- saved state: differences read back from their files.

It runs on IEEE 14 and 30, CIGRE MV, pandapower's multi-voltage example (also
loaded from CGMES, where every object a difference changes must be one of the
model's own) and MV Oberrhein (add `--large` for PEGASE 1354 and 9241, with
timings), and writes a manifest. Every check gives the same network.

**Switches the converter does not read.** pandapower's converter reads breakers,
disconnectors, load-break switches and generic switches; a fuse, jumper,
recloser, sectionaliser or ground disconnector it leaves out, and a fuse in series
then leaves everything beyond it cut off. Each is added at import as a switch
between its two nodes, open or closed as the model has it, with its class, so
the topology is the model's.

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
| N-1 screen | a scope: the outages of one feeder, substation or voltage level; for a network loaded as CIM, the model's temporary limits (TATL): each overload after an outage says how long it may be held, and the result the time left to act |
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

A proposal *is* a difference to the CIM model (above). For the agent,
`describe_network(network=<proposal>, about="changes")` reads it element by
element: each element its difference touches, by mRID, added, removed or
changed, with the statements that change (a line's r and x, its limit). Elements
keep their mRIDs between the base and a proposal, so the layer's facts are
compared element by element too (`cim_export.difference`: a feeder's members, a
unit's source, a bus's level), written as a difference model of the layer
(`GET /api/network/cim/difference?network=<proposal>`).

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
