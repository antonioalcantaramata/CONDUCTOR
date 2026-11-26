# What the network is: the CIM layer

A network file is a set of tables: buses, lines, generators, loads. It rarely
says what an operator needs to know: which units are solar, which switch ties
two feeders, whether a value was measured or assumed. An assistant that guesses
these from names gets them wrong. A diesel unit renamed "Solar Farm North" is
counted as solar, and a study that halves solar output then changes the wrong
units.

CONDUCTOR therefore describes every loaded network in the **Common Information
Model (CIM)**, the IEC standard in which system operators exchange their grid
models, and records **how each fact is known**. This description is the **CIM
layer**. The agent answers from it, the studies read it (a solar scenario
changes exactly the units the layer says are solar), and every answer is
checked against it. You see it, and correct it, under
**System → 🧭 CIM layer**.

## How each fact is known

Every fact carries one of four labels:

| Label | Meaning | Example (CIGRE MV) |
| --- | --- | --- |
| **stated** | The network says so (a type column, a CIM class), or you declared it | `Solar Farm North`, type `CHP diesel`: thermal |
| **named** | Only the element's name suggests it: a likely reading, said as such | a unit called `Wind Park` with a generic type: wind |
| **inferred** | Worked out from the topology, by a rule named with the fact | `S1`, normally open, ties the two feeders |
| **unknown** | The network does not say | every unit of PGLib's IEEE 30: the case records no fuel |

An unknown fact is never filled in with a guess. If a study needs it, the
answer says it is unknown, and you can tell CONDUCTOR what it is.

The layer also knows things that no table stores, worked out from the
topology as the network is switched now:

- each **feeder** (named after its first line) and what it supplies;
- the **role of each switch**: a tie between two feeders, a normally open point
  closing a loop within one, a sectionalising switch;
- what an **outage cuts off**, and which normally open **ties could restore**
  supply. Each tie is checked with a power flow at the current operating point:
  what it restores, the lowest voltage, the highest loading, and whether the
  network stays within its limits. It is one moment's check, not a switching
  plan.

And for each value of the operating point, where it comes from: **measured**,
a share of its bus's measurement (**allocated**), **derived** from an
assumption such as the load power factor, or **solved** by the power flow.

## System → CIM layer

| Part | What it shows |
| --- | --- |
| **Headline** | The network at a glance: its voltage levels, its substations, how many units have a known source and loads a known category, and where the external grid connects |
| **Picture** | Units by energy source, or loads by category, by voltage level and substation. A lighter shade: named or inferred, not stated. Grey: unknown |
| **Tabs** | Units, loads, lines, transformers, buses, switches, feeders, shunts, and **Data** (where each value comes from), with how each fact is known |
| **Look around an element** | Pick a bus, line, transformer, unit, load or substation: where it sits, how it is supplied, and what would lose supply without it |
| **Export as CIM…** | The whole layer as CIM RDF/XML or JSON-LD, for other CIM tools |

### Correcting a fact

In the **Units** or **Loads** tab, **Edit** on a row opens *Set what it is*:
choose the unit's energy source or the load's category and **Apply**. Your
setting is marked *declared*, overrides what a name suggests, and every later
study uses it. **Forget** goes back to what the network says; **Forget
everything I set** undoes all of them.

You can do the same in the chat: *"Fuel cell 1 has been replaced by a wind
turbine."* CONDUCTOR records it and says so. Declarations are kept for that
network across restarts, like renames.

## Your own CIM models

A **CGMES** model (CIM 2.4.15 or 3.0) can be uploaded as a `.zip` of its
profiles, with the boundary set when it needs one (see
[Data and the operating point](data-and-operating-point.md)). Its identifiers,
substations, unit types and switch states are then *stated* by the model, not
inferred. What CIM itself does not store, such as feeders and what an outage
cuts off, the layer still works out from the topology.

The whole model is kept, not only what the studies use. The layer reads from it
the names of substations, voltage levels and busbars, every limit on a line
(also the ones that may be exceeded for a while), the controls of machines,
shunts and tap changers, and how each load follows the voltage. Ask *"what does
the model have that the studies do not use?"*: the agent says, for example,
that the studies check only permanent limits, or that a shunt regulating a
voltage in the model is held fixed. When the model carries its own power-flow
solution, CONDUCTOR's power flow is compared with it on loading; a difference is
listed among the data issues.

In **System → CIM layer**, the picture *The CIM model* shows what the layer does
with each kind of object in your model, and the *CIM model* tab gives the
details. The network map draws a node-breaker model as its electrical nodes,
named after their busbars, and uses your model's own diagram when it has one, so
the network looks as it does in the tool it came from. **Export as CIM** then
also offers *your model + CONDUCTOR's additions*: your files exactly as uploaded,
plus one file with what CONDUCTOR adds, by your model's own identifiers.

A network that did not come as CIM gets a CIM model too: CONDUCTOR writes it as
CGMES from its data, and **Export as CIM** offers it as *The network as CGMES*,
for other CIM tools. Under **As CGMES**, *Check that it reads back the same*
reads it back and compares the power flow element by element.

What you change is kept as changes to that model: your generator models and
renames, and each proposal, as CIM difference files against it. The studies run
on the model with them applied. Both downloads of **Export as CIM** carry them
(the operator's changes, `proposals/<id>.xml`, and a manifest saying what each
file is), so another CIM tool can take the network with your changes as one
package.

**Export as CIM** writes the layer back out in the distribution namespace of
CIM (CIM100). How each fact is known goes in a namespace of its own beside the
standard properties, so a tool that knows only CIM still reads a valid model.

## In your answers

- **Questions about the network itself** are answered from the layer. See
  [Asking questions](asking-questions.md#what-the-network-is) for examples.
- **What an answer says about an element is checked** against the layer: a unit
  called the wrong source, a voltage the bus does not have, a value called
  measured that was assumed. See
  [Reading an answer](reading-an-answer.md#answer-checks).

For the full description (every CIM class, the rules behind each inference,
the export and the checks), see
[What the network is](../methods/what-the-network-is.md) under *How it works*.
