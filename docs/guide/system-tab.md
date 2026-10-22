# The System tab

The System tab shows the loaded network and everything CONDUCTOR assumes about
it. At the top: the network's name and size, and a strip of **assumption chips**
(load power factor, generators, costs, nominal point). A chip is highlighted when
the assumption differs from the default; clicking it opens the assumptions.

Three views: **🗺️ Network map**, **⚙️ Model assumptions** and **🏷️ Names**.

## Network map

A schematic of the network (connectivity, not geography), coloured by the
operating point in the sidebar: bus voltages, line and transformer loading,
generation and demand, the external grid. The subtitle names the moment and any
assumption that is not the default.

With proposals, choose which network to draw: the base or a proposal, and for a
proposal its own map or the **difference from the base**.

## Model assumptions

What every study assumes beyond the data. Changes apply from the next study on;
nothing has to be regenerated. CONDUCTOR can read and change all of these when
you ask. Earlier answers are not recomputed; they are marked when computed under
other settings (see [Reading an answer](reading-an-answer.md#answers-computed-under-other-settings)).

### Generators

Each unit, grouped by how it is modelled, with an **Edit** button:

| Setting | Meaning |
| --- | --- |
| **Control** | *Voltage control*: holds its bus voltage; its reactive power is whatever that takes. *Fixed power factor*: reactive power in a fixed ratio to active power. *Fixed Q*: a set reactive output. Large synchronous machines usually control voltage; inverter-based units usually run at a fixed power factor |
| **V set-point** | The voltage a voltage-controlled unit (or condenser) holds |
| **Power factor** | For fixed-power-factor units, injecting or absorbing |
| **Reactive output** | For fixed-Q units: positive injects, negative absorbs |
| **Reactive range** | The unit's reactive limits. The optimisations respect them; power flows report each unit's reactive power against them |
| **Dispatchable** | Whether the optimisations may change the unit's active power |

Synchronous condensers have no active power: they always control voltage and
are never dispatched. **Reset all to as loaded** undoes every change.

When your data carry measured reactive generation, voltage-controlled units at
those buses can be switched to follow it with one click.

### Costs

Each unit's costs, in two tables with an **Edit** button per row.

| Figure | Meaning |
| --- | --- |
| **€/MWh, €/MW²h, €/h** | Production cost: €/h + €/MWh × P + €/MW²h × P² |
| **Up €/MWh** | What each MWh of upward redispatch costs |
| **Down €/MWh** | What each MWh of downward redispatch costs (for renewables, curtailment) |
| **Q €/Mvarh** | What each Mvarh of reactive power the unit is asked to inject or absorb costs |
| **External grid** | Its import price (production €/MWh); reactive exchange is free within a band (\|Q\| ≤ tan φ · \|P\|) and charged per Mvarh beyond it |

**Source** says where each figure comes from: *case* (the network file's own
generation costs), *synthetic* (a fixed rule, shown under the tables) or
*changed* (by you or CONDUCTOR). In the Edit form, an emptied field goes back
to its default; **Back to defaults** resets one unit, **Reset all to defaults**
every unit.

The costs price the [corrective options](asking-questions.md#corrective-options)
and decide which one is cheapest. The default corrective dispatch minimises the
change in dispatch, so a cost does not change it. CONDUCTOR says so if asked.

### Load power factor

The power factor that gives each measured load its reactive power when the data
carry active power only (0.95 lagging by default). With measured reactive power,
it only fills gaps. One study can use another value without changing this
setting: *"rerun at power factor 0.90"*.

## Names

Give buses, lines, transformers, loads and units the names your team uses. One
tab per kind of element; **Edit** on a row opens the rename form, and
**Restore** gives back the name it was loaded with.

- Every chart, table and answer uses the new name from the next study on, and a
  bus's lines and transformers are relabelled with it.
- Your data stay matched to the names the network was loaded with: a renamed
  bus keeps its measurements. The Data tab shows it as "North (data: Bus_3)".
- Names must be unique within a kind; `|`, `[` and `]` are not allowed.
- Earlier answers keep the names they ran with and say so.
- CONDUCTOR can rename too: *"Rename Bus 3 to North."*

Renames, costs, generator models and the power factor are kept for the network
across restarts; see [Sessions and scenarios](sessions-and-scenarios.md). When
CONDUCTOR changes any of them in an answer, the answer says so and offers
**Undo** (see [Reading an answer](reading-an-answer.md#answers-that-changed-a-setting)).
