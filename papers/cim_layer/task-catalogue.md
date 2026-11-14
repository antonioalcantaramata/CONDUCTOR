# Task catalogue

Every task of the evaluation suites, generated from the code that runs them (`evaluation/catalogue.py`). Each check is a fact of the network or of the study's own record, computed rather than judged: no model grades another, and nothing is labelled by hand. The facts the reasoning and look-up tasks are graded against are themselves checked against the networks by `tests/test_ablation_ground_truth.py`.

| Suite | Tasks | Turns per repeat |
|---|---:|---:|
| [whatif](#whatif) | 10 | 12 |
| [investigation](#investigation) | 19 | 21 |
| [options](#options) | 11 | 13 |
| [clarify](#clarify) | 5 | 7 |
| [narration](#narration) | 6 | 6 |
| [semantics-cigre](#semantics-cigre) | 5 | 5 |
| [semantics-ieee30](#semantics-ieee30) | 5 | 5 |
| [reasoning](#reasoning) | 27 | 28 |
| **all** | **88** | **97** |

## whatif

Building the proposal the operator describes and studying it.

Every task here is also graded on: named the proposal; no refused edit.

**`dc_flat`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> What if we connect a 2 MW data centre at Bus 9? Compare with today.

- Check: edit: 2 MW load at Bus 9
- Check: looked beyond one moment

**`dc_follow_worst`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Add a 1.5 MW load at Bus 4 that follows the bus's own demand. When in the week is it worst compared with today?

- Check: edit: 1.5 MW follow_bus load at Bus 4
- Check: studied the week

**`factory_shift`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> A 3 MW factory would run from 08:00 to 18:00 at Bus 11. Would it cause violations during the week?

- Check: edit: 3 MW shift 08–18 at Bus 11
- Check: studied the week

**`reinforce_after_load`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> 1. Create a proposal with a 4 MW load at Bus 9 and compare it with today at 2000-01-03 19:00.
> 2. Would doubling the line Bus 8 -> Bus 9 help?

- Check: edit: 4 MW load at Bus 9
- Check: edit: line L6 ×2
- Check: compared at the same moment

**`new_site_n1`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Build a new substation 5 km from Bus 9 with the same conductor as the line Bus 8 -> Bus 9 and put a 1 MW data centre there. Is it N-1 secure?

- Check: edit: new bus 5 km from Bus 9 like L6
- Check: edit: 1 MW load at the new bus
- Check: N-1 on the proposal
- Check: said a single-line site loses supply

**`open_line_n1`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> What happens if we permanently open the line Bus 3 -> Bus 4 [L2]? Check N-1 too.

- Check: edit: open L2
- Check: N-1 on the proposal
- Check: said what loses supply

**`wind_week`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Add a 2 MW wind farm at Bus 11 and compare the external grid import with today over the week.

- Check: edit: 2 MW wind at Bus 11
- Check: studied the week

**`trafo_upgrade`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Would upgrading the transformer Bus 0 -> Bus 1 [T0] to 40 MVA change anything at the worst hour of the week?

- Check: edit: T0 to 40 MVA
- Check: found the worst hour

**`classical_generator`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> 1. Add a 2 MW generator at Bus 4 and tell me whether the grid stays secure.
> 2. A classical one that holds its bus voltage — use 1.0 p.u.

- Check: asked how the unit is modelled before building it
- Check: edit: 2 MW unit on voltage control at Bus 4

**`new_substation_trafo`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Build a new 0.4 kV substation at Bus 9 with two 0.63 MVA transformers (vk 6 %, vkr 1 %) and connect a 0.8 MW data centre to it. Is it N-1 secure?

- Check: edit: 2 × 0.63 MVA transformer to a 0.4 kV bus at Bus 9
- Check: edit: 0.8 MW load at the new bus
- Check: N-1 on the proposal
- Check: reported how the units are modelled

## investigation

Investigating a "why" about the grid — and only then.

**`why_bus29`** — IEEE 30 (PGLib; unit types not stated)

> Why is Bus_29 below its voltage limit at 20:00 on 6 January?

- Check: investigated
- Check: named the demand near Bus_29 as the main cause
- Check: recorded the investigation
- Check: gave nothing untested as a finding

**`what_caused_overload`** — IEEE 30 (PGLib; unit types not stated)

> What caused the overload of the line from the external grid at 20:00 on 6 January?

- Check: investigated

**`why_insecure`** — IEEE 30 (PGLib; unit types not stated)

> Why is the grid insecure at 20:00 on 6 January when it was fine in the afternoon?

- Check: asked which element, or investigated

**`no_symptom`** — IEEE 30 (PGLib; unit types not stated)

> Why is Bus_29 below its voltage limit at 04:00 on 6 January?

- Check: checked with the investigation
- Check: said it is within its limits

**`why_physics`** — IEEE 30 (PGLib; unit types not stated)

> Why does reactive power affect voltage?

- Check: did not investigate

**`what_if`** — IEEE 30 (PGLib; unit types not stated)

> What happens if Gen_bus4 produces 10 MW less at 20:00 on 6 January?

- Check: did not investigate

**`why_method`** — IEEE 30 (PGLib; unit types not stated)

> 1. Is the grid secure at 20:00 on 6 January?
> 2. Why did you use that power factor for the loads?

- Check: did not investigate

**`own_hypotheses`** — IEEE 30 (PGLib; unit types not stated)

> On proposal two_loads, Bus_29 is below its limit at 18:00 on 6 January but fine on the base. Is it the extra demand at Bus_23 or at Bus_25, or both?

- Set-up: proposal `two_loads`: {"bus": 23, "element_type": "load", "factor": 2.0, "op": "scale"}; {"bus": 25, "element_type": "load", "factor": 1.6, "op": "scale"}
- Check: tested its own hypotheses (Bus_23, Bus_25)
- Check: recorded Bus_23 as supported
- Check: recorded Bus_25 as supported

**`untestable`** — IEEE 30 (PGLib; unit types not stated)

> Could the hot weather that evening explain the overload of the line from the external grid at 20:00 on 6 January?

- Check: recorded the weather as untested, with the tool needed
- Check: said it could not be tested

**`why_outage`** — IEEE 30 (PGLib; unit types not stated)

> Why would losing the line Bus_28 -> Bus_29 be critical at 18:00 on 6 January when it was not at 15:00?

- Check: investigated after the outage
- Check: recorded the investigation

**`why_window`** — IEEE 30 (PGLib; unit types not stated)

> Why was the line from the external grid outside its limit for longer on 6 January than on 5 January?

- Check: investigated the window

**`why_so_close`** — IEEE 30 (PGLib; unit types not stated)

> At 16:30 on 6 January Bus_29 is within its limits but close to them. Why is it so close, compared with a moment when it had a comfortable margin?

- Check: compared with a comfortable moment

**`classical_views`** — IEEE 30 (PGLib; unit types not stated)

> Why is the line from the external grid overloaded at 20:00 on 6 January? Who is using that line, and how far would a sensitivity-based estimate get?

- Check: asked for the classical views
- Check: said who uses the line

**`recurring`** — IEEE 30 (PGLib; unit types not stated)

> Is the undervoltage at Bus_29 a recurring problem this week, and is it always the same cause?

- Check: investigated every episode
- Check: named the recurring cause

**`prevention`** — IEEE 30 (PGLib; unit types not stated)

> 1. Could the undervoltage at Bus_29 on the evening of 5 January have been avoided, and how much warning did we have?
> 2. The one around 18:30.

- Check: checked prevention
- Check: said how much warning there was

**`look_ahead`** — IEEE 30 (PGLib; unit types not stated)

> What should I watch for in the next day of the forecast, and what would prevent it?

- Check: looked ahead
- Check: said when to decide
- Check: said it is a forecast

**`deep_why`** — IEEE 30 (PGLib; unit types not stated) — Deep Research on the first question

> Why is Bus_29 below its voltage limit at 20:00 on 6 January?

- Check: investigated
- Check: checked prevention
- Check: looked for recurrence
- Check: recorded the investigation
- Check: proposed an analysis of the grid CONDUCTOR lacks
- Check: proposed data for what lies behind the cause
- Check: said what was not tested, and nothing untested as a finding

**`deep_analysis`** — IEEE 30 (PGLib; unit types not stated) — Deep Research on the first question

> At 20:00 on 6 January, give me options to clear the violations.

- Check: did not investigate
- Check: asked the solver for options
- Check: recorded as answered without Deep Research

**`injected_setpoint`** — IEEE 30 (PGLib; unit types not stated)

> On proposal inj_setpoint, Bus_29 is below its limit at 18:00 on 6 January but on the base it is fine. Why?

- Set-up: proposal `inj_setpoint`: {"element": "Gen_bus12", "element_type": "gen", "op": "set_parameter", "values": {"vm_pu": 0.95}}
- Check: investigated against the base
- Check: named Gen_bus12's set-point

## options

Corrective options, not a winner.

Every task here is also graded on: left the choice to the operator.

**`options_peak`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January, give me options to clear the violations.

- Check: asked the solver for options

**`rule_out_unit`** — IEEE 30 (PGLib; unit types not stated)

> 1. At 20:00 on 6 January, give me options to clear the violations.
> 2. I don't trust Gen_bus1 today.

- Check: re-solved with Gen_bus1 held
- Check: did not take Gen_bus1 out of service

**`near_optimal`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January, the cheapest corrective action is fine on cost, but are there other ways that cost about the same and use different units?

- Check: asked for near-optimal alternatives
- Check: said which units an alternative avoids or adds

**`near_optimal_margin`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January, which corrective actions cost at most 25 % more than the cheapest one?

- Check: used the margin asked for

**`branch_cap`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January, give me options to clear the violations that keep L30 below 80 % of its rating.

- Check: held L30 below 80 % in every option

**`n1_secure`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January, give me corrective options that stay secure if a line trips.

- Check: ran the N-1 security study
- Check: said what could not be secured

**`n1_corrective`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January, which dispatch lets us fix any critical line outage with a quick redispatch afterwards, and what would we do after each?

- Check: asked for the corrective N-1 options
- Check: gave the corrective plan

**`last_resort`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January with loads at 125 %, give me options to clear the violations.

- Check: reached the last resort
- Check: said redispatch alone is not enough and named demand response

**`hold_option`** — IEEE 30 (PGLib; unit types not stated)

> 1. At 20:00 on 6 January, give me options to clear the violations.
> 2. Hold the cheapest one for the next 4 hours: until when does it keep every limit?

- Check: held the chosen option

**`plan_peak`** — IEEE 30 (PGLib; unit types not stated)

> Plan the corrective action for the evening peak on 6 January, from 17:00 to 22:00, keeping instructions steady.

- Check: planned the window as one

**`no_action`** — IEEE 30 (PGLib; unit types not stated)

> At 03:00 on 6 January, give me corrective options for any violations.

- Check: established that no action is needed
- Check: said no action is needed

## clarify

Asking first when the question leaves the answer open.

**`which_line`** — IEEE 30 (PGLib; unit types not stated)

> 1. At 20:00 on 6 January, simulate losing the line into Bus_23.
> 2. The one from Bus_22.

- Check: asked which line before running
- Check: offered numbered choices
- Check: then simulated the outage of L31

**`unstated_margin`** — IEEE 30 (PGLib; unit types not stated)

> 1. At 20:00 on 6 January, give me options that keep L30 comfortably below its rating.
> 2. 75 %.

- Check: asked how far below before running
- Check: then held L30 at 75 % in every option

**`secure_now`** — IEEE 30 (PGLib; unit types not stated)

> Is the grid secure right now?

- Check: assessed without asking

**`n1_unspecified`** — IEEE 30 (PGLib; unit types not stated)

> At 20:00 on 6 January, give me options that stay secure if a line trips.

- Check: ran the N-1 study without asking
- Check: covered preventive and corrective (n1=both)

**`worst_outage`** — IEEE 30 (PGLib; unit types not stated)

> Which single outage is the most critical at 20:00 on 6 January?

- Check: screened every outage without asking

## narration

Reporting a study as its own result says.

**`hosting`** — IEEE 30 (PGLib; unit types not stated)

> How much more generation can Bus_13 host?

- Check: ran the hosting capacity
- Check: glossary: unity
- Check: glossary: fixed power factor
- Check: glossary: the reactive proxy is a proxy

**`probabilistic`** — IEEE 30 (PGLib; unit types not stated)

> What is the probability of a violation at 20:00 on 6 January?

- Check: ran the probabilistic assessment
- Check: led with the probability of any violation
- Check: explained the uncertainty modelled

**`robust`** — IEEE 30 (PGLib; unit types not stated)

> Give me a dispatch at 20:00 on 6 January that stays within limits with 95 % confidence.

- Check: ran the robust dispatch
- Check: reported the risk out of sample
- Check: explained the back-off, or why it fell short

**`envelope`** — IEEE 30 (PGLib; unit types not stated)

> How much reactive power can Gen_bus7 inject or absorb right now?

- Check: ran the flexibility envelope
- Check: said it holds at this operating point only
- Check: named the binding constraint

**`historical_risk`** — IEEE 30 (PGLib; unit types not stated)

> How often does Bus_29 violate its limits over the week, and how often does it come close?

- Check: ran the historical risk
- Check: gave the near misses
- Check: told the worst episodes

**`renewables`** — IEEE 30 (PGLib; unit types not stated)

> What if renewable output drops 30 %? Compare with the baseline over the week.

- Check: ran the scenario scan
- Check: compared with the baseline
- Check: stated the uniform-scaling limitation

## semantics-cigre

Looking up what elements are, on a network that states nearly everything.

**`cigre_renewables`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which units in this network are renewable?

- Check: named the wind turbine and PV units

**`cigre_levels`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> At what voltage are the PV units connected, and where does the network connect to the external grid?

- Check: 20 kV, and Bus 0 at 110 kV

**`cigre_storage`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Is there any storage in this network, and where?

- Check: both batteries and their buses

**`cigre_outage`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> If Line 2-3 trips, which loads lose supply, and which normally open tie could restore them?

- Check: named what loses supply, as far as Bus 11
- Check: named the tie Line 14-8

**`cigre_feeder`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which feeder supplies Load R11, and through which lines does its power come from the external grid?

- Check: feeder Line 1-2 and the route through Line 3-8 to Line 10-11

## semantics-ieee30

Looking up what elements are, on a network that states almost nothing: often the right answer is that it does not say.

**`ieee30_renewables`** — IEEE 30 (PGLib; unit types not stated)

> Which units in this network are wind or solar?

- Check: said the network does not say

**`ieee30_customers`** — IEEE 30 (PGLib; unit types not stated)

> What kind of customers does the load at Bus_7 serve?

- Check: said the network does not say

**`ieee30_level`** — IEEE 30 (PGLib; unit types not stated)

> What is the nominal voltage of this network?

- Check: 135 kV

**`ieee30_radial`** — IEEE 30 (PGLib; unit types not stated)

> If the line Bus_24 -> Bus_25 [L33] trips, does any load lose supply?

- Check: named what is cut off
- Check: said it loses supply

**`ieee30_external`** — IEEE 30 (PGLib; unit types not stated)

> Which lines carry the power coming in from the external grid?

- Check: L0 and L1

## reasoning

Reasoning from what the elements are: misleading names, interventions (a fact changed, the answer and the study must follow), composition, switches, what is measured, consistency.

### reasoning · names

**`names_cigre`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which units in this network are solar, and how many are there?

- Set-up: element sgen 10 renamed “Solar Farm North”
- Set-up: element sgen 11 renamed “PV Park East”
- Check: named the PV units
- Check: counted eight
- Check: did not take 'Solar Farm North' (a CHP unit) for solar
- Check: did not take 'PV Park East' (a fuel cell) for solar

**`names_ieee30`** — IEEE 30 (PGLib; unit types not stated)

> Which units in this network are solar?

- Set-up: element gen 0 renamed “Solar Plant A”
- Check: said 'Solar Plant A' is solar only by its name

**`names_cigre_wind`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which units in this network are wind turbines?

- Set-up: element sgen 1 renamed “Wind Park West”
- Check: named WKA 7
- Check: did not take 'Wind Park West' (a PV unit) for wind

**`names_cigre_storage`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Where is the storage in this network?

- Set-up: element sgen 9 renamed “Battery Storage 3”
- Check: named Battery 1 and Battery 2
- Check: did not take 'Battery Storage 3' (a fuel cell) for storage

**`names_ieee30_wind`** — IEEE 30 (PGLib; unit types not stated)

> Which units in this network are wind?

- Set-up: element gen 3 renamed “Offshore Wind 1”
- Check: said 'Offshore Wind 1' is wind only by its name

### reasoning · intervention

**`drop_ieee30_none`** — IEEE 30 (PGLib; unit types not stated)

> If wind and solar output dropped by 50 % at 2000-01-03 12:00, would the grid stay secure, and which units would that affect?

- Check: said no unit is stated wind or solar
- Check: named no unit as affected

**`drop_ieee30_declared`** — IEEE 30 (PGLib; unit types not stated)

> If wind and solar output dropped by 50 % at 2000-01-03 12:00, would the grid stay secure, and which units would that affect?

- Set-up: declared by the operator: Gen_bus12 is wind
- Check: named Gen_bus12 as affected
- Check: the study scaled Gen_bus12 only

**`drop_cigre_stated`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> If wind and solar output dropped by 50 % at 2000-01-03 12:00, would the grid stay secure, and which units would that affect?

- Check: named the wind and PV units
- Check: did not count CHP diesel 1
- Check: the study scaled the nine wind and PV units

**`drop_cigre_declared`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> If wind and solar output dropped by 50 % at 2000-01-03 12:00, would the grid stay secure, and which units would that affect?

- Set-up: declared by the operator: CHP diesel 1 is solar
- Check: named CHP diesel 1 as affected (declared solar)
- Check: the study scaled CHP diesel 1 too

### reasoning · composition

**`compose_t0`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> If the transformer Bus 0 -> Bus 1 [T0] trips, which generating units lose their connection, how many of them are solar, and could supply be restored?

- Check: named the wind turbine among them
- Check: eight of them solar
- Check: the tie Line 14-8 restores it

**`compose_battery`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> At 18:00 on 3 January, is Battery 1 on the same feeder as the bus with the lowest voltage?

- Check: found Bus 11 lowest
- Check: said yes, the same feeder
- Check: named the feeder Line 1-2

**`compose_l33`** — IEEE 30 (PGLib; unit types not stated)

> If the line Bus_24 -> Bus_25 [L33] trips, how much load loses supply, and is there a tie that could restore it?

- Check: 1.84 MW
- Check: no tie restores it

**`compose_tie`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> If Line 2-3 trips, which switch could restore supply to the far end of the feeder, and is it normally open?

- Check: named the tie S1 (Line 14-8)
- Check: said it is normally open

### reasoning · switches

**`switches_open`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which switches in this network are normally open, and which of them ties two feeders?

- Check: named the three normally open switches
- Check: said S1 is the tie between feeders

**`switches_loops`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which normally open switches close a loop within one feeder, rather than tying two feeders together?

- Check: named S2 and S3
- Check: did not count S1 (it ties two feeders)

**`switches_breakers`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which switches in this network are circuit breakers, and what do they switch?

- Check: named Switch 6 and Switch 7
- Check: said they switch the transformers

**`switches_line148`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which switch decides whether Line 14-8 carries power, and what is its normal state?

- Check: named its switch S1
- Check: said it is normally open

### reasoning · data

**`data_cigre`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Is the reactive demand of Load R5 measured? And is PV 9's output measured?

- Check: Load R5's Q: derived from the power factor, not measured
- Check: PV 9's output: a share of its bus's measurement

**`data_ieee30`** — IEEE 30 (PGLib; unit types not stated)

> Is the reactive power of Load 5 measured?

- Check: said it is derived from the power factor

**`data_cigre_wind`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which output is measured directly, and which is estimated: WKA 7's or PV 9's?

- Check: WKA 7's output: measured
- Check: PV 9's output: a share of its bus's measurement

**`data_cigre_battery`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Is Battery 1's output in the measurement data?

- Check: held at its set-point; the data carry no storage

**`data_cigre_load_split`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Is Load R1's demand measured on its own?

- Check: a share of Bus 1's measurement, split with Load CI1

**`data_ieee30_solved`** — IEEE 30 (PGLib; unit types not stated)

> Is Gen_bus12's reactive power measured?

- Check: said the power flow solves it

### reasoning · consistency

**`para_weather`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which units in this network depend on the weather?

- Check: named the wind and PV units
- Check: did not count CHP diesel 1

**`para_vary`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> Which generators' output varies with the wind or the sun?

- Check: named the wind and PV units
- Check: did not count CHP diesel 1

**`para_variable`** — CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)

> List the variable renewable units in this network.

- Check: named the wind and PV units
- Check: did not count CHP diesel 1

**`persist_ieee30`** — IEEE 30 (PGLib; unit types not stated)

> 1. Gen_bus12 is a wind farm.
> 2. If wind output dropped by 50 % at 2000-01-03 12:00, which units would be affected, and would the grid stay secure?

- Check: named Gen_bus12 as affected
- Check: a study lowered Gen_bus12 only
