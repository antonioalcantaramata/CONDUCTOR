# Investigating a cause

*Why is Bus_29 below its limit?* is a comparison: below its limit now, and not
below it at some other moment. The investigation makes that comparison
explicit and measures it. It takes the element's value now and in a
**reference state** where the element was fine, lists everything that differs
between the two states, and divides the change among those differences. Each
share comes from AC power flows, and the shares add up exactly to the change.

It answers a different question from the
[violation attribution](contingency-analysis.md#violation-attribution):

| | Attribution | Investigation |
| --- | --- | --- |
| Question | What moves this element now, and what would I have to move? | What changed since it was fine, and how much of the change is each cause? |
| Method | Sensitivities: one small step per source, at this moment | Every combination of the real changes between two states |
| Result | A ranking per MW or Mvar, and the movement each source would need | A share per factor that adds up to the change, with a verdict |
| What it cannot see | A set-point, a line out, the external grid's voltage (not injections) | Nothing a power flow sees is left out |

What changed is the **cause**, but it is not necessarily the **remedy**: demand
that rose in an area explains an undervoltage, yet the operator cannot lower
it. An investigation therefore ends with the attribution's levers, the things
to move now, kept apart from the causes.

## What can be investigated

| Scope | The quantity divided | Asked as |
| --- | --- | --- |
| One element at one moment | a bus voltage (p.u.) or a branch's loading (% of its rating) | `element` |
| The whole grid at one moment | the number of elements outside their limits; with only one outside, that element's own voltage or loading, which divides the causes far more finely than a count of one (the result says so in `narrowed`) | `element: "all"` |
| After an outage | the same, with a line or transformer out in both states | `outage` |
| A window | an element's exceedance (how far outside its limits, times how long), its average if it never left them, or the grid's element-hours outside limits | `start_timestamp`, `end_timestamp` |
| Every episode of a window | each episode outside the limits, investigated at its worst moment against the moment just before it began, and the main causes gathered | `recurring` |

A why about **cost** ("why does the redispatch cost twice as much as
yesterday?") is not investigated. Each combination would need an optimal power
flow instead of a power flow, 16 to 64 of them at seconds each. The
optimiser's local optima would also put noise into shares that the method
otherwise computes exactly.

## The comparison

The state investigated is one network at one moment: the measurements or the
forecast, the loaded network or a proposal. The reference is one of these:

- **By default, the nearest earlier moment at which the element was within its
  limits**: the last moment before the problem began. The search goes back up
  to 672 ticks (a week of 15-minute data), one power flow per tick, and then
  forward if it found nothing. The result also says when the element went
  outside its limits (`outside_limits_since`). Such a reference often sits just
  inside the limit, so the shares explain the last push over it. When it is
  within 0.005 p.u. or 5 points of the limit, the result says so
  (`reference_note`).
- **A moment well inside the limits** (`reference_mode: "comfortable"`): the
  nearest earlier moment at least 0.01 p.u. or 10 points of loading inside
  them, and at least twice the margin the element has now (`symptom.margin`).
  This shows why the element got close to its limit at all. Without the second
  condition, an element 0.015 p.u. inside its band met the fixed margin itself,
  and the reference came 15 minutes earlier and explained nothing.
- **The same moment the day before** (`reference_mode: "previous_day"`).
- **A moment the operator names** (`reference_timestamp`), such as the same hour
  yesterday or this morning.
- **The other dataset at the same moment** (`reference_data_source`): why the
  forecast differs from what was measured.
- **Another network at the same moment** (`reference_network`): why a proposal
  has a violation the base does not.

These can be combined, for example a proposal now against the base an hour
earlier.

**After an outage** (`outage`), both states have the line or transformer out:
why the outage is critical now and was not at the reference. With
`reference_outage` the reference has another outage, or `none` for the intact
network. A mix of the two states can then have both outages at once. If the
network cannot survive that (a bus left without supply), the change cannot be
divided, and the result says to compare each outage with the intact network.

**Over a window**, the reference is the window of the same length just before
it (or the one given, or the same window in the other dataset or network). The
moments are paired in order. Every combination solves every pair, as many as a
budget of 1,600 power flows allows for the number of factors: all 96 moments
of a day with three factors, never fewer than 24. The factors are categories
that keep their name from one moment to the next: the demand near the element,
the demand elsewhere, the units' output, non-dispatchable generation, voltage
set-points, topology and unit status, elements added. With the agent's own
hypotheses, the factors are those hypotheses instead.

The study refuses to explain what is not there. An element within its limits,
with no reference given, returns `no_symptom`, and the answer says the premise
is wrong instead of explaining it. With several elements outside their limits
and none named, it returns `choose`, and the agent asks the operator which one.

## What differs between the two states

Both states are built exactly as every study builds a moment (see
[Network, data and the operating point](network-and-data.md)). Every column a
power flow reads is then compared, element by element:

| Table | Columns compared |
| --- | --- |
| Loads, static generators | active and reactive power, scaling, in service, bus |
| Voltage-controlled units | active power, voltage set-point, scaling, in service, bus |
| External grid | voltage magnitude and angle, in service |
| Shunts | active and reactive power, step, in service, bus |
| Lines | in service, impedances, length, parallel circuits, rating, ends |
| Transformers | in service, rating, voltages, impedances, losses, tap, phase shift, parallel units, ends |
| Buses | in service |

An element only one state has (a proposal's addition) is out of service in the
other. Columns a power flow does not read, such as an OPF limit, a cost or a
name, are not compared, because they change nothing the study could measure.

## Factors

The differences are grouped into the factors an operator would name:

| Factor | What it holds |
| --- | --- |
| Demand near the element | the loads within two buses of it (of either end of a branch), one member per bus |
| Demand elsewhere | every other load that changed |
| A unit's output, or the units' output | dispatchable units whose output changed: each its own factor when there are at most two, otherwise one factor with a member per unit |
| Non-dispatchable generation | static generators the operator does not dispatch |
| A voltage set-point | a unit's or the external grid's, each its own factor |
| A unit in or out of service | each its own factor |
| A branch, bus or shunt changed | each its own factor: out of service, back in service, or parameters changed |
| Elements added (or removed) | a proposal's additions, a member per element |

When there are more than six factors, the six that remain are the five with the
largest effect and the rest merged into "other changes". The effect of a factor
is measured both ways, as the larger of applying it alone and undoing it
alone, two power flows per factor.

## Shares: Shapley values

Let $N$ be the $k \le 6$ factors and $x(S)$ the quantity (a bus voltage in
p.u., or a branch loading in percent of its rating) when the factors in $S$
are as they are now and every other factor is as it was in the reference.
$x(\varnothing)$ is the reference state and $x(N)$ the state now. All $2^k$
combinations are solved, at most 64 AC power flows. The share of factor $i$ is
its Shapley value:

$$
\varphi_i = \sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!\,(k-|S|-1)!}{k!}\,\bigl(x(S \cup \{i\}) - x(S)\bigr)
$$

This is the factor's effect averaged over every order in which the changes
could have happened. The values add up to the change,
$\sum_i \varphi_i = x(N) - x(\varnothing)$, interactions included. An
interaction is shared between the factors that cause it, so it is neither
counted twice nor left out. The result reports the interaction as
`interaction_pct`: the part of the change that the factors applied one at a
time do not account for.

**Closure.** The two real states are also solved on their own. The
combinations $x(\varnothing)$ and $x(N)$ must reproduce them to within
$10^{-5}$ p.u. or 0.01 percentage points. Otherwise the two states differ in
something the study cannot separate, and it reports `not_decomposed` instead of
shares. A combination with no power-flow solution has the same outcome.

**Interactions.** The same table gives the Shapley interaction index of each
pair of factors:

$$
I_{ij} = \sum_{S \subseteq N \setminus \{i,j\}} \frac{|S|!\,(k-|S|-2)!}{(k-1)!}
\bigl(x(S \cup \{i,j\}) - x(S \cup \{i\}) - x(S \cup \{j\}) + x(S)\bigr)
$$

A pair whose interaction is at least 10 % of the change is reported. The pair
**amplifies** when, together, the two push further in the direction of the
change than each does alone; it **dampens** when they hold each other back.
For example, a line opened and a load raised at the bus it feeds amplify each
other by 24 % of the voltage fall: each makes the other worse.

**Does the conclusion depend on the reference?** Shapley shares depend on the
reference: that is the method's known weak point. So when the reference was
chosen automatically, the decomposition is repeated against the other two: the
last moment within limits, a moment well inside them, and the day before. The
result says whether the main cause holds against all of them
(`reference_robustness`). It compares the main cause by kind (demand, units'
output, a set-point…), since the same demand groups differently against
another moment. On the demo's evening peak, the undervoltage's main cause is
demand against all three references (71 %, 72 %, 71 %). When the operator
names the comparison, this check runs only if asked (`check_references`).

## Verdicts and counterfactuals

Each factor's share, $\varphi_i$ divided by the change, gives its verdict:

| Share of the change | Verdict |
| --- | --- |
| 15 % or more | **cause** |
| 3 % to 15 % | **contributes** |
| between −3 % and 3 % | **ruled out**: no measurable effect |
| −3 % or less | **counteracts**: it pushed the other way |

The main cause is the cause with the largest share. When no factor reaches
15 %, there is none, and the answer says the change is spread over many small
ones.

The same table gives two counterfactuals per factor, at no extra cost:

- **Undoing only this change**, $x(N \setminus \{i\})$, and whether that would
  bring the element back within its limits. For example, "undoing only the
  demand near Bus_29 would bring it back to 0.964 p.u.".
- **This change alone**, $x(\{i\})$: the reference state with only this factor
  changed.

## Inside a group

A group (demand near the element, the units' output…) that explains at least
a quarter of the change is divided among its members in the same way, for at
most two groups. The other factors stay as they are now, and the members'
values add up to the group's effect with those factors present. This is how a
"units' output" factor becomes "Gen_bus1 37 %, Gen_bus4 23 %…", or "demand near
Bus_29" becomes "the load at Bus_29 87 %".

## Classical views, when the agent asks for them

Power-system engineering has older, cheaper ways to look at the same question.
The agent may ask for them (`classical`) beside the exact shares, to show where
a linear reading would hold or who is using a branch. They inform; the
verdicts stay the exact ones, and the agent says they are approximations.

**A linear estimate from sensitivities** (`"linear"`). For each factor, the
directional derivative of the quantity along the factor's change is measured
at both end states with a step of 5 % of the change, then averaged (the
trapezoid rule):

$$
\hat\varphi_i = \tfrac12\bigl(\partial_{\Delta_i} x(\text{reference}) + \partial_{\Delta_i} x(\text{now})\bigr)
$$

This is a two-point approximation of the **Aumann–Shapley** value, the
continuous counterpart of the Shapley value used in loss and cost allocation.
It costs two power flows per factor. A switching change has no sensitivity:
an element taken in or out of service, a moved element. When a branch's
loading is investigated and the factor takes one branch out, the classical
estimate is instead the **DC line outage distribution factor**: the monitored
flow picks up LODF × the outaged flow, and its loading scales with its flow.
Any other switching factor is left without an estimate, and the result says
why. A count of violations has none either, because it jumps at the limit.
The result gives each factor's linear share beside its exact share, the error
in points, the share of the change the estimates cover, and whether the linear
main cause agrees with the exact one.

**Power-flow tracing** (`"tracing"`, for a line or transformer). On the solved
AC flows of both states, proportional sharing (Bialek) traces which sources
feed the branch's active flow and which loads draw it. It is lossless: each
branch's flow is taken as the mean of its two ends. This is a picture of who
uses the branch, not a cause. That the external grid supplies all of a feeder's
flow says nothing about what changed.

On the demo's evening peak, the two agree closely on a change between nearby
moments: the changes are small and nearly linear. They part where the method
matters. A line opened and a load raised at the bus it feeds: the line has no
sensitivity, and the linear estimates cover only 53 % of the change. A
branch's outage estimated by its DC LODF: 120 % against an exact 100 %.

## Physical, or a fault in the data?

An investigation names what changed. When what changed is a measured load or
unit, the change may not have happened at all: a spike from a faulty meter, or
a value frozen when the telemetry stopped updating. Acting on it would mean
acting on something that is not there. So before a measured change is
reported as a cause, each load or unit behind it is compared with its own
history in the same data:

- **a jump**: its change between the two moments against every change it shows
  over the same number of steps. One beyond the 99.5th percentile, and more
  than 1.5 times its usual largest, is unusual;
- **a level**: its value now against everything it shows elsewhere in the
  data. More than 10 % of that range outside it is unusual;
- **frozen**: the same value for eight moments in a row (two hours of
  15-minute data), from an element that otherwise varies.

A suspicious member goes into the result's `data_check`, and the message begins
with it: *Check the data before acting on this: Demand at Bus_29 shows a jump of
…*. When nothing is unusual, the result says the measured changes look physical.
A flag is a reason to check the data, not a verdict that the data is wrong,
because a real event can be unusual too. It changes no share or verdict.

The answer must say it. An answer that names a flagged cause without a word of
doubt about the data is flagged by the answer check (`data_doubt_unsaid`), which
the evaluation counts as a failure. On the demo's clean data nothing is
flagged. A demand at Bus_29 multiplied by four at one moment is flagged as both
a jump and a level, and the investigation still names it as the main cause,
with the doubt said first.

## The levers

When the element is outside its limits, the
[attribution](contingency-analysis.md#violation-attribution) is run at the
state investigated. Its controllable drivers for this element (at most three),
its voltage set-point moves for a voltage violation (each checked by a power
flow) and its recommended action are added under `levers`. The answer gives
that action verbatim, after the causes.

## The agent leads: its hypotheses, its record, the tools it lacks

The study above judges; the agent decides what to ask it. Three things give it
room to investigate in its own way, without its conclusions floating free of
the studies.

**Its own hypotheses.** Instead of the standard grouping, the agent can pass
its own factors (`hypotheses`, at most five), each a name and the elements it
covers:

- a bus name covers the demand and shunts at that bus;
- a unit name covers its output and its set-point;
- a line or transformer, by name or tag (`L38`), covers its state and
  parameters;
- `external grid` covers the grid's voltage;
- `demand` and `units` cover all of them.

Everything else that changed becomes one more factor, "other changes". The
shares and verdicts are measured exactly as before, so a wrong hypothesis is
visible in the result. **How much the hypotheses explain** is measured too
(`hypotheses_coverage_pct`, the share not in "other changes"). Below 75 %, the
result asks the agent to split or add hypotheses, or to say what remains
unexplained. An answer that does neither is flagged (`coverage_unsaid`). One that names nothing that changed between the two
states is **ruled out**, and the result says so (`nothing_changed`). If the
real cause lies elsewhere, "other changes" takes the share. For example, two
loads that rose in different areas are one factor in the standard grouping; as
two hypotheses, they come back as 63 % and 37 %.

**Any study, then a record.** The agent may test a hypothesis with any tool,
including the scopes and references above:
another reference, a proposal compared with the base, an element over time, an
outage, the forecast against the measurements. Before answering, it calls
`record_investigation` once, with every hypothesis it considered, refuted
ones included. Each hypothesis carries a verdict, the tools whose results
decide it, and what they showed. The record is checked in code before it is
accepted:

| Verdict | Accepted only if |
| --- | --- |
| supported, refuted | it names tools that returned a result (not an error) earlier in the same question |
| supported, refuted, citing `investigate` | when the hypothesis is one of the study's factors (named in `factor`, or written in its text), the study's verdict agrees: supported needs a cause or a contributor, refuted needs ruled out or counteracting |
| untested | it describes what would test it (`missing_tool`), says whether that is an analysis or data (`missing_kind`), and is not something CONDUCTOR already does |

The last rule exists because the agent once recorded hypotheses it had not yet
tested, while waiting for the operator's answer, as "untested, needing
`investigate`": a step not taken, not a tool CONDUCTOR lacks.

**The answer against the record.** The answer review, the audit panel and the
grader compare the answer with the record:

- **an untested hypothesis stated as a finding** (a failure): a sentence with
  the hypothesis's own distinctive words and a causal verb (caused, explains,
  due to…) and nothing hedging it (could, may, whether, not, no data…);
- **untested hypotheses never said to be untested** (a warning);
- **a supported hypothesis whose elements the answer never names** (a
  warning).

These checks read words, not meaning. A miss is possible either way, and each
finding quotes the sentence so a reader can judge it.

A record that fails is refused with the reason, and the agent corrects it. The
agent chooses what to test, but not what counts as tested.

**What it lacks: tools to build, data to integrate.** A hypothesis no tool can
test is recorded as untested, with what would test it (`missing_tool`) and
which of two things that is (`missing_kind`):

- **an analysis** of the grid that CONDUCTOR could compute from what it has
  and does not: how far a bus is from voltage collapse (a stability margin),
  whether the units could have followed in time (their ramps), how the grid
  moves through an event (dynamics), what a cause cost. This is a tool to
  build.
- **data** it does not have: the weather, customer metering, the calendar,
  operator or protection logs. This is an input to integrate, not a tool.

An analysis request that only describes something CONDUCTOR already computes
(an N-1 screen, an OPF, a hosting capacity, sensitivities, a probabilistic
assessment, a proposal with a capacitor) is refused, because it is a study to
run. With something new beside it (*an OPF with the units' ramp limits over
time*), it is accepted. The answer must say the hypothesis could not be tested
and what would test it, and never present it as a finding.

Each request is appended to `tool_requests.jsonl` in the local cache.
`python -m evaluation.tool_requests` ranks them in two lists, **tools to
build** and **data to integrate**, by how often they were needed: what to do
next, taken from real questions. The app's sidebar shows the same lists under
**Tools investigations needed**. The agent words one need in many ways, so
requests are grouped by the needs they ask for:

- for analyses: voltage stability, dynamics, time-coupled studies, costs,
  short circuit and protection, multiple contingencies, power quality, thermal
  models, reactive capability;
- for data: weather, calendar and events, customer data, operator logs,
  protection events, equipment history, line ratings, prices.

A request counts under each need it asks for: *a continuation power flow and
a ramp study* counts under voltage stability and under time-coupled studies.
A need named only in saying why CONDUCTOR cannot do it (*…; no CONDUCTOR tool
provides weather or customer data*) is not counted. One that asks for none is
grouped by its wording.

**When it suggests one.** In an ordinary answer, the agent records a
hypothesis as untested when the question raises one (*could the hot weather
explain it?*). Left to itself, it stops at the cause the studies find (*the
demand near Bus_29*): in the live runs before this was added, 0 of 40
investigations proposed anything of their own. With **Deep Research**, it asks
two more questions. Which analysis of the grid would confirm the cause, or say
how serious it is, and CONDUCTOR cannot do? It first tries what the tools can
do, such as a proposal with a capacitor. And what lies behind the cause,
outside the grid? Why demand rose, why a set-point changed. Each is its own
hypothesis. The reminder asks once if the record has no analysis among them.

A hypothesis that a change is a fault in the data is judged by that factor's
data check, not by its share.

The record is the agent's own note, so its findings are **never a source** for
the answer's figures: provenance and the evidence levels ignore it. A figure
the answer gives must still come from a study. Under the answer, the record is
shown as a card with one row per hypothesis: its verdict, what showed it and
which studies, or the tool it would need.

## Every episode: is the cause recurring?

With `recurring`, every moment of a window (the whole dataset by default, two
weeks at most) is solved once. Each run of moments outside the limits is an
episode. With no element named, the element outside its limits most often is
taken, and the result says so. The ten most severe episodes are each
investigated at their worst moment against the moment just before they began,
and their main causes are gathered by kind. A cause that is the main one in at
least 60 % of the episodes **recurs**, which points to a structural problem
(a reinforcement, a flexibility contract) rather than an operating one. On the
demo's week, Bus_29 leaves its limits in 12 episodes, and in all 10
investigated the main cause is the demand near it (71–72 % of the change). The
week takes about 15 s.

## Could it have been avoided?

`test_prevention` takes a moment in an episode and returns:

- **the episode**: the run of moments outside the limits around that moment;
- **the warning**: the run of moments just before it, within a small margin of
  the limit (0.01 p.u. or 10 points), and the minutes it gave;
- **for each candidate action** the agent proposes (a voltage set-point, a
  change of output, or several together), held from the warning to the end of
  the episode: whether it keeps the element within its limits at every moment,
  with no element outside its limits that was not before and the unit within
  its range. When it does, the smallest share of it that still does (found by
  halving). When it falls short with nothing else wrong, how much more would
  be needed (up to four times). Otherwise, what stops it.

Each moment is checked by an AC power flow, up to 24 moments evenly spread.
Without actions, the attribution's levers at that moment are tested.

The model is static: an action held from the warning on is, at each moment,
the same as one taken just before the episode. What the warning measures is
the time there was to decide. On the demo, the episode of 5 January had 30
minutes of warning, and raising Gen_bus7's set-point by 0.009 p.u. would have
avoided it. On 6 January, no set-point alone or together with others would
have: the one at Gen_bus7 overloads the line Bus_5 → Bus_7, and the one at
Gen_bus12 puts its own bus outside the band before it is enough.

## What is coming: looking ahead

The same questions can be asked of the future. `look_ahead` reads the forecast
(24 hours by default, 72 at most, from the simulation clock or the forecast's
first moment) and works out, without being asked for each step:

1. **What is coming.** Every moment of the horizon is solved by an AC power
   flow. The moments of one element outside its limits form an **episode**,
   and a return inside the limits for at most an hour does not split it. An
   episode that runs past the horizon is marked as such.
2. **When to decide.** The warning before each episode is found as in the
   prevention check: the moments within a small margin of the limit. Its start
   is the time to decide by. With no warning moment, the answer says to decide
   before the episode, or now when the warning has already begun.
3. **Why.** The episode's worst moment is investigated against the last
   comfortable moment before the warning, so the cause covers the whole
   approach to the limit, not only the last push over it.
4. **What would prevent it.** The prevention check runs on the episode, from
   the warning to its end, with the attribution's levers and all of them
   together. The smallest sufficient share of the first action that works is
   kept.
5. **Whether that holds.** That action, at its smallest share, is checked
   again where the forecast errs against it: with demand 5 % above the
   forecast, and with renewable output 5 % below it (`uncertainty_pct`). When
   it no longer holds, the result says why: the element still out, a new
   violation, or the unit out of its range. When more of the same action would
   still be enough, it says how much.

The five most severe episodes are worked out in full (`max_episodes`); the
others are counted. Everything is marked as an expectation from the forecast,
not a fact. The chart is a timeline with one row per episode, the warning in
amber before the episode in red, and a label saying whether an action was
found and whether it holds if the forecast errs.

The result says which errors were checked (`forecast_errors_checked`). The
demo's five units are all conventional, so on the demo only the error in
demand is checked. On a proposal that adds a 3 MW PV unit at Bus_29, the
morning action also holds with that unit's output 5 % lower.

On the demo's first forecast day, four episodes are expected:

- the line Bus_0 → Bus_1 in the morning, with no warning. Raising Gen_bus1's
  output by 8 MW prevents it, but with 5 % more demand 18.6 MW is needed;
- the same line in the evening, with 15 minutes of warning. No unit can carry
  enough;
- Bus_29 from 18:00, with an hour of warning. Every set-point that would hold
  it needs more reactive power than its unit has;
- Bus_28 briefly at 19:15. Raising Gen_bus7's set-point by 0.005 p.u. prevents
  it, but with 5 % more demand that overloads the line Bus_5 → Bus_7.

The day takes about 12 s, and one element over 12 hours about 2 s.

## The report

Every answer that investigated something offers **Download the investigation
report**. This is a Markdown file built from the turn's record, not from the
model's prose beyond quoting the answer. It contains:

- the question and the answer;
- each investigation: the comparison, the factors with their exact shares and
  verdicts, what is inside a group, the causes that interact, the check against
  other references, the coverage of the agent's hypotheses, the classical views
  and the levers;
- recurring episodes, prevention checks and the look-ahead (each episode,
  when to decide, the cause, the action and whether it holds if the forecast errs);
- the data check, when a cause looked like a fault in the data;
- the hypotheses the agent recorded, and the tools untested ones would need;
- what backs the answer: evidence levels, answer checks, the reading check;
- every study that ran, with its arguments;
- the method and its limits.

## Deep Research

The operator can switch on **Deep Research** for one question. It asks for
depth; it does not decide whether the question is an investigation. For that
question only:

- the system prompt gains a note: first decide whether the question is a *why*
  about a problem in the grid; if it is not, answer as usual; if it is,
  investigate in depth (own hypotheses until they explain most of the change,
  the references, the classical views when they help, prevention, recurrence,
  what lies behind the main cause where the tools stop, the record);
- the agent has 12 more steps (`DT_DEEP_EXTRA_TURNS`);
- **a reminder, in code**: when an investigation is about to answer, the loop
  checks what depth it still lacks:
  - prevention, when the element is outside its limits;
  - recurrence;
  - hypotheses that explain too little;
  - the record;
  - a hypothesis about what lies behind the main cause (why demand rose, why a
    set-point changed), recorded as untested with the tool that would test it.

  It names what is missing once. The agent does it, or says why it does not
  apply. The draft and the reminder are taken out of the conversation it
  keeps;
- the turn's record says Deep Research was asked for and whether the agent
  investigated (`deep_research`), which the answer's badge shows.

In the live run, the same *why* went from a basic investigation (one study, 26
s) to all of it (five studies, 57 s: the cause, references, prevention,
recurrence, the record). An options question with the switch on was answered as
an analysis.

## Around the investigation

- **The reading check** re-runs an investigation under the readings the
  question left open: another load power factor, the 90 % loading limit, the
  ±5 % band, the forecast. It compares the main causes by kind (demand, units'
  output, a set-point, an element added, or the factor itself), since the same
  demand can be grouped differently under another reading. A different main cause,
  or no symptom at all under another reading, **changes** the conclusion; the
  same main cause with a share that moves by more than 10 % **shifts** it. A
  window investigation is not re-run, because it takes too long.
- **The advisor**, after an investigation, proposes testing a hypothesis the
  record left open or did not consider, and how: the agent's own hypotheses,
  another reference, an outage, a window, the forecast. It never re-proposes
  one recorded as untested.

## Cost

On the IEEE 30-bus demo, an investigation of one moment runs about 100 power
flows and takes about a second; most of that is the attribution and the two
drill-downs. The reference search adds one power flow per tick it looks at. A
window costs up to about 1,600 power flows: about 3 s for five hours with
three factors, and 8 s for a whole day.

## How it has been tested

The bench in `evaluation/investigation.py` injects known causes into the demo
as proposals and asks why the element they affect changed. It has 15 cases:

- six causes alone;
- four pairs of causes, two of them interacting;
- two cases with the agent's own hypotheses: two loads the standard grouping
  joins, and a wrong hypothesis that must be ruled out;
- two other scopes: the whole grid with two causes, and a cause after an outage;
- three causes with background, compared with the base an hour earlier while
  demand and the schedule moved too.

The injected cause must come back as the main cause, both causes of a pair
must be causes, a wrong hypothesis must be ruled out, and nothing else may be
blamed. All 17 cases pass, in about half a second each.

The linear (sensitivity) estimate is scored on the same cases. Where it can
estimate a factor, it is within 0.4 points of the exact share on average, and
it names the main cause in 81 % of the cases. It has no estimate for a line
taken out when a voltage is investigated, which is five cases.

For comparison, the attribution's top driver names the injected cause in only
12 % of the cases: a set-point, a line out or the external grid's voltage is
not a source it can rank. Undoing each factor alone also ranks the causes
correctly on this bench, but its effects add up to 124 % of the change when the
causes interact. See
[Using CONDUCTOR in research](../research/using-conductor-in-research.md#the-investigation-evaluation).

## Assumptions and limits

- **What it divides.** A voltage, a loading, a count of violations or their
  duration. A why about a cost is not investigated (see above). A window's
  shares come from the moments solved, each standing for the ones around it
  when the window is thinned (the result says so).
- **Shares depend on the reference.** Against 18:30 or against 16:00, the same
  undervoltage has different shares, because the changes are different. Every
  answer names its reference state and why it was chosen.
- **Grouping is a choice.** "Near" means within two buses; more than two units
  are grouped together. The drill-down shows what is inside a group, and with
  six factors at most, the smallest are merged. The agent can replace the
  grouping with its own hypotheses.
- **A record states what was tested, not that it was tested well.** The check
  makes sure each verdict cites studies that ran, not that those studies
  decide it. That reading is left to the operator, with the studies one click
  away in the answer's details.
- **A forecast is an expectation.** A look-ahead is only as good as the
  forecast. The checks with more demand and less renewable output cover the
  two errors that work against an action, 5 % each, not every error.
- **A data flag is a doubt, not a diagnosis.** The data check compares an
  element with its own history; a real but unusual event is flagged too, and
  a fault that stays within the element's usual range is not.
- **Physics, not intent.** A change in a unit's output is reported as a cause
  even when it followed the schedule: the study says what changed the power
  flow, not who decided it.

## In the code

| Part | Where |
| --- | --- |
| The decomposition | `backend/investigation.py` (`diff`, `group`, `shapley`, `decompose`) |
| The classical views | `backend/classical.py` (`linear_estimates`, `trace_branch`) |
| The endpoint | `POST /api/investigate` in `backend/main_backend.py` (reference, symptom, levers) |
| The tool | `investigate` in `llm_agent/agent/tools.py`, schema in `tool_schemas.py` |
| The chart | `render_investigation` in `llm_agent/agent/renderers.py` |
| The record of an investigation | `record_investigation` in `llm_agent/agent/tools.py`; its checks in `llm_agent/agent/investigation_record.py`; the card in `llm_agent/app.py` |
| The answer against the record (grader) | `evaluation/graders/investigation.py` |
| The tools investigations lacked | `evaluation/tool_requests.py` (`python -m evaluation.tool_requests`) |
| Recurring causes, prevention | `_investigate_recurring`, `POST /api/investigate/prevention` in `backend/main_backend.py`; the tool `test_prevention` |
| Physical or a fault in the data | `backend/plausibility.py` (`check`); `_data_inspector` in `backend/main_backend.py` |
| Looking ahead | `POST /api/investigate/look_ahead` in `backend/main_backend.py`; the tool `look_ahead`; `render_look_ahead` |
| The report | `llm_agent/agent/investigation_report.py`; the download in `llm_agent/app.py` |
| The bench and the agent tasks | `evaluation/investigation.py` |
| Tests | `tests/test_investigation.py` |
