# ENTSO-E SmallGrid test configuration

`SmallGrid_BusBranch_v3.0.0.zip` holds ENTSO-E's CGMES 2.4.15 **SmallGrid** test
configuration, version 3.0.0 (bus-branch base case, complete: EQ, SSH, TP, SV, DL,
GL), and its boundary set (EQ_BD, TP_BD), as two zips inside one. The model files
are unmodified.

- **Owner:** ENTSO-E (European Network of Transmission System Operators for
  Electricity), CGMES Conformity Assessment Scheme test configurations — see
  <https://www.entsoe.eu/data/cim/cim-conformity-and-interoperability/>.
- **Licence:** Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0, <https://creativecommons.org/licenses/by-sa/4.0/>). These files
  stay under that licence; CONDUCTOR's code is under the Apache License 2.0.
- **Obtained from:** the copy PowSyBl redistributes
  (`powsybl-core`, `cgmes/cgmes-conformity`, `conformity/cas-1.1.3-data-4.0.3/SmallGrid/BusBranch`).
- ENTSO-E provides the models "as is", without warranty.

CONDUCTOR uses the model as its *ENTSO-E SmallGrid* demo
(`backend/network_profiles/entsoe_smallgrid.yaml`).
