# Closed-form baseline on fully_meshed - summary

Session-median symmetrized ranges, one construction per session/variant;
evaluation: rigid Umeyama alignment (reflection allowed) against the
surveyed anchors of the unmasked session, 2-D (x, y).

<!-- BEGIN GENERATED (summarize.py) -->
| campaign | session | variant       | n_placed | n_anchors | pairs_available | rmse_m | max_error_m | reflection | z_spread_ref_m |
| -------- | ------- | ------------- | -------- | --------- | --------------- | ------ | ----------- | ---------- | -------------- |
| set_b    | s03     | faithful      | 10       | 10        | 45/45           | 1.694  | 4.658       | False      | 0.040          |
| set_b    | s03     | disambiguated | 10       | 10        | 45/45           | 1.694  | 4.658       | False      | 0.040          |
| set_a    | s21     | faithful      | 11       | 12        | 65/66           | 5.277  | 16.199      | False      | 0.103          |
| set_a    | s21     | disambiguated | 11       | 12        | 65/66           | 18.551 | 37.462      | True       | 0.103          |
<!-- END GENERATED (summarize.py) -->

Notes:

- `faithful` reproduces the original positioning.py (always +y);
  `disambiguated` resolves the y sign against a third placed anchor.
  The sign test relies on a single reference range, so at an NLOS-rich
  site it can itself be fooled and flip anchors the wrong way.
- The construction is planar while ranges are slope distances; the
  surveyed z-spread per site (last column) bounds that approximation.
- set_a is the NLOS-rich site: degradation vs set_b is
  expected and reported as-is, not tuned away.
- `epoch_success_rate` in the estimate CSVs is the fraction of epochs
  in which an anchor could be placed from that epoch's matrix alone
  (faithful placement rule), i.e. the original per-epoch diagnostic.
- Replication (benchmark.parquet): s22 reproduces the s21
  pathology exactly - `disambiguated` flips into the reflected
  geometry again (18.56 m RMSE, reflection=True, vs faithful 5.50 m),
  confirming the sign test is systematically fooled by this site's
  NLOS pattern, not by an s21-specific draw.
