# Vicon registration: cross-check and error budget

The canonical Vicon->ENU registration of the set_b dynamic sessions is the
wand tie in `Data/set_b/<session>/session.yaml` (`transform_vicon_to_local`),
computed by `Code/datasets/radiocal/scripts/02_build_set_b.py` from the two
surveyed wand points. It uses no UWB range.

`vicon_to_enu_<campaign>_<session>.yaml`: an independent vicon->enu
transform fitted from the radio data by `estimate_vicon_transform.py`
(robust per-epoch multilateration of DD98 from the surveyed anchors,
time-matched to the Vicon poses on the shared clock, robust planar Umeyama).
Format: `radiocal.frames.Transform.to_dict()` plus a `provenance` block
(matches, residuals, script, date) whose `wand_comparison` holds the
difference to the wand tie. The `date` changes on every run, and the last
digits of the floats can differ across platforms. It is a cross-check and
never enters the benchmark.

`error_budget.yaml`: the ground-truth error budget of `gt_error_budget.py`.
