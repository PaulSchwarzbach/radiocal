"""Parser for the Qorvo autopositioning backup ``.log`` files (Torgau, sets A and B).

File format: one Python-repr record per logical line, but numpy matrices are
pretty-printed across many physical lines. Record shapes observed:

  None                                                  (first line of file)
  [t, None, None, None, None, None]                     (idle, no ranging data)
  [t, seq, ['4197', ...], array([[nan, 14.85, ...]])]   (data record, set B)
  (None, None, None, None, None)                        (idle, set A)
  (seq, ['4197', ...], array([[nan, 15.85, ...]]))      (data record, set A - no timestamp)

`t` (when present) is the Unix time of the logging PC's Qorvo pipeline
("embedded clock"); it lags the MQTT broker clock by a per-run offset (see
convert.estimate_clock_offset). Records without a timestamp yield t_raw = NaN.
The NxN matrix is a directed distance matrix: row = measuring node, column =
target node, entries in metres, missing ranges NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_EVAL_GLOBALS = {"__builtins__": {}, "array": np.array, "nan": float("nan"), "None": None}


def iter_log_records(path: str | Path):
    """Yield evaluated records (lists) from a Qorvo backup log.

    Physical lines are accumulated until the square brackets balance, then the
    chunk is evaluated with a restricted namespace (only ``array`` and ``nan``).
    Bare ``None`` lines are skipped.
    """
    buf: list[str] = []
    depth = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not buf:
                if not stripped.startswith(("[", "(")):
                    continue  # e.g. the leading literal "None"
            buf.append(line)
            depth += (line.count("[") + line.count("(")) - (line.count("]") + line.count(")"))
            if depth == 0 and buf:
                chunk = "".join(buf)
                buf, depth = [], 0
                try:
                    yield eval(chunk, _EVAL_GLOBALS)  # noqa: S307 - trusted repr data
                except Exception as exc:  # malformed tail record etc.
                    raise ValueError(f"unparseable record in {path}: {chunk[:120]}...") from exc
    if buf:
        raise ValueError(f"truncated trailing record in {path} ({len(buf)} lines)")


def parse_qorvo_log(path: str | Path) -> pd.DataFrame:
    """Parse a backup log into a directed long-format range table.

    Returns columns: t_raw (embedded clock), seq, from_id, to_id, range_m.
    NaN entries and the diagonal are dropped. Idle records contribute nothing.
    """
    rows_t, rows_seq, rows_from, rows_to, rows_r = [], [], [], [], []
    for rec in iter_log_records(path):
        if len(rec) == 4 and rec[1] is not None:        # [t, seq, nodes, matrix]
            t_raw, seq, node_list, mat = rec
        elif len(rec) == 3 and rec[0] is not None:      # (seq, nodes, matrix) - no clock
            t_raw, (seq, node_list, mat) = float("nan"), rec
        else:
            continue  # idle record
        mat = np.asarray(mat, dtype=float)
        n = len(node_list)
        if mat.shape != (n, n):
            raise ValueError(f"matrix shape {mat.shape} != ({n},{n}) at seq={seq} in {path}")
        ii, jj = np.nonzero(~np.isnan(mat))
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        rows_t.append(np.full(len(ii), t_raw))
        rows_seq.append(np.full(len(ii), seq, dtype=np.int64))
        rows_from.append(np.array(node_list)[ii])
        rows_to.append(np.array(node_list)[jj])
        rows_r.append(mat[ii, jj])
    if not rows_t:
        return pd.DataFrame(columns=["t_raw", "seq", "from_id", "to_id", "range_m"])
    df = pd.DataFrame({
        "t_raw": np.concatenate(rows_t),
        "seq": np.concatenate(rows_seq),
        "from_id": np.concatenate(rows_from),
        "to_id": np.concatenate(rows_to),
        "range_m": np.concatenate(rows_r),
    })
    # seq is the epoch counter and monotone in file order; t_raw may be NaN (set A logs)
    return df.sort_values("seq", kind="stable").reset_index(drop=True)
