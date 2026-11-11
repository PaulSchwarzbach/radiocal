"""QA reports for every converted session of the Torgau campaigns.

    python -B Code/datasets/radiocal/scripts/03_qa_all.py --data rebuild --out rebuild

--data names the folder that holds the campaign folders (default: Data/ at
the repository root). --out names the folder that receives
<campaign>/<session>/qa/ (default: the --data folder itself). The default
run is destructive: it overwrites the shipped reports under Data/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radiocal.qa import run_qa

DATA = Path(__file__).resolve().parents[4] / "Data"
CAMPAIGNS = ["set_b", "set_a"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA, metavar="DIR",
                   help="folder with the campaign folders (default: Data/ at the "
                        "repository root)")
    p.add_argument("--out", type=Path, default=None, metavar="DIR",
                   help="folder that receives <campaign>/<session>/qa/ (default: the "
                        "--data folder; destructive, overwrites the shipped reports)")
    p.add_argument("--campaigns", nargs="+", default=CAMPAIGNS, metavar="SET",
                   help="campaign folders to report on (default: set_b set_a)")
    args = p.parse_args()
    out = args.out if args.out is not None else args.data
    for campaign in args.campaigns:
        sessions = sorted((args.data / campaign).glob("*/session.yaml"))
        if not sessions:
            raise SystemExit(f"03_qa_all: no session under {args.data / campaign}")
        for session_yaml in sessions:
            sdir = session_yaml.parent
            report = run_qa(sdir, out_dir=out / campaign / sdir.name / "qa")
            print(f"-> {report}")


if __name__ == "__main__":
    main()
