"""Every variant, several seeds, one run after another.

    python scripts/run_ablations.py --data ~/work/memes-data
    python scripts/run_ablations.py --variants full concat --seeds 0 1 2

Each run is a separate process writing results/<variant>/seed<seed>.json, and a
run whose file already exists is skipped, so this can be stopped and started
again without losing anything. Seeds are the outer loop: if it is cut short,
every variant has the same number of seeds done.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from ca_agfn.config import ROOT, VARIANTS

ORDER = ["full", "text_only", "image_only", "concat", "no_clash",
         "no_entropy", "no_captions", "xlmr"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=ORDER,
                        choices=sorted(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--logs", default=str(ROOT / "logs"))
    parser.add_argument("--checkpoint", default=None,
                        help="Save the weights of the first full run here.")
    args, passthrough = parser.parse_known_args()

    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    failures = []

    for seed in args.seeds:
        for variant in args.variants:
            out = Path(args.results) / variant / f"seed{seed}.json"
            if out.exists():
                print(f"{variant} seed {seed}: done already")
                continue

            command = [sys.executable, str(ROOT / "scripts" / "train.py"),
                       "--variant", variant, "--seed", str(seed),
                       "--results", args.results, *passthrough]
            if args.checkpoint and variant == "full" and seed == args.seeds[0]:
                command += ["--checkpoint", args.checkpoint]

            log = logs / f"{variant}_seed{seed}.log"
            print(f"{variant} seed {seed}: running, log in {log}", flush=True)
            started = time.time()
            with log.open("w", encoding="utf-8") as handle:
                code = subprocess.run(command, stdout=handle,
                                      stderr=subprocess.STDOUT).returncode
            minutes = (time.time() - started) / 60
            if code:
                failures.append((variant, seed))
                print(f"  failed after {minutes:.0f} min, exit {code}")
            else:
                print(f"  done in {minutes:.0f} min", flush=True)

    if failures:
        print(f"\n{len(failures)} runs failed: {failures}")
        sys.exit(1)
    print("\nAll runs done.")


if __name__ == "__main__":
    main()
