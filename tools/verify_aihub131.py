# -*- coding: utf-8 -*-
"""Integrity check + downloader-remnant cleanup for the AI-Hub 131 corpus.

AI-Hub's INNORIX downloader leaves partial-transfer remnants named
``<archive>.zip.irx<N>`` next to the real archive when a download is
interrupted and restarted. They are not openable archives and waste tens
of GB. This tool:

  1. CRC-verifies every real ``*.zip`` under the dataset root
     (``zipfile.testzip`` reads and checksums every member — slow on the
     53 GB training archive, but it is the only real proof of integrity);
  2. lists ``*.irx*`` remnants with sizes;
  3. with ``--delete-remnants``, deletes remnants **only for archives
     whose verification passed in the same invocation**.

Usage:
  python tools/verify_aihub131.py [--root PATH] [--delete-remnants]
"""

import argparse
import glob
import json
import os
import sys
import time
import zipfile

DEFAULT_ROOT = (
    r"C:\Users\Baek\Phomene"
    "\\131.인공지능 학습을 위한 외국인 한국어 발화 음성 데이터 (일본어 모어 화자)"
)


def gb(n):
    return f"{n / 1024**3:.1f} GB"


def verify_zip(path):
    t0 = time.time()
    try:
        with zipfile.ZipFile(path) as z:
            n = len(z.infolist())
            bad = z.testzip()      # None = every member's CRC checks out
    except zipfile.BadZipFile as e:
        return {"ok": False, "error": f"BadZipFile: {e}"}
    return {
        "ok": bad is None,
        "members": n,
        "first_bad_member": bad,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--delete-remnants", action="store_true",
                    help="delete *.irx* remnants of archives that verified OK")
    args = ap.parse_args()

    zips, remnants = [], []
    for dirpath, _, filenames in os.walk(args.root):
        for f in filenames:
            p = os.path.join(dirpath, f)
            if f.lower().endswith(".zip"):
                zips.append(p)
            elif ".irx" in f.lower():
                remnants.append(p)

    print(f"archives: {len(zips)}, remnants: {len(remnants)}\n")
    results = {}
    for p in sorted(zips):
        size = os.path.getsize(p)
        print(f"verifying {os.path.relpath(p, args.root)} ({gb(size)}) ...",
              flush=True)
        r = verify_zip(p)
        results[p] = r
        status = "OK" if r["ok"] else f"CORRUPT ({r.get('first_bad_member') or r.get('error')})"
        print(f"  -> {status}  "
              f"[{r.get('members', '?')} members, {r.get('seconds', '?')}s]\n")

    total_remnant = 0
    for p in sorted(remnants):
        size = os.path.getsize(p)
        total_remnant += size
        owner = p[:p.lower().index(".irx")]
        owner_ok = results.get(owner, {}).get("ok")
        verdict = ("safe to delete (archive verified OK)" if owner_ok
                   else "KEEP — owning archive missing or failed verification")
        print(f"remnant: {os.path.relpath(p, args.root)} ({gb(size)}) — {verdict}")
        if args.delete_remnants and owner_ok:
            os.remove(p)
            print("  deleted.")
    if remnants:
        print(f"\nremnant total: {gb(total_remnant)}")

    all_ok = all(r["ok"] for r in results.values())
    print(f"\nSUMMARY: {'ALL ARCHIVES OK' if all_ok else 'FAILURES PRESENT'}")
    print(json.dumps({os.path.basename(k): v for k, v in results.items()},
                     ensure_ascii=False, indent=2))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
