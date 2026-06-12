"""Fetch the 36 EC-Earth3 extension members from US ESGF nodes.

Reads ``manifest.json`` written by ``scripts/audit_ec_earth3_msftyz.py``,
filters to the canonical-physics, paired (tas + msftyz), not-already-local
candidate set (24 historical + 12 ssp245 members), flattens to a per-file
work plan, and downloads in parallel via plain HTTPS (no Globus client,
no ESGF cert dance).

Outputs land flat at
``/work/bu1431/T_EU_AMOC/CMIP6/upload/ec-earth3/`` matching the uo1075
naming convention. **No processing happens here** — the
``data_process_paper.ipynb`` cell ~239 recipe (msftyz → amoc26 yearly)
is a separate task.
"""

# %%
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# %%
# CONFIG ---------------------------------------------------------------------

AUDIT_DIR = '/home/m/m300940/teu_amoc/data/ec_earth3_audit_2026-05-25'
DEFAULT_MANIFEST = f'{AUDIT_DIR}/manifest.json'
DEFAULT_GAP_TABLE = f'{AUDIT_DIR}/gap_table.csv'
DEFAULT_TARGET = '/work/bu1431/T_EU_AMOC/CMIP6/upload/ec-earth3'
DEFAULT_WORKERS = 8

# Per-file fetch tuning
HTTP_TIMEOUT_S = 120
CHUNK_BYTES = 1 << 20  # 1 MiB stream chunk
RETRY_DELAYS_S = [5, 15, 45]  # three attempts, exponential

# JSONL append lock for thread-safety
import threading
_LOG_LOCK = threading.Lock()


# %%
# CELL 1: Load + filter manifest ---------------------------------------------

def load_candidates(manifest_path, gap_table_path):
    """Return the per-file work plan as a list of dicts.

    The candidate set is *paired* (scenario, member) pairs where:
      - both tas and msftyz are present on ESGF for that (sce, mem),
      - the variant is canonical (r*i1p1f1),
      - neither variable is already on disk (in_local_raw == False).

    These are exactly the 36 (scenario, member) pairs surfaced in the
    audit report's "Pipeline relevance" section. The cell halts if the
    count diverges — defensive against future audits adding stray
    forcings.
    """
    # Parse gap_table.csv to find the candidate (sce, member) pairs.
    # gap_table.csv columns:
    #   model, scenario, variable_id, variant_label,
    #   in_local_raw, in_local_yearly, in_pool, in_dkrz_replica,
    #   in_dkrz_reachable, gap_all, gap_actionable,
    #   esgf_version, local_pool_version, version_drift,
    #   grid_label, require_regrid, non_canonical_variant, pair_status
    with open(gap_table_path) as f:
        gap_rows = list(csv.DictReader(f))

    def truthy(s):
        return s == 'True'

    pres = {}  # (sce, mem) -> {var: row}
    for r in gap_rows:
        if truthy(r['non_canonical_variant']):
            continue
        key = (r['scenario'], r['variant_label'])
        pres.setdefault(key, {})[r['variable_id']] = r

    candidates = []
    for (sce, mem), vmap in pres.items():
        if 'tas' not in vmap or 'msftyz' not in vmap:
            continue
        if truthy(vmap['tas']['in_local_raw']):
            continue
        if truthy(vmap['msftyz']['in_local_raw']):
            continue
        candidates.append((sce, mem))
    candidates.sort()
    n_hist = sum(1 for s, _ in candidates if s == 'historical')
    n_ssp245 = sum(1 for s, _ in candidates if s == 'ssp245')
    print(f'[cell 1] candidate (scenario, member) pairs: {len(candidates)} '
          f'(historical={n_hist}, ssp245={n_ssp245})')
    if len(candidates) != 36:
        raise SystemExit(
            f'[cell 1] expected 36 candidate pairs but got {len(candidates)}; '
            'halting. Inspect gap_table.csv and the audit "Pipeline '
            'relevance" section, then re-run.'
        )
    candidate_set = set(candidates)

    # Now read manifest.json and flatten matching gaps to a per-file plan.
    with open(manifest_path) as f:
        manifest = json.load(f)

    plan = []
    for g in manifest['gaps']:
        sce, mem, vid = g['scenario'], g['variant_label'], g['variable_id']
        if (sce, mem) not in candidate_set:
            continue
        if vid not in ('tas', 'msftyz'):
            continue
        if g.get('non_canonical_variant'):
            continue
        for f_ in g['files']:
            url = f_.get('url')
            if not url:
                continue
            plan.append({
                'scenario': sce,
                'variant_label': mem,
                'variable_id': vid,
                'grid_label': g.get('grid_label'),
                'url': url,
                'expected_size': f_.get('size'),
                'checksum': f_.get('checksum'),
                'preferred_data_node': g.get('preferred_data_node'),
            })
    print(f'[cell 1] expanded to per-file work plan: {len(plan)} files')
    return plan


# %%
# CELL 2: Target paths + skip-already-present --------------------------------

def assign_targets(plan, target_dir):
    """Add ``target_path`` and ``status`` columns. status ∈
    {pending, skip_size_ok, plan}."""
    os.makedirs(target_dir, exist_ok=True)
    seen = {}
    duplicates = []
    for row in plan:
        basename = os.path.basename(row['url'])
        if not basename.endswith('.nc'):
            row['status'] = 'invalid_basename'
            continue
        target = os.path.join(target_dir, basename)
        row['target_path'] = target
        if basename in seen and seen[basename]['url'] != row['url']:
            duplicates.append(basename)
        seen[basename] = row
        if os.path.exists(target):
            sz = os.path.getsize(target)
            if (row['expected_size'] is not None
                    and sz == row['expected_size']):
                row['status'] = 'skip_size_ok'
                row['actual_size_pre'] = sz
            else:
                row['status'] = 'pending_size_mismatch'
                row['actual_size_pre'] = sz
        else:
            row['status'] = 'pending'
    if duplicates:
        print(f'[cell 2] WARNING: {len(duplicates)} duplicate basenames '
              f'across the plan (different URLs, same filename); the '
              f'last-seen URL wins on disk.')
    n_pending = sum(1 for r in plan if r.get('status') == 'pending')
    n_mismatch = sum(1 for r in plan if r.get('status') == 'pending_size_mismatch')
    n_skip = sum(1 for r in plan if r.get('status') == 'skip_size_ok')
    n_invalid = sum(1 for r in plan if r.get('status') == 'invalid_basename')
    bytes_to_fetch = sum((r['expected_size'] or 0) for r in plan
                        if r.get('status') in ('pending', 'pending_size_mismatch'))
    print(f'[cell 2] new={n_pending}, resume(size-mismatch)={n_mismatch}, '
          f'skip(size-ok)={n_skip}, invalid={n_invalid}; '
          f'bytes to fetch: {bytes_to_fetch / 1e9:.2f} GB')
    return plan


def write_plan_csv(plan, out_csv):
    cols = ['scenario', 'variant_label', 'variable_id', 'grid_label',
            'url', 'expected_size', 'checksum',
            'preferred_data_node', 'target_path', 'status']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in plan:
            w.writerow(r)
    print(f'[cell 2] wrote work plan to {out_csv}')


# %%
# CELL 3: Download (parallel) -------------------------------------------------

def _log(log_path, entry):
    entry = {'t': dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
             **entry}
    with _LOG_LOCK:
        with open(log_path, 'a') as f:
            json.dump(entry, f)
            f.write('\n')


def _fetch_one(row, log_path):
    """Stream-download one file with retries.

    Returns the updated row with keys ``status``, ``actual_size``,
    ``sha256``, ``t_elapsed``, ``attempt``.
    """
    target = row['target_path']
    partial = target + '.partial'
    url = row['url']
    last_err = None
    t0 = time.time()
    for attempt, delay in enumerate([0] + RETRY_DELAYS_S):
        if delay:
            time.sleep(delay)
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_S) as r:
                r.raise_for_status()
                h = hashlib.sha256()
                with open(partial, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        f.write(chunk)
                        h.update(chunk)
                sha = h.hexdigest()
            actual = os.path.getsize(partial)
            if row['expected_size'] and actual != row['expected_size']:
                last_err = (f'size mismatch: expected {row["expected_size"]} '
                            f'got {actual}')
                os.remove(partial)
                continue
            os.rename(partial, target)
            elapsed = time.time() - t0
            row['status'] = 'ok'
            row['actual_size'] = actual
            row['sha256'] = sha
            row['t_elapsed'] = elapsed
            row['attempt'] = attempt
            _log(log_path, {'event': 'ok', 'url': url, 'target': target,
                            'bytes': actual, 'sha256': sha,
                            'elapsed_s': round(elapsed, 1),
                            'attempt': attempt})
            return row
        except (requests.RequestException, OSError) as e:
            last_err = str(e)
            if os.path.exists(partial):
                try:
                    os.remove(partial)
                except OSError:
                    pass
    elapsed = time.time() - t0
    row['status'] = 'fail'
    row['error'] = last_err
    row['t_elapsed'] = elapsed
    _log(log_path, {'event': 'fail', 'url': url, 'target': target,
                    'error': last_err, 'elapsed_s': round(elapsed, 1)})
    return row


def download(plan, log_path, workers=DEFAULT_WORKERS, limit_bytes=None):
    todo = [r for r in plan
            if r.get('status') in ('pending', 'pending_size_mismatch')]
    if limit_bytes is not None:
        cum = 0
        capped = []
        for r in todo:
            cum += r.get('expected_size') or 0
            capped.append(r)
            if cum >= limit_bytes:
                break
        print(f'[cell 3] --limit-bytes {limit_bytes / 1e9:.2f} GB: '
              f'capped to first {len(capped)} files')
        todo = capped
    print(f'[cell 3] starting download: {len(todo)} files, '
          f'{workers} workers')
    t0 = time.time()
    results = []
    done = 0
    bytes_done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, r, log_path): r for r in todo}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            if res.get('actual_size'):
                bytes_done += res['actual_size']
            if done % 50 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate_mbps = (bytes_done / 1e6) / max(elapsed, 1)
                print(f'[cell 3] {done:5}/{len(todo)}  '
                      f'{bytes_done/1e9:.2f} GB  '
                      f'{rate_mbps:.1f} MB/s elapsed={elapsed/60:.1f}min',
                      flush=True)
    n_ok = sum(1 for r in results if r['status'] == 'ok')
    n_fail = sum(1 for r in results if r['status'] == 'fail')
    print(f'[cell 3] done: ok={n_ok}, fail={n_fail}, '
          f'bytes={bytes_done/1e9:.2f} GB, '
          f'wall={time.time() - t0:.1f} s')
    return results


# %%
# CELL 4: Verify + status CSV -------------------------------------------------

def verify(plan, target_dir, audit_dir=AUDIT_DIR):
    """Re-scan target dir, compute final size/sha checks, write
    download_status.csv. Halts on impossible state but otherwise just
    reports."""
    rows = []
    for r in plan:
        target = r.get('target_path')
        if not target:
            rows.append({**r, 'final_status': 'no_target',
                         'actual_size_final': None, 'size_match': False})
            continue
        if not os.path.exists(target):
            rows.append({**r, 'final_status': 'missing',
                         'actual_size_final': None, 'size_match': False})
            continue
        actual = os.path.getsize(target)
        match = (r['expected_size'] is not None
                 and actual == r['expected_size'])
        rows.append({**r, 'final_status': 'ok' if match else 'size_mismatch',
                     'actual_size_final': actual, 'size_match': match})
    n_ok = sum(1 for r in rows if r['final_status'] == 'ok')
    n_mm = sum(1 for r in rows if r['final_status'] == 'size_mismatch')
    n_miss = sum(1 for r in rows if r['final_status'] == 'missing')
    print(f'[cell 4] verify: ok={n_ok}, size_mismatch={n_mm}, '
          f'missing={n_miss}, total={len(rows)}')
    out = f'{audit_dir}/download_status.csv'
    cols = ['scenario', 'variant_label', 'variable_id', 'grid_label',
            'url', 'expected_size', 'actual_size_final', 'size_match',
            'preferred_data_node', 'target_path', 'final_status']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[cell 4] wrote {out}')
    return rows


# %%
# CELL 5: Summary writer ------------------------------------------------------

def write_report(rows, audit_dir=AUDIT_DIR):
    total = len(rows)
    ok = sum(1 for r in rows if r['final_status'] == 'ok')
    mm = sum(1 for r in rows if r['final_status'] == 'size_mismatch')
    miss = sum(1 for r in rows if r['final_status'] == 'missing')
    bytes_ok = sum(r['actual_size_final'] or 0 for r in rows
                   if r['final_status'] == 'ok')

    # Per-(scenario, variable_id, member) tally
    per = {}
    for r in rows:
        k = (r['scenario'], r['variable_id'], r['variant_label'])
        d = per.setdefault(k, {'total': 0, 'ok': 0, 'fail': 0})
        d['total'] += 1
        if r['final_status'] == 'ok':
            d['ok'] += 1
        else:
            d['fail'] += 1

    lines = []
    lines.append(f'# EC-Earth3 extension download — '
                 f'{dt.date.today().isoformat()}\n')
    lines.append('Generated by `scripts/fetch_ec_earth3_extension.py`. '
                 'Source manifest: `manifest.json` from the 2026-05-25 '
                 'EC-Earth3 ESGF audit.\n')
    lines.append('## Totals\n')
    lines.append(f'- Planned: {total}')
    lines.append(f'- Successful: {ok}')
    lines.append(f'- Size mismatch: {mm}')
    lines.append(f'- Missing: {miss}')
    lines.append(f'- Bytes downloaded: {bytes_ok / 1e9:.2f} GB\n')

    lines.append('## Per (scenario, var, member) success rate\n')
    lines.append('| Scenario | Var | Member | OK | Failed | Total |')
    lines.append('|---|---|---|---|---|---|')
    for k in sorted(per):
        sce, vid, mem = k
        d = per[k]
        lines.append(f'| {sce} | {vid} | {mem} | {d["ok"]} | {d["fail"]} | '
                     f'{d["total"]} |')
    lines.append('')

    failed = [r for r in rows if r['final_status'] != 'ok']
    if failed:
        lines.append(f'## Failed URLs ({len(failed)})\n')
        for r in failed[:50]:
            lines.append(f'- `{r["url"]}` → {r["final_status"]} '
                         f'(expected={r["expected_size"]}, '
                         f'actual={r.get("actual_size_final")})')
        if len(failed) > 50:
            lines.append(f'\n_(plus {len(failed) - 50} more — see '
                         f'download_status.csv)_')
        lines.append('')

    lines.append('## Forensics\n')
    lines.append('- Per-file event log: `download_log.jsonl`')
    lines.append('- Full status: `download_status.csv`')
    lines.append('- Re-run idempotently: `python scripts/fetch_ec_earth3_extension.py`\n')

    out = f'{audit_dir}/download_report.md'
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[cell 5] wrote {out}')


# %%
# Main entry point -----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=DEFAULT_MANIFEST)
    p.add_argument('--gap-table', default=DEFAULT_GAP_TABLE)
    p.add_argument('--target', default=DEFAULT_TARGET)
    p.add_argument('--audit-dir', default=AUDIT_DIR)
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    p.add_argument('--dry-run', action='store_true',
                   help='Stop after cell 2; write download_plan.csv only.')
    p.add_argument('--limit-bytes', type=str, default=None,
                   help='Cap downloaded bytes (suffixes: K, M, G). '
                        'E.g. --limit-bytes 50M for smoke test.')
    args = p.parse_args()

    limit_bytes = None
    if args.limit_bytes:
        s = args.limit_bytes.strip().upper()
        mult = 1
        if s.endswith('G'): mult, s = 1e9, s[:-1]
        elif s.endswith('M'): mult, s = 1e6, s[:-1]
        elif s.endswith('K'): mult, s = 1e3, s[:-1]
        limit_bytes = int(float(s) * mult)

    plan = load_candidates(args.manifest, args.gap_table)
    plan = assign_targets(plan, args.target)
    plan_csv = f'{args.audit_dir}/download_plan.csv'
    write_plan_csv(plan, plan_csv)

    if args.dry_run:
        print('--dry-run: stopping after cell 2.')
        return

    log_path = f'{args.audit_dir}/download_log.jsonl'
    download(plan, log_path, workers=args.workers, limit_bytes=limit_bytes)
    rows = verify(plan, args.target, audit_dir=args.audit_dir)
    write_report(rows, audit_dir=args.audit_dir)
    print(f'\n=== EC-Earth3 extension download complete → {args.target} ===')


if __name__ == '__main__':
    main()
