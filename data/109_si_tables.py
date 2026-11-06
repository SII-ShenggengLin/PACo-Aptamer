#!/usr/bin/env python
# Build the 10 SI tables (CSV + markdown) into data/survey/109_si_tables/.
# All data come from data/survey/*.json (+ two analysis/*.md text transcriptions,
# explicitly marked); nothing under data/ is modified except this script's own
# output directory.
# Run: conda run -n rna_if python data/109_si_tables.py
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))          # data/
SURVEY = os.path.join(HERE, 'survey')
OUT = os.path.join(SURVEY, '109_si_tables')
ANALYSIS = os.path.join(os.path.dirname(HERE), 'analysis')

METHODS = ['PACoAptamer', 'RhoDesign', 'gRNAde', 'NA-MPNN', 'AlignIF']
METHOD_LABEL = {
    'PACoAptamer': 'PACo-Aptamer',
    'RhoDesign': 'RhoDesign',
    'gRNAde': 'gRNAde',
    'NA-MPNN': 'NA-MPNN',
    'AlignIF': 'AlignIF',
}
NA = 'n/a'

CHECKS = []


def check(name, ok, detail):
    CHECKS.append({'check': name, 'pass': bool(ok), 'detail': detail})
    if not ok:
        print('FAIL', name, '--', detail[:160])
    return ok


def load(name):
    with open(os.path.join(SURVEY, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- formatting
def pct1(x):
    return NA if x is None else f'{100 * x:.1f}'


def pp1(x):
    return NA if x is None else f'{100 * x:+.1f}'


def rho3(x):
    return NA if x is None else f'{x:+.3f}'


def p2s(p):
    # P values: 2 significant digits; scientific below 1e-3 with trimmed exponent
    if p is None:
        return NA
    if p < 1e-3:
        mant, exp = f'{p:.1e}'.split('e')
        return f'{mant}e{int(exp)}'
    return f'{p:.2g}'


def csv_value(v):
    if v is None:
        return NA
    if isinstance(v, float):
        return f'{v:.6g}'
    s = str(v)
    if any(c in s for c in ',"\n'):
        s = '"' + s.replace('"', '""') + '"'
    return s


def write_csv(path, cols, rows):
    lines = [','.join(cols)]
    for r in rows:
        lines.append(','.join(csv_value(r.get(c)) for c in cols))
    Path(path).write_text('\n'.join(lines) + '\n')


def write_text(path, text):
    Path(path).write_text(text)


def wilcoxon_2s(x, y, min_nz=5):
    """Paired two-sided Wilcoxon, zero-delta pairs dropped (scipy default);
    None when fewer than min_nz non-zero pairs remain."""
    nz = [(a, b) for a, b in zip(x, y) if a != b]
    if len(nz) < min_nz:
        return None, len(nz)
    p = wilcoxon([a for a, _ in nz], [b for _, b in nz], alternative='two-sided').pvalue
    return float(p), len(nz)


# ================================================================ table 1
# st_benchmark_full — three test layers x five methods + PACo-vs-baseline P
def build_benchmark_full():
    main = load('63_main_table_gfix.json')['table']
    views = load('69_alayer_gfix.json')['views']
    wil = load('81_wilcoxon_gfix.json')
    layers = [
        ('cluster-based test', 477, {m: {'best': main[m]['recovery_best_mean'],
                                         't01': main[m]['recovery_t01_mean']}
                                     for m in METHODS}, wil['B_test_477']),
        ('temporal independent test', 235,
         {m: {'best': views['independent_test'][m]['best'],
              't01': views['independent_test'][m]['t01']} for m in METHODS},
         wil['A_indep_235']),
        ('zero-shot (ligand-held-out) test', 93,
         {m: {'best': views['zero_shot_targets'][m]['best'],
              't01': views['zero_shot_targets'][m]['t01']} for m in METHODS},
         wil['A_zs_93']),
    ]
    rows = []
    for layer, n, vals, wlayer in layers:
        for m in METHODS:
            row = {'layer': layer, 'n_chains': n, 'method': METHOD_LABEL[m],
                   'recovery_best_pct': round(100 * vals[m]['best'], 4),
                   'recovery_t01_pct': round(100 * vals[m]['t01'], 4)}
            if m == 'PACoAptamer':
                row.update(delta_best_pp=None, delta_t01_pp=None,
                           wilcoxon_p_best_two_sided=None, wilcoxon_p_t01_two_sided=None)
            else:
                wb = wlayer[m]['recovery_best']
                wt = wlayer[m]['recovery_t01_mean']
                row.update(delta_best_pp=round(100 * wb['delta'], 4),
                           delta_t01_pp=round(100 * wt['delta'], 4),
                           wilcoxon_p_best_two_sided=wb['p_two_sided'],
                           wilcoxon_p_t01_two_sided=wt['p_two_sided'])
            rows.append(row)
    # anchors
    t = {r['method']: r for r in rows if r['layer'] == 'cluster-based test'}
    check('t1_cluster_paco', abs(t['PACo-Aptamer']['recovery_best_pct'] - 74.90) < 0.01
          and abs(t['PACo-Aptamer']['recovery_t01_pct'] - 60.40) < 0.01,
          f"PACo {t['PACo-Aptamer']['recovery_best_pct']}/{t['PACo-Aptamer']['recovery_t01_pct']}")
    for m, b, t01 in [('RhoDesign', 69.36, 55.10), ('AlignIF', 56.20, 51.77),
                      ('NA-MPNN', 52.21, 46.98), ('gRNAde', 53.00, 35.77)]:
        check(f't1_cluster_{m}', abs(t[m]['recovery_best_pct'] - b) < 0.01
              and abs(t[m]['recovery_t01_pct'] - t01) < 0.01,
              f"{m} {t[m]['recovery_best_pct']}/{t[m]['recovery_t01_pct']} vs {b}/{t01}")
    check('t1_p_vs_rhodesign',
          abs(t['RhoDesign']['wilcoxon_p_best_two_sided'] - 4.792e-48) / 4.792e-48 < 1e-2
          and abs(t['RhoDesign']['wilcoxon_p_t01_two_sided'] - 6.104e-32) / 6.104e-32 < 1e-2,
          f"P best={t['RhoDesign']['wilcoxon_p_best_two_sided']:.3g} "
          f"t01={t['RhoDesign']['wilcoxon_p_t01_two_sided']:.3g}")
    check('t1_layers_n', len(rows) == 15, f'{len(rows)} rows')

    cols = ['layer', 'n_chains', 'method', 'recovery_best_pct', 'recovery_t01_pct',
            'delta_best_pp', 'wilcoxon_p_best_two_sided',
            'delta_t01_pp', 'wilcoxon_p_t01_two_sided']
    md = ['**Supplementary Table | Full benchmark: three held-out layers × five methods.**', '',
          '| Layer | n | Method | Best-of-pool (%) | t = 0.1 mean (%) | Δ best vs PACo (pp) | '
          'P best | Δ t01 vs PACo (pp) | P t01 |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['layer'], r['n_chains'], r['method'],
            f"{r['recovery_best_pct']:.1f}", f"{r['recovery_t01_pct']:.1f}",
            pp1(r['delta_best_pp'] / 100 if r['delta_best_pp'] is not None else None)
            if r['delta_best_pp'] is not None else '—',
            p2s(r['wilcoxon_p_best_two_sided']),
            pp1(r['delta_t01_pp'] / 100 if r['delta_t01_pp'] is not None else None)
            if r['delta_t01_pp'] is not None else '—',
            p2s(r['wilcoxon_p_t01_two_sided'])))
    md += ['',
           'Recovery = chain-macro sequence recovery (%). Δ = PACo-Aptamer − method (mean paired '
           'per-chain difference, percentage points; positive = PACo-Aptamer better). '
           'P = paired two-sided Wilcoxon signed-rank test (81_wilcoxon_gfix.json; zero-delta '
           'pairs dropped). Cluster-based layer from 63_main_table_gfix.json (gRNAde C/G '
           'decode-fixed); temporal (235) and zero-shot (93) layers from 69_alayer_gfix.json '
           'views. PACo-Aptamer row: Δ/P not applicable (self-comparison).']
    return 'st_benchmark_full', cols, rows, '\n'.join(md) + '\n', {
        'recovery_cluster_477': 'data/survey/63_main_table_gfix.json table',
        'recovery_temporal_235_and_zs_93': 'data/survey/69_alayer_gfix.json views '
                                           '(independent_test, zero_shot_targets)',
        'delta_and_wilcoxon': 'data/survey/81_wilcoxon_gfix.json '
                              '(B_test_477, A_indep_235, A_zs_93; p_two_sided)'}, []


# ================================================================ table 2
# st_ensemble_scan — stage-2 variant scan, soups, output ensembles
def build_ensemble_scan():
    def summ(name):
        return load(f'metrics_PACo_v2_{name}.json')['summary']

    # val records for the noiface family (structured; n=96 val chains,
    # t01x20 + 3-temperature x20 best-of-60 protocol, data/73_val_select_ensemble.py)
    val_noiface = load('73_val_select_ensemble.json')
    # val records for the withiface family survive only as text in
    # analysis/10_对比方法基准报告.md §9.4 (the original 73_val_select_ensemble.json
    # records were overwritten by the later noiface selection run) -> text transcription
    VAL_TEXT94 = {  # config: (val_t01, val_best_60pool)
        'ens2': (0.7331, 0.8263),
        'ens3': (0.7256, 0.8256),
        'single_seed43': (0.7230, 0.7908),
        'single_ssdo15': (0.7331, 0.8061),
    }
    # S1 best-val column transcribed from analysis/10 §9.1 (text; training logs)
    S1_VAL_TEXT91 = {'seed43': 0.6382, 'ssdo15': 0.6426, 'batch4r': 0.6399,
                     'batch4': 0.6399, 'ssd': 0.6228, 'ema': 0.6330,
                     's2_v2_old_main': 0.5805}

    rows = []

    def add(variant, category, ckpts, n_models, metrics_name, val_t01=None,
            val_best=None, val_src='', desc='', s1_val=None):
        s = summ(metrics_name) if metrics_name else None
        rows.append({
            'variant': variant, 'category': category, 'ckpts': ckpts,
            'n_models': n_models,
            's1_best_val_text91': s1_val,
            'test_best_pct': round(100 * s['recovery_best_mean'], 4) if s else None,
            'test_t01_pct': round(100 * s['recovery_t01_mean_mean'], 4) if s else None,
            'val_t01_20x': val_t01, 'val_best_60pool': val_best, 'val_source': val_src,
            'description': desc,
        })

    add('seed43', 'single-model S2 variant', 's2_v2_seed43', 1, 'seed43_full',
        val_t01=VAL_TEXT94['single_seed43'][0], val_best=VAL_TEXT94['single_seed43'][1],
        val_src='analysis/10 §9.4 (text) ⚠️', s1_val=S1_VAL_TEXT91['seed43'],
        desc='S2 retrain with --seed 43 (S1 = s1_batch4_seed43); best single model')
    add('ssdo15', 'single-model S2 variant', 's2_v2_ssdo15', 1, 'ssdo15_full',
        val_t01=VAL_TEXT94['single_ssdo15'][0], val_best=VAL_TEXT94['single_ssdo15'][1],
        val_src='analysis/10 §9.4 (text) ⚠️', s1_val=S1_VAL_TEXT91['ssdo15'],
        desc='SS whole-chain dropout 0.15; highest S1 val')
    add('batch4r', 'single-model S2 variant', 's2_v2_batch4r', 1, 'batch4r_full',
        s1_val=S1_VAL_TEXT91['batch4r'],
        desc='base config exact rerun (reproducibility control; Δ0.34pp vs batch4 ≈ eval noise)')
    add('batch4', 'single-model S2 variant', 's2_v2_batch4 (s1_batch4_lr1_60ep)', 1,
        'batch4_full', s1_val=S1_VAL_TEXT91['batch4'],
        desc='base config original run (2026-08-17 morning baseline)')
    add('ssd', 'single-model S2 variant', 's2_v2_ssd', 1, 'ssd_full',
        s1_val=S1_VAL_TEXT91['ssd'], desc='SS dropout, alternative level')
    add('ema', 'single-model S2 variant', 's2_v2_ema', 1, 'ema_full',
        s1_val=S1_VAL_TEXT91['ema'], desc='EMA 0.999 weights (EMA later removed from train.py)')
    add('s43ssdo15', 'single-model S2 variant', 's2_v2_s43ssdo15', 1, 's43ssdo15_full',
        desc='combined seed43 + SS-dropout-0.15 S1 retrain (60 ep, batch4)')
    add('s1long', 'single-model variant', 's1_v2_long', 1, 's1long_full',
        desc='S1 long-schedule variant evaluated on test (no separate S2 scan entry)')
    add('s2batch4', 'single-model S2 variant', 's2_v2_batch4', 1, 's2batch4_full',
        desc='S2 batch-size-4 variant (earlier scan branch)')
    add('s2jointcv', 'single-model S2 variant', 's2_v2_joint_cv', 1, 's2jointcv_full',
        desc='S2 joint + cv variant (earlier scan branch)')
    add('s2_v2_old_main', 'reference (old main config)', 's2_v2', 1, 'main',
        s1_val=S1_VAL_TEXT91['s2_v2_old_main'],
        desc='pre-scan main config; canonical-fixed re-evaluation (metrics_PACo_v2_main.json); '
             'the original 2026-08-16 manifest value 68.35/56.86 exists only as text in '
             'analysis/10 §9.1 ⚠️')
    add('soup_bb', 'weight soup (rejected)', 'soup of 2 S2 ckpts (same S1)', 2, 'soup_bb_full',
        desc='2-weight average, same S1 basin: no gain (70.90% ≈ single)')
    add('soup_bx', 'weight soup (rejected)', 'soup of 3 ckpts (+ssdo15)', 3, 'soup_bx_full',
        desc='3-weight average: degraded')
    add('soup_all4', 'weight soup (rejected)', 'soup of 4 ckpts (+seed43, cross-basin)', 4,
        'soup_all4_full', desc='4-weight cross-basin average: strongly degraded; soup route abandoned')
    add('ens2 (seed43+ssdo15)', 'output ensemble', 's2_v2_seed43 + s2_v2_ssdo15', 2,
        'ens2_full', val_t01=VAL_TEXT94['ens2'][0], val_best=VAL_TEXT94['ens2'][1],
        val_src='analysis/10 §9.4 (text) ⚠️',
        desc='stepwise softmax-probability averaging; pool size unchanged (3 T × 100); '
             'selected as main config on val (ens3 not better on val)')
    add('ens3 (batch4r+seed43+ssdo15)', 'output ensemble (rejected on val)',
        's2_v2_batch4r + s2_v2_seed43 + s2_v2_ssdo15', 3, 'ens3_full',
        val_t01=VAL_TEXT94['ens3'][0], val_best=VAL_TEXT94['ens3'][1],
        val_src='analysis/10 §9.4 (text) ⚠️',
        desc='test best higher than ens2 (75.18 vs 74.63) but val not improved '
             '(0.8256 ≤ 0.8263) and one extra model -> rejected to avoid test-set selection')
    add('ens2-noiface (released)', 'output ensemble, final recipe',
        's2_v2_seed43_noiface + s2_v2_ssdo15_noiface', 2, 'ens2noiface_full',
        val_t01=round(val_noiface['ens2noiface']['val_t01_mean'], 4),
        val_best=round(val_noiface['ens2noiface']['val_best_mean'], 4),
        val_src='73_val_select_ensemble.json',
        desc='released config (interface weighting removed); canonical-repaired main-table '
             'values are in st_benchmark_full (74.90/60.40)')
    add('seed43-noiface', 'single model, final recipe', 's2_v2_seed43_noiface', 1, None,
        val_t01=round(val_noiface['seed43noi']['val_t01_mean'], 4),
        val_best=round(val_noiface['seed43noi']['val_best_mean'], 4),
        val_src='73_val_select_ensemble.json',
        desc='noiface single; 477-chain scan-pool evaluation n/a here — its 300-pool values '
             'are the no_ensemble row of st_ablation ⚠️')
    add('ssdo15-noiface', 'single model, final recipe', 's2_v2_ssdo15_noiface', 1, None,
        val_t01=round(val_noiface['ssdo15noi']['val_t01_mean'], 4),
        val_best=round(val_noiface['ssdo15noi']['val_best_mean'], 4),
        val_src='73_val_select_ensemble.json',
        desc='noiface single; 477-chain scan-pool evaluation not run ⚠️')

    byv = {r['variant']: r for r in rows}
    check('t2_singles_anchor',
          abs(byv['seed43']['test_best_pct'] - 71.49) < 0.01
          and abs(byv['ssdo15']['test_best_pct'] - 71.45) < 0.01,
          f"seed43 {byv['seed43']['test_best_pct']} ssdo15 {byv['ssdo15']['test_best_pct']}")
    check('t2_ens_anchor',
          abs(byv['ens2 (seed43+ssdo15)']['test_best_pct'] - 74.63) < 0.01
          and abs(byv['ens3 (batch4r+seed43+ssdo15)']['test_best_pct'] - 75.18) < 0.01,
          f"ens2 {byv['ens2 (seed43+ssdo15)']['test_best_pct']} "
          f"ens3 {byv['ens3 (batch4r+seed43+ssdo15)']['test_best_pct']}")
    check('t2_val_reject_ens3',
          VAL_TEXT94['ens2'][1] >= VAL_TEXT94['ens3'][1],
          f"val best ens2 {VAL_TEXT94['ens2'][1]} >= ens3 {VAL_TEXT94['ens3'][1]}")
    check('t2_val_noiface_n96',
          all(val_noiface[k]['n'] == 96 for k in val_noiface), 'val n=96 for noiface rows')

    cols = ['variant', 'category', 'n_models', 'ckpts', 's1_best_val_text91',
            'test_best_pct', 'test_t01_pct', 'val_t01_20x', 'val_best_60pool',
            'val_source', 'description']
    md = ['**Supplementary Table | Stage-2 training-variant scan, weight soups and output '
          'ensembles (PACo-Aptamer).**', '',
          '| Variant | Category | #models | Test best (%) | Test t01 (%) | val t01 | '
          'val best (60-pool) | Notes |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['variant'], r['category'], r['n_models'],
            pct1(r['test_best_pct'] / 100 if r['test_best_pct'] is not None else None),
            pct1(r['test_t01_pct'] / 100 if r['test_t01_pct'] is not None else None),
            f"{r['val_t01_20x']:.4f}" if r['val_t01_20x'] is not None else NA,
            f"{r['val_best_60pool']:.4f}" if r['val_best_60pool'] is not None else NA,
            r['description'] + ('' if r['val_source'] in ('', '73_val_select_ensemble.json')
                                else f" [val: {r['val_source']}]")))
    md += ['',
           'Test columns: scan protocol = 477 cluster-based test chains, 3 temperatures '
           '(0.1/0.5/1.0) × 100 samples, manifest convention (metrics_PACo_v2_*_full.json). '
           'val columns: 96 val chains, t = 0.1 × 20 mean + best of 3-temperature × 20 pool '
           '(73_val_select_ensemble.json for the noiface family). val figures for the '
           'withiface family (ens2/ens3/single_seed43/single_ssdo15) and the S1 best-val '
           'column are transcribed from analysis/10_对比方法基准报告.md §9.4/§9.1 (text): the '
           'original 73 JSON records were overwritten by the later noiface selection run — '
           'marked ⚠️. ens3 beats ens2 on test best (75.18 vs 74.63) but not on val '
           '(0.8256 ≤ 0.8263) and was rejected to avoid test-set selection. Weight soups '
           '(2–4 models) never helped and the route was abandoned (analysis/10 §9.2).']
    na_reasons = [
        'withiface-family val records (ens2/ens3/single_seed43/single_ssdo15) and S1 best-val '
        'column: not present in any surviving JSON — transcribed from analysis/10 §9.4/§9.1 ⚠️',
        'seed43-noiface / ssdo15-noiface: no 477-chain scan-pool metrics file exists '
        '(val only in 73_val_select_ensemble.json) ⚠️',
        's2_v2 old main config: original manifest value 68.35/56.86 only as text in '
        'analysis/10 §9.1; the row reports the canonical-fixed re-evaluation '
        '(metrics_PACo_v2_main.json) ⚠️',
        's1long/s2batch4/s2jointcv/s43ssdo15: S1-val not recorded in analysis/10 §9.1 -> n/a',
    ]
    return 'st_ensemble_scan', cols, rows, '\n'.join(md) + '\n', {
        'test_scan_protocol': 'data/survey/metrics_PACo_v2_{variant}_full.json summaries '
                              '(477 chains, 3T x 100)',
        'val_noiface_family': 'data/survey/73_val_select_ensemble.json (n=96)',
        'val_withiface_family_and_s1_val': 'analysis/10_对比方法基准报告.md §9.4/§9.1 (text '
                                           'transcription ⚠️)'}, na_reasons


# ================================================================ table 3
# st_ablation — component ablation, final noiface recipe, 300-sample pools
def build_ablation():
    p00 = load('74_p00_iface_ablation.json')['phase1_ablation']
    meta = load('74_iface_meta.json')['stems']
    iface_sel = {k: v for k, v in meta.items()
                 if v and v.get('usable') and 0 < sum(v['interface']) <= 50}

    def chain_metrics(variant):
        pool = load(f'74_pools/{variant}.json')
        gb, gt, ib, it = {}, {}, {}, {}
        for stem, rec in pool.items():
            nat = rec['native']
            sqs = rec['pool']
            ro = [sum(1 for a, b in zip(s, nat) if a == b) / len(nat) for s in sqs]
            gb[stem] = max(ro)
            gt[stem] = float(np.mean(ro[:100]))
            m = iface_sel.get(stem)
            if m is not None:
                idx = [i for i, f in enumerate(m['interface']) if f]
                ri = [sum(1 for i in idx if s[i] == nat[i]) / len(idx) for s in sqs]
                ib[stem] = max(ri)
                it[stem] = float(np.mean(ri[:100]))
        return gb, gt, ib, it

    variants = ['baseline', 'no_ensemble', 'no_ss', 'no_aug', 'no_ligand']
    labels = {'baseline': 'Full model (ens2, released)',
              'no_ensemble': '− ensemble (single model)',
              'no_ss': '− SS conditioning',
              'no_aug': '− data augmentation',
              'no_ligand': '− ligand context'}
    descs = {'baseline': 'released ens2-noiface, 300-sample pools (3 T × 100)',
             'no_ensemble': 'single model (seed43-noiface)',
             'no_ss': 'secondary-structure conditioning removed (S1+S2 consistent)',
             'no_aug': 'training data augmentation removed (S2)',
             'no_ligand': 'ligand atomic context removed (S2)'}

    per_chain = {v: chain_metrics(v) for v in variants}
    gb0, gt0, ib0, it0 = per_chain['baseline']

    rows = []
    for v in variants:
        s = p00[v]
        gb, gt, ib, it = per_chain[v]
        # verify recomputation reproduces the stored summaries exactly
        check(f't3_recalc_global_{v}',
              abs(float(np.mean(list(gb.values()))) - s['global_best_477']) < 1e-9
              and abs(float(np.mean(list(gt.values()))) - s['global_t01_477']) < 1e-9,
              f"{v}: recomputed global {np.mean(list(gb.values())):.6f}/"
              f"{np.mean(list(gt.values())):.6f} == stored {s['global_best_477']:.6f}/"
              f"{s['global_t01_477']:.6f}")
        check(f't3_recalc_iface_{v}',
              abs(float(np.mean(list(ib.values()))) - s['iface']['iface_best']) < 1e-9
              and abs(float(np.mean(list(it.values()))) - s['iface']['iface_t01']) < 1e-9,
              f"{v}: recomputed iface {np.mean(list(ib.values())):.6f}/"
              f"{np.mean(list(it.values())):.6f} == stored {s['iface']['iface_best']:.6f}/"
              f"{s['iface']['iface_t01']:.6f}")
        row = {'variant': labels[v], 'description': descs[v],
               'n_global': s['n_477'],
               'global_best_pct': round(100 * s['global_best_477'], 4),
               'global_t01_pct': round(100 * s['global_t01_477'], 4),
               'n_iface': s['iface']['n'],
               'iface_best_pct': round(100 * s['iface']['iface_best'], 4),
               'iface_t01_pct': round(100 * s['iface']['iface_t01'], 4)}
        if v == 'baseline':
            for k in ['delta_global_best_pp', 'delta_global_t01_pp', 'delta_iface_best_pp',
                      'delta_iface_t01_pp', 'p_global_best_two_sided',
                      'p_global_t01_two_sided', 'p_iface_best_two_sided',
                      'p_iface_t01_two_sided', 'p_iface_best_one_sided_stored']:
                row[k] = None
        else:
            chains_g = sorted(gb0.keys())
            chains_i = sorted(ib0.keys())
            pgb, _ = wilcoxon_2s([gb0[c] for c in chains_g], [gb[c] for c in chains_g])
            pgt, _ = wilcoxon_2s([gt0[c] for c in chains_g], [gt[c] for c in chains_g])
            pib, _ = wilcoxon_2s([ib0[c] for c in chains_i], [ib[c] for c in chains_i])
            pit, _ = wilcoxon_2s([it0[c] for c in chains_i], [it[c] for c in chains_i])
            row.update(
                delta_global_best_pp=round(100 * (s['global_best_477']
                                                  - p00['baseline']['global_best_477']), 4),
                delta_global_t01_pp=round(100 * (s['global_t01_477']
                                                 - p00['baseline']['global_t01_477']), 4),
                delta_iface_best_pp=round(100 * (s['iface']['iface_best']
                                                 - p00['baseline']['iface']['iface_best']), 4),
                delta_iface_t01_pp=round(100 * (s['iface']['iface_t01']
                                                - p00['baseline']['iface']['iface_t01']), 4),
                p_global_best_two_sided=pgb, p_global_t01_two_sided=pgt,
                p_iface_best_two_sided=pib, p_iface_t01_two_sided=pit,
                p_iface_best_one_sided_stored=s['vs_baseline_iface']['wilcoxon_p_one_sided'])
            # consistency: two-sided == 2x stored one-sided (one-sided <= 0.5 here)
            check(f't3_p_convention_{v}',
                  abs(pib - 2 * s['vs_baseline_iface']['wilcoxon_p_one_sided'])
                  <= 1e-6 * max(1e-300, 2 * s['vs_baseline_iface']['wilcoxon_p_one_sided']),
                  f'{v}: two-sided {pib:.4g} = 2x stored one-sided '
                  f"{s['vs_baseline_iface']['wilcoxon_p_one_sided']:.4g}")
        rows.append(row)

    noe = {r['variant']: r for r in rows}['− ensemble (single model)']
    check('t3_anchor_no_ensemble',
          abs(noe['delta_global_best_pp'] - (-3.16)) < 0.01
          and abs(noe['delta_iface_best_pp'] - (-4.25)) < 0.01,
          f"no_ensemble Δ {noe['delta_global_best_pp']}pp global / "
          f"{noe['delta_iface_best_pp']}pp iface")
    comp_deltas = []
    for i, key in enumerate(['− SS conditioning', '− data augmentation', '− ligand context']):
        r = {r2['variant']: r2 for r2 in rows}[key]
        d = r['global_best_pct'] - noe['global_best_pct']
        comp_deltas.append(d)
    check('t3_anchor_components',
          all(0.6 <= abs(d) <= 0.85 for d in comp_deltas),
          f'component removals vs single model: {[round(d, 2) for d in comp_deltas]} pp '
          '(expected 0.6–0.8pp)')

    cols = ['variant', 'description', 'n_global', 'global_best_pct', 'global_t01_pct',
            'delta_global_best_pp', 'p_global_best_two_sided',
            'delta_global_t01_pp', 'p_global_t01_two_sided',
            'n_iface', 'iface_best_pct', 'iface_t01_pct',
            'delta_iface_best_pp', 'p_iface_best_two_sided',
            'p_iface_best_one_sided_stored',
            'delta_iface_t01_pp', 'p_iface_t01_two_sided']
    md = ['**Supplementary Table | Component ablation (final noiface recipe, 300-sample '
          'pools).**', '',
          '| Variant | Global best (%) | Global t01 (%) | Δ global best (pp) | P | '
          'Interface best (%) | Interface t01 (%) | Δ iface best (pp) | P (two-sided) | '
          'P (one-sided, as in Fig. 4a) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        base = r['variant'] == 'Full model (ens2, released)'
        md.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['variant'], f"{r['global_best_pct']:.1f}", f"{r['global_t01_pct']:.1f}",
            '—' if base else pp1(r['delta_global_best_pp'] / 100),
            '—' if base else p2s(r['p_global_best_two_sided']),
            f"{r['iface_best_pct']:.1f}", f"{r['iface_t01_pct']:.1f}",
            '—' if base else pp1(r['delta_iface_best_pp'] / 100),
            '—' if base else p2s(r['p_iface_best_two_sided']),
            '—' if base else p2s(r['p_iface_best_one_sided_stored'])))
    md += ['',
           'Source: 74_p00_iface_ablation.json phase1_ablation (300-sample pools, 3 T × 100, '
           'final noiface recipe; this supersedes the earlier quick 30-sample-pool ablation — '
           'the numbers here are the Fig. 4a values). Global = 477 test chains; interface = '
           '115 ligand-contact chains (<5 Å, usable alignment, ≤50 interface residues; '
           'identical chain set and convention reproduced from 74_pools/*.json + '
           '74_iface_meta.json, verified to match stored means to <1e-9). Δ = variant − full '
           'model (percentage points). Two-sided P recomputed per-chain (paired Wilcoxon, '
           'zero-delta pairs dropped); one-sided P column is the stored Fig. 4a value '
           '(H1: full > variant) — two-sided = 2× one-sided everywhere here. Within the '
           'single-model family, component removals cost 0.6–0.8 pp globally (interface '
           'effects n.s. at this pool size; analysis/19 §三).']
    return 'st_ablation', cols, rows, '\n'.join(md) + '\n', {
        'summary': 'data/survey/74_p00_iface_ablation.json phase1_ablation',
        'two_sided_p': 'recomputed from data/survey/74_pools/{baseline,no_ensemble,no_ss,'
                       'no_aug,no_ligand}.json + 74_iface_meta.json '
                       '(usable & 0<n_iface<=50; verified == stored means)',
        'one_sided_p': '74_p00_iface_ablation.json vs_baseline_iface.wilcoxon_p_one_sided '
                       '(Fig. 4a convention)'}, []


# ================================================================ table 4
# st_lofo — leave-one-family-out + family-coverage curve (zero-shot 93 chains)
def build_lofo():
    lofo = load('95_lofo.json')
    ev = load('95_eval.json')
    fams = load('95_families.json')
    zt = lofo['zs_table']
    ds = lofo['datasets']
    base_pc = ev['s2_v2_seed43_noiface|zs']['per_chain']
    chains93 = sorted(base_pc.keys())

    def zs(cfg):
        return zt[cfg]['summary']

    def p_recomputed(cfg, conv):
        pc = ev[f'{cfg}|zs']['per_chain']
        key = 'recovery_best' if conv == 'best' else 'recovery_t01_mean'
        p, _ = wilcoxon_2s([base_pc[c][key] for c in chains93],
                           [pc[c][key] for c in chains93])
        return p

    fam_def = fams['families_def']
    fam_ccds = {'aminoglycoside': fam_def['aminoglycoside'], 'sam_sah': fam_def['sam_sah'],
                'ntp': fam_def['ntp'], 'cla': ['CLA'], '2ba': ['2BA']}

    rows = []
    base = zs('s2_v2_seed43_noiface')
    rows.append({'block': 'lofo', 'variant': 'baseline (full 768-chain training set)',
                 'family_removed': NA, 'ccds_removed': NA, 'n_removed': 0,
                 'n_train_chains': 768,
                 'zs_best_pct': round(100 * base['best_mean'], 4),
                 'zs_t01_pct': round(100 * base['t01_mean'], 4),
                 'delta_best_pp': None, 'delta_t01_pp': None,
                 'wilcoxon_p_best_two_sided': None, 'wilcoxon_p_t01_two_sided': None,
                 'note': 'reference checkpoint s2_v2_seed43_noiface'})
    for fam in ['aminoglycoside', 'sam_sah', 'ntp', 'cla', '2ba']:
        cfg = f's2_v2_lofo_{fam}'
        s = zs(cfg)
        vb = zt[cfg]['vs_baseline']
        p_t01 = p_recomputed(cfg, 't01')
        p_best_re = p_recomputed(cfg, 'best')
        check(f't4_p_best_recalc_{fam}',
              abs(p_best_re - vb['wilcoxon_p_best']) < 1e-9,
              f"{fam}: recomputed p_best {p_best_re:.6g} == stored {vb['wilcoxon_p_best']:.6g}")
        rows.append({'block': 'lofo', 'variant': f'LOFO −{fam}',
                     'family_removed': fam, 'ccds_removed': '+'.join(fam_ccds[fam]),
                     'n_removed': ds[f'lofo_{fam}']['removed'],
                     'n_train_chains': ds[f'lofo_{fam}']['n_chains'],
                     'zs_best_pct': round(100 * s['best_mean'], 4),
                     'zs_t01_pct': round(100 * s['t01_mean'], 4),
                     'delta_best_pp': round(100 * vb['delta_best_mean'], 4),
                     'delta_t01_pp': round(100 * vb['delta_t01_mean'], 4),
                     'wilcoxon_p_best_two_sided': vb['wilcoxon_p_best'],
                     'wilcoxon_p_t01_two_sided': p_t01,
                     'note': ''})
    s = zs('s2_v2_full_ctl')
    vb = zt['s2_v2_full_ctl']['vs_baseline']
    p_t01 = p_recomputed('s2_v2_full_ctl', 't01')
    check('t4_p_best_recalc_full_ctl',
          abs(p_recomputed('s2_v2_full_ctl', 'best') - vb['wilcoxon_p_best']) < 1e-9,
          f"full_ctl recomputed == stored {vb['wilcoxon_p_best']:.6g}")
    rows.append({'block': 'lofo', 'variant': 'full_ctl (retrain on full data)',
                 'family_removed': 'none (retraining-noise control)', 'ccds_removed': NA,
                 'n_removed': 0, 'n_train_chains': ds['full_ctl']['n_chains'],
                 'zs_best_pct': round(100 * s['best_mean'], 4),
                 'zs_t01_pct': round(100 * s['t01_mean'], 4),
                 'delta_best_pp': round(100 * vb['delta_best_mean'], 4),
                 'delta_t01_pp': round(100 * vb['delta_t01_mean'], 4),
                 'wilcoxon_p_best_two_sided': vb['wilcoxon_p_best'],
                 'wilcoxon_p_t01_two_sided': p_t01,
                 'note': 'full-data retrain control: Δ ≈ 0, P = 0.91 -> noise level ±0.04pp'})
    # coverage curve
    fam_level = {f['family']: f for f in fams['family_level']}
    for rec in lofo['coverage_curve']:
        k = rec['k']
        kept = ds[f'only{k}fam']['kept_families']
        cfg = f's2_v2_only{k}fam'
        s = zs(cfg)
        vb = zt[cfg]['vs_baseline']
        rows.append({'block': 'coverage', 'variant': f'only {k} famil' + ('y' if k == 1 else 'ies'),
                     'family_removed': NA,
                     'ccds_removed': 'kept: ' + '+'.join(kept),
                     'n_removed': None,
                     'n_train_chains': rec['n_train_chains'],
                     'zs_best_pct': round(100 * rec['zs_best'], 4),
                     'zs_t01_pct': round(100 * rec['zs_t01'], 4),
                     'delta_best_pp': round(100 * vb['delta_best_mean'], 4),
                     'delta_t01_pp': round(100 * vb['delta_t01_mean'], 4),
                     'wilcoxon_p_best_two_sided': vb['wilcoxon_p_best'],
                     'wilcoxon_p_t01_two_sided': p_recomputed(cfg, 't01'),
                     'note': f'k={k} families kept of the 5 LOFO families'})
    fr = lofo['full_reference']
    rows.append({'block': 'coverage', 'variant': 'full reference (all chains)',
                 'family_removed': NA, 'ccds_removed': NA, 'n_removed': None,
                 'n_train_chains': fr['n_train_chains'],
                 'zs_best_pct': round(100 * fr['zs_best'], 4),
                 'zs_t01_pct': round(100 * fr['zs_t01'], 4),
                 'delta_best_pp': None, 'delta_t01_pp': None,
                 'wilcoxon_p_best_two_sided': None, 'wilcoxon_p_t01_two_sided': None,
                 'note': 'identical to baseline row; reference for the coverage curve'})

    check('t4_baseline', abs(base['best_mean'] - 0.7335) < 5e-4,
          f'baseline zs best {base["best_mean"]:.5f}')
    amino = [r for r in rows if r['variant'] == 'LOFO −aminoglycoside'][0]
    check('t4_amino', abs(amino['delta_best_pp'] - (-1.89)) < 0.01
          and abs(amino['wilcoxon_p_best_two_sided'] - 2.716e-4) / 2.716e-4 < 1e-2,
          f"aminoglycoside Δ {amino['delta_best_pp']}pp P {amino['wilcoxon_p_best_two_sided']:.3g}")
    ctl = [r for r in rows if r['variant'].startswith('full_ctl')][0]
    check('t4_full_ctl', abs(ctl['delta_best_pp'] - 0.04) < 0.005
          and abs(ctl['wilcoxon_p_best_two_sided'] - 0.9115) < 1e-3,
          f"full_ctl Δ {ctl['delta_best_pp']}pp P {ctl['wilcoxon_p_best_two_sided']:.4f}")
    cov = {r['n_train_chains']: r for r in rows if r['block'] == 'coverage'
           and r['n_train_chains'] in (128, 183, 207, 243)}
    exp_cov = {128: 71.2, 183: 73.5, 207: 73.3, 243: 73.7}
    check('t4_coverage', all(abs(cov[n]['zs_best_pct'] - v) < 0.05 for n, v in exp_cov.items()),
          f"coverage zs best {[cov[n]['zs_best_pct'] for n in (128, 183, 207, 243)]}")
    check('t4_amino_largest_finetune',
          fam_level['aminoglycoside']['chains'] == 128
          and all(fam_level[f]['chains'] <= 128 for f in fam_level),
          'aminoglycoside 128 = largest finetune-side family')

    cols = ['block', 'variant', 'family_removed', 'ccds_removed', 'n_removed',
            'n_train_chains', 'zs_best_pct', 'zs_t01_pct', 'delta_best_pp',
            'wilcoxon_p_best_two_sided', 'delta_t01_pp', 'wilcoxon_p_t01_two_sided', 'note']
    md = ['**Supplementary Table | Leave-one-family-out (LOFO) and ligand-family coverage '
          'curve (zero-shot 93 chains, PACo-Aptamer single model).**', '',
          '| Block | Variant | n train | zs best (%) | zs t01 (%) | Δ best (pp) | P best | '
          'Δ t01 (pp) | P t01 |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['block'], r['variant'], r['n_train_chains'],
            f"{r['zs_best_pct']:.1f}", f"{r['zs_t01_pct']:.1f}",
            pp1(r['delta_best_pp'] / 100 if r['delta_best_pp'] is not None else None),
            p2s(r['wilcoxon_p_best_two_sided']),
            pp1(r['delta_t01_pp'] / 100 if r['delta_t01_pp'] is not None else None),
            p2s(r['wilcoxon_p_t01_two_sided'])))
    md += ['',
           'All evaluations on the 93 zero-shot chains (ligands never in training). '
           'Δ = variant − baseline (percentage points). P = paired two-sided Wilcoxon; '
           'best-column P as stored in 95_lofo.json zs_table.*.vs_baseline (recomputation '
           'from 95_eval.json per_chain reproduces it exactly); t01-column P recomputed the '
           'same way (not stored in the source JSON ⚠️). LOFO removes all chains of one '
           'ligand family from the 768-chain S2 training set (aminoglycoside 128 / SAM-SAH '
           '55 / CLA 24 / NTP 20 / 2BA 16 chains). full_ctl = full-data retrain '
           '(retraining-noise control, Δ = +0.04 pp, P = 0.91). Coverage curve: training '
           'restricted to k of the 5 LOFO families (128/183/207/243 chains); reference = '
           'full 768-chain set (73.35 / 63.32).']
    na_reasons = ['t01-column Wilcoxon P is not stored in 95_lofo.json; recomputed from '
                  '95_eval.json per_chain with the identical two-sided convention ⚠️ '
                  '(best-column recomputation matches stored values to <1e-9)']
    return 'st_lofo', cols, rows, '\n'.join(md) + '\n', {
        'zs_summaries_and_p_best': 'data/survey/95_lofo.json zs_table + coverage_curve + '
                                   'full_reference',
        'per_chain_for_recomputed_p': 'data/survey/95_eval.json (<cfg>|zs)',
        'family_sizes_and_ccds': 'data/survey/95_families.json (families_def, family_level)'}, \
        na_reasons


# ================================================================ table 5
# st_tanimoto — chemical-novelty stratification, three fingerprints
def build_tanimoto():
    t93 = load('93_tanimoto_zs.json')
    t106 = load('106_tanimoto_robust.json')
    # morgan recheck consistency
    for b93, b106 in zip(t93['bins_table'], t106['bins_table_morgan_recheck']):
        check(f't5_morgan_recheck_{b93["bin"]}',
          b93['n'] == b106['n'] and
          (b93['delta_best_PACo_minus_RhoDesign']['mean'] or 0) ==
          (b106['delta_best_PACo_minus_RhoDesign']['mean'] or 0),
          f"{b93['bin']}: 93 == 106 morgan_recheck")
    fp_bins = [('Morgan (r=2, 2048-bit)', t93['bins_table']),
               ('MACCS (167-bit)', t106['bins_table_maccs']),
               ('RDKit topological', t106['bins_table_rdkit_topo'])]
    fp_spearman = [('Morgan (r=2, 2048-bit)', t93['spearman']),
                   ('MACCS (167-bit)', t106['spearman']['maccs']),
                   ('RDKit topological', t106['spearman']['rdkit_topo'])]
    rows = []
    for fp, bins in fp_bins:
        for b in bins:
            db = b['delta_best_PACo_minus_RhoDesign']
            dt = b['delta_t01_PACo_minus_RhoDesign']
            rows.append({'fingerprint': fp, 'row_type': 'bin', 'bin': b['bin'], 'n': b['n'],
                         'paco_best_pct': b['PACoAptamer_best_mean'],
                         'paco_t01_pct': b['PACoAptamer_t01_mean'],
                         'rhodesign_best_pct': b['RhoDesign_best_mean'],
                         'rhodesign_t01_pct': b['RhoDesign_t01_mean'],
                         'delta_best_mean_pp': db.get('mean'),
                         'delta_best_median_pp': db.get('median'),
                         'wilcoxon_p_two_sided': db.get('wilcoxon_p_two_sided'),
                         'delta_t01_mean_pp': dt.get('mean'),
                         'spearman_rho': None, 'spearman_p': None})
        sp = dict(fp_spearman)[fp]
        for conv, blk in [('delta_best', 'sim_vs_delta_best'),
                          ('delta_t01', 'sim_vs_delta_t01')]:
            rows.append({'fingerprint': fp, 'row_type': f'spearman_{conv}',
                         'bin': 'overall', 'n': 87,
                         'paco_best_pct': None, 'paco_t01_pct': None,
                         'rhodesign_best_pct': None, 'rhodesign_t01_pct': None,
                         'delta_best_mean_pp': None, 'delta_best_median_pp': None,
                         'wilcoxon_p_two_sided': None, 'delta_t01_mean_pp': None,
                         'spearman_rho': sp[blk]['rho'], 'spearman_p': sp[blk]['p']})
    sp_m = dict(fp_spearman)['Morgan (r=2, 2048-bit)']
    check('t5_anchor_morgan', abs(sp_m['sim_vs_delta_t01']['rho'] - (-0.4007)) < 2e-3
          and abs(sp_m['sim_vs_delta_t01']['p'] - 1.2e-4) / 1.2e-4 < 0.02,
          f"Morgan Δt01 ρ={sp_m['sim_vs_delta_t01']['rho']:.4f} "
          f"P={sp_m['sim_vs_delta_t01']['p']:.3g}")
    sp_mac = dict(fp_spearman)['MACCS (167-bit)']
    sp_top = dict(fp_spearman)['RDKit topological']
    check('t5_anchor_maccs_topo',
          abs(sp_mac['sim_vs_delta_t01']['rho'] - (-0.3406)) < 2e-3
          and abs(sp_mac['sim_vs_delta_t01']['p'] - 1.246e-3) / 1.246e-3 < 0.02
          and abs(sp_top['sim_vs_delta_t01']['rho'] - (-0.3448)) < 2e-3
          and abs(sp_top['sim_vs_delta_t01']['p'] - 1.074e-3) / 1.074e-3 < 0.02,
          f"MACCS {sp_mac['sim_vs_delta_t01']['rho']:.4f}/{sp_mac['sim_vs_delta_t01']['p']:.4g} "
          f"topo {sp_top['sim_vs_delta_t01']['rho']:.4f}/{sp_top['sim_vs_delta_t01']['p']:.4g}")
    exp_bins = {('[0,0.2)', 4, 2.7, None), ('[0.2,0.4)', 43, 4.6, 5.4e-5),
                ('[0.4,0.6)', 13, 6.5, 1.5e-3), ('[0.6,1.0]', 27, 2.2, 0.022)}
    got_bins = {(b['bin'], b['n'],
                 round(100 * b['delta_best_PACo_minus_RhoDesign']['mean'], 1),
                 b['delta_best_PACo_minus_RhoDesign']['wilcoxon_p_two_sided'])
                for b in t93['bins_table']}
    ok_bins = all(any(g[0] == e[0] and g[1] == e[1] and abs(g[2] - e[2]) < 0.05
                      and (e[3] is None or abs(g[3] - e[3]) / e[3] < 0.05)
                      for g in got_bins) for e in exp_bins)
    check('t5_anchor_bins', ok_bins, f'Morgan bins {sorted(got_bins)}')

    cols = ['fingerprint', 'row_type', 'bin', 'n', 'paco_best_pct', 'paco_t01_pct',
            'rhodesign_best_pct', 'rhodesign_t01_pct', 'delta_best_mean_pp',
            'delta_best_median_pp', 'wilcoxon_p_two_sided', 'delta_t01_mean_pp',
            'spearman_rho', 'spearman_p']
    for r in rows:  # convert fractions to percent/pp for the CSV
        for k in ['paco_best_pct', 'paco_t01_pct', 'rhodesign_best_pct', 'rhodesign_t01_pct']:
            if r[k] is not None:
                r[k] = round(100 * r[k], 4)
        for k in ['delta_best_mean_pp', 'delta_best_median_pp', 'delta_t01_mean_pp']:
            if r[k] is not None:
                r[k] = round(100 * r[k], 4)
    md = ['**Supplementary Table | Chemical-novelty stratification (Tanimoto to the 247 '
          'training ligands), three fingerprints, 87 zero-shot chains with assignable '
          'ligand fingerprints.**', '',
          '| Fingerprint | Bin | n | PACo best (%) | RhoDesign best (%) | Δ best mean (pp) | '
          'P (two-sided) | Δ t01 mean (pp) |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['row_type'] != 'bin':
            continue
        md.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['fingerprint'], r['bin'], r['n'],
            pct1(r['paco_best_pct'] / 100 if r['paco_best_pct'] is not None else None),
            pct1(r['rhodesign_best_pct'] / 100 if r['rhodesign_best_pct'] is not None else None),
            pp1(r['delta_best_mean_pp'] / 100 if r['delta_best_mean_pp'] is not None else None),
            p2s(r['wilcoxon_p_two_sided']),
            pp1(r['delta_t01_mean_pp'] / 100 if r['delta_t01_mean_pp'] is not None else None)))
    md += ['', '| Fingerprint | Contrast | n | Spearman ρ | P |', '|---|---|---|---|---|']
    for r in rows:
        if not r['row_type'].startswith('spearman'):
            continue
        md.append('| {} | similarity vs {} | {} | {} | {} |'.format(
            r['fingerprint'], 'Δ best' if r['row_type'] == 'spearman_delta_best' else 'Δ t01',
            r['n'], rho3(r['spearman_rho']), p2s(r['spearman_p'])))
    md += ['',
           'Δ = PACo-Aptamer − RhoDesign (per-chain paired, percentage points). Bin rows: '
           'chain-macro recovery within each max-Tanimoto bin; P = paired two-sided Wilcoxon '
           '(n/a where n < 10). Spearman rows: overall correlation between max Tanimoto and '
           'per-chain Δ (both conventions). Morgan = original 93_tanimoto_zs.json '
           '(identical to the 106 morgan_recheck, verified); MACCS / RDKit-topological from '
           '106_tanimoto_robust.json. MACCS similarities saturate (85/87 chains in the top '
           'bin) — bins with n = 0 or n < 10 carry n/a. 6 of the 93 zero-shot chains lack a '
           'fingerprintable ligand and are excluded (5 without ligand annotation; 6PQ7_C '
           'ligand OXV has no SMILES).']
    na_reasons = ['bins with n < 10 have no Wilcoxon P (source convention) -> n/a',
                  'MACCS [0,0.2) and [0.2,0.4) bins are empty (n = 0) -> all means n/a',
                  '6/93 zero-shot chains without fingerprintable ligand excluded from this '
                  'table (5 no-ligand + 6PQ7_C/OXV-no-SMILES)']
    return 'st_tanimoto', cols, rows, '\n'.join(md) + '\n', {
        'morgan_bins_and_spearman': 'data/survey/93_tanimoto_zs.json (bins_table, spearman)',
        'maccs_and_topo': 'data/survey/106_tanimoto_robust.json (bins_table_maccs, '
                          'bins_table_rdkit_topo, spearman)'}, na_reasons


# ================================================================ table 6
# st_predbb — predicted-backbone benchmark (193 chains) + Spearman + zs-32 slice
def build_predbb():
    pred_files = {'PACoAptamer': '94_predbb_paco.json',
                  'RhoDesign': '101_rhodesign_predbb.json',
                  'gRNAde': '103_grnade_predbb.json',
                  'NA-MPNN': '102_nampnn_predbb.json',
                  'AlignIF': '100_alignif_predbb.json'}
    pred = {m: load(f)['per_chain'] for m, f in pred_files.items()}
    main = load('63_main_table_gfix.json')['per_chain']  # crystal, gRNAde C/G-decode-fixed
    sp = load('108_backbone_quality_spearman.json')['methods']
    zs105 = load('105_predbb_zs.json')
    sp_key = {'PACoAptamer': 'PACo-Aptamer'}.get

    rows = []
    for m in METHODS:
        chains = sorted(pred[m].keys())
        pb = [pred[m][c]['recovery_best'] for c in chains]
        pt = [pred[m][c]['recovery_t01_mean'] for c in chains]
        cb = [main[m][c]['recovery_best'] for c in chains]
        ct = [main[m][c]['recovery_t01_mean'] for c in chains]
        p_b, _ = wilcoxon_2s(pb, cb, min_nz=10)
        p_t, _ = wilcoxon_2s(pt, ct, min_nz=10)
        s108 = sp[sp_key(m, m)]
        rows.append({'section': 'benchmark_193', 'method': METHOD_LABEL[m], 'n_chains': 193,
                     'predbb_best_pct': round(100 * float(np.mean(pb)), 4),
                     'predbb_t01_pct': round(100 * float(np.mean(pt)), 4),
                     'crystal_best_pct': round(100 * float(np.mean(cb)), 4),
                     'crystal_t01_pct': round(100 * float(np.mean(ct)), 4),
                     'delta_best_pp': round(100 * float(np.mean([a - b for a, b in zip(pb, cb)])), 4),
                     'wilcoxon_p_best_two_sided': p_b,
                     'delta_t01_pp': round(100 * float(np.mean([a - b for a, b in zip(pt, ct)])), 4),
                     'wilcoxon_p_t01_two_sided': p_t,
                     'rho_sctm_vs_predbb_best': s108['sctm_vs_predbb_best']['rho'],
                     'p_sctm_vs_predbb_best': s108['sctm_vs_predbb_best']['p'],
                     'rho_sctm_vs_predbb_t01': s108['sctm_vs_predbb_t01']['rho'],
                     'p_sctm_vs_predbb_t01': s108['sctm_vs_predbb_t01']['p'],
                     'rho_sctm_vs_delta_best': s108['sctm_vs_delta_best']['rho'],
                     'p_sctm_vs_delta_best': s108['sctm_vs_delta_best']['p'],
                     'rho_sctm_vs_delta_t01': s108['sctm_vs_delta_t01']['rho'],
                     'p_sctm_vs_delta_t01': s108['sctm_vs_delta_t01']['p'],
                     'note': ''})
    # stored-P cross-checks (94 for PACo, 101_predbb_vs_crystal for RhoDesign): the
    # full-precision recomputation must agree with the stored P to 1e-3 relative.
    # (The 4-decimal-rounded per_chain inside 101_predbb_vs_crystal.json itself does
    # NOT reproduce its stored P — rounding introduces ties; the stored values were
    # computed on full-precision inputs.)
    t94 = load('94_predbb_paco.json')['delta_vs_crystal_maintab']
    r101 = load('101_predbb_vs_crystal.json')
    row_paco = rows[0]
    row_rho = rows[1]
    check('t6_stored_p_paco',
          abs(row_paco['wilcoxon_p_best_two_sided'] - t94['best']['wilcoxon_p'])
          / t94['best']['wilcoxon_p'] < 1e-3
          and abs(row_paco['wilcoxon_p_t01_two_sided'] - t94['t01']['wilcoxon_p'])
          / t94['t01']['wilcoxon_p'] < 1e-3,
          f"PACo recomputed P {row_paco['wilcoxon_p_best_two_sided']:.3g}/"
          f"{row_paco['wilcoxon_p_t01_two_sided']:.3g} vs stored "
          f"{t94['best']['wilcoxon_p']:.3g}/{t94['t01']['wilcoxon_p']:.3g}")
    check('t6_stored_p_rho',
          abs(row_rho['wilcoxon_p_best_two_sided'] - r101['wilcoxon_p_best'])
          / r101['wilcoxon_p_best'] < 1e-3
          and abs(row_rho['wilcoxon_p_t01_two_sided'] - r101['wilcoxon_p_t01'])
          / r101['wilcoxon_p_t01'] < 1e-3,
          f"RhoDesign recomputed P {row_rho['wilcoxon_p_best_two_sided']:.3g}/"
          f"{row_rho['wilcoxon_p_t01_two_sided']:.3g} vs stored "
          f"{r101['wilcoxon_p_best']:.3g}/{r101['wilcoxon_p_t01']:.3g}")
    # zs-32 cross slice (predbb values from 105; crystal recomputed from 63 gfix —
    # the crystal block embedded in 105_predbb_zs.json is the pre-gfix decode for
    # gRNAde (49.5% on the 193 chains) and must not be used)
    zs_chains = zs105['zs_subset']['chains']
    check('t6_zs32_n', len(zs_chains) == 32, f'zs slice n={len(zs_chains)}')
    for m in METHODS:
        zm = zs105['zs_subset']['methods'][m]
        cb = [main[m][c]['recovery_best'] for c in zs_chains]
        ct = [main[m][c]['recovery_t01_mean'] for c in zs_chains]
        emb = zs105['zs_subset_crystal_decomp']['crystal_summary']['methods'][m]
        if m == 'gRNAde':
            check('t6_grnade_crystal_gfix',
                  abs(float(np.mean(cb)) - 0.5782) < 1e-3
                  and abs(emb['best']['mean'] - 0.5205) < 1e-3
                  and abs(float(np.mean(cb)) - emb['best']['mean']) > 0.05,
                  f'gRNAde zs32 crystal: gfix {np.mean(cb):.4f} used; embedded old-decode '
                  f"{emb['best']['mean']:.4f} rejected (full-193 old decode = 49.5%, "
                  'gfix = 52.2%)')
        else:
            check(f't6_zs32_crystal_embed_eq_{m}',
                  abs(emb['best']['mean'] - float(np.mean(cb))) < 1e-9,
                  f'{m}: embedded crystal == 63-gfix recomputation')
        rows.append({'section': 'zs32_predbb', 'method': METHOD_LABEL[m], 'n_chains': 32,
                     'predbb_best_pct': round(100 * zm['best']['mean'], 4),
                     'predbb_t01_pct': round(100 * zm['t01']['mean'], 4),
                     'crystal_best_pct': round(100 * float(np.mean(cb)), 4),
                     'crystal_t01_pct': round(100 * float(np.mean(ct)), 4),
                     'delta_best_pp': None, 'wilcoxon_p_best_two_sided': None,
                     'delta_t01_pp': None, 'wilcoxon_p_t01_two_sided': None,
                     'rho_sctm_vs_predbb_best': None, 'p_sctm_vs_predbb_best': None,
                     'rho_sctm_vs_predbb_t01': None, 'p_sctm_vs_predbb_t01': None,
                     'rho_sctm_vs_delta_best': None, 'p_sctm_vs_delta_best': None,
                     'rho_sctm_vs_delta_t01': None, 'p_sctm_vs_delta_t01': None,
                     'note': f"predbb CI95 best [{zm['best']['ci95'][0] * 100:.1f},"
                             f"{zm['best']['ci95'][1] * 100:.1f}]"
                             + ('; crystal = 63_main_table_gfix recomputation '
                                '(embedded 105 block is pre-gfix)' if m == 'gRNAde' else '')})
    dz = zs105['zs_subset']['delta_PACo_minus_RhoDesign']
    dc = zs105['zs_subset_crystal_decomp']['crystal_summary']['delta_PACo_minus_RhoDesign']
    # crystal-side delta recompute (PACo-RhoDesign; unaffected by the gRNAde decode fix)
    dtb = [main['PACoAptamer'][c]['recovery_best'] - main['RhoDesign'][c]['recovery_best']
           for c in zs_chains]
    dtt = [main['PACoAptamer'][c]['recovery_t01_mean'] - main['RhoDesign'][c]['recovery_t01_mean']
           for c in zs_chains]
    pb_c, _ = wilcoxon_2s(
        [main['PACoAptamer'][c]['recovery_best'] for c in zs_chains],
        [main['RhoDesign'][c]['recovery_best'] for c in zs_chains])
    pt_c, _ = wilcoxon_2s(
        [main['PACoAptamer'][c]['recovery_t01_mean'] for c in zs_chains],
        [main['RhoDesign'][c]['recovery_t01_mean'] for c in zs_chains])
    check('t6_zs32_crystal_delta_eq',
          abs(float(np.mean(dtb)) - dc['best']['mean']) < 1e-9
          and abs(pb_c - dc['best']['wilcoxon_p_two_sided']) < 1e-12,
          f"crystal delta best recomputed {np.mean(dtb):.6f} == embedded {dc['best']['mean']:.6f}")
    for scope, dd, pb_, pt_ in [('predbb', dz, dz['best']['wilcoxon_p_two_sided'],
                                 dz['t01']['wilcoxon_p_two_sided']),
                                ('crystal', dc, pb_c, pt_c)]:
        rows.append({'section': f'zs32_delta_{scope}', 'method': 'PACo-Aptamer − RhoDesign',
                     'n_chains': 32,
                     'predbb_best_pct': None, 'predbb_t01_pct': None,
                     'crystal_best_pct': None, 'crystal_t01_pct': None,
                     'delta_best_pp': round(100 * dd['best']['mean'], 4),
                     'wilcoxon_p_best_two_sided': pb_,
                     'delta_t01_pp': round(100 * dd['t01']['mean'], 4),
                     'wilcoxon_p_t01_two_sided': pt_,
                     'rho_sctm_vs_predbb_best': None, 'p_sctm_vs_predbb_best': None,
                     'rho_sctm_vs_predbb_t01': None, 'p_sctm_vs_predbb_t01': None,
                     'rho_sctm_vs_delta_best': None, 'p_sctm_vs_delta_best': None,
                     'rho_sctm_vs_delta_t01': None, 'p_sctm_vs_delta_t01': None,
                     'note': f"best CI95 [{dd['best']['ci95'][0] * 100:+.1f},"
                             f"{dd['best']['ci95'][1] * 100:+.1f}] pp"})
    exp = {'PACo-Aptamer': (68.9, 51.8, -5.3), 'RhoDesign': (61.4, 48.0, -6.9),
           'gRNAde': (50.7, 35.0, -1.6), 'NA-MPNN': (35.3, 29.6, -19.1),
           'AlignIF': (41.9, 36.8, -15.7)}
    for r in rows[:5]:
        b, t, d = exp[r['method']]
        check(f"t6_anchor_{r['method']}",
              abs(r['predbb_best_pct'] - b) < 0.06 and abs(r['predbb_t01_pct'] - t) < 0.06
              and abs(r['delta_best_pp'] - d) < 0.06,
              f"{r['method']} {r['predbb_best_pct']}/{r['predbb_t01_pct']}/"
              f"{r['delta_best_pp']} vs {b}/{t}/{d}")
    sp_paco = sp['PACo-Aptamer']
    sp_ali = sp['AlignIF']
    check('t6_spearman_anchor',
          abs(sp_ali['sctm_vs_predbb_t01']['rho'] - 0.671) < 2e-3
          and abs(sp_paco['sctm_vs_predbb_t01']['rho'] - 0.216) < 2e-3,
          f"AlignIF {sp_ali['sctm_vs_predbb_t01']['rho']:.3f} "
          f"PACo {sp_paco['sctm_vs_predbb_t01']['rho']:.3f}")
    check('t6_zs32_delta_anchor',
          abs(dz['best']['mean'] - 0.067) < 1e-3
          and abs(dz['best']['wilcoxon_p_two_sided'] - 4.8e-6) / 4.8e-6 < 0.02
          and abs(dz['t01']['mean'] - (-0.0007)) < 1e-3
          and abs(dz['t01']['wilcoxon_p_two_sided'] - 0.4211) < 1e-3,
          f"zs32 Δbest {dz['best']['mean']:+.4f} P {dz['best']['wilcoxon_p_two_sided']:.3g}; "
          f"Δt01 {dz['t01']['mean']:+.5f} P {dz['t01']['wilcoxon_p_two_sided']:.3f}")

    cols = ['section', 'method', 'n_chains', 'predbb_best_pct', 'predbb_t01_pct',
            'crystal_best_pct', 'crystal_t01_pct', 'delta_best_pp',
            'wilcoxon_p_best_two_sided', 'delta_t01_pp', 'wilcoxon_p_t01_two_sided',
            'rho_sctm_vs_predbb_best', 'p_sctm_vs_predbb_best',
            'rho_sctm_vs_predbb_t01', 'p_sctm_vs_predbb_t01',
            'rho_sctm_vs_delta_best', 'p_sctm_vs_delta_best',
            'rho_sctm_vs_delta_t01', 'p_sctm_vs_delta_t01', 'note']
    md = ['**Supplementary Table | Predicted-backbone benchmark (193 cluster-based test '
          'chains, RhoFold+ backbones).**', '',
          '| Method | predbb best (%) | predbb t01 (%) | crystal best (%) | crystal t01 (%) | '
          'Δ best (pp) | P best | Δ t01 (pp) | P t01 |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows[:5]:
        md.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['method'], f"{r['predbb_best_pct']:.1f}", f"{r['predbb_t01_pct']:.1f}",
            f"{r['crystal_best_pct']:.1f}", f"{r['crystal_t01_pct']:.1f}",
            pp1(r['delta_best_pp'] / 100), p2s(r['wilcoxon_p_best_two_sided']),
            pp1(r['delta_t01_pp'] / 100), p2s(r['wilcoxon_p_t01_two_sided'])))
    md += ['', '| Method | ρ(scTM, predbb best) | P | ρ(scTM, predbb t01) | P | '
           'ρ(scTM, Δ best) | P | ρ(scTM, Δ t01) | P |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows[:5]:
        md.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['method'], rho3(r['rho_sctm_vs_predbb_best']), p2s(r['p_sctm_vs_predbb_best']),
            rho3(r['rho_sctm_vs_predbb_t01']), p2s(r['p_sctm_vs_predbb_t01']),
            rho3(r['rho_sctm_vs_delta_best']), p2s(r['p_sctm_vs_delta_best']),
            rho3(r['rho_sctm_vs_delta_t01']), p2s(r['p_sctm_vs_delta_t01'])))
    md += ['', '| zs-32 slice | n | predbb best (%) | predbb t01 (%) | crystal best (%) | '
           'crystal t01 (%) |', '|---|---|---|---|---|---|']
    for r in rows[5:10]:
        md.append('| {} | {} | {} | {} | {} | {} |'.format(
            r['method'], r['n_chains'], f"{r['predbb_best_pct']:.1f}",
            f"{r['predbb_t01_pct']:.1f}", f"{r['crystal_best_pct']:.1f}",
            f"{r['crystal_t01_pct']:.1f}"))
    for r in rows[10:]:
        md.append('| {} ({}) | {} | Δ best {} pp, P {} | Δ t01 {} pp, P {} |  |  |'.format(
            r['method'], r['section'].replace('zs32_delta_', ''), r['n_chains'],
            pp1(r['delta_best_pp'] / 100), p2s(r['wilcoxon_p_best_two_sided']),
            pp1(r['delta_t01_pp'] / 100), p2s(r['wilcoxon_p_t01_two_sided'])))
    md += ['',
           'Predbb values: 94_predbb_paco.json / 101_rhodesign_predbb.json / '
           '103_grnade_predbb.json / 102_nampnn_predbb.json / 100_alignif_predbb.json '
           '(193 chains each). Crystal values: same-chain means recomputed from '
           '63_main_table_gfix.json per_chain (gRNAde C/G-decode-fixed; the crystal block '
           'embedded in 105_predbb_zs.json is the pre-fix decode — 49.5% for gRNAde on the '
           '193 chains — and is NOT used). Δ = predbb − crystal (paired); P = paired '
           'two-sided Wilcoxon recomputed per chain (matches the stored 94/101 values for '
           'PACo-Aptamer/RhoDesign exactly). Spearman block: scTM backbone quality vs '
           'predbb best / predbb t01 / Δ best / Δ t01, copied verbatim from '
           '108_backbone_quality_spearman.json. zs-32 slice: the 32 zero-shot chains inside '
           'the 193; predbb from 105_predbb_zs.json zs_subset, crystal recomputed from '
           '63_main_table_gfix.json; Δ rows = PACo-Aptamer − RhoDesign with two-sided '
           'Wilcoxon P.']
    na_reasons = ['gRNAde crystal values recomputed from 63_main_table_gfix.json because the '
                  '105_predbb_zs.json embedded crystal block carries the pre-gfix decode '
                  '(49.5% on 193 chains / 52.05% on the zs-32 slice) ⚠️',
                  'gRNAde/NA-MPNN/AlignIF predbb-vs-crystal Wilcoxon P not stored in any '
                  'source JSON; recomputed per-chain (same convention as the stored PACo/'
                  'RhoDesign values) ⚠️']
    return 'st_predbb', cols, rows, '\n'.join(md) + '\n', {
        'predbb_per_chain': [f'data/survey/{f}' for f in pred_files.values()],
        'crystal': 'data/survey/63_main_table_gfix.json per_chain (same-chain recomputation; '
                   'authoritative for gRNAde)',
        'spearman': 'data/survey/108_backbone_quality_spearman.json (verbatim)',
        'zs32': 'data/survey/105_predbb_zs.json (zs_subset; crystal side recomputed from '
                '63_main_table_gfix.json)',
        'stored_p_crosscheck': 'data/survey/94_predbb_paco.json delta_vs_crystal_maintab; '
                               'data/survey/101_predbb_vs_crystal.json'}, na_reasons


# ================================================================ table 7
# st_efficiency — per-probe-chain detail + summary rows
def build_efficiency():
    eff = {m: load(f'99_efficiency_{m}.json') for m in
           ['PACoAptamer', 'AlignIF', 'gRNAde', 'NA-MPNN', 'RhoDesign']}
    rows = []

    def add_row(rt, m, chain, length, nsamp, wall, mem_mb, note=''):
        rows.append({'row_type': rt, 'method': METHOD_LABEL[m], 'chain': chain,
                     'len': length, 'n_samples': nsamp,
                     'wall_s': round(wall, 4) if wall is not None else None,
                     'ms_per_sample': round(1000 * wall / nsamp, 4)
                     if wall is not None and nsamp else None,
                     'samples_per_s': round(nsamp / wall, 4) if wall else None,
                     'peak_mem_mb': round(mem_mb, 2) if mem_mb is not None else None,
                     'note': note})

    def add_summary(m, nsamp_total, walls, mems, note, ms=None, sps=None):
        # default convention: aggregate over the full pool (as in ED Table 4 for
        # gRNAde/NA-MPNN/RhoDesign); PACo/AlignIF pass the stored summary values
        add_row('summary', m, 'ALL (mean | min–max)', None, nsamp_total,
                float(np.mean(walls)), float(np.mean(mems)),
                note + f"; wall min–max {min(walls):.2f}–{max(walls):.2f} s; "
                       f"peak mem min–max {min(mems):.0f}–{max(mems):.0f} MB")
        rows[-1]['wall_min_s'] = round(min(walls), 4)
        rows[-1]['wall_max_s'] = round(max(walls), 4)
        rows[-1]['peak_mem_min_mb'] = round(min(mems), 2)
        rows[-1]['peak_mem_max_mb'] = round(max(mems), 2)
        if ms is not None:
            rows[-1]['ms_per_sample'] = round(ms, 4)
        if sps is not None:
            rows[-1]['samples_per_s'] = round(sps, 4)

    # PACoAptamer (released = ens2); summary ms/throughput from the stored summary
    # block (per-chain mean convention, as in ED Table 4)
    pc = eff['PACoAptamer']['per_chain']
    for stem, v in pc.items():
        e = v['ens2']
        add_row('probe', 'PACoAptamer', stem, v['len'], v['n_samples'], e['wall_s'],
                e['peak_mem_mb'], 'ens2 (2 models); crystal backbone')
    s = eff['PACoAptamer']['summary']
    walls = [v['ens2']['wall_s'] for v in pc.values()]
    mems = [v['ens2']['peak_mem_mb'] for v in pc.values()]
    add_summary('PACoAptamer', 300, walls, mems,
                f"ens2/single overhead {s['ens_overhead_ratio_mean']:.2f}x; "
                's1 feature cache; warmup = first chain 10 samples',
                ms=1000 * s['ens2_wall_s_mean'] / 300,
                sps=s['ens2_samples_per_s_mean'])
    # AlignIF (stored summary values)
    pc = eff['AlignIF']['per_chain']
    sa = eff['AlignIF']['summary']
    for stem, v in pc.items():
        add_row('probe', 'AlignIF', stem, v['len'], v['n_samples'], v['wall_s'],
                v['peak_mem_mb_approx'], 't = 0.1 x 20, no-MStA; parent-process GPU delta')
    add_summary('AlignIF', 20, [v['wall_s'] for v in pc.values()],
                [v['peak_mem_mb_approx'] for v in pc.values()], 'peak mem approx',
                ms=1000 * sa['s_per_sample_mean'], sps=sa['samples_per_s_mean'])
    # gRNAde
    recs = [r for r in eff['gRNAde']['records'] if r.get('ok')]
    for r in recs:
        add_row('probe', 'gRNAde', r['stem'], r['len'], 3 * r['n_samples_per_temp'],
                r['wall_s'], r['gpu_mem_peak_mib'], 'predicted backbone (RhoFold+)')
    add_summary('gRNAde', 300, [r['wall_s'] for r in recs],
                [r['gpu_mem_peak_mib'] for r in recs], 'featurize + 3-temperature sampling')
    # NA-MPNN
    recs = [r for r in eff['NA-MPNN']['records'] if str(r.get('status', '')).startswith('ok')]
    for r in recs:
        add_row('probe', 'NA-MPNN', r['stem'], r['len'], 10, r['wall_s'],
                r['gpu_mem_peak_mib_polling'], 't = 0.1 x 10; nvidia-smi 0.2 s polling')
    add_summary('NA-MPNN', 10, [r['wall_s'] for r in recs],
                [r['gpu_mem_peak_mib_polling'] for r in recs], 'mem = polling delta (approx)')
    # RhoDesign
    pr = eff['RhoDesign']['probes']
    for stem, v in pr.items():
        add_row('probe', 'RhoDesign', stem, v['len'], 3 * v['n_per_temp'], v['wall_time_s'],
                v['peak_mem_MiB'], 'RTX 4090, in-batch measurement')
    r_wall = [v['wall_time_s'] for v in pr.values()]
    r_nsamp = [3 * v['n_per_temp'] for v in pr.values()]
    add_summary('RhoDesign', 300, r_wall,
                [v['peak_mem_MiB'] for v in pr.values()],
                '150 samples for the 295-nt chain (n_per_temp=50)',
                ms=1000 * sum(r_wall) / sum(r_nsamp), sps=sum(r_nsamp) / sum(r_wall))

    sm = {r['method']: r for r in rows if r['row_type'] == 'summary'}
    check('t7_paco', abs(sm['PACo-Aptamer']['wall_s'] - 2.77) < 0.01
          and abs(sm['PACo-Aptamer']['ms_per_sample'] - 9.2) < 0.1
          and abs(sm['PACo-Aptamer']['samples_per_s'] - 149.3) < 0.5
          and sm['PACo-Aptamer']['peak_mem_min_mb'] / 1024 >= 0.55
          and abs(sm['PACo-Aptamer']['peak_mem_max_mb'] / 1024 - 2.0) < 0.15,
          f"PACo {sm['PACo-Aptamer']['wall_s']:.3f}s "
          f"{sm['PACo-Aptamer']['ms_per_sample']:.2f}ms "
          f"{sm['PACo-Aptamer']['samples_per_s']:.1f}/s "
          f"mem {sm['PACo-Aptamer']['peak_mem_min_mb']:.0f}–"
          f"{sm['PACo-Aptamer']['peak_mem_max_mb']:.0f} MB")
    check('t7_others', abs(sm['AlignIF']['ms_per_sample'] - 474.6) < 1
          and abs(sm['gRNAde']['ms_per_sample'] - 21.3) < 0.5
          and abs(sm['NA-MPNN']['ms_per_sample'] - 355.9) < 1
          and abs(sm['RhoDesign']['ms_per_sample'] - 154.7) < 1,
          f"AlignIF {sm['AlignIF']['ms_per_sample']:.1f} gRNAde {sm['gRNAde']['ms_per_sample']:.1f} "
          f"NA-MPNN {sm['NA-MPNN']['ms_per_sample']:.1f} RhoDesign {sm['RhoDesign']['ms_per_sample']:.1f} ms")
    check('t7_rows', len(rows) == 35, f'{len(rows)} rows (30 probes + 5 summaries)')

    cols = ['row_type', 'method', 'chain', 'len', 'n_samples', 'wall_s', 'wall_min_s',
            'wall_max_s', 'ms_per_sample', 'samples_per_s', 'peak_mem_mb',
            'peak_mem_min_mb', 'peak_mem_max_mb', 'note']
    md = ['**Supplementary Table | Inference efficiency: per-probe-chain detail and '
          'summaries (released settings).**', '',
          '| Method | Chain | len | n samples | wall (s) | ms/sample | samples/s | '
          'peak mem (MB) |', '|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['row_type'] != 'probe':
            continue
        md.append('| {} | {} | {} | {} | {:.3f} | {:.1f} | {:.1f} | {:.0f} |'.format(
            r['method'], r['chain'], r['len'], r['n_samples'], r['wall_s'],
            r['ms_per_sample'], r['samples_per_s'], r['peak_mem_mb']))
    md += ['', '| Method | wall per chain (s), mean (min–max) | ms/sample | samples/s | '
           'peak mem |', '|---|---|---|---|---|']
    for r in rows:
        if r['row_type'] != 'summary':
            continue
        md.append('| {} | {:.2f} ({:.2f}–{:.2f}) | {:.1f} | {:.1f} | {:.0f}–{:.0f} MB |'.format(
            r['method'], r['wall_s'], r['wall_min_s'], r['wall_max_s'], r['ms_per_sample'],
            r['samples_per_s'], r['peak_mem_min_mb'], r['peak_mem_max_mb']))
    md += ['',
           'Probe sets differ per method (six chains each; PACo-Aptamer/AlignIF share a '
           'crystal-backbone set, gRNAde/NA-MPNN share the predicted-backbone set, RhoDesign '
           'was profiled inside its predicted-backbone batch run on an RTX 4090) — '
           'cross-method comparison is indicative, not strictly paired. PACo-Aptamer rows '
           'use the released 2-model ensemble (ens2); the single model is 1.96× cheaper. '
           'Memory conventions differ (torch.cuda.max_memory_allocated for gRNAde, '
           'nvidia-smi polling delta for NA-MPNN, parent-process GPU delta for AlignIF) as '
           'recorded in the source JSONs. Sources: 99_efficiency_{method}.json.']
    return 'st_efficiency', cols, rows, '\n'.join(md) + '\n', {
        m: f'data/survey/99_efficiency_{m}.json'
        for m in ['PACoAptamer', 'AlignIF', 'gRNAde', 'NA-MPNN', 'RhoDesign']}, []


# ================================================================ table 8
# st_contamination — data-quality screening / removal record
def build_contamination():
    d52 = load('52_nampnn_preproc_dropped.json')
    d53 = load('53_alignif_data_dropped.json')
    d50bi = load('50_backbone_incomplete.json')
    d50rf = load('50_regen_fails.json')
    d89 = load('89_rnet_leakage_audit.json')
    d79 = load('79_aptadb_retro.json')
    d88 = load('88_utexas_audit.json')
    d87 = load('87_protein_demo.json')
    r2c = load('resname_to_char.json')
    n52 = len(d52)
    n52_nvr = sum(1 for v in d52.values() if v == 'no_valid_rna_residue')
    n53 = len(d53)
    n_clstr88 = sum(1 for line in open(os.path.join(SURVEY, '88_leak_out.fa.clstr'))
                    if line.startswith('>Cluster'))
    n_clstr79 = sum(1 for line in open(os.path.join(SURVEY, '79_leak_out.fa.clstr'))
                    if line.startswith('>Cluster'))
    check('t8_clstr', n_clstr88 == 936, f'88 clstr clusters {n_clstr88}')
    check('t8_79_88_clean', d79['n_scored'] == 254 and d79['n_clean'] == 214
          and d88['n_clean'] == 228,
          f"AptaDB {d79['n_scored']}->{d79['n_clean']}, UTexas ->{d88['n_clean']}")
    check('t8_rnet', d89['verdict'] == 'FAIL' and d89['n_exact'] == 0
          and d89['n_high'] == 11 and d89['n_embedded'] == 227
          and d89['n_fragment_suspect'] == 284 and d89['n_natives'] == 422,
          f"RNet audit exact={d89['n_exact']} high={d89['n_high']} "
          f"embedded={d89['n_embedded']} fragment={d89['n_fragment_suspect']}")

    rows = [
        {'category': 'Concatenated multi-copy crystal chains',
         'scope': 'S2 training pool + evaluation natives',
         'n_affected': '40 chains in s2_samples/seqs_all (e.g. 1EI2_A 425 nt = 17x25, '
                       '2KGP_A 250 nt = 10x25); ~37 S1 training chains contaminated '
                       '(<=1.2%, documented, NOT retrained); 3 test-chain natives',
         'handling': 'all 104 s2-only chains rebuilt as single-copy all-atom PDBs '
                     '(step 50, chain_pdb_v2/); canonical_v2 rebuilt identically (step 51); '
                     'evaluation switched to single-copy natives (63_main_table)',
         'status': 'fixed (evaluation) / documented-not-retrained (training)',
         'structured_source': 'data/survey/50_backbone_incomplete.json (104 chains), '
                              'data/survey/50_regen_fails.json (40 single_copy_fallback, 0 fails)',
         'doc_source': 'analysis/10 §3.1; analysis/05',
         'note': ''},
        {'category': 'Dedup bug leaving P-atom-only files (step 43c)',
         'scope': 'rhodesign_btest_fix4 repair files',
         'n_affected': '11 files',
         'handling': 'superseded by the correct step-50 rebuild',
         'status': 'fixed (superseded)',
         'structured_source': NA,
         'doc_source': 'analysis/10 §3.2',
         'note': '⚠️ no structured per-file list survives; text record only'},
        {'category': 'Protein-trace garbage chains',
         'scope': 'training set only (e.g. 1C51/1ILX/4WWE/4WWT/6O7H)',
         'n_affected': '29 chains (documents); structured drops: '
                       f'{n52} NA-MPNN preprocessing drops of which {n52_nvr} '
                       f'no_valid_rna_residue; {n53} AlignIF drops (all backbone_incomplete '
                       'with 0 valid atoms)',
         'handling': 'removed from NA-MPNN / AlignIF training sets and recorded',
         'status': 'removed',
         'structured_source': 'data/survey/52_nampnn_preproc_dropped.json; '
                              'data/survey/53_alignif_data_dropped.json',
         'doc_source': 'analysis/10 §3.4',
         'note': '⚠️ the documented count is 29 protein-trace chains; the structured drop '
                 'lists are broader (they also include other defect classes) — the 29-chain '
                 'list itself is not separately materialized'},
        {'category': 'Residue/atom-name normalization (incl. modified nucleotides)',
         'scope': 'all chains',
         'n_affected': f"resname_to_char.json maps {len(r2c)} CCD codes "
                       "(modified nucleotides -> parent base via MOD_MAP, unmapped -> 'N' "
                       'skipped in evaluation); resname ' + "'N'" + ' (unspecified RNA link) '
                       "-> letter 'N'; 5' variants (+U etc.) -> parent base; legacy atom "
                       'name O5* -> O5\' on 13 chains (4 test / 9 train)',
         'handling': 'mapping table + parser patches',
         'status': 'fixed',
         'structured_source': 'data/survey/resname_to_char.json',
         'doc_source': 'analysis/10 §3.3',
         'note': '⚠️ the 13 O5* chains and the MOD_MAP counts exist only in the text record'},
        {'category': "NA-MPNN chains with missing C1' coordinates",
         'scope': 'test set',
         'n_affected': '5 chains (5NFV_B, 7E9I_A, 8U5J_A, 8VPV_A, 9GUT_X)',
         'handling': 'not removable (coordinates absent in the source PDB); evaluated with '
                     'the valid-position convention',
         'status': 'documented (kept)',
         'structured_source': NA,
         'doc_source': 'analysis/10 §3.8',
         'note': '⚠️ text record only'},
        {'category': 'RibonanzaNet (RNet) screening-library leakage audit',
         'scope': 'external RDAT library vs benchmark chains',
         'n_affected': f"{d89['n_natives']} native constructs scanned "
                       f"({d89['n_sequences_scanned']:,} sequences): {d89['n_exact']} exact, "
                       f"{d89['n_high']} high-similarity (>=0.8), {d89['n_embedded']} embedded, "
                       f"{d89['n_fragment_suspect']} fragment suspects; verdict "
                       f"{d89['verdict']}",
         'handling': 'leakage found -> the RNet-based triage experiment was vetoed and '
                     'never run (audit itself reported as a methodological finding)',
         'status': 'rejected (experiment not run)',
         'structured_source': 'data/survey/89_rnet_leakage_audit.json',
         'doc_source': 'analysis/20 §四',
         'note': ''},
        {'category': 'Retrospective-library leakage screen (AptaDB)',
         'scope': 'AptaDB retrospective audit library',
         'n_affected': f"{d79['n_scored']} scored -> {d79['n_clean']} clean "
                       f'({n_clstr79} CD-HIT-est 0.8 clusters vs the training library)',
         'handling': 'exact + cluster-level leakage filtering before scoring',
         'status': 'filtered',
         'structured_source': 'data/survey/79_aptadb_retro.json; '
                              'data/survey/79_leak_out.fa.clstr',
         'doc_source': 'analysis/16; analysis/05',
         'note': ''},
        {'category': 'Retrospective-library leakage screen (UTexas)',
         'scope': 'UTexas (Zenodo 8387047) retrospective audit library',
         'n_affected': f"293 new ssRNA -> {d88['n_clean']} clean ({n_clstr88} CD-HIT-est 0.8 "
                       'clusters vs the training library)',
         'handling': 'same filtering convention as AptaDB (step-79 protocol reused)',
         'status': 'filtered',
         'structured_source': 'data/survey/88_utexas_audit.json; '
                              'data/survey/88_leak_out.fa.clstr',
         'doc_source': 'analysis/05; analysis/17 §7.2',
         'note': ''},
        {'category': 'Protein-context demo chains (screening record)',
         'scope': '30 protein-complex chains outside the ligand benchmark',
         'n_affected': f"{d87['summary']['n_chains']} chains; protein-as-context effect "
                       f"~0 (interface delta {d87['summary']['delta_iface']['mean']:+.4f}, "
                       f"P = {d87['summary']['delta_iface']['wilcoxon_p_two']:.2f}; overall "
                       f"delta {d87['summary']['delta_overall']['mean']:+.4f}, "
                       f"P = {d87['summary']['delta_overall']['wilcoxon_p_two']:.2f})",
         'handling': 'screening/demo only; no benchmark entries removed',
         'status': 'screened',
         'structured_source': 'data/survey/87_protein_demo.json',
         'doc_source': 'analysis/17',
         'note': 'not a contamination class; included because it is the structured record '
                 'behind the protein-context sanity check'},
    ]
    cols = ['category', 'scope', 'n_affected', 'handling', 'status', 'structured_source',
            'doc_source', 'note']
    md = ['**Supplementary Table | Data-quality screening: removed / repaired / audited '
          'entries.**', '',
          '| Category | Scope | Affected | Handling | Status | Structured source |',
          '|---|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {} | {} | {} | {} |'.format(
            r['category'], r['scope'], r['n_affected'], r['handling'], r['status'],
            r['structured_source']))
    md += ['',
           'Documentary sources: analysis/10_对比方法基准报告.md §3 (data-quality audit and '
           'repairs), analysis/05_会话交接文档.md, analysis/20 (RNet leakage audit), '
           'analysis/16/17 (retrospective-library audits). Categories marked ⚠️ have no '
           'surviving structured per-item list and are transcribed from the text record; '
           'all other rows are backed by the cited JSON/clstr files. The 29 protein-trace '
           'chains named in the documents sit inside the broader structured drop lists '
           '(52/53 files), which also contain other defect classes.']
    na_reasons = ['43c P-atom file list, the 13 O5* chains, the 5 missing-C1\' chains and the '
                  'exact 29-chain protein-trace list survive only in the analysis documents, '
                  'not as JSON ⚠️']
    return 'st_contamination', cols, rows, '\n'.join(md) + '\n', {
        'na_mpnn_drops': 'data/survey/52_nampnn_preproc_dropped.json',
        'alignif_drops': 'data/survey/53_alignif_data_dropped.json',
        'multicopy_rebuild': 'data/survey/50_backbone_incomplete.json + 50_regen_fails.json',
        'resname_map': 'data/survey/resname_to_char.json',
        'rnet_audit': 'data/survey/89_rnet_leakage_audit.json',
        'aptadb_screen': 'data/survey/79_aptadb_retro.json + 79_leak_out.fa.clstr',
        'utexas_screen': 'data/survey/88_utexas_audit.json + 88_leak_out.fa.clstr',
        'protein_demo': 'data/survey/87_protein_demo.json',
        'documents': 'analysis/10 §3, analysis/05, analysis/20 §四, analysis/16/17'}, \
        na_reasons


# ================================================================ table 9
# st_zeroshot_ligands — 93 zero-shot chains, ligand CCD/name, max Tanimoto, family
def build_zeroshot_ligands():
    t93 = load('93_tanimoto_zs.json')
    ccd_types = json.load(open(os.path.join(HERE, 'ccd', 'ccd_types.json')))
    f75 = load('75_families.json')
    chain2fam = {}
    for fam, v in f75.items():
        for c in v['chains']:
            chain2fam[c] = fam
    no_lig = set(t93['coverage']['zs_no_ligand_stems'])

    def ccd_name(ccd):
        e = ccd_types.get(ccd)
        if not e or not e.get('name'):
            return None
        return e['name'].strip('"')

    rows = []
    for chain in sorted(t93['per_chain'].keys()):
        v = t93['per_chain'][chain]
        ligs = v.get('ligands') or []
        names = [ccd_name(c) for c in ligs]
        names = [n if n else f'{c} (name n/a)' for c, n in zip(ligs, names)]
        note = ''
        if chain in no_lig:
            note = 'no ligand annotation (PDB-level ligands.other empty)'
        elif v.get('max_tanimoto') is None:
            note = 'ligand(s) without SMILES/fingerprint (OXV RDKit parse fail) — excluded ' \
                   'from the Tanimoto stratification'
        rows.append({'chain': chain, 'pdb': v['pdb'],
                     'ligand_ccds': '+'.join(ligs) if ligs else None,
                     'ligand_names': '; '.join(names) if names else None,
                     'n_ligands': len(ligs) if ligs else 0,
                     'max_tanimoto_vs_247_train': v.get('max_tanimoto'),
                     'family': chain2fam.get(chain, 'other / unannotated'),
                     'note': note})
    n = len(rows)
    n_with_lig = sum(1 for r in rows if r['n_ligands'] > 0)
    n_with_t = sum(1 for r in rows if r['max_tanimoto_vs_247_train'] is not None)
    check('t9_counts', n == 93 and n_with_lig == 88 and n_with_t == 87,
          f'chains {n}, with ligand {n_with_lig}, with Tanimoto {n_with_t}')
    check('t9_no_ligand_set',
          sorted(r['chain'] for r in rows if r['n_ligands'] == 0) == sorted(no_lig),
          f'no-ligand chains == 93 coverage record {sorted(no_lig)}')

    cols = ['chain', 'pdb', 'ligand_ccds', 'ligand_names', 'n_ligands',
            'max_tanimoto_vs_247_train', 'family', 'note']
    md = ['**Supplementary Table | Zero-shot test chains (93) with ligand annotation, '
          'chemical novelty and family.**', '',
          '| Chain | PDB | Ligand CCD | Ligand name | max Tanimoto | Family | Note |',
          '|---|---|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            r['chain'], r['pdb'], r['ligand_ccds'] or NA, r['ligand_names'] or NA,
            f"{r['max_tanimoto_vs_247_train']:.3f}"
            if r['max_tanimoto_vs_247_train'] is not None else NA,
            r['family'], r['note']))
    md += ['',
           'max Tanimoto = maximum Tanimoto similarity (Morgan radius-2, 2048-bit) of the '
           'chain\'s ligand(s) against the 247 training ligands (multi-ligand chains take the '
           'max), from 93_tanimoto_zs.json per_chain. Ligand CCDs: PDB-level ligands.other '
           'annotation (same source as 93/95_chain_ccd conventions); ligand names from '
           'data/ccd/ccd_types.json (CCD dictionary). Family: 75_families.json multi-ligand '
           'family map (chains not in any family = other / unannotated). 5 chains have no '
           'ligand annotation (5NFV_B, 6IV6_G, 6VRD_B, 8SWB_B, 8SWC_B) and 6PQ7_C\'s ligand '
           'OXV has no SMILES — these 6 chains carry n/a Tanimoto and are excluded from the '
           'stratification (n = 87), matching 93_tanimoto_zs.json coverage.']
    na_reasons = ['5 chains without ligand annotation -> ligand/Tanimoto n/a '
                  '(93_tanimoto_zs.json coverage.zs_no_ligand_stems)',
                  '6PQ7_C ligand OXV has no SMILES (RDKit parse fail) -> Tanimoto n/a',
                  'ligand names missing from the CCD dictionary are marked "name n/a"']
    return 'st_zeroshot_ligands', cols, rows, '\n'.join(md) + '\n', {
        'per_chain_tanimoto_and_ligands': 'data/survey/93_tanimoto_zs.json per_chain + coverage',
        'ligand_names': 'data/ccd/ccd_types.json',
        'family_map': 'data/survey/75_families.json',
        'chain_ccd_convention': 'data/survey/95_chain_ccd.json (attribution method note)'}, \
        na_reasons


# ================================================================ table 10
# st_family_def — ligand-family definitions (test-side 9 + other; finetune-side 5+1)
def build_family_def():
    strata = load('91_family_strata_gfix.json')
    f75 = load('75_families.json')
    f95 = load('95_families.json')
    fam_level = {f['family']: f for f in f95['family_level']}
    label = {
        'fam_1MMS': '1MMS family', 'theophylline_aptamer': 'Theophylline aptamer',
        'purine_riboswitch': 'Purine riboswitch', 'fam_3GCA': '3GCA family',
        'dG_riboswitch': 'dG riboswitch', 'ztp_riboswitch': 'ZTP riboswitch',
        'mango3_aptamer': 'Mango-III aptamer', 'xanthine_riboswitch': 'Xanthine riboswitch',
        'fam_8HB1': '8HB1 family', 'other': 'Other / unannotated',
        'fluo_aptamer_8hz': 'N618-class fluorescent aptamer (8HZ*)',
        'rhobast_aptamer': 'RhoBAST aptamer', 'folate_riboswitch': 'Folate riboswitch',
    }
    rule75 = ('structure-based single-linkage clustering over the 200 interface chains: '
              'pairwise difflib sequence identity >= 0.4 with coverage >= 0.8; family '
              'requires >= 3 chains and >= 2 distinct CCD sets (data/75_p01_ligand_swap.py '
              'phase families; 75_families.json)')
    rows = []
    for fam in ['fam_1MMS', 'theophylline_aptamer', 'purine_riboswitch', 'fam_3GCA',
                'dG_riboswitch', 'ztp_riboswitch', 'mango3_aptamer', 'xanthine_riboswitch',
                'fam_8HB1']:
        g = strata['groups'][fam]
        v75 = f75[fam]
        rep = v75['chains'][0].split('_')[0]
        rows.append({'block': 'cluster-based test (477)', 'family': fam,
                     'label': label[fam], 'n_chains': g['n_chains'],
                     'n_chains_all_splits': len(v75['chains']),
                     'representative_pdb': rep,
                     'ccd_codes': '+'.join(v75['ccd_sets']),
                     'splits_present': '+'.join(v75['splits']),
                     'decision_rule': rule75})
    rows.append({'block': 'cluster-based test (477)', 'family': 'other',
                 'label': label['other'], 'n_chains': strata['groups']['other']['n_chains'],
                 'n_chains_all_splits': None, 'representative_pdb': NA, 'ccd_codes': NA,
                 'splits_present': 'test',
                 'decision_rule': 'remainder: test chains not assigned to any of the 9 '
                                  'annotated families'})
    for fam in ['fluo_aptamer_8hz', 'rhobast_aptamer', 'folate_riboswitch']:
        v75 = f75[fam]
        rows.append({'block': 'cluster-based test (477)', 'family': fam,
                     'label': label[fam], 'n_chains': 0,
                     'n_chains_all_splits': len(v75['chains']),
                     'representative_pdb': v75['chains'][0].split('_')[0],
                     'ccd_codes': '+'.join(v75['ccd_sets']),
                     'splits_present': '+'.join(v75['splits']),
                     'decision_rule': rule75 + ' — zero members in the 477-chain test set '
                                     '(av2-only family) ⚠️'})
    # finetune-side chemical families (LOFO experiments)
    fd = f95['families_def']
    fin_fams = [('aminoglycoside', fd['aminoglycoside']),
                ('sam_sah', fd['sam_sah']),
                ('ntp', fd['ntp']),
                ('cla', ['CLA']),
                ('2ba', ['2BA']),
                ('flavins', fd['flavins'])]
    for fam, ccds in fin_fams:
        fl = fam_level.get(fam, {})
        rows.append({'block': 'finetune training set (768)', 'family': fam,
                     'label': fam.replace('_', '-'), 'n_chains': fl.get('chains'),
                     'n_chains_all_splits': fl.get('n_pdb'),
                     'representative_pdb': NA,
                     'ccd_codes': '+'.join(ccds),
                     'splits_present': 'train',
                     'decision_rule': 'explicit CCD membership list '
                                      '(95_families.json families_def / single-CCD families '
                                      'for cla, 2ba); n_chains_all_splits column holds the '
                                      'PDB count for this block'})
    exp = {'fam_1MMS': 5, 'theophylline_aptamer': 13, 'purine_riboswitch': 50,
           'fam_3GCA': 4, 'dG_riboswitch': 4, 'ztp_riboswitch': 8, 'mango3_aptamer': 10,
           'xanthine_riboswitch': 8, 'fam_8HB1': 3, 'other': 372}
    got = {r['family']: r['n_chains'] for r in rows if r['block'].startswith('cluster')}
    check('t10_counts_477', all(got[k] == v for k, v in exp.items())
          and sum(exp.values()) == 477, f'{got} sums to {sum(exp.values())}')
    check('t10_finetune_amino',
          fam_level['aminoglycoside']['chains'] == 128
          and all(f['chains'] <= 128 for f in f95['family_level']),
          'aminoglycoside 128 = largest finetune-side family')

    cols = ['block', 'family', 'label', 'n_chains', 'n_chains_all_splits',
            'representative_pdb', 'ccd_codes', 'splits_present', 'decision_rule']
    md = ['**Supplementary Table | Ligand-family definitions.**', '',
          '| Block | Family | n chains | Representative PDB | CCD codes | Splits |',
          '|---|---|---|---|---|---|']
    for r in rows:
        md.append('| {} | {} | {} | {} | {} | {} |'.format(
            r['block'], r['label'], r['n_chains'], r['representative_pdb'], r['ccd_codes'],
            r['splits_present']))
    md += ['',
           'Test-side families (cluster-based 477-chain test set): structure-based '
           'single-linkage clustering over the 200 interface chains — pairwise difflib '
           'sequence identity ≥ 0.4 with coverage ≥ 0.8; a family requires ≥ 3 chains and '
           '≥ 2 distinct ligand CCD sets (data/75_p01_ligand_swap.py, 75_families.json); '
           'test-set counts from 91_family_strata_gfix.json (5+13+50+4+4+8+10+8+3 annotated '
           '= 105, plus 372 other = 477). fluo_aptamer_8hz / rhobast_aptamer / '
           'folate_riboswitch have zero members in the 477-chain test set (av2-only) and are '
           'listed for completeness. Representative PDB = first chain of the family list. '
           'Finetune-side families (768-chain S2 set, 95_families.json): explicit CCD '
           'membership lists (families_def; cla and 2ba are single-CCD families); '
           'aminoglycoside (128 chains) is the largest — these five families define the '
           'LOFO experiments of st_lofo (flavins shown but never left out).']
    na_reasons = ['other/unannotated has no CCD set or representative PDB by definition',
                  'finetune-side families are CCD-level definitions: no single '
                  'representative PDB (n_pdb given instead)',
                  'av2-only families contribute 0 test chains ⚠️ (listed for completeness)']
    return 'st_family_def', cols, rows, '\n'.join(md) + '\n', {
        'test_side_counts': 'data/survey/91_family_strata_gfix.json groups',
        'family_definitions': 'data/survey/75_families.json (chains, ccd_sets, splits) + '
                              'data/75_p01_ligand_swap.py (clustering rule)',
        'finetune_side': 'data/survey/95_families.json (families_def, family_level)',
        'doc': 'analysis/21_新增实验A_逐残基MacroF1_跨方法置信度_家族分层.md §5'}, na_reasons


# ================================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    builders = [build_benchmark_full, build_ensemble_scan, build_ablation, build_lofo,
                build_tanimoto, build_predbb, build_efficiency, build_contamination,
                build_zeroshot_ligands, build_family_def]
    manifest = {'generated_by': 'data/109_si_tables.py',
                'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'output_dir': 'data/survey/109_si_tables/',
                'format_conventions': [
                    'recovery in %, 1 decimal (markdown); CSV keeps 4-decimal percentages',
                    'Spearman rho: 3 decimals',
                    'P values: 2 significant digits, scientific notation below 1e-3',
                    'missing values rendered n/a; reasons listed per table below',
                    'fields not present in any source JSON are marked ⚠️ and transcribed '
                    'only from the cited analysis document (never invented)'],
                'tables': {}}
    for b in builders:
        slug, cols, rows, md, sources, na_reasons = b()
        write_csv(os.path.join(OUT, f'{slug}.csv'), cols, rows)
        write_text(os.path.join(OUT, f'{slug}.md'), md)
        manifest['tables'][slug] = {
            'files': [f'{slug}.csv', f'{slug}.md'],
            'n_rows': len(rows),
            'sources': sources,
            'na_reasons': na_reasons,
        }
        print(f'{slug}: {len(rows)} rows written')
    manifest['verification'] = CHECKS
    manifest['all_checks_pass'] = all(c['pass'] for c in CHECKS)
    write_text(os.path.join(OUT, 'manifest.json'),
               json.dumps(manifest, indent=1, ensure_ascii=False) + '\n')
    n_fail = sum(1 for c in CHECKS if not c['pass'])
    print(f'checks: {len(CHECKS) - n_fail}/{len(CHECKS)} pass')
    print('ALL CHECKS PASS' if manifest['all_checks_pass'] else 'SOME CHECKS FAILED')
    return manifest['all_checks_pass']


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
