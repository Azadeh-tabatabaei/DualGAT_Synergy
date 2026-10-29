"""Pair-disjoint target adaptation. No deep-learning dependencies.
All model selection is performed inside the labelled support budget.
This module never treats support resampling as independent test replication.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score

ALPHAS = (0.1, 1., 10., 100., 1000.)

def lock_manifest(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Locked protocol differs: {path}. Use a new run directory, not an overwrite.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True))


def partition_pairs(meta, seed=2027, query_fraction=.5):
    """Target pair split ignores labels. All cells/replicates of a pair stay together."""
    groups = np.array(sorted(meta.pair_key.unique()))
    if len(groups) < 20:
        raise ValueError('Too few independent target pairs for the predeclared experiment.')
    rng = np.random.default_rng(seed)
    groups = rng.permutation(groups)
    nq = int(np.ceil(query_fraction * len(groups)))
    query = meta.pair_key.isin(groups[:nq]).to_numpy()
    assert not set(meta.loc[query, 'pair_key']) & set(meta.loc[~query, 'pair_key'])
    return np.flatnonzero(~query), np.flatnonzero(query)


def draw_support(meta, pool, max_budget, seed):
    """One labelled row per distinct unordered pair; prefixes give nested budgets.
    Select pairs and rows without looking at response labels; not per-cell K-shot.
    """
    rng = np.random.default_rng(seed)
    groups = rng.permutation(sorted(meta.iloc[pool].pair_key.unique()))
    if len(groups) < max_budget:
        raise ValueError(f'Budget {max_budget} exceeds {len(groups)} support-pool pairs.')
    chosen=[]
    for group in groups[:max_budget]:
        candidates=pool[meta.iloc[pool].pair_key.to_numpy()==group]
        chosen.append(int(rng.choice(candidates)))
    return np.array(chosen)


def fit_ridge(x, y, alpha):
    # Fit normalization on current training/support rows ONLY.
    scaler=StandardScaler().fit(x)
    model=Ridge(alpha=float(alpha), solver='cholesky').fit(scaler.transform(x), y)
    return scaler, model


def select_and_predict(xs, ys, xq, alphas=ALPHAS):
    """Support-only inner CV; alpha chosen by MSE, never query PCC.
    Support rows have unique pair keys, so KFold is also pair-disjoint.
    """
    xs=np.asarray(xs, dtype=float); ys=np.asarray(ys, dtype=float)
    if len(ys)<6: raise ValueError('At least six labelled support pairs are required.')
    cv=list(KFold(n_splits=3, shuffle=True, random_state=818).split(xs))
    losses=[]
    for alpha in alphas:
        loss=[]
        for tr, va in cv:
            scaler, model=fit_ridge(xs[tr], ys[tr], alpha)
            loss.append(mean_squared_error(ys[va], model.predict(scaler.transform(xs[va]))))
        losses.append(float(np.mean(loss)))
    alpha=float(alphas[int(np.argmin(losses))])
    scaler, model=fit_ridge(xs, ys, alpha)
    return model.predict(scaler.transform(xq)), alpha


def score(y, p, raw, cells):
    y=np.asarray(y); p=np.asarray(p); raw=np.asarray(raw); cells=np.asarray(cells)
    corr=lambda a,b: float(pearsonr(a,b).statistic) if np.std(a)>1e-10 and np.std(b)>1e-10 else np.nan
    binary=raw>10
    out=dict(pcc=corr(y,p), scc=float(spearmanr(y,p).statistic) if np.std(p)>1e-10 else np.nan,
             rmse_source_z=float(np.sqrt(mean_squared_error(y,p))), prevalence=float(binary.mean()))
    out['auc']=float(roc_auc_score(binary,p)) if np.unique(binary).size==2 else np.nan
    out['ap']=float(average_precision_score(binary,p)) if binary.any() else np.nan
    within=[]
    for cell in np.unique(cells):
        m=cells==cell
        if m.sum()>=30:
            r=corr(y[m],p[m])
            if np.isfinite(r):within.append((int(m.sum()),r))
    out['within_cell_pcc']=float(np.average([r for n,r in within],weights=[n for n,r in within])) if within else np.nan
    out['within_cell_eligible_rows']=sum(n for n,r in within)
    return out


def run_bundle(bundle_path, out_dir, budgets=(16,32,64,128), draws=(101,102,103,104,105), split_seed=2027):
    """Fit adapters, then evaluate fixed query. Bundle consists of .npz plus .csv.
    y is standardized with SOURCE training statistics. No target-wide scaling.
    """
    bundle_path=Path(bundle_path); out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    z=np.load(bundle_path,allow_pickle=False)
    meta=pd.read_csv(bundle_path.with_suffix('.csv'), dtype={'row_id':str,'pair_key':str,'cell_line':str})
    required={'base','uniform','independent','interaction','fingerprint','label','raw_target'}
    assert required<=set(z.files)
    assert all(len(z[k])==len(meta) for k in required)
    assert meta.row_id.is_unique
    for k in required:
        if not np.isfinite(z[k]).all(): raise ValueError('Nonfinite bundle values: '+k)
    pool,q=partition_pairs(meta,seed=split_seed)
    manifest=dict(protocol='pair_disjoint_one_labelled_row_per_support_pair_v1', split_seed=int(split_seed),
                  budgets=list(map(int,budgets)), draws=list(map(int,draws)), alphas=list(ALPHAS),
                  query_rows=meta.iloc[q].row_id.tolist(),pool_rows=meta.iloc[pool].row_id.tolist(),
                  feature_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest())
    lock_manifest(out/'locked_protocol.json',manifest)
    # Query labels are not passed to any fitting or alpha-selection function.
    y=z['label']; base=z['base']; full=np.concatenate([z['independent'],z['interaction']],axis=1)
    rows=[]; records=[]
    for ds in draws:
        support_order=draw_support(meta,pool,max(budgets),ds)
        for budget in budgets:
            s=support_order[:budget]
            assert len(set(meta.iloc[s].pair_key))==budget
            assert not set(meta.iloc[s].pair_key)&set(meta.iloc[q].pair_key)
            support_info=dict(draw=int(ds),budget=int(budget),
                              n_support_rows=len(s),n_support_pairs=meta.iloc[s].pair_key.nunique(),
                              n_support_cells=meta.iloc[s].cell_line.nunique(),
                              n_query_rows=len(q),n_query_pairs=meta.iloc[q].pair_key.nunique())
            support_info['support_rows']=meta.iloc[s].row_id.tolist()
            lock_manifest(out/f'support_draw{ds}_n{budget}.json',support_info)
            predictions={'source_only_sym':(base[q],np.nan),
                         'source_uniform_attention_intervention':(z['uniform'][q],np.nan),
                         'support_mean_residual':(base[q]+float(np.mean(y[s]-base[s])),np.nan)}
            # This is an ADAPTER FEATURE ablation, NOT retraining/removing source IMCA.
            for name,x,residual in [
                ('target_only_fingerprint_ridge',z['fingerprint'],False),
                ('target_head_frozen_full',full,False),
                ('residual_independent_features',z['independent'],True),
                ('residual_full_features',full,True)]:
                fit_y=y[s]-base[s] if residual else y[s]
                p,alpha=select_and_predict(x[s],fit_y,x[q])
                predictions[name]=(p+base[q] if residual else p,alpha)
            for name,(p,alpha) in predictions.items():
                metrics=score(y[q],p,z['raw_target'][q],meta.iloc[q].cell_line.to_numpy())
                rows.append({**{k:v for k,v in support_info.items() if k!='support_rows'},'method':name,'alpha':alpha,**metrics})
                records.append(pd.DataFrame(dict(row_id=meta.iloc[q].row_id.to_numpy(),pair_key=meta.iloc[q].pair_key.to_numpy(),
                    cell_line=meta.iloc[q].cell_line.to_numpy(),draw=ds,budget=budget,method=name,
                    target=y[q],raw_target=z['raw_target'][q],prediction=p)))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(out/'query_predictions.csv.gz',index=False)
    # Descriptive only: these draws share the same query and are NOT independent experiments.
    metrics.groupby(['budget','method']).agg(pcc_mean=('pcc','mean'),pcc_support_draw_sd=('pcc','std'),
                ap_mean=('ap','mean'),within_cell_mean=('within_cell_pcc','mean')).to_csv(out/'descriptive_summary.csv')
    return metrics


def paired_pair_bootstrap(predictions, method_a='residual_full_features',method_b='residual_independent_features',
                          budget=128,n_boot=2000,seed=919):
    """Average paired PCC differences across training seeds/support draws within each resample.
    One shared cluster resample across all runs. Pair clusters ignore shared-drug dependence.
    predictions must contain seed, draw, budget, method, row_id, pair_key, target, prediction.
    """
    d=predictions[predictions.budget==budget]
    ids=['seed','draw','row_id','pair_key','target']
    a=d[d.method==method_a][ids+['prediction']].rename(columns={'prediction':'a'})
    b=d[d.method==method_b][ids+['prediction']].rename(columns={'prediction':'b'})
    merged=a.merge(b,on=ids,validate='one_to_one')
    assert len(merged)==len(a)==len(b)
    groups=np.array(sorted(merged.pair_key.unique()))
    if len(groups)<20:raise ValueError('Too few pairs for a useful interval.')
    runs=[]
    for _,r in merged.groupby(['seed','draw']):
        r=r.reset_index(drop=True)
        runs.append((r, {g:np.flatnonzero(r.pair_key.to_numpy()==g) for g in groups}))
    def stat(sample):
        vals=[]
        for r,lookup in runs:
            ix=np.concatenate([lookup[g] for g in sample]); t=r.target.to_numpy()[ix]
            if np.std(t)>1e-10 and np.std(r.a.to_numpy()[ix])>1e-10 and np.std(r.b.to_numpy()[ix])>1e-10:
                vals.append(pearsonr(t,r.a.to_numpy()[ix]).statistic-pearsonr(t,r.b.to_numpy()[ix]).statistic)
        return float(np.mean(vals)) if vals else np.nan
    estimate=stat(groups);rng=np.random.default_rng(seed)
    boots=np.array([stat(rng.choice(groups,len(groups),replace=True)) for _ in range(n_boot)])
    lo,hi=np.nanquantile(boots,[.025,.975])
    return dict(method_a=method_a,method_b=method_b,budget=budget,delta_pcc=estimate,
                ci_low=float(lo),ci_high=float(hi),n_pairs=len(groups),
                warning='Exploratory; fixed query, few source seeds; shared-drug dependence not modeled.')
