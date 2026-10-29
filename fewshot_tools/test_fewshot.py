import sys,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from fewshot_core import *
rng=np.random.default_rng(44)
# Synthetic software tests, not biomedical results.
meta=pd.DataFrame({'row_id':[str(i) for i in range(320)],'pair_key':np.repeat([f'p{i:03}' for i in range(80)],4),'cell_line':np.tile(['a','b','c','d'],80)})
pool,q=partition_pairs(meta)
s=draw_support(meta,pool,16,101)
assert len(s)==16 and len(set(meta.iloc[s].pair_key))==16
assert not set(meta.iloc[s].pair_key)&set(meta.iloc[q].pair_key)
assert np.array_equal(draw_support(meta,pool,8,101),s[:8])
x=rng.normal(size=(320,12));y=x[:,0]+rng.normal(size=320)*.2
p,alpha=select_and_predict(x[s],y[s],x[q])
y2=y.copy();y2[q]=rng.normal(size=len(q))*100000
p2,a2=select_and_predict(x[s],y2[s],x[q])
assert np.allclose(p,p2) and alpha==a2,'Query labels reached fitting'
with tempfile.TemporaryDirectory() as td:
 td=Path(td);f=td/'toy.npz';meta.to_csv(f.with_suffix('.csv'),index=False)
 np.savez_compressed(f,base=.5*x[:,0],uniform=.2*x[:,0],independent=x[:,:6],interaction=x[:,6:],fingerprint=x,label=y,raw_target=y*12)
 m=run_bundle(f,td/'out',budgets=(8,16),draws=(101,102))
 assert len(m)==2*2*7
 assert (td/'out'/'query_predictions.csv.gz').exists()
 # Mean correction cannot change PCC.
 a=m[m.method=='source_only_sym'].pcc.to_numpy();b=m[m.method=='support_mean_residual'].pcc.to_numpy()
 assert np.allclose(a,b)
 pred=pd.read_csv(td/'out'/'query_predictions.csv.gz',dtype={'row_id':str});pred['seed']=42
 b=paired_pair_bootstrap(pred,budget=16,n_boot=20)
 assert np.isfinite(b['delta_pcc'])
 run_bundle(f,td/'out',budgets=(8,16),draws=(101,102))
 try:run_bundle(f,td/'out',budgets=(8,),draws=(101,102))
 except ValueError:pass
 else:raise AssertionError('Protocol lock was not enforced')
print('PASS: pair disjointness, exact/nested label budgets, query-label isolation, constant-shift PCC, deterministic rerun, protocol lock, paired bootstrap, end-to-end synthetic outputs.')
