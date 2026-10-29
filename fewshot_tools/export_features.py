"""Export frozen symmetric source representations. Run once; no target adaptation here."""
import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def export_all(root, checkpoint_dir=None, seeds=(42,43,44), batch_size=128):
    root=Path(root);os.environ['DGS_ROOT']=str(root)
    checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else root/'cross_dataset'
    directions=('DrugComb->ONeil','ONeil->DrugComb')
    paths={(d,s):checkpoint_dir/f"best_model_xdataset_unified_loewe_{d.replace('->','_to_')}_seed{s}.pth"
           for d in directions for s in seeds}
    needed=[root/'data'/f for f in ['drugcomb_full_v1.5.csv','smiles_cache.csv','oneil_summary_idx.csv',
            'drugs.csv','cell_lines.csv','drug_to_chemberta.pt']]+list(paths.values())
    missing=[str(p) for p in needed if not p.exists()]
    if missing:raise FileNotFoundError('Missing required ORIGINAL data/checkpoints:\n'+'\n'.join(missing))
    import source_runtime as r
    if Path(r.ROOT).resolve()!=root.resolve():
        raise RuntimeError("ROOT changed after importing runtime; restart the notebook runtime before exporting.")
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_dense_batch
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from fewshot_core import lock_manifest
    out=root/'fewshot_extension_v1';out.mkdir(parents=True,exist_ok=True)
    shipped_cell_map=root/'splits'/'cell_line_to_idx.json'
    if shipped_cell_map.exists():
        expected=json.loads(shipped_cell_map.read_text())
        assert expected=={str(k):int(v) for k,v in r.cell_line_to_idx.items()},'Cell embedding index differs from the shipped source mapping.'
    # Operational equivalence rule: transitive normalized name OR largest-fragment/no-stereo SMILES.
    names=sorted(map(str,r.all_drugs));parent={d:d for d in names}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    name_seen={};struct_seen={}
    for d in names:
        mol=Chem.MolFromSmiles(r.drug_to_smiles[d]); assert mol is not None
        fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
        largest=max(fragments,key=lambda m:m.GetNumAtoms())
        sk=Chem.MolToSmiles(largest,isomericSmiles=False)
        for key,seen in [(r._norm_name(d),name_seen),(sk,struct_seen)]:
            if key in seen:union(d,seen[key])
            else:seen[key]=d
    equiv={d:find(d) for d in names}
    def pair_key(a,b):return json.dumps(sorted([equiv[a],equiv[b]]),separators=(',',':'))
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    fingerprints={d:fpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(r.drug_to_smiles[d])).astype(np.float32) for d in names}
    def sym(a,b):return torch.cat([a+b,torch.abs(a-b)],dim=1)
    @torch.no_grad()
    def batch_features(model,a,b):
        h1=model.encode_atoms(a.x,a.edge_index,a.edge_attr)
        h2=model.encode_atoms(b.x,b.edge_index,b.edge_attr)
        g1,g2=model.pool(h1,a.batch),model.pool(h2,b.batch)
        d1,m1=to_dense_batch(h1,a.batch);d2,m2=to_dense_batch(h2,b.batch)
        t1,w1=model.cross_attention(d1,d2,d2,key_padding_mask=~m2,need_weights=True,average_attn_weights=False)
        t2,w2=model.cross_attention(d2,d1,d1,key_padding_mask=~m1,need_weights=True,average_attn_weights=False)
        i1,i2=model.pool(t1[m1],a.batch),model.pool(t2[m2],b.batch)
        c1,c2=model.chemberta_proj(a.cb),model.chemberta_proj(b.cb)
        cell=model.cell_embedding(a.cell_idx)
        pred=lambda v1,v2:model.predictor(torch.cat([g1,g2,v1,v2,c1,c2,cell],dim=1)).flatten()
        p1=pred(i1,i2)
        p2=model.predictor(torch.cat([g2,g1,i2,i1,c2,c1,cell],dim=1)).flatten()
        independent=torch.cat([sym(g1,g2),sym(c1,c2),cell],dim=1)
        interaction=sym(i1,i2)
        # Inference intervention: exactly uniform partner attention, same learned V/out projections.
        # Not a retrained no-IMCA model; tests functional dependence of this fixed checkpoint.
        mha=model.cross_attention
        assert mha._qkv_same_embed_dim and mha.bias_k is None and mha.bias_v is None and not mha.add_zero_attn
        vw=mha.in_proj_weight.chunk(3)[2]
        vb=mha.in_proj_bias.chunk(3)[2] if mha.in_proj_bias is not None else None
        def uniform(dense,mask):
            v=F.linear(dense,vw,vb)
            mean=(v*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
            return mha.out_proj(mean)
        u1,u2=uniform(d2,m2),uniform(d1,m1)
        pu1=pred(u1,u2)
        pu2=model.predictor(torch.cat([g2,g1,u2,u1,c2,c1,cell],dim=1)).flatten()
        diagnostics=[]
        for weights,mq,mk in [(w1,m1,m2),(w2,m2,m1)]:
            for j in range(len(mq)):
                w=weights[j][:,mq[j],:][:,:,mk[j]].clamp_min(1e-12)
                mean=w.mean(dim=1,keepdim=True)
                # Mean KL of each atom-query attention to the molecule's average attention.
                # Zero means all queries attend identically (not necessarily uniform across keys).
                querydiv=(w*(w.log()-mean.log())).sum(-1).mean()
                ent=-(w*w.log()).sum(-1).mean()/np.log(max(2,int(mk[j].sum())))
                diagnostics.append([float(querydiv),float(ent)])
        # Manual order-averaged prediction must equal the original forward implementation.
        fwd,_=model(a,b);rev,_=model(b,a)
        assert torch.allclose((p1+p2)/2,(fwd.flatten()+rev.flatten())/2,atol=2e-5,rtol=2e-4)
        independent_rev=torch.cat([sym(g2,g1),sym(c2,c1),cell],dim=1)
        assert torch.allclose(independent,independent_rev,atol=1e-6)
        return dict(base=((p1+p2)/2).cpu().numpy(), uniform=((pu1+pu2)/2).cpu().numpy(),
                    independent=independent.cpu().numpy(),interaction=interaction.cpu().numpy()),diagnostics
    outputs=[]
    for direction in directions:
        src,tgt=(r.dc_loe_src,r.df_on) if direction=='DrugComb->ONeil' else (r.df_on,r.dc_loe_src)
        tr,va,target,info=r.build_cross_dataset_split(src,tgt,seed=42)
        source_pairs={pair_key(a,b) for a,b in src[['drug_1','drug_2']].itertuples(index=False,name=None)}
        target=target.copy();target['row_id']=target.index.astype(str)
        target['pair_key']=[pair_key(a,b) for a,b in target[['drug_1','drug_2']].itertuples(index=False,name=None)]
        before=len(target)
        # Stricter than the old Table 9: remove whole pair if present ANYWHERE in source (train or validation).
        target=target[~target.pair_key.isin(source_pairs)].copy().reset_index(drop=True)
        ds=r.UnifiedSynergyDataset(target,r.drug_to_smiles,r.drug_to_chemberta,r.cell_line_to_idx)
        assert len(ds.valid_indices)==len(target),'Invalid molecular rows: inspect instead of silently changing query.'
        tag=direction.replace('->','_to_')
        lock_manifest(out/f'{tag}_data_audit.json',dict(direction=direction,original_split=info,
            removed_source_pair_overlap_rows=before-len(target),target_rows=len(target),
            target_pairs=target.pair_key.nunique(),equivalence_groups=len(set(equiv.values())),
            cell_mapping={str(k):int(v) for k,v in r.cell_line_to_idx.items()},
            target_row_ids=target.row_id.tolist(),target_pair_keys=target.pair_key.tolist(),
            source_model_split='Original identifier-level source split, unchanged; alias validation risk not retroactively fixed.'))
        fp=np.stack([np.concatenate([fingerprints[a]+fingerprints[b],np.abs(fingerprints[a]-fingerprints[b])])
            for a,b in target[['drug_1','drug_2']].itertuples(index=False,name=None)])
        onehot=np.eye(r.n_cell_lines,dtype=np.float32)[target.cell_line.map(r.cell_line_to_idx).to_numpy()]
        fp=np.concatenate([fp,onehot],axis=1)
        for seed in seeds:
            checkpoint=paths[direction,seed]
            model=r.DualGATSynergy(**r.MODEL_CFG).to(r.DEVICE)
            model.load_state_dict(torch.load(checkpoint,map_location=r.DEVICE,weights_only=True));model.eval()
            checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            lock_manifest(out/f'{tag}_seed{seed}_checkpoint.json',dict(path=str(checkpoint),sha256=checkpoint_hash))
            pieces=[];diagnostics=[]
            offset=0
            for a,b,y in DataLoader(ds,batch_size=batch_size,shuffle=False):
                part,diag=batch_features(model,a.to(r.DEVICE),b.to(r.DEVICE));pieces.append(part)
                size=len(y)
                for side in range(2):
                    for j in range(size):
                        row=target.iloc[offset+j]
                        diagnostics.append(dict(row_id=row.row_id,pair_key=row.pair_key,
                            direction='A_to_B' if side==0 else 'B_to_A',
                            query_attention_diversity=diag[side*size+j][0],
                            normalized_attention_entropy=diag[side*size+j][1]))
                offset+=size
            arrays={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            arrays.update(fingerprint=fp,label=target.label.to_numpy(),raw_target=target.raw_target.to_numpy())
            path=out/f'{tag}_seed{seed}.npz'
            # Do not overwrite an existing exported bundle silently (GPU roundoff may differ).
            if path.exists():
                old=np.load(path)
                assert set(old.files)==set(arrays) and all(np.allclose(old[k],v,atol=2e-5,rtol=2e-4) for k,v in arrays.items()),'Export differs: use a new versioned directory.'
            else:np.savez_compressed(path,**arrays)
            target[['row_id','pair_key','cell_line','drug_1','drug_2','n_unseen_drugs']].to_csv(path.with_suffix('.csv'),index=False)
            pd.DataFrame(diagnostics).to_csv(out/f'{tag}_seed{seed}_attention_audit.csv',index=False)
            outputs.append(str(path));print('EXPORTED',path)
            del model
            if r.DEVICE=='cuda':torch.cuda.empty_cache()
    return outputs
