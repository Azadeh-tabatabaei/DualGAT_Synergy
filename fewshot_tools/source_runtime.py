# Reused verbatim from the supplied 04 notebook, selected preparation/definition cells only.
# No source retraining, old test evaluation loop, or manuscript files are executed.
# ===== Cell 1: Environment =====
# Colab only. The package versions used for the reported results are pinned in requirements.txt.

# Standard library
import io, json, os, sys
from pathlib import Path

# Numerics and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

# Graph learning
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import to_dense_batch

# Chemistry
from rdkit import Chem, RDLogger, Geometry
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from PIL import Image
from rdkit.Chem import Draw, rdDepictor


# Pretrained chemical language model (ChemBERTa)
from transformers import AutoTokenizer, AutoModel

# Classical ML and metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")
ROOT = Path(os.environ['DGS_ROOT'])
DATA_DIR=ROOT/'data'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
SEED=42
XD_SETTINGS=('unified_loewe',)
XD_SEEDS=(42,43,44)
XD_BASELINES=False


# ===== Cell 3: Load and filter the raw data =====
# --- 3.1 DrugComb v1.5 ---
DRUGCOMB_COLS = ["drug_row", "drug_col", "cell_line_name", "synergy_hsa", "study_name", "tissue_name"]
drugcomb_raw = pd.read_csv(DATA_DIR / "drugcomb_full_v1.5.csv", usecols=DRUGCOMB_COLS, low_memory=False)
print(f"Raw records:                 {len(drugcomb_raw):,}   (paper: 1,432,351)")

# Remove non-oncology studies
NON_ONCOLOGY_KEYWORDS = ["malaria", "covid", "sars", "plasmodium", "tuberculosis"]
is_non_oncology = (drugcomb_raw["study_name"].astype(str).str.lower()
                   .str.contains("|".join(NON_ONCOLOGY_KEYWORDS), na=False))
drugcomb_raw = drugcomb_raw[~is_non_oncology].copy()

# Drop records without an HSA score or a cell-line name (removes nothing in v1.5)
drugcomb_raw = drugcomb_raw.dropna(subset=["synergy_hsa", "cell_line_name"])
n_compounds = len(set(drugcomb_raw["drug_row"]) | set(drugcomb_raw["drug_col"]))
print(f"After filtering:             {len(drugcomb_raw):,}   (paper: 1,432,146) | compounds: {n_compounds:,} (paper: 4,608)")

# --- 3.2 Compound -> SMILES cache (resolved through PubChem) ---
cache = pd.read_csv(DATA_DIR / "smiles_cache.csv")        # columns: name, smiles (empty if not resolved)
drug_to_smiles = dict(zip(cache["name"], cache["smiles"]))
known_drugs = {name for name, smi in drug_to_smiles.items() if isinstance(smi, str) and smi}
print(f"Cache: {len(drug_to_smiles)} names, {len(known_drugs)} resolved   (paper: 452 / 418)")

# Keep only records where BOTH compounds have a SMILES string
drugcomb = drugcomb_raw[drugcomb_raw["drug_row"].isin(known_drugs) &
                        drugcomb_raw["drug_col"].isin(known_drugs)].copy()
n_drugs = len(set(drugcomb["drug_row"]) | set(drugcomb["drug_col"]))
print(f"DrugComb subset: {len(drugcomb):,} records, {n_drugs} drugs   (paper: 61,111 / 276)")

# --- 3.3 O'Neil ---
oneil      = pd.read_csv(DATA_DIR / "oneil_summary_idx.csv")
drugs_meta = pd.read_csv(DATA_DIR / "drugs.csv")
cells_meta = pd.read_csv(DATA_DIR / "cell_lines.csv")
drug_map = dict(zip(drugs_meta["id"], drugs_meta["name"]))
cell_map = dict(zip(cells_meta["id"], cells_meta["name"]))
oneil["drug_row"]       = oneil["drug_row_idx"].map(drug_map)
oneil["drug_col"]       = oneil["drug_col_idx"].map(drug_map)
oneil["cell_line_name"] = oneil["cell_line_idx"].map(cell_map)
print(f"O'Neil: {len(oneil):,} records   (paper: 22,737) | unmapped names: "
      f"{int(oneil[['drug_row', 'drug_col', 'cell_line_name']].isna().sum().sum())}")

# ===== Cell 4: O'Neil records with SMILES =====
oneil["smiles_row"] = oneil["drug_row"].map(drug_to_smiles)
oneil["smiles_col"] = oneil["drug_col"].map(drug_to_smiles)
oneil_processed = oneil.dropna(subset=["smiles_row", "smiles_col"])[
    ["drug_row", "drug_col", "smiles_row", "smiles_col", "cell_line_name", "synergy_loewe"]].copy()
oneil_processed.columns = ["drug_1", "drug_2", "smiles_1", "smiles_2", "cell_line", "synergy_score"]
print(f"O'Neil: {len(oneil_processed):,} of {len(oneil):,} records have SMILES for both drugs   (paper: 22,737 of 22,737)")

# ===== Cell 5: ChemBERTa embeddings (frozen pretrained model, [CLS] token) =====
CHEMBERTA_NAME       = "seyonec/ChemBERTa-zinc-base-v1"
CHEMBERTA_SHIPPED    = DATA_DIR / "drug_to_chemberta.pt"               # 418 compounds x 768 numbers (about 1.3 MB)
CHEMBERTA_RECOMPUTED = DATA_DIR / "drug_to_chemberta_recomputed.pt"    # never overwrites the shipped file
RECOMPUTE_CHEMBERTA  = False   # False: load the shipped file (no download, CPU is fine)
                               # True : recompute with the pretrained model (inference only, no training)

if not RECOMPUTE_CHEMBERTA:
    drug_to_chemberta = torch.load(CHEMBERTA_SHIPPED, map_location="cpu")
else:
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_NAME)
    chemberta = AutoModel.from_pretrained(CHEMBERTA_NAME).eval().to(DEVICE)
    print("Model revision:", getattr(chemberta.config, "_commit_hash", "unknown"))

    @torch.no_grad()
    def chemberta_embedding(smiles):
        """[CLS]-token embedding of one SMILES string (truncated to 128 tokens)."""
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        return chemberta(**inputs).last_hidden_state[:, 0, :].squeeze(0).cpu()

    drug_to_chemberta = {name: chemberta_embedding(drug_to_smiles[name]) for name in sorted(known_drugs)}
    torch.save(drug_to_chemberta, CHEMBERTA_RECOMPUTED)

CHEMBERTA_DIM = next(iter(drug_to_chemberta.values())).shape[0]
print(f"ChemBERTa embeddings: {len(drug_to_chemberta)} compounds | dim = {CHEMBERTA_DIM}   (paper pipeline: 418 | 768)")
assert set(drug_to_chemberta) == known_drugs, "embedding file and SMILES cache do not match"

# ===== Cell 6: Molecular graphs =====
NODE_FEATURES = ["atomic_number", "degree", "explicit_valence", "implicit_valence", "is_aromatic", "in_ring",
                 "formal_charge", "num_radical_electrons", "hybridization", "chiral_tag", "atomic_mass"]
NODE_DIM, EDGE_DIM = len(NODE_FEATURES), 1      # edge feature: bond order (1.0, 1.5, 2.0 or 3.0)

def _valences(atom):
    """(explicit, implicit) valence. GetExplicitValence/GetImplicitValence produced the reported results;
    newer RDKit versions replace them with GetValence(...)."""
    try:
        return atom.GetExplicitValence(), atom.GetImplicitValence()
    except AttributeError:
        return atom.GetValence(Chem.ValenceType.EXPLICIT), atom.GetValence(Chem.ValenceType.IMPLICIT)

def smiles_to_graph(smiles):
    """SMILES string -> PyG graph (node features: NODE_FEATURES; edge feature: bond order).
    Returns None if the SMILES is missing or cannot be parsed."""
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = []
    for atom in mol.GetAtoms():
        explicit_valence, implicit_valence = _valences(atom)
        features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            explicit_valence,
            implicit_valence,
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
            int(atom.GetChiralTag()),       # RDKit chiral tag (none / clockwise / counter-clockwise / other)
            atom.GetMass(),
        ])
    x = torch.tensor(features, dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = bond.GetBondTypeAsDouble()
        edge_index += [(s, e), (e, s)]           # undirected: both directions
        edge_attr  += [order, order]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float).view(-1, 1)
    else:                                         # single-atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1),   dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Sanity check on all resolved compounds
graph_check = {name: smiles_to_graph(drug_to_smiles[name]) for name in known_drugs}
n_failed   = sum(g is None for g in graph_check.values())
n_no_bonds = sum(g is not None and g.edge_index.shape[1] == 0 for g in graph_check.values())
print(f"{len(graph_check)} cached compounds | failed to parse: {n_failed} | without any bond: {n_no_bonds} "
      f"((the paper's statement concerns the 309 compounds used in the experiments; that check is in notebook 04)")
assert all(g.x.shape[1] == NODE_DIM for g in graph_check.values() if g is not None)
del graph_check

# ===== Cell 7: Combine DrugComb (HSA) and O'Neil (Loewe) =====
drugcomb_unified = (drugcomb.dropna(subset=["synergy_hsa", "cell_line_name"])
                    [["drug_row", "drug_col", "synergy_hsa", "cell_line_name"]]
                    .rename(columns={"drug_row": "drug_1", "drug_col": "drug_2",
                                     "synergy_hsa": "raw_target", "cell_line_name": "cell_line"}))
drugcomb_unified["dataset_type"] = "drugcomb"

oneil_unified = (oneil_processed.dropna(subset=["synergy_score"])
                 [["drug_1", "drug_2", "synergy_score", "cell_line"]]
                 .rename(columns={"synergy_score": "raw_target"}))
oneil_unified["dataset_type"] = "oneil"

df_combined = pd.concat([drugcomb_unified, oneil_unified], ignore_index=True)

# One integer per cell line (order of first appearance) and the list of all compounds
all_cell_lines   = df_combined["cell_line"].unique()
cell_line_to_idx = {cl: i for i, cl in enumerate(all_cell_lines)}
n_cell_lines     = len(all_cell_lines)
all_drugs        = pd.concat([df_combined["drug_1"], df_combined["drug_2"]]).unique()

print(f"Records: {len(df_combined):,} (paper: 83,848) | cell-line identifiers: {n_cell_lines} (paper: 251) "
      f"| compounds: {len(all_drugs)} (paper: 309)")

# ===== Cell 9: Dataset =====
_mol_graph_cache = {}

def get_cached_graph(smiles):
    """Build each molecular graph once; return a copy so per-sample attributes never leak into the cache."""
    if smiles not in _mol_graph_cache:
        _mol_graph_cache[smiles] = smiles_to_graph(smiles)
    cached = _mol_graph_cache[smiles]
    return cached.clone() if cached is not None else None


class UnifiedSynergyDataset(Dataset):
    """Sample = (graph of drug 1, graph of drug 2, standardized synergy label).
    A record is kept only if both drugs have a ChemBERTa embedding and a SMILES that RDKit can parse."""
    def __init__(self, dataframe, drug_smiles_dict, drug_chemberta_dict, cell_line_to_idx):
        self.data        = dataframe.reset_index(drop=True)
        self.drug_smiles = drug_smiles_dict
        self.drug_cb     = drug_chemberta_dict
        self.cell_to_idx = cell_line_to_idx

        valid_drugs = {name for name in drug_chemberta_dict
                       if isinstance(drug_smiles_dict.get(name), str)
                       and Chem.MolFromSmiles(drug_smiles_dict[name]) is not None}
        keep = self.data["drug_1"].isin(valid_drugs) & self.data["drug_2"].isin(valid_drugs)
        self.valid_indices = list(self.data.index[keep])
        print(f"Valid samples: {len(self.valid_indices):,} of {len(self.data):,}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.data.iloc[self.valid_indices[idx]]
        g1 = get_cached_graph(self.drug_smiles[row["drug_1"]])
        g2 = get_cached_graph(self.drug_smiles[row["drug_2"]])
        g1.cb = self.drug_cb[row["drug_1"]].unsqueeze(0)
        g2.cb = self.drug_cb[row["drug_2"]].unsqueeze(0)
        cl_idx = self.cell_to_idx[row["cell_line"]]
        g1.cell_idx = torch.tensor([cl_idx], dtype=torch.long)
        g2.cell_idx = torch.tensor([cl_idx], dtype=torch.long)
        target = torch.tensor(row["label"], dtype=torch.float)
        return g1, g2, target

# ===== Cell 10: DualGATSynergy =====
class DualGATSynergy(nn.Module):
    def __init__(self, num_atom_features, chemberta_dim, n_cell_lines, hidden_dim=64, heads=2):
        super().__init__()
        # Weight-shared two-layer GATv2 atom encoder (bond order is the edge feature)
        self.conv1 = GATv2Conv(num_atom_features, hidden_dim, heads=heads, edge_dim=1, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, edge_dim=1, dropout=0.2)

        # Gated-attention pooling (one gate network, used for all four pooled streams)
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)))

        # IMCA: 4-head atom-level cross-attention; the same weights are used in both directions
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)

        # Projection of the 768-d ChemBERTa vector; LayerNorm aligns its scale with the GAT vectors
        self.chemberta_proj = nn.Sequential(
            nn.Linear(chemberta_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.3))

        self.cell_embedding = nn.Embedding(n_cell_lines, hidden_dim)

        # 7 streams x 64 = 448 inputs
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def encode_atoms(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index, edge_attr=edge_attr)

    def forward(self, d1, d2):
        atoms1 = self.encode_atoms(d1.x, d1.edge_index, d1.edge_attr)     # per-atom embeddings, before any pooling
        atoms2 = self.encode_atoms(d2.x, d2.edge_index, d2.edge_attr)
        emb1 = self.pool(atoms1, d1.batch)
        emb2 = self.pool(atoms2, d2.batch)

        dense1, mask1 = to_dense_batch(atoms1, d1.batch)                   # pad molecules to equal size
        dense2, mask2 = to_dense_batch(atoms2, d2.batch)
        attn_1to2, w_1to2 = self.cross_attention(dense1, dense2, dense2, key_padding_mask=~mask2)
        attn_2to1, w_2to1 = self.cross_attention(dense2, dense1, dense1, key_padding_mask=~mask1)
        pooled_1to2 = self.pool(attn_1to2[mask1], d1.batch)                # drop padded atoms, then pool
        pooled_2to1 = self.pool(attn_2to1[mask2], d2.batch)

        cb1_proj = self.chemberta_proj(d1.cb)
        cb2_proj = self.chemberta_proj(d2.cb)
        cell_emb = self.cell_embedding(d1.cell_idx)

        combined = torch.cat([emb1, emb2, pooled_1to2, pooled_2to1, cb1_proj, cb2_proj, cell_emb], dim=1)
        return self.predictor(combined), (w_1to2, w_2to1)


MODEL_CFG = dict(num_atom_features=NODE_DIM, chemberta_dim=CHEMBERTA_DIM, n_cell_lines=n_cell_lines, hidden_dim=64)
_m = DualGATSynergy(**MODEL_CFG)
n_params = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}   (paper: 260,482)")
assert n_params == 260_482
del _m

# ===== Cell 11: Training and evaluation functions =====
def evaluate_model(model, loader):
    """RMSE, PCC and the raw predictions / targets of `model` on `loader`."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            d1, d2, target = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            pred, _ = model(d1, d2)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    pcc, _ = pearsonr(all_targets, all_preds)
    return float(rmse), float(pcc), all_preds, all_targets


def train_one_fold(model, train_loader, val_loader, checkpoint_path, num_epochs=200, patience_epochs=40):
    """Adam + Smooth-L1. ONLY the validation set drives the LR scheduler, early stopping and checkpoint selection
    (best validation PCC); the test set is never used here. Returns the best validation PCC."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss()
    best_val_pcc, epochs_since_improve = -1.0, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            d1, d2, targets = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            optimizer.zero_grad()
            predictions, _ = model(d1, d2)
            loss = criterion(predictions, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 2 == 0:                          # validate every 2 epochs
            val_rmse, val_pcc, _, _ = evaluate_model(model, val_loader)
            scheduler.step(val_pcc)
            print(f"  Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val PCC: {val_pcc:.4f}")
            if val_pcc > best_val_pcc:
                best_val_pcc, epochs_since_improve = val_pcc, 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_since_improve += 2
            if epochs_since_improve >= patience_epochs:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    return best_val_pcc

# ===== Cell 13: Helpers – name normalization, structure keys and metrics =====
import re, time, random
from scipy.stats import spearmanr
from rdkit.Chem import rdFingerprintGenerator


def _norm_name(name):
    """Make cell-line names comparable: 'HS 578T' and 'Hs578T' become the same string."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _map_unique(series, fn):
    mapping = {x: fn(x) for x in series.unique()}
    return series.map(mapping)


_flat_smiles_cache = {}

def _flat_smiles(drug):
    """SMILES without stereochemistry, so that one molecule listed under two names is counted once."""
    if drug not in _flat_smiles_cache:
        s   = drug_to_smiles.get(drug)
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        _flat_smiles_cache[drug] = Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None
    return _flat_smiles_cache[drug]


def _drug_key(drug):
    return _flat_smiles(drug) or str(drug)


def _combo_keys(df):
    """Order-free key of a record: drug A + drug B + cell line."""
    a  = _map_unique(df["drug_1"], _drug_key)
    b  = _map_unique(df["drug_2"], _drug_key)
    lo = a.where(a <= b, b)
    hi = b.where(a <= b, a)
    return lo + "|" + hi + "@" + _map_unique(df["cell_line"], _norm_name)


def _within_cell(p, t, cells, min_n=30):
    """Mean PCC / SCC inside each cell line, weighted by the number of records (removes differences between cell-line means).
    Only cell lines with at least `min_n` records are used."""
    rows = []
    for c in pd.unique(cells):
        m = (cells == c)
        if m.sum() >= min_n and np.std(p[m]) > 1e-8 and np.std(t[m]) > 1e-8:
            rows.append((int(m.sum()), pearsonr(t[m], p[m])[0], spearmanr(t[m], p[m])[0]))
    if not rows:
        return np.nan, np.nan, 0
    n = np.array([r[0] for r in rows], dtype=float)
    w = n / n.sum()
    return (float((w * np.array([r[1] for r in rows])).sum()),
            float((w * np.array([r[2] for r in rows])).sum()), len(rows))


def cross_metrics(preds, targets, raw, mask, cells=None, min_n=30):
    """Metrics of one subset of records. Returns None if there are too few records or the predictions are constant."""
    p, t, r = preds[mask], targets[mask], raw[mask]
    if len(p) < min_n or np.std(p) < 1e-8 or np.std(t) < 1e-8:
        return None
    pcc  = pearsonr(t, p)[0]
    scc  = spearmanr(t, p)[0]
    rmse = np.sqrt(mean_squared_error(t, p))
    y_cls = (r > 10).astype(int)          # the same AUC / AUPR threshold as in the five-fold experiments
    if y_cls.min() == y_cls.max():
        auc = aupr = np.nan
    else:
        auc  = roc_auc_score(y_cls, p)
        aupr = average_precision_score(y_cls, p)
    if cells is not None:
        pcc_w, scc_w, n_cells_w = _within_cell(p, t, cells[mask], min_n=min_n)
    else:
        pcc_w = scc_w = np.nan; n_cells_w = 0
    return {"n_test": int(len(p)), "pcc": pcc, "scc": scc, "rmse": rmse, "auc": auc, "aupr": aupr,
            "prevalence": float(y_cls.mean()),
            "pcc_within_cell": pcc_w, "scc_within_cell": scc_w, "n_cells_within": n_cells_w}

# ===== Cell 14: Cross-dataset split =====
XD_SPLIT_INFO = {}          # statistics of every split (used for the dataset documentation)


def build_cross_dataset_split(src_df, tgt_df, seed=42, val_frac=0.15):
    """
    src_df: source dataset (train + validation) | tgt_df: target dataset (test only; the model never sees it).
    - Validation drugs are drawn from the source drugs (cold-drug split, used only for early stopping).
    - The label scaler is fitted on the source TRAINING rows only; validation and target rows are only transformed.
      No statistic of the target dataset is used for training, checkpoint selection or labelling.
    - Only target rows whose cell line also occurs in the source training data are kept.
    """
    src_df, tgt_df = src_df.copy(), tgt_df.copy()
    n_src_rows, n_tgt_before = len(src_df), len(tgt_df)
    n_tgt_cells_before = tgt_df["cell_line"].nunique()

    src_drugs = pd.concat([src_df["drug_1"], src_df["drug_2"]]).unique()
    rng       = np.random.RandomState(seed)
    val_drugs = set(rng.choice(src_drugs, size=int(val_frac * len(src_drugs)), replace=False))
    is_val    = src_df["drug_1"].isin(val_drugs) | src_df["drug_2"].isin(val_drugs)
    train_df  = src_df[~is_val].copy()
    val_df    = src_df[is_val].copy()

    scaler = StandardScaler()
    train_df["label"] = scaler.fit_transform(train_df[["raw_target"]])
    val_df["label"]   = scaler.transform(val_df[["raw_target"]])

    # --- cell lines shared with the source training data (names matched after normalization) ---
    train_cells   = set(train_df["cell_line"])
    norm_to_train = {}
    for c in sorted(train_cells):
        norm_to_train.setdefault(_norm_name(c), c)

    def to_train_cell(c):
        return c if c in train_cells else norm_to_train.get(_norm_name(c))

    tgt_df["cell_line_orig"] = tgt_df["cell_line"]
    tgt_df["cell_line"]      = _map_unique(tgt_df["cell_line"], to_train_cell)
    n_alias = int((tgt_df["cell_line"].notna() & (tgt_df["cell_line"] != tgt_df["cell_line_orig"])).sum())
    tgt_df  = tgt_df.dropna(subset=["cell_line"]).copy()
    val_df  = val_df[val_df["cell_line"].isin(train_cells)].copy()
    assert len(tgt_df) > 0, "no cell line is shared between the two datasets - check the cell-line names."

    # --- target labels with the SOURCE scaler (not target statistics). PCC / SCC / AUC / AUPR do not depend on a
    #     fixed linear scale; RMSE is in source z-units and therefore also contains the shift between the datasets ---
    tgt_df["label"] = scaler.transform(tgt_df[["raw_target"]])

    # --- harder subsets, relative to the training data (validation drugs only served for early stopping) ---
    src_keys = set(_combo_keys(train_df))
    tgt_df["novel_combo"] = ~_combo_keys(tgt_df).isin(src_keys)

    src_flat = set(_map_unique(pd.concat([train_df["drug_1"], train_df["drug_2"]]), _drug_key))
    unseen1  = ~_map_unique(tgt_df["drug_1"], _drug_key).isin(src_flat)
    unseen2  = ~_map_unique(tgt_df["drug_2"], _drug_key).isin(src_flat)
    tgt_df["n_unseen_drugs"] = unseen1.astype(int) + unseen2.astype(int)

    info = {
        "n_src_rows": n_src_rows, "n_src_drugs": len(src_drugs), "n_val_drugs": len(val_drugs),
        "n_src_cell_lines_train": len(train_cells),
        "n_train_rows": len(train_df), "n_val_rows": len(val_df),
        "n_tgt_rows_before_filter": n_tgt_before, "n_tgt_rows": len(tgt_df),
        "n_tgt_cell_lines_before": n_tgt_cells_before, "n_tgt_cell_lines_shared": tgt_df["cell_line"].nunique(),
        "n_tgt_alias_rows": n_alias,
        "n_tgt_novel_combo": int(tgt_df["novel_combo"].sum()),
        "n_tgt_ge1_unseen_drug": int((tgt_df["n_unseen_drugs"] > 0).sum()),
        "n_tgt_both_unseen": int((tgt_df["n_unseen_drugs"] == 2).sum()),
    }
    print(f"  source -> train: {info['n_train_rows']} | validation: {info['n_val_rows']} | "
          f"source drugs: {info['n_src_drugs']} (validation: {info['n_val_drugs']})")
    print(f"  target -> {info['n_tgt_rows']} of {n_tgt_before} rows ({info['n_tgt_rows']/n_tgt_before*100:.1f}%) after the cell-line filter "
          f"| {info['n_tgt_cell_lines_shared']} of {n_tgt_cells_before} cell lines shared ({n_alias} rows matched after name normalization)")
    print(f"  target -> entirely new combinations: {info['n_tgt_novel_combo']} ({info['n_tgt_novel_combo']/len(tgt_df)*100:.1f}%) "
          f"| at least one unseen drug: {info['n_tgt_ge1_unseen_drug']} ({info['n_tgt_ge1_unseen_drug']/len(tgt_df)*100:.1f}%) "
          f"| both unseen: {info['n_tgt_both_unseen']} ({info['n_tgt_both_unseen']/len(tgt_df)*100:.1f}%)")
    return train_df, val_df, tgt_df, info
n_dc=int((df_combined.dataset_type=="drugcomb").sum())

# ===== Cell 17: Data for the cross-dataset experiments =====
# df_combined does not contain the study name or the Loewe score of DrugComb, so both are attached here.
assert len(drugcomb) == n_dc and (df_combined["dataset_type"].iloc[:n_dc] == "drugcomb").all()
assert (df_combined["drug_1"].iloc[:n_dc].values == drugcomb["drug_row"].values).all(), "row order differs"
xd_study = pd.Series(drugcomb["study_name"].astype(str).values, index=df_combined.index[:n_dc])

# Raw Loewe score of DrugComb, aligned through the original CSV row index (kept by `drugcomb`)
loewe_all = pd.to_numeric(pd.read_csv(DATA_DIR / "drugcomb_full_v1.5.csv", usecols=["synergy_loewe"],
                                      low_memory=False)["synergy_loewe"], errors="coerce")
xd_loewe = loewe_all.loc[drugcomb.index].values
assert len(xd_loewe) == n_dc

# DrugComb redistributes the O'Neil screen as its "ONEIL" study: these records are the same data as the O'Neil dataset
# and are removed from every DrugComb set used as source or target (leakage control).
xd_is_oneil = xd_study.str.lower().str.contains(r"o['_ ]?neil", regex=True)
print("Study names that look like O'Neil in DrugComb:", xd_study[xd_is_oneil].unique().tolist())
print(f"Records removed: {int(xd_is_oneil.sum()):,} of {n_dc:,}   (paper: 10,380)")
oneil_idx = xd_study.index[xd_is_oneil]

df_dc_hsa = df_combined[df_combined["dataset_type"] == "drugcomb"].copy()          # raw_target = HSA (the paper's pipeline)
df_dc_loe = df_dc_hsa.copy()
df_dc_loe["raw_target"] = xd_loewe                                                   # raw_target = Loewe
df_dc_loe = df_dc_loe.loc[~np.isnan(xd_loewe)]
df_on     = df_combined[df_combined["dataset_type"] == "oneil"].copy()               # Loewe

dc_hsa_src = df_dc_hsa[~df_dc_hsa.index.isin(oneil_idx)]
dc_loe_src = df_dc_loe[~df_dc_loe.index.isin(oneil_idx)]
print(f"DrugComb source: {len(dc_loe_src):,} records (Loewe; paper: 50,719) / {len(dc_hsa_src):,} records (HSA; paper: 50,731) "
      f"| O'Neil: {len(df_on):,} records")

# --- Jobs: (setting, direction, source, target, reuse_from, baselines) ---
# unified_loewe : both datasets use the Loewe score, so source and target labels share one synergy definition (primary setting)
# paper_protocol: the label definitions of the main pipeline (DrugComb = HSA, O'Neil = Loewe)
# In the direction O'Neil -> DrugComb the source (O'Neil, Loewe) is identical in both settings, so the models of 'unified_loewe'
# are not trained twice: they are only evaluated against the HSA labels.
_reuse = "unified_loewe" if "unified_loewe" in XD_SETTINGS else None
jobs = []
if "unified_loewe" in XD_SETTINGS:
    jobs += [("unified_loewe", "DrugComb->ONeil", dc_loe_src, df_on, None, XD_BASELINES),
             ("unified_loewe", "ONeil->DrugComb", df_on, dc_loe_src, None, XD_BASELINES)]
if "paper_protocol" in XD_SETTINGS:
    jobs += [("paper_protocol", "DrugComb->ONeil", dc_hsa_src, df_on, None, XD_BASELINES),
             ("paper_protocol", "ONeil->DrugComb", df_on, dc_hsa_src, _reuse, False if _reuse else XD_BASELINES)]
print(f"{len(jobs)} jobs x {len(XD_SEEDS)} seeds")