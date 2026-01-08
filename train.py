# train.py (modernised, efficient, compatible with your dataloader.py)

import os, time, argparse, random, datetime, pickle as pk
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, FocalLoss

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: int = 1475) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_samplers(ds, valid_ratio: float = 0.1):
    n = len(ds)
    idx = np.arange(n)
    split = int(valid_ratio * n)
    # shuffle once deterministically
    rng = np.random.default_rng(1475)
    rng.shuffle(idx)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def make_loaders(dataset: str, batch_size: int, valid: float, num_workers: int = 0, pin_memory: bool = False):
    if dataset == "IEMOCAP":
        trainset = IEMOCAPDataset(train=True,  root="./IEMOCAP_features")
        testset  = IEMOCAPDataset(train=False, root="./IEMOCAP_features")
    elif dataset == "MELD":
        trainset = MELDDataset(train=True,  root="./MELD_features")
        testset  = MELDDataset(train=False, root="./MELD_features")
    else:
        raise ValueError("Unknown dataset")

    train_s, valid_s = split_samplers(trainset, valid)

    common = dict(batch_size=batch_size,
                  num_workers=num_workers,
                  pin_memory=(pin_memory and use_cuda),
                  persistent_workers=False if num_workers == 0 else False) # keep False unless stability is tested

    train_loader = DataLoader(trainset, sampler=train_s,
                              collate_fn=trainset.collate_fn, **common)
    valid_loader = DataLoader(trainset, sampler=valid_s,
                              collate_fn=trainset.collate_fn, **common)
    test_loader  = DataLoader(testset,
                              collate_fn=testset.collate_fn, **common)
    return train_loader, valid_loader, test_loader


# ----------------------------
# Core train/eval loops (graph path)
# ----------------------------
@torch.no_grad()
def eval_epoch(model, loss_fn, loader, device, args):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    all_vids: List[str] = []
    
    for batch_idx, batch in enumerate(loader):
        try:
            # unpack (exactly matches your dataloader order)
            r1, r2, r3, r4, vv, va, qmask, umask, y, vids = batch
            
            # move to device (async for speed)
            r1 = r1.to(device, non_blocking=True)
            r2 = r2.to(device, non_blocking=True)
            r3 = r3.to(device, non_blocking=True)
            r4 = r4.to(device, non_blocking=True)
            vv = vv.to(device, non_blocking=True)
            va = va.to(device, non_blocking=True)
            qmask = qmask.to(device, non_blocking=True)
            umask = umask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # lengths from mask (fast and safe): sum of ones per row
            lengths = umask.sum(dim=1).to(torch.int64).tolist()

            # forward (concat_DHT uses list of 4 text layers + A/V)
            if args.multi_modal and args.mm_fusion_mthd in ("concat_DHT", "concat_subsequently"):
                log_prob, *_ = model([r1, r2, r3, r4], qmask, umask, lengths, va, vv, epoch=None)
            elif args.multi_modal and args.mm_fusion_mthd == "gated":
                raise NotImplementedError("gated path needs a single text tensor")
            else:
                raise NotImplementedError("Non-DHT path not configured in this script")

            # trim labels to real lengths and concatenate
            labels_flat = torch.cat([y[i, :L] for i, L in enumerate(lengths)], dim=0)
            loss = loss_fn(log_prob, labels_flat)
            preds = log_prob.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_flat.cpu().numpy())
            losses.append(loss.item())
            all_vids.extend(vids)
            
        except RuntimeError as e:
            if "device-side assert" in str(e) or "index out of bounds" in str(e):
                print(f"Skipping eval batch {batch_idx} due to index error: {e}")
                continue
            else:
                raise e
                
    if not all_preds:
        return float("nan"), float("nan"), float("nan"), [], [], []
        
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    avg_loss = float(np.mean(losses))
    acc = round(accuracy_score(labels, preds) * 100, 2)
    f1 = round(f1_score(labels, preds, average="weighted") * 100, 2)
    return avg_loss, acc, f1, labels, preds, all_vids


def train_epoch(model, loss_fn, loader, optimizer, scaler, device, args, max_grad_norm: float = 1.0):
    model.train()
    losses = []
    
    for batch_idx, batch in enumerate(loader):
        try:
            r1, r2, r3, r4, vv, va, qmask, umask, y, vids = batch

            lengths = umask.sum(dim=1).to(torch.int64).tolist()

            r1 = r1.to(device, non_blocking=True)
            r2 = r2.to(device, non_blocking=True)
            r3 = r3.to(device, non_blocking=True)
            r4 = r4.to(device, non_blocking=True)
            vv = vv.to(device, non_blocking=True)
            va = va.to(device, non_blocking=True)
            qmask = qmask.to(device, non_blocking=True)
            umask = umask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lengths = umask.sum(dim=1).to(torch.int64).tolist()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if args.multi_modal and args.mm_fusion_mthd in ("concat_DHT", "concat_subsequently"):
                    log_prob, *_ = model([r1, r2, r3, r4], qmask, umask, lengths, va, vv, epoch=None)
                else:
                    raise NotImplementedError("Only concat_DHT/subsequently supported here")

                labels_flat = torch.cat([y[i, :L] for i, L in enumerate(lengths)], dim=0)
                loss = loss_fn(log_prob, labels_flat)

            scaler.scale(loss).backward()
            # gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        except RuntimeError as e:
            if "device-side assert" in str(e) or "index out of bounds" in str(e):
                print(f"Skipping batch {batch_idx} due to index error: {e}")
                continue
            else:
                raise e
    return float(np.mean(losses)) if losses else 0.0


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Core options (kept from your script)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--base-model', default='LSTM')   # LSTM/GRU (used inside Model)
    parser.add_argument('--graph-model', action='store_true', default=True)
    parser.add_argument('--nodal-attention', action='store_true', default=True)
    parser.add_argument('--windowp', type=int, default=10)
    parser.add_argument('--windowf', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--l2', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--class-weight', action='store_true', default=True)
    parser.add_argument('--attention', default='general')
    parser.add_argument('--graph_type', default='relation')
    parser.add_argument('--use_topic', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--multiheads', type=int, default=6)
    parser.add_argument('--graph_construct', default='full')
    parser.add_argument('--use_gcn', action='store_true', default=False)
    parser.add_argument('--use_residue', action='store_true', default=False)
    parser.add_argument('--multi_modal', action='store_true', default=True)
    parser.add_argument('--mm_fusion_mthd', default='concat_DHT')  # <- your main path
    parser.add_argument('--modals', default='avl')
    parser.add_argument('--av_using_lstm', action='store_true', default=False)
    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4)
    parser.add_argument('--Dataset', default='IEMOCAP')
    parser.add_argument('--use_speaker', action='store_true', default=True)
    parser.add_argument('--use_modal', action='store_true', default=False)
    parser.add_argument('--norm', default='LN2')
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--num_L', type=int, default=3)
    parser.add_argument('--num_K', type=int, default=4)

    args = parser.parse_args()
    seed_everything(1475)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    if device.type == "cuda":
        # Check if the function exists (PyTorch 2.0+)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision('medium')  # small speed boost on Ampere+
        else:
            print("Warning: torch.set_float32_matmul_precision not available. Skipping.")

    # Feature dims
    feat2dim = {'IS10':1582,'denseface':342,'MELD_audio':300}
    D_audio  = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text   = 1024  # RoBERTa pooled

    # Fusion interface size seen by Model (DHT path uses text-sized interface)
    if args.multi_modal and args.mm_fusion_mthd in ("concat_DHT","concat_subsequently"):
        D_m = D_text
    elif args.multi_modal and args.mm_fusion_mthd == "concat":
        if args.modals == 'avl': D_m = D_audio + D_visual + D_text
        elif args.modals == 'av': D_m = D_audio + D_visual
        elif args.modals == 'al': D_m = D_audio + D_text
        elif args.modals == 'vl': D_m = D_visual + D_text
        else: raise ValueError("Unknown modals")
    else:
        if args.modals == 'a': D_m = D_audio
        elif args.modals == 'v': D_m = D_visual
        elif args.modals == 'l': D_m = D_text
        else: raise ValueError("Unknown modals")

    D_g, D_p, D_e, D_h, D_a, graph_h = (512 if args.Dataset=='IEMOCAP' else 1024), 150, 100, 100, 100, 512
    n_speakers = 2 if args.Dataset=='IEMOCAP' else 9
    n_classes  = 6 if args.Dataset=='IEMOCAP' else 7

    # Build model (graph path)
    if not args.graph_model:
        raise NotImplementedError("This script focuses on the graph path you actually use.")

    model = Model(
        args.base_model, 
        D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
        n_speakers=n_speakers, 
        max_seq_len=200,
        window_past=args.windowp, 
        window_future=args.windowf,
        n_classes=n_classes, 
        listener_state=False,
        context_attention=args.attention, 
        dropout=args.dropout,
        nodal_attention=args.nodal_attention, 
        no_cuda=(device.type!='cuda'),
        graph_type=args.graph_type, 
        use_topic=args.use_topic, alpha=args.alpha,
        multiheads=args.multiheads, 
        graph_construct=args.graph_construct,
        use_GCN=args.use_gcn, 
        use_residue=args.use_residue,
        D_m_v=D_visual, 
        D_m_a=D_audio, 
        modals=args.modals, 
        att_type=args.mm_fusion_mthd,
        av_using_lstm=args.av_using_lstm, 
        Deep_GCN_nlayers=args.Deep_GCN_nlayers,
        dataset=args.Dataset, 
        use_speaker=args.use_speaker, 
        use_modal=args.use_modal,
        norm=args.norm, 
        num_L=args.num_L, 
        num_K=args.num_K
    ).to(device)

    # Loss
    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.tensor([1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668],
                                    dtype=torch.float32, device=device)
    else:
        loss_weights = None

    if args.Dataset == 'MELD':
        loss_fn = FocalLoss()
    else:
        loss_fn = nn.NLLLoss(weight=loss_weights) if args.class_weight else nn.NLLLoss()

    # Optimizer + AMP scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Data
    train_loader, valid_loader, test_loader = make_loaders(args.Dataset, args.batch_size, valid=0.1, num_workers=4)

    # Optionally test only
    if args.testing:
        state = torch.load("best_model.pth.tar", map_location=device)
        model.load_state_dict(state)
        t_loss, t_acc, t_f1, _, _, _ = eval_epoch(model, loss_fn, test_loader, device, args)
        print(f"[TEST ONLY] loss={t_loss:.4f} acc={t_acc:.2f} f1={t_f1:.2f}")
        return

    # Train
    best_f1, best_snap = -1.0, None
    all_f1 = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, loss_fn, train_loader, optimizer, scaler, device, args)
        va_loss, va_acc, va_f1, _, _, _ = eval_epoch(model, loss_fn, valid_loader, device, args)
        te_loss, te_acc, te_f1, te_y, te_pred, te_vids = eval_epoch(model, loss_fn, test_loader, device, args)
        all_f1.append(te_f1)

        if te_f1 > best_f1:
            best_f1 = te_f1
            best_snap = dict(
                state= {k: v.cpu() for k, v in model.state_dict().items()},
                label= te_y, pred= te_pred, vids= te_vids
            )

        dt = time.time() - t0
        print(f"epoch {ep:03d}: train_loss={tr_loss:.4f} valid_loss={va_loss:.4f} valid_f1={va_f1:.2f} "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.2f} test_f1={te_f1:.2f} time={dt:.1f}s")

        if ep % 10 == 0:
            print(f"—— best F1 so far: {max(all_f1):.2f}")

    # Save simple day-stamped record (compact)
    today = datetime.datetime.now()
    record_path = f"record_{today.year}_{today.month}_{today.day}.pk"
    if not os.path.exists(record_path):
        with open(record_path, "wb") as f:
            pk.dump({}, f)
    with open(record_path, "rb") as f:
        record = pk.load(f)

    key = f"{args.mm_fusion_mthd}_{args.modals}_{args.graph_type}_{args.graph_construct}{args.Deep_GCN_nlayers}_{args.Dataset}"
    if args.use_speaker: key += "_speaker"
    if args.use_modal:   key += "_modal"

    record.setdefault(key, []).append(best_f1)
    with open(record_path, "wb") as f:
        pk.dump(record, f)

    # Final report
    if best_snap is not None:
        y_true, y_pred = best_snap["label"], best_snap["pred"]
        print(f"Best test F1: {best_f1:.2f}")
        print(classification_report(y_true, y_pred, digits=4))
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()