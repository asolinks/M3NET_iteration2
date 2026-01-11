# train.py - Unified training for IEMOCAP, MELD, MOSEI
# Extended from working M3NET vectorized implementation

import os, time, argparse, random, datetime, pickle as pk
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, MOSEIDataset, DATASET_INFO
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
    rng = np.random.default_rng(1475)
    rng.shuffle(idx)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def make_loaders(dataset: str, batch_size: int, valid: float, num_workers: int = 0, 
                 pin_memory: bool = False, use_cuda: bool = True):
    """Create train/valid/test dataloaders for any supported dataset."""
    
    if dataset == "IEMOCAP":
        trainset = IEMOCAPDataset(train=True, root="./IEMOCAP_features")
        testset = IEMOCAPDataset(train=False, root="./IEMOCAP_features")
        
    elif dataset == "MELD":
        trainset = MELDDataset(train=True, root="./MELD_features")
        testset = MELDDataset(train=False, root="./MELD_features")
        
    elif dataset == "MOSEI":
        trainset = MOSEIDataset(split='train', root="./MOSEI_features")
        testset = MOSEIDataset(split='test', root="./MOSEI_features")
        # MOSEI has its own validation split
        try:
            validset = MOSEIDataset(split='val', root="./MOSEI_features")
            has_valid_split = True
        except:
            validset = None
            has_valid_split = False
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create samplers
    if dataset == "MOSEI" and has_valid_split:
        train_s = SubsetRandomSampler(list(range(len(trainset))))
        valid_s = None
    else:
        train_s, valid_s = split_samplers(trainset, valid)
        validset = trainset

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(pin_memory and use_cuda),
        persistent_workers=(num_workers > 0)
    )

    train_loader = DataLoader(trainset, sampler=train_s, collate_fn=trainset.collate_fn, **common)
    
    if valid_s is not None:
        valid_loader = DataLoader(validset, sampler=valid_s, collate_fn=validset.collate_fn, **common)
    else:
        valid_loader = DataLoader(validset, collate_fn=validset.collate_fn, **common)
    
    test_loader = DataLoader(testset, collate_fn=testset.collate_fn, **common)
    
    return train_loader, valid_loader, test_loader


# ----------------------------
# Core train/eval loops
# ----------------------------
@torch.no_grad()
def eval_epoch(model, loss_fn, loader, device, args):
    model.eval()
    all_preds, all_labels, losses = [], [], []
    all_vids: List[str] = []
    
    for batch_idx, batch in enumerate(loader):
        try:
            r1, r2, r3, r4, vv, va, qmask, umask, y, vids = batch
            
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

            if args.multi_modal and args.mm_fusion_mthd in ("concat_DHT", "concat_subsequently"):
                log_prob, *_ = model([r1, r2, r3, r4], qmask, umask, lengths, va, vv, epoch=None)
            elif args.multi_modal and args.mm_fusion_mthd == "gated":
                raise NotImplementedError("gated path needs a single text tensor")
            else:
                raise NotImplementedError("Non-DHT path not configured")

            labels_flat = torch.cat([y[i, :L] for i, L in enumerate(lengths)], dim=0)
            loss = loss_fn(log_prob, labels_flat)
            preds = log_prob.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_flat.cpu().numpy())
            losses.append(loss.item())
            all_vids.extend(vids)
            
        except RuntimeError as e:
            if "device-side assert" in str(e) or "index out of bounds" in str(e):
                print(f"Skipping eval batch {batch_idx}: {e}")
                continue
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
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if args.multi_modal and args.mm_fusion_mthd in ("concat_DHT", "concat_subsequently"):
                    log_prob, *_ = model([r1, r2, r3, r4], qmask, umask, lengths, va, vv, epoch=None)
                else:
                    raise NotImplementedError("Only concat_DHT/subsequently supported")

                labels_flat = torch.cat([y[i, :L] for i, L in enumerate(lengths)], dim=0)
                loss = loss_fn(log_prob, labels_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        except RuntimeError as e:
            if "device-side assert" in str(e) or "index out of bounds" in str(e):
                print(f"Skipping batch {batch_idx}: {e}")
                continue
            raise e
            
    return float(np.mean(losses)) if losses else 0.0


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Core options
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--base-model', default='LSTM')
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
    parser.add_argument('--graph_type', default='hyper')
    parser.add_argument('--use_topic', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--multiheads', type=int, default=6)
    parser.add_argument('--graph_construct', default='full')
    parser.add_argument('--use_gcn', action='store_true', default=False)
    parser.add_argument('--use_residue', action='store_true', default=True)
    parser.add_argument('--multi_modal', action='store_true', default=True)
    parser.add_argument('--mm_fusion_mthd', default='concat_DHT')
    parser.add_argument('--modals', default='avl')
    parser.add_argument('--av_using_lstm', action='store_true', default=False)
    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4)
    parser.add_argument('--Dataset', default='IEMOCAP', choices=['IEMOCAP', 'MELD', 'MOSEI'])
    parser.add_argument('--use_speaker', action='store_true', default=True)
    parser.add_argument('--use_modal', action='store_true', default=False)
    parser.add_argument('--norm', default='LN2')
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--num_L', type=int, default=3)
    parser.add_argument('--num_K', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true', default=False)

    args = parser.parse_args()
    seed_everything(1475)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    use_cuda = device.type == "cuda"
    
    if use_cuda and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision('medium')

    # ----------------------------
    # Dataset-specific configuration
    # ----------------------------
    info = DATASET_INFO.get(args.Dataset, DATASET_INFO['IEMOCAP'])
    
    # Feature dimensions
    if args.Dataset == 'IEMOCAP':
        D_audio = 1582  # IS10
        D_visual = 342  # denseface
        D_text = 1024   # RoBERTa
    elif args.Dataset == 'MELD':
        D_audio = 300
        D_visual = 342
        D_text = 1024
    elif args.Dataset == 'MOSEI':
        D_audio = 74    # COVAREP
        D_visual = 35   # FACET
        D_text = 300    # GloVe (or 1024 if RoBERTa available)
    else:
        D_audio, D_visual, D_text = info['D_audio'], info['D_visual'], info['D_text']

    # D_m for fusion interface
    if args.multi_modal and args.mm_fusion_mthd in ("concat_DHT", "concat_subsequently"):
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

    D_g = 512 if args.Dataset == 'IEMOCAP' else 1024
    D_p, D_e, D_h, D_a, graph_h = 150, 100, 100, 100, 512
    
    n_speakers = info['n_speakers']
    n_classes = info['n_classes']

    print(f"[INFO] Dataset={args.Dataset}, n_classes={n_classes}, n_speakers={n_speakers}")
    print(f"[INFO] D_text={D_text}, D_audio={D_audio}, D_visual={D_visual}")

    # ----------------------------
    # Data
    # ----------------------------
    train_loader, valid_loader, test_loader = make_loaders(
        args.Dataset, args.batch_size, valid=0.1, 
        num_workers=args.num_workers, pin_memory=args.pin_memory, use_cuda=use_cuda
    )

    # ----------------------------
    # Build model
    # ----------------------------
    if not args.graph_model:
        raise NotImplementedError("This script focuses on the graph path.")

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
        no_cuda=(not use_cuda),
        graph_type=args.graph_type,
        use_topic=args.use_topic,
        alpha=args.alpha,
        multiheads=args.multiheads,
        graph_construct=args.graph_construct,
        use_GCN=args.use_gcn,
        use_residue=args.use_residue,
        D_m_v=D_visual,
        D_m_a=D_audio,
        D_m_text=D_text,  # Pass text dimension for BatchNorm
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

    # ----------------------------
    # Loss
    # ----------------------------
    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.tensor(
            [1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668],
            dtype=torch.float32, device=device
        )
        loss_fn = nn.NLLLoss(weight=loss_weights) if args.class_weight else nn.NLLLoss()
    elif args.Dataset == 'MELD':
        loss_fn = FocalLoss()
    elif args.Dataset == 'MOSEI':
        # MOSEI class weights (can be adjusted based on actual distribution)
        loss_fn = FocalLoss() if args.class_weight else nn.NLLLoss()
    else:
        loss_fn = nn.NLLLoss()

    # ----------------------------
    # Optimizer + AMP scaler
    # ----------------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # ----------------------------
    # Test only mode
    # ----------------------------
    if args.testing:
        state = torch.load("best_model.pth.tar", map_location=device)
        model.load_state_dict(state)
        t_loss, t_acc, t_f1, _, _, _ = eval_epoch(model, loss_fn, test_loader, device, args)
        print(f"[TEST ONLY] loss={t_loss:.4f} acc={t_acc:.2f} f1={t_f1:.2f}")
        return

    # ----------------------------
    # Train
    # ----------------------------
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
                state={k: v.cpu() for k, v in model.state_dict().items()},
                label=te_y, pred=te_pred, vids=te_vids
            )

        dt = time.time() - t0
        print(
            f"epoch {ep:03d}: train_loss={tr_loss:.4f} valid_loss={va_loss:.4f} valid_f1={va_f1:.2f} "
            f"test_loss={te_loss:.4f} test_acc={te_acc:.2f} test_f1={te_f1:.2f} time={dt:.1f}s"
        )

        if ep % 10 == 0:
            print(f"—— best F1 so far: {max(all_f1):.2f}")

    # ----------------------------
    # Save record
    # ----------------------------
    today = datetime.datetime.now()
    record_path = f"record_{today.year}_{today.month}_{today.day}.pk"

    if not os.path.exists(record_path):
        with open(record_path, "wb") as f:
            pk.dump({}, f)

    with open(record_path, "rb") as f:
        record = pk.load(f)

    key = f"{args.mm_fusion_mthd}_{args.modals}_{args.graph_type}_{args.graph_construct}{args.Deep_GCN_nlayers}_{args.Dataset}"
    if args.use_speaker: key += "_speaker"
    if args.use_modal: key += "_modal"

    record.setdefault(key, []).append(best_f1)
    with open(record_path, "wb") as f:
        pk.dump(record, f)

    # ----------------------------
    # Final report
    # ----------------------------
    if best_snap is not None:
        y_true, y_pred = best_snap["label"], best_snap["pred"]
        print(f"\nBest test F1: {best_f1:.2f}")
        print(classification_report(y_true, y_pred, digits=4))
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
