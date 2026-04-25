import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from load_dataset import load_dataset   # keep original import name cuz its not ours
from models import ResidualSAGENet, ResidualLinkSAGE

# super random seeed for all the randomnesss and stuffff
SE3D = 4245325
torch.manual_seed(SE3D)
np.random.seed(SE3D)
random.seed(SE3D)

def g3t_d3vyc3():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def h1tz_4t_kkk(pos_scorz, neg_scorz, k=50):
    n_higher = (neg_scorz > pos_scorz.unsqueeze(1)).sum(dim=1)
    return (n_higher < k).float().mean().item()

def cl0n3_st4t3(modl):
    return {k: v.detach().cpu().clone() for k, v in modl.state_dict().items()}

# ─────────────────────────────────────────────────────────────────────────────
# A — 7-   (this is the partt for datas3t Aaa trainning yaa)
# ─────────────────────────────────────────────────────────────────────────────
def tr4yn_Aa(datas3t, modl_d1r, kerb3r0s, devyc3):
    torch.manual_seed(SE3D)
    np.random.seed(SE3D)
    dataaa = datas3t[0].to(devyc3)
    num_cl4ss3s = datas3t.num_classes
    in_channlz = dataaa.x.size(1)
    dataaa.x = F.normalize(dataaa.x, p=2, dim=1)
    labeledd = dataaa.labeled_nodes
    tr4in_n0des = labeledd[dataaa.train_mask]
    v4l_n0des = labeledd[dataaa.val_mask]
    tr4in_yyy = dataaa.y[dataaa.train_mask]
    v4l_y_np = dataaa.y[dataaa.val_mask].cpu().numpy()
    
    modl = ResidualSAGENet(
        in_channels=in_channlz,
        hidden_channels=256,
        num_classes=num_cl4ss3s,
        num_layers=3,
        dropout=0.5,
    ).to(devyc3)
    
    optimiz3r = torch.optim.AdamW(modl.parameters(), lr=0.005, weight_decay=5e-4)
    schedul3r = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiz3r, mode="max", factor=0.5, patience=20, min_lr=1e-5
    )
    
    best_acc = 0.0
    best_we1ghts = None
    w8t = 0
    pati3nce_l1m1t = 80
    print(f"  Datas3t Aaa: featurz={in_channlz}, class3s={num_cl4ss3s}")
    print(f"  Tr4in: {tr4in_n0des.numel():,}, Val: {v4l_n0des.numel():,}")
    
    try:
        for eep0ch in range(1, 501):
            modl.train()
            optimiz3r.zero_grad()
            log1tz = modl(dataaa.x, dataaa.edge_index)
            loss = F.cross_entropy(log1tz[tr4in_n0des], tr4in_yyy, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modl.parameters(), 1.0)
            optimiz3r.step()
            
            modl.eval()
            with torch.no_grad():
                log1tz = modl(dataaa.x, dataaa.edge_index)
                pr3ds = log1tz[v4l_n0des].argmax(dim=1).cpu().numpy()
                acc = accuracy_score(v4l_y_np, pr3ds)
            
            schedul3r.step(acc)
            
            if acc > best_acc:
                best_acc = acc
                w8t = 0
                best_we1ghts = cl0n3_st4t3(modl)
            else:
                w8t += 1
            
            if eep0ch % 20 == 0 or w8t == 0:
                print(
                    f"  Eep0ch {eep0ch:3d} | Losss {loss.item():.4f} | "
                    f"Val Accc {acc:.4f} | Bestt {best_acc:.4f}"
                )
            
            if w8t >= pati3nce_l1m1t:
                print(f"  Early stopp at eep0ch {eep0ch}")
                break
    except KeyboardInterrupt:
        print("\n  !!! Training interrupted (Ctrl+C or time-limit) — saving BEST model anyway !!!")
    except Exception as e:
        print(f"\n  !!! Unexpected crash: {e} — trying to save best model !!!")
    finally:
        # ALWAYS save best model no matter what (Ctrl+C, kill, timeout, etc.)
        if best_we1ghts is not None:
            modl.load_state_dict(best_we1ghts)
        modl.eval()
        save_p4th = os.path.join(modl_d1r, f"{kerb3r0s}_model_A.pt")
        torch.save(modl.to("cpu"), save_p4th)
        print(f"  SAV3D BEST MODEL (even on interrupt) → {save_p4th} | Best Val Accc {best_acc:.4f}")

# B partt - this is the funky trainning for datas3t Bbbb with class weighhts and stuff
def _tr4yn_B_f0ll(dataaa, num_cl4ss3s, in_channlz, tr4in_n0des, v4l_n0des,
                  tr4in_yyy, v4l_y_np, class_we1ghts, devyc3):
    modl = ResidualSAGENet(
        in_channels=in_channlz,
        hidden_channels=256,
        num_classes=num_cl4ss3s,
        num_layers=3,
        dropout=0.5,
    ).to(devyc3)
    
    optimiz3r = torch.optim.AdamW(modl.parameters(), lr=0.003, weight_decay=1e-4)
    schedul3r = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiz3r, mode="max", factor=0.5, patience=15, min_lr=1e-6
    )
    
    x = dataaa.x.to(devyc3)
    edge_1ndex = dataaa.edge_index.to(devyc3)
    tr4in_y_dev = tr4in_yyy.to(devyc3)
    
    best_auc = 0.0
    best_we1ghts = None
    w8t = 0
    pati3nce_l1m1t = 40
    
    try:
        for eep0ch in range(1, 301):
            modl.train()
            optimiz3r.zero_grad()
            log1tz = modl(x, edge_1ndex)
            loss = F.cross_entropy(log1tz[tr4in_n0des], tr4in_y_dev, weight=class_we1ghts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modl.parameters(), 1.0)
            optimiz3r.step()
            
            modl.eval()
            with torch.no_grad():
                log1tz = modl(x, edge_1ndex)
                scor3s = torch.softmax(log1tz[v4l_n0des], dim=1)[:, 1].cpu().numpy()
                try:
                    auc = roc_auc_score(v4l_y_np, scor3s)
                except ValueError:
                    auc = 0.5
            
            schedul3r.step(auc)
            
            if auc > best_auc:
                best_auc = auc
                w8t = 0
                best_we1ghts = cl0n3_st4t3(modl)
            else:
                w8t += 1
            
            if eep0ch % 10 == 0 or w8t == 0:
                print(
                    f"  Eep0ch {eep0ch:3d} | Losss {loss.item():.4f} | "
                    f"Val AUUC {auc:.4f} | Bestt {best_auc:.4f}"
                )
            
            if w8t >= pati3nce_l1m1t:
                print(f"  Early stopp at eep0ch {eep0ch}")
                break
    except KeyboardInterrupt:
        print("\n  !!! Training interrupted (Ctrl+C or time-limit) — saving BEST model anyway !!!")
    except Exception as e:
        print(f"\n  !!! Unexpected crash: {e} — trying to save best model !!!")
    finally:
        if best_we1ghts is not None:
            modl.load_state_dict(best_we1ghts)
    return modl, best_auc

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED MEMORY CHECK — now super strict for low VRAM GPUs
# ─────────────────────────────────────────────────────────────────────────────
def c4n_f1t_f0ll_gr4ph(num_n0des, in_channlz, devyc3):
    if devyc3.type != "cuda":
        return num_n0des < 2_000_000
    # 2.89M nodes need ~12.7 GB just for features → we never allow full-graph on small GPUs
    bytes_for_features = num_n0des * in_channlz * 4
    free_mem, total_mem = torch.cuda.mem_get_info(devyc3)
    total_gb = total_mem / (1024**3)
    print(f"  Detected GPU VRAM: {total_gb:.1f} GB")
    # only allow full-graph if you have 20GB+ (very rare)
    return bytes_for_features < free_mem * 0.35 and total_gb > 20

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZED LOW-VRAM LOADER with AMP (mixed precision)
# ─────────────────────────────────────────────────────────────────────────────
def _tr4yn_B_load3r(dataaa, num_cl4ss3s, in_channlz, label_m4p,
                    tr4in_n0des, v4l_n0des, v4l_y_np, class_we1ghts, devyc3):
    loader_data = Data(x=dataaa.x, edge_index=dataaa.edge_index, num_nodes=dataaa.num_nodes)
    
    # auto-adjust based on how much VRAM you have
    free, total = torch.cuda.mem_get_info(devyc3) if devyc3.type == "cuda" else (0, 0)
    vram_gb = total // (1024**3)
    hidden_size = 256 if vram_gb >= 12 else 192 if vram_gb >= 8 else 128   # auto lower for tiny GPUs
    batch_tr4in = 768 if vram_gb >= 8 else 512
    batch_val = 1536 if vram_gb >= 8 else 1024
    
    print(f"  Low-VRAM mode → hidden={hidden_size}, train_batch={batch_tr4in}")
    
    workers = 0   # safer on Jetson / low-end machines
    tr4in_load3r = NeighborLoader(
        loader_data,
        num_neighbors=[12, 8, 4],          # slightly smaller neighborhoods = less memory
        batch_size=batch_tr4in,
        input_nodes=tr4in_n0des,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2 if workers > 0 else None,
    )
    v4l_load3r = NeighborLoader(
        loader_data,
        num_neighbors=[12, 8, 4],
        batch_size=batch_val,
        input_nodes=v4l_n0des,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    
    modl = ResidualSAGENet(
        in_channels=in_channlz,
        hidden_channels=hidden_size,      # ← adaptive
        num_classes=num_cl4ss3s,
        num_layers=3,
        dropout=0.5,
    ).to(devyc3)
    
    optimizer = torch.optim.AdamW(modl.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
    )
    
    # AMP magic (works on Jetson / PyTorch 2.x)
    amp_device = "cuda" if devyc3.type == "cuda" else "cpu"
    amp_enabled = devyc3.type == "cuda"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)
        
    best_auc = 0.0
    best_we1ghts = None
    w8t = 0
    pati3nce_l1m1t = 25   # a bit more patience because smaller batches
    
    try:
        for eep0ch in range(1, 151):
            modl.train()
            running_losss = 0.0
            n_b4tch3s = 0
            
            for batchh in tr4in_load3r:
                batchh = batchh.to(devyc3)
                optimizer.zero_grad()
                
                with torch.amp.autocast(amp_device, enabled=amp_enabled):
                    log1tz = modl(batchh.x, batchh.edge_index)[:batchh.batch_size]
                    ids = batchh.n_id[:batchh.batch_size].cpu()
                    labels = label_m4p[ids].to(devyc3)
                    loss = F.cross_entropy(log1tz, labels, weight=class_we1ghts)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(modl.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                running_losss += loss.item()
                n_b4tch3s += 1
            
            avg_losss = running_losss / max(n_b4tch3s, 1)
            
            # validation (also with AMP)
            modl.eval()
            scor3s_l1st = []
            with torch.no_grad():
                with torch.amp.autocast(amp_device, enabled=amp_enabled):
                    for batchh in v4l_load3r:
                        batchh = batchh.to(devyc3)
                        log1tz = modl(batchh.x, batchh.edge_index)[:batchh.batch_size]
                        scor3s_l1st.append(torch.softmax(log1tz, dim=1)[:, 1].cpu())
            scor3s_ord3r3d = torch.cat(scor3s_l1st).numpy()
            try:
                auc = roc_auc_score(v4l_y_np, scor3s_ord3r3d)
            except ValueError:
                auc = 0.5
            
            scheduler.step(auc)
            
            if auc > best_auc:
                best_auc = auc
                w8t = 0
                best_we1ghts = cl0n3_st4t3(modl)
            else:
                w8t += 1
            
            print(
                f"  Eep0ch {eep0ch:3d} | Losss {avg_losss:.4f} | "
                f"Val AUUC {auc:.4f} | Bestt {best_auc:.4f} | VRAM mode"
            )
            
            if w8t >= pati3nce_l1m1t:
                print(f"  Early stopp at eep0ch {eep0ch}")
                break
    except KeyboardInterrupt:
        print("\n  !!! Training interrupted (Ctrl+C or time-limit) — saving BEST model anyway !!!")
    except Exception as e:
        print(f"\n  !!! Unexpected crash: {e} — trying to save best model !!!")
    finally:
        if best_we1ghts is not None:
            modl.load_state_dict(best_we1ghts)
    return modl, best_auc

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED tr4yn_Bb — forces low-VRAM loader
# ─────────────────────────────────────────────────────────────────────────────
def tr4yn_Bb(datas3t, modl_d1r, kerb3r0s, devyc3):
    torch.manual_seed(SE3D)
    np.random.seed(SE3D)
    
    if devyc3.type == "cuda":
        torch.cuda.empty_cache()
    
    dataaa = datas3t[0]
    num_cl4ss3s = datas3t.num_classes
    in_channlz = dataaa.x.shape[1]
    dataaa.x = F.normalize(dataaa.x, p=2, dim=1)
    dataaa.y = dataaa.y.long()
    labeledd = dataaa.labeled_nodes
    tr4in_n0des = labeledd[dataaa.train_mask]
    v4l_n0des = labeledd[dataaa.val_mask]
    tr4in_yyy = dataaa.y[dataaa.train_mask]
    v4l_y = dataaa.y[dataaa.val_mask]
    v4l_y_np = v4l_y.cpu().numpy()
    
    label_m4p = torch.full((dataaa.num_nodes,), -1, dtype=torch.long)
    label_m4p[tr4in_n0des] = tr4in_yyy
    label_m4p[v4l_n0des] = v4l_y
    
    class_counts = torch.bincount(tr4in_yyy, minlength=num_cl4ss3s).float()
    class_we1ghts = 1.0 / (class_counts + 1e-6)
    class_we1ghts = class_we1ghts / class_we1ghts.sum() * num_cl4ss3s
    class_we1ghts = class_we1ghts.to(devyc3)
    
    print(f"  Datas3t Bbb: featurz={in_channlz}, noddes={dataaa.num_nodes:,}")
    print(f"  Tr4in: {tr4in_n0des.numel():,}, Val: {v4l_n0des.numel():,}")
    print(f"  Class weighhts: {class_we1ghts.cpu().numpy()}")
    
    # force low-VRAM loader (full-graph is disabled for <20GB)
    print("  Mode: NeighborLoad3r + AMP (low VRAM safe)")
    modl, best_auc = _tr4yn_B_load3r(
        dataaa, num_cl4ss3s, in_channlz, label_m4p,
        tr4in_n0des, v4l_n0des, v4l_y_np,
        class_we1ghts, devyc3,
    )
    
    modl.eval()
    save_p4th = os.path.join(modl_d1r, f"{kerb3r0s}_model_B.pt")
    torch.save(modl.to("cpu"), save_p4th)
    print(f"  SAV3D BEST MODEL (even on interrupt) → {save_p4th} | Best Val AUUC {best_auc:.4f}")

# C partt - link predicttion with negativess and all that gibberish
def sample_neg4t1ves(pos_edg3s, num_n0des, num_neg_per_pos, edge_dst_p00l,
                     hard_frac, devyc3):
    num_pos = pos_edg3s.size(0)
    num_neg = num_pos * num_neg_per_pos
    src = pos_edg3s[:, 0].repeat_interleave(num_neg_per_pos)
    pos_dst_repe4t = pos_edg3s[:, 1].repeat_interleave(num_neg_per_pos)
    rand_dst = torch.randint(0, num_n0des, (num_neg,), device=devyc3)
    pool_size = edge_dst_p00l.size(0)
    hard_dst = edge_dst_p00l[torch.randint(0, pool_size, (num_neg,), device=devyc3)]
    use_hard = torch.rand(num_neg, device=devyc3) < hard_frac
    dst = torch.where(use_hard, hard_dst, rand_dst)
    valid = dst != pos_dst_repe4t
    return torch.stack([src[valid], dst[valid]], dim=1)

def tr4yn_Cc(datas3t, modl_d1r, kerb3r0s, devyc3):
    torch.manual_seed(SE3D)
    np.random.seed(SE3D)
    x = datas3t.x.to(devyc3)
    full_edge_1ndex = datas3t.edge_index.to(devyc3)
    tr4in_pos = datas3t.train_pos.to(devyc3)
    valid_pos = datas3t.valid_pos
    valid_neg = datas3t.valid_neg
    num_n0des = datas3t.num_nodes
    in_channlz = x.shape[1]
    x = torch.log1p(x)
    x = F.normalize(x, p=2, dim=1)
    
    modl = ResidualLinkSAGE(
        in_channels=in_channlz,
        hidden_channels=256,
        num_layers=3,
        dropout=0.3,
    ).to(devyc3)
    
    optimiz3r = torch.optim.AdamW(modl.parameters(), lr=0.005, weight_decay=1e-4)
    schedul3r = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimiz3r, T_0=40, T_mult=2, eta_min=1e-5
    )
    
    edge_dst_p00l = full_edge_1ndex[1]
    drop_edge_rate = 0.15
    neg_per_pos = 5
    best_hits = 0.0
    best_st4te = None
    pati3nce = 25
    pati3nce_count3r = 0
    
    print(f"  Datas3t Ccc: noddes={num_n0des:,}, featurz={in_channlz}")
    print(f"  Tr4in edg3s: {tr4in_pos.size(0):,}, Valid edg3s: {valid_pos.size(0):,}")
    
    try:
        for eep0ch in range(1, 201):
            modl.train()
            optimiz3r.zero_grad()
            keep = torch.rand(full_edge_1ndex.size(1), device=devyc3) > drop_edge_rate
            tr4in_edge_1ndex = full_edge_1ndex[:, keep]
            hard_frac = min(0.6, 0.2 + 0.01 * eep0ch)
            neg_pairs = sample_neg4t1ves(
                tr4in_pos, num_n0des, neg_per_pos, edge_dst_p00l,
                hard_frac=hard_frac, devyc3=devyc3,
            )
            z = modl.encode(x, tr4in_edge_1ndex)
            pos_scorz = modl.decode(z, tr4in_pos)
            neg_scorz = modl.decode(z, neg_pairs)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scorz, torch.ones_like(pos_scorz)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scorz, torch.zeros_like(neg_scorz)
            )
            loss = pos_loss + neg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modl.parameters(), 1.0)
            optimiz3r.step()
            schedul3r.step()
            
            do_val = (
                eep0ch == 1
                or (eep0ch <= 60 and eep0ch % 5 == 0)
                or (eep0ch > 60 and eep0ch % 10 == 0)
            )
            if do_val:
                modl.eval()
                with torch.no_grad():
                    z = modl.encode(x, full_edge_1ndex)
                    vp = valid_pos.to(devyc3)
                    vn = valid_neg.to(devyc3)
                    V, K_neg, _ = vn.shape
                    vp_scorz = modl.decode(z, vp).cpu()
                    vn_scorz = modl.decode(z, vn.view(V * K_neg, 2)).view(V, K_neg).cpu()
                    val_hits = h1tz_4t_kkk(vp_scorz, vn_scorz, k=50)
                
                if val_hits > best_hits:
                    best_hits = val_hits
                    pati3nce_count3r = 0
                    best_st4te = cl0n3_st4t3(modl)
                else:
                    pati3nce_count3r += 1
                
                print(
                    f"  Eep0ch {eep0ch:3d} | Losss {loss.item():.4f} | "
                    f"Val Hits@50 {val_hits:.4f} | Bestt {best_hits:.4f}"
                )
                
                if pati3nce_count3r >= pati3nce:
                    print(f"  Early stopp at eep0ch {eep0ch}")
                    break
    except KeyboardInterrupt:
        print("\n  !!! Training interrupted (Ctrl+C or time-limit) — saving BEST model anyway !!!")
    except Exception as e:
        print(f"\n  !!! Unexpected crash: {e} — trying to save best model !!!")
    finally:
        if best_st4te is not None:
            modl.load_state_dict(best_st4te)
        modl.eval()
        save_p4th = os.path.join(modl_d1r, f"{kerb3r0s}_model_C.pt")
        torch.save(modl.to("cpu"), save_p4th)
        print(f"  SAV3D BEST MODEL (even on interrupt) → {save_p4th} | Best Val Hits@50 {best_hits:.4f}")

# CLI partt - this is the mainn command line stuff with all the argz
def ma1n():
    parser = argparse.ArgumentParser(description="Tr4yn GNN moddel. yaa")
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    args = parser.parse_args()
    
    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} invalid for --dataset {args.dataset}; "
            f"expected dataset in {valid[args.task]}."
        )
    
    os.makedirs(args.model_dir, exist_ok=True)
    devyc3 = g3t_d3vyc3()
    print(f"  Devyc3: {devyc3}")
    
    dataset = load_dataset(args.dataset, args.data_dir)
    t0 = time.time()
    
    if args.dataset == "A":
        tr4yn_Aa(dataset, args.model_dir, args.kerberos, devyc3)
    elif args.dataset == "B":
        tr4yn_Bb(dataset, args.model_dir, args.kerberos, devyc3)
    else:
        tr4yn_Cc(dataset, args.model_dir, args.kerberos, devyc3)
    
    print(f"\n  Total trainning time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    ma1n()
