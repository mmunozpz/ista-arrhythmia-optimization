"""
Bag-of-Codes + Logistic Regression with (lambda/p)||w||_p^p via ISTA (with backtracking)

 python opti_approach.py \
      --trainval-file tvtsv_trainval_2.npy \
      --test-file tvtsv_test_2.npy \
      --label-encoder tvtsv_test_2_label_encoder.npy \
      --pretrained-model ./models/canelita_full_12/best.pt \
      --hidden-size 64 \
      --k 128 \
      --enc-layernorm \
      --device cuda \
      --output-folder ./results/canelita_full_12/boc_ista_results_p_l \
      --lambda-grid 0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2 \
      --max-iter 5000 \
      --k-folds 5

"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from modules_v2 import VectorQuantizedVAE

# VQ-VAE loader


def load_vqvae(model_path, device, hidden_size, k, enc_layernorm=False):
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = VectorQuantizedVAE(12, hidden_size, k,
                               use_enc_layernorm=enc_layernorm,
                               debug_shapes=False).to(device)
    try:
        model.load_state_dict(state, strict=True)
        return model
    except RuntimeError as e:
        msg = str(e)
        if "Unexpected key(s) in state_dict" in msg and "encoder." in msg:
            alt_flag = not enc_layernorm
            model_alt = VectorQuantizedVAE(12, hidden_size, k,
                                           use_enc_layernorm=alt_flag,
                                           debug_shapes=False).to(device)
            model_alt.load_state_dict(state, strict=True)
            print(
                f"[INFO] Auto-switched use_enc_layernorm → {alt_flag} to match checkpoint.")
            return model_alt
        else:
            raise

# construct BoC


@torch.no_grad()
def extract_boc_features(vqvae_model, ecgs, device, K, normalize=True, log1p=False):
    vqvae_model.eval()
    feats = []
    for ecg in ecgs:
        x = torch.tensor(ecg, dtype=torch.float32, device=device)
        x = x.permute(1, 0).unsqueeze(0).unsqueeze(2)
        _, _, _, indices = vqvae_model(x)
        idx = indices.view(-1)
        hist = torch.bincount(idx, minlength=K).float()
        if normalize:
            total = hist.sum().clamp_min(1.0)
            hist = hist / total
        if log1p:
            hist = torch.log1p(hist)
        feats.append(hist.cpu().numpy())
    return np.vstack(feats)

# logistic loss and its gradient


def logistic_loss_and_grad(w, b, X, y):
    y_pm = 2*y - 1
    z = X @ w + b
    yz = y_pm * z
    f = np.mean(np.log1p(np.exp(-yz)))
    sigma = 1.0 / (1.0 + np.exp(yz))
    grad_w = - (y_pm * sigma) @ X / X.shape[0]
    grad_b = - np.mean(y_pm * sigma)
    return f, grad_w, grad_b

# psi value


def psi_value(w, lam, p):
    return (lam/p)*np.sum(np.abs(w)**p)

# proximal operator update


def prox_lp(v, alpha, lam, p, newton_tol=1e-10, newton_maxit=50):
    if lam == 0 or alpha == 0:
        return v.copy()
    u = v.copy()
    if p == 1.0:
        # lasso
        thr = alpha * lam
        return np.sign(u) * np.maximum(np.abs(u) - thr, 0.0)
    if p == 2.0:
        # ridge
        return u / (1.0 + alpha * lam)

    # 1 < p < 2
    al = alpha * lam
    x = np.empty_like(u)
    absu = np.abs(u)
    sgn = np.sign(u)

    # Coordinates where v == 0 -> solution 0 (by symmetry/optimality)
    zero_mask = (absu == 0)
    x[zero_mask] = 0.0

    # For non-zero v, solve s >= 0: s + al * s^{p-1} = a
    idx = ~zero_mask
    a = absu[idx]

    # Good initialization: s0 = max(a - al, 0) works for p near 1; for stability, start at a/(1+al)
    s = a / (1.0 + al)

    for _ in range(newton_maxit):
        # phi(s) = s + al*s^{p-1} - a
        spm1 = np.power(s, p-1)  # s^{p-1}
        phi = s + al * spm1 - a
        # phi'(s) = 1 + al*(p-1)*s^{p-2}
        spm2 = np.power(s, p-2, where=s > 0, out=np.zeros_like(s))
        dphi = 1.0 + al * (p-1.0) * spm2
        step = phi / dphi
        s_new = s - step
        # Project to s>=0 to keep feasibility
        s_new = np.maximum(s_new, 0.0)
        if np.max(np.abs(s_new - s)) < newton_tol:
            s = s_new
            break
        s = s_new

    x[idx] = sgn[idx] * s
    return x

# ista


def ista_logreg_lp(X, y, lam, p, max_iter=500, tol=1e-6, L0=1e-6, L_mult=2.0, standardize=False, verbose=False):
    """
    Returns dict with: w, b, losses, iters
    """
    N, K = X.shape
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xn = scaler.fit_transform(X)
    else:
        scaler = None
        Xn = X

    # init w,b = 0
    w = np.zeros(K, dtype=np.float64)
    b = 0.0

    # helper: full objective
    def full_obj(w, b):
        f, _, _ = logistic_loss_and_grad(w, b, Xn, y)
        return f + psi_value(w, lam, p)

    losses = []
    L = L0
    for t in range(max_iter):
        f, g_w, g_b = logistic_loss_and_grad(w, b, Xn, y)

        # Backtracking on L
        while True:
            # gradient step
            w_bar = w - (1.0 / L) * g_w
            b_bar = b - (1.0 / L) * g_b

            # prox on w only
            w_new = prox_lp(w_bar, alpha=1.0/L, lam=lam, p=p)
            b_new = b_bar  # sin regularizar b

            # Check majorization: f(y) <= f(x) + <g, y-x> + (L/2)||y-x||^2
            dw = w_new - w
            db = b_new - b
            # RHS (Taylor + quadratic)
            rhs = f + g_w.dot(dw) + g_b * db + 0.5 * L * (dw.dot(dw) + db*db)
            f_new, _, _ = logistic_loss_and_grad(w_new, b_new, Xn, y)

            if f_new <= rhs + 1e-12:
                break
            L *= L_mult  # increase L and retry

        w, b = w_new, b_new
        obj = f_new + psi_value(w, lam, p)
        losses.append(obj)

        if verbose and (t % 20 == 0 or t == max_iter-1):
            print(f"[ISTA] it={t:03d} L={L:.3e} obj={obj:.6f}")

        # stop if small step
        if np.linalg.norm(dw)**2 + db*db < tol**2:
            break

    return {
        "w": w, "b": b, "losses": losses, "iters": len(losses),
        "scaler_mean": (scaler.mean_ if scaler is not None else None),
        "scaler_scale": (scaler.scale_ if scaler is not None else None),
        "standardize": standardize
    }


def predict_logits(X, model):
    if model["standardize"]:
        mu, sc = model["scaler_mean"], model["scaler_scale"]
        X = (X - mu) / sc
    return X @ model["w"] + model["b"]


def predict_labels(X, model):
    z = predict_logits(X, model)
    # logistic prob = sigmoid(z); threshold 0.5
    return (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)


def plot_confusion_matrix(y_true, y_pred, out_dir, title, acc, class_names, fname):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16})
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title(f"{title} - Acc={acc:.4f}", fontsize=16)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def cv_select_lambda_ista(X, y, groups, lambdas, p, k_folds=5,
                          max_iter=500, tol=1e-6, L0=1e-6, standardize=False, verbose=False):
    gkf = GroupKFold(n_splits=k_folds)
    best_lam, best_mean = None, -1.0
    results = []
    for lam in lambdas:
        fold_accs = []
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
            model = ista_logreg_lp(
                X[tr], y[tr], lam=lam, p=p, max_iter=max_iter, tol=tol, L0=L0,
                standardize=standardize, verbose=False
            )
            preds = predict_labels(X[va], model)
            acc = accuracy_score(y[va], preds)
            fold_accs.append(float(acc))
            if verbose:
                print(f"[CV λ={lam:.2e}] fold {fold}/{k_folds} acc={acc:.4f}")
        mean_acc = float(np.mean(fold_accs))
        results.append(
            {"lambda": lam, "fold_accs": fold_accs, "mean_acc": mean_acc})
        if mean_acc > best_mean:
            best_mean, best_lam = mean_acc, lam
        print(f"[CV] λ={lam:.2e} mean acc={mean_acc:.4f}")
    return best_lam, results


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load VQ-VAE
    vqvae = load_vqvae(args.pretrained_model, device, args.hidden_size, args.k,
                       enc_layernorm=args.enc_layernorm)
    vqvae.eval()

    # Load splits
    trainval = np.load(args.trainval_file, allow_pickle=True).item()
    test = np.load(args.test_file, allow_pickle=True).item()
    X_tv_raw, y_tv, ids_tv = trainval["ecgs"], trainval["labels"], trainval["ids"]
    X_test_raw, y_test, ids_test = test["ecgs"], test["labels"], test["ids"]

    # BoC features
    X_tv = extract_boc_features(vqvae, X_tv_raw, device, args.k,
                                normalize=not args.no_norm, log1p=args.log1p)
    X_test = extract_boc_features(vqvae, X_test_raw, device, args.k,
                                  normalize=not args.no_norm, log1p=args.log1p)
    np.save(os.path.join(args.output_folder, "boc_trainval.npy"),
            {"X": X_tv, "y": y_tv, "ids": ids_tv})
    np.save(os.path.join(args.output_folder, "boc_test.npy"),
            {"X": X_test, "y": y_test, "ids": ids_test})
    # Lambda grid
    if args.lambda_grid.strip():
        lambdas = np.array([float(s) for s in args.lambda_grid.split(",")])
    else:
        lambdas = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3])

    # CV doble sobre p y lambda
    p_grid = [1.0, 1.25, 1.5, 1.75, 2.0]
    full_cv_results = []
    best_score = -1
    best_pair = None

    for p in p_grid:
        print("="*60)
        print(f"=== CV for p={p} ===")
        best_lam, cv_results = cv_select_lambda_ista(
            X_tv, y_tv, ids_tv, lambdas, p=p, k_folds=args.k_folds,
            max_iter=args.max_iter, tol=args.tol, L0=args.L0,
            standardize=args.standardize, verbose=args.verbose
        )

        # score del mejor lambda
        best_cv = [r for r in cv_results if r["lambda"]
                   == best_lam][0]["mean_acc"]

        full_cv_results.append({
            "p": float(p),
            "best_lambda": float(best_lam),
            "best_cv_acc": float(best_cv),
            "cv_results": cv_results
        })

        if best_cv > best_score:
            best_score = best_cv
            best_pair = (p, best_lam)

    save_json(full_cv_results,
              os.path.join(args.output_folder, "cv_summary_p_lambda.json"))

    best_p, best_lam = best_pair
    print(f"\n[GLOBAL BEST] p={best_p} λ={best_lam} score={best_score:.4f}")

    # train final model with best params
    model = ista_logreg_lp(
        X_tv, y_tv, lam=best_lam, p=best_p,
        max_iter=args.max_iter, tol=args.tol, L0=args.L0,
        standardize=args.standardize, verbose=args.verbose
    )

    # Save model + meta
    meta = {
        "best_p": float(best_p),
        "best_lambda": float(best_lam),
        "iters": int(model["iters"]),
        "standardize": bool(args.standardize),
        "no_norm_hist": bool(args.no_norm),
        "log1p_hist": bool(args.log1p)
    }
    save_json(meta, os.path.join(args.output_folder, "model_meta.json"))

    # Test eval
    y_pred = predict_labels(X_test, model)
    acc = accuracy_score(y_test, y_pred)

    # confusion matrix for the best model
    if args.label_encoder is not None and os.path.exists(args.label_encoder):
        class_names = list(np.load(args.label_encoder, allow_pickle=True))
    else:
        class_names = ["Class0", "Class1"]

    plot_confusion_matrix(
        y_test,
        y_pred,
        args.output_folder,
        title=f"BoC+ISTA (p={best_p}, λ={best_lam})",
        acc=acc,
        class_names=class_names,
        fname="confusion_matrix_test.png"
    )

    with open(os.path.join(args.output_folder, "final_test_metrics.txt"), "w") as f:
        f.write(f"Final Test Accuracy: {acc:.6f}\n")
        f.write(f"Best p: {best_p}\n")
        f.write(f"Best lambda: {best_lam}\n")

    print(f"[RESULT] Test accuracy: {acc:.4f}")

    print("[DONE] Artifacts saved to:", args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BoC + Logistic with (lambda/p)||w||_p^p via ISTA")
    parser.add_argument("--trainval-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--label-encoder", type=str, default=None)
    parser.add_argument("--pretrained-model", type=str, required=True)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--enc-layernorm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-folder", type=str,
                        default="boc_ista_results")

    parser.add_argument("--p", type=float, default=2.0,
                        help="norm exponent p (1 for L1, 2 for L2, 1<p<2 allowed)")
    parser.add_argument("--lambda-grid", type=str,
                        default="0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2")
    parser.add_argument("--k-folds", type=int, default=5)

    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--L0", type=float, default=1e-6)
    parser.add_argument("--standardize", action="store_true",
                        help="standardize X before optimization")
    parser.add_argument("--no-norm", action="store_true",
                        help="disable BoC normalization")
    parser.add_argument("--log1p", action="store_true",
                        help="apply log1p to BoC")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
