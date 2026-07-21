import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def split_and_shuffle(X, y, val_frac=0.2, seed=0):
    """Hold out the last `val_frac` of rows as validation (in original order),
    then shuffle only the training rows.

    Returns X_train, y_train (shuffled), X_val, y_val.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    n_val = int(val_frac * n)

    cut = n - n_val
    X_train, y_train = X[:cut], y[:cut]      # everything up to the cut
    X_val,   y_val   = X[cut:], y[cut:]      # the bottom slice, left in order

    perm = rng.permutation(len(X_train))     # shuffle the training rows only
    return X_train[perm], y_train[perm], X_val, y_val



class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_sizes=(256, 256, 128),
                 dropout=None, batchnorm=False):
        super().__init__()

        # dropout defaults to "none anywhere"; otherwise it must line up with the layers
        if dropout is None:
            dropout = [0.0] * len(hidden_sizes)
        if len(dropout) != len(hidden_sizes):
            raise ValueError(
                f"dropout has {len(dropout)} entries but there are "
                f"{len(hidden_sizes)} hidden layers — they need to match."
            )

        layers = []
        prev = n_inputs
        for h, p in zip(hidden_sizes, dropout):
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))    # normalize before the nonlinearity
            layers.append(nn.ReLU())
            if p > 0:
                layers.append(nn.Dropout(p))
            prev = h
        layers.append(nn.Linear(prev, n_outputs))   # bare output for regression
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




class TwoBranchMLP(nn.Module):
    def __init__(self, n_in_a, n_in_b, n_outputs,
                 branch_a_hidden=(128, 64),
                 branch_b_hidden=(128, 64),
                 head_hidden=(128,),
                 activation=nn.ReLU):
        super().__init__()

        def stack(n_in, hidden):
            layers, prev = [], n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), activation()]
                prev = h
            return nn.Sequential(*layers), prev   # also hand back the output width

        self.branch_a, out_a = stack(n_in_a, branch_a_hidden)
        self.branch_b, out_b = stack(n_in_b, branch_b_hidden)

        # the head starts where the two branches meet
        self.head, head_out = stack(out_a + out_b, head_hidden)
        self.output = nn.Linear(head_out, n_outputs)

    def forward(self, x_a, x_b):
        a = self.branch_a(x_a)
        b = self.branch_b(x_b)
        merged = torch.cat([a, b], dim=1)   # join along the feature axis
        return self.output(self.head(merged))




def train_nn(
    X_train, y_train,
    epochs=20,
    mlp_args=None,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=512,
    X_val=None, y_val=None,
    patience=None,             # epochs of no val improvement before stopping; None disables
    min_delta=0.0,             # how much better counts as a real improvement
    num_workers=0,
    save_path=None,
    verbose=True,
):
    mlp_args = mlp_args or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if patience is not None and X_val is None:
        raise ValueError("early stopping needs a validation set — pass X_val and y_val.")

    X_mean, X_std = X_train.mean(0), X_train.std(0)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    X_std = np.where(X_std == 0, 1.0, X_std)
    y_std = np.where(y_std == 0, 1.0, y_std)
    scaler = {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}

    def prep(X, y):
        Xs = torch.tensor((X - X_mean) / X_std, dtype=torch.float32)
        ys = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
        return Xs, ys

    Xt, yt = prep(X_train, y_train)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)

    if X_val is not None:
        Xv, yv = prep(X_val, y_val)
        Xv, yv = Xv.to(device), yv.to(device)

    model = MLP(n_inputs=Xt.shape[1], n_outputs=yt.shape[1], **mlp_args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val = np.inf
    best_state = None            # weights at the lowest val loss seen
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_loss = running / len(Xt)
        history["train"].append(train_loss)

        val_loss = None
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xv), yv).item()
            history["val"].append(val_loss)

            # did this epoch meaningfully beat the best so far?
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

        if verbose:
            msg = f"epoch {epoch:3d} | train {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}"
            print(msg)

        if patience is not None and epochs_since_improve >= patience:
            if verbose:
                print(f"stopping early at epoch {epoch}; "
                      f"best was epoch {best_epoch} (val {best_val:.4f})")
            break

    # restore the best weights rather than keeping the last ones
    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "mlp_args": mlp_args,
                "n_inputs": Xt.shape[1],
                "n_outputs": yt.shape[1],
                "scaler": scaler,
            },
            save_path,
        )
        if verbose:
            print(f"saved model + scaler to {save_path}")

    return model, scaler, history

def load_nn(save_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(save_path, map_location=device)
    model = MLP(n_inputs=ckpt["n_inputs"], n_outputs=ckpt["n_outputs"],
                **ckpt["mlp_args"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["scaler"]



def evaluate_nn(model, scaler, X_test, y_test, batch_size=4096):
    """Run a trained MLP on a test set and report metrics in physical units."""
    device = next(model.parameters()).device
    model.eval()

    Xm, Xs = scaler["X_mean"], scaler["X_std"]
    ym, ys = scaler["y_mean"], scaler["y_std"]

    # predict in batches so a large test set doesn't overwhelm GPU memory
    preds = []
    with torch.no_grad():
        Xn = torch.tensor((X_test - Xm) / Xs, dtype=torch.float32)
        for i in range(0, len(Xn), batch_size):
            xb = Xn[i:i + batch_size].to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds) * ys + ym       # undo output scaling -> real units

    # metrics computed on the real-unit values
    err = pred - y_test
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))

    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_test - y_test.mean(axis=0)) ** 2, axis=0)
    r2_per_output = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    r2_overall = np.nanmean(r2_per_output)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_overall": r2_overall,
        "r2_per_output": r2_per_output,
    }















def train_two_branch(
    Xa_train, Xb_train, y_train,
    branch_args=None,          # dict: branch_a_hidden, branch_b_hidden, head_hidden, ...
    epochs=20,
    lr=1e-3,
    weight_decay=0.0,
    batch_size=512,
    Xa_val=None, Xb_val=None, y_val=None,
    patience=None,
    min_delta=0.0,
    num_workers=0,
    save_path=None,
    verbose=True,
):
    branch_args = branch_args or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    has_val = Xa_val is not None
    if patience is not None and not has_val:
        raise ValueError("early stopping needs a validation set — pass the val inputs.")

    # each input branch gets its own scaling; the target is scaled once
    def fit_scaler(arr):
        m, s = arr.mean(0), arr.std(0)
        return m, np.where(s == 0, 1.0, s)

    Xa_mean, Xa_std = fit_scaler(Xa_train)
    Xb_mean, Xb_std = fit_scaler(Xb_train)
    y_mean, y_std   = fit_scaler(y_train)

    scaler = {
        "Xa_mean": Xa_mean, "Xa_std": Xa_std,
        "Xb_mean": Xb_mean, "Xb_std": Xb_std,
        "y_mean":  y_mean,  "y_std":  y_std,
    }

    def prep(Xa, Xb, y):
        ta = torch.tensor((Xa - Xa_mean) / Xa_std, dtype=torch.float32)
        tb = torch.tensor((Xb - Xb_mean) / Xb_std, dtype=torch.float32)
        ty = torch.tensor((y  - y_mean)  / y_std,  dtype=torch.float32)
        return ta, tb, ty

    ta, tb, ty = prep(Xa_train, Xb_train, y_train)
    loader = DataLoader(TensorDataset(ta, tb, ty), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)

    if has_val:
        va, vb, vy = prep(Xa_val, Xb_val, y_val)
        va, vb, vy = va.to(device), vb.to(device), vy.to(device)

    model = TwoBranchMLP(n_in_a=ta.shape[1], n_in_b=tb.shape[1],
                         n_outputs=ty.shape[1], **branch_args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, epochs_since_improve = np.inf, None, -1, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xa, xb, yb in loader:
            xa, xb, yb = xa.to(device), xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xa, xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xa)
        train_loss = running / len(ta)
        history["train"].append(train_loss)

        val_loss = None
        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(va, vb), vy).item()
            history["val"].append(val_loss)
            if val_loss < best_val - min_delta:
                best_val, best_epoch, epochs_since_improve = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                epochs_since_improve += 1

        if verbose:
            msg = f"epoch {epoch:3d} | train {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}"
            print(msg)

        if patience is not None and epochs_since_improve >= patience:
            if verbose:
                print(f"stopping early at epoch {epoch}; "
                      f"best was epoch {best_epoch} (val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "branch_args": branch_args,
                "n_in_a": ta.shape[1], "n_in_b": tb.shape[1],
                "n_outputs": ty.shape[1],
                "scaler": scaler,
            },
            save_path,
        )
        if verbose:
            print(f"saved model + scaler to {save_path}")

    return model, scaler, history







def evaluate_two_branch(model, scaler, Xa_test, Xb_test, y_test, batch_size=4096):
    """Run a trained two-branch model on a test set; metrics in physical units."""
    device = next(model.parameters()).device
    model.eval()

    # unpack the per-branch input scalers and the target scaler
    Xa_m, Xa_s = scaler["Xa_mean"], scaler["Xa_std"]
    Xb_m, Xb_s = scaler["Xb_mean"], scaler["Xb_std"]
    ym,   ys   = scaler["y_mean"],  scaler["y_std"]

    Xa_n = (Xa_test - Xa_m) / Xa_s
    Xb_n = (Xb_test - Xb_m) / Xb_s

    preds = []
    with torch.no_grad():
        for i in range(0, len(Xa_n), batch_size):
            xa = torch.tensor(Xa_n[i:i + batch_size], dtype=torch.float32).to(device)
            xb = torch.tensor(Xb_n[i:i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xa, xb).cpu().numpy())
    pred = np.concatenate(preds) * ys + ym          # undo target scaling -> real units

    err = pred - y_test
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))

    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_test - y_test.mean(axis=0)) ** 2, axis=0)
    r2_per_output = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    r2_overall = np.nanmean(r2_per_output)

    return {
        "mse": mse, "rmse": rmse, "mae": mae,
        "r2_overall": r2_overall, "r2_per_output": r2_per_output,
    }









def build_pca_features(Xa_train, Xa_val=None, Xa_test=None, n_components=24):
    """Fit PCA on branch-1 training features only; transform all splits.
    Returns the reduced arrays plus the fitted objects (to save alongside the model)."""
    # PCA is variance-based, so branch 1 must be standardized first —
    # otherwise large-magnitude features dominate through their units alone.
    pre = StandardScaler().fit(Xa_train)
    pca = PCA(n_components=n_components).fit(pre.transform(Xa_train))

    def transform(X):
        return None if X is None else pca.transform(pre.transform(X))

    evr = pca.explained_variance_ratio_.sum()
    print(f"PCA: {Xa_train.shape[1]} -> {n_components} components, "
          f"retaining {evr:.1%} of branch-1 variance")

    return transform(Xa_train), transform(Xa_val), transform(Xa_test), (pre, pca)


def combine(Xa_reduced, Xb):
    """Attach the PCA components to branch-2's untouched features."""
    return np.hstack([Xa_reduced, Xb])


def train_pca_pipeline(
    Xa_train, Xb_train, y_train,
    Xa_val=None, Xb_val=None, y_val=None,
    n_components=24,
    **train_kwargs,            # epochs, mlp_args, lr, patience, save_path, ...
):
    # 1. reduce branch 1 (fit on train, apply to val)
    Xa_tr, Xa_va, _, pca_objs = build_pca_features(
        Xa_train, Xa_val, None, n_components=n_components
    )

    # 2. concatenate the components with branch 2's raw features
    X_train = combine(Xa_tr, Xb_train)
    X_val = combine(Xa_va, Xb_val) if Xa_va is not None else None

    print(f"combined feature count: {X_train.shape[1]} "
          f"({n_components} PCA + {Xb_train.shape[1]} raw)")

    # 3. train the ordinary single-branch MLP on the combined features
    model, scaler, history = train_nn(
        X_train, y_train, X_val=X_val, y_val=y_val, **train_kwargs
    )

    return model, scaler, history, pca_objs


def predict_pca_pipeline(model, scaler, pca_objs, Xa_new, Xb_new):
    pre, pca = pca_objs
    Xa_r = pca.transform(pre.transform(Xa_new))     # same fitted transform, never refit
    X = combine(Xa_r, Xb_new)

    Xm, Xs = scaler["X_mean"], scaler["X_std"]
    ym, ys = scaler["y_mean"], scaler["y_std"]

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.tensor((X - Xm) / Xs, dtype=torch.float32).to(device)
        return model(xt).cpu().numpy() * ys + ym










class ConvScalarNet(nn.Module):
    def __init__(
        self,
        n_conv_vars=4, n1=58, n_scalar=0, n_outputs=1,
        conv_channels=(16, 16), kernel_size=3,
        conv_hidden=(64,), scalar_hidden=(32,), head_hidden=(128, 64),
        dropout=0.0,                     # <- new: applied in dense stacks and after convs
        activation=nn.ReLU,
    ):
        super().__init__()
        self.n_conv_vars, self.n1, self.n_scalar = n_conv_vars, n1, n_scalar

        def dense(n_in, hidden):
            layers, prev = [], n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), activation()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            return nn.Sequential(*layers), prev

        self.conv_branches = nn.ModuleList()
        self.conv_heads = nn.ModuleList()
        merged_width = 0
        for _ in range(n_conv_vars):
            layers, in_ch = [], 1
            for out_ch in conv_channels:
                layers += [nn.Conv1d(in_ch, out_ch, kernel_size,
                                     padding=kernel_size // 2), activation()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_ch = out_ch
            self.conv_branches.append(nn.Sequential(*layers))
            head, out = dense(in_ch * n1, conv_hidden)
            self.conv_heads.append(head)
            merged_width += out

        if n_scalar > 0:
            self.scalar_branch, s_out = dense(n_scalar, scalar_hidden)
            merged_width += s_out
        else:
            self.scalar_branch = None

        self.head, head_out = dense(merged_width, head_hidden)
        self.output = nn.Linear(head_out, n_outputs)

    def forward(self, x_conv, x_scalar=None):
        feats = []
        for i in range(self.n_conv_vars):
            xi = x_conv[:, i:i + 1, :]
            z = self.conv_branches[i](xi).flatten(1)
            feats.append(self.conv_heads[i](z))
        if self.scalar_branch is not None:
            feats.append(self.scalar_branch(x_scalar))
        merged = torch.cat(feats, dim=1)
        return self.output(self.head(merged))



def train_conv_scalar(
    Xc_train, Xs_train, y_train,
    Xc_val=None, Xs_val=None, y_val=None,
    model_args=None,
    epochs=100, lr=1e-3, weight_decay=0.0, batch_size=512,
    patience=None, min_delta=0.0, num_workers=0,
    save_path=None, verbose=True,
):
    model_args = model_args or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_val = Xc_val is not None
    has_scalar = Xs_train is not None and Xs_train.shape[1] > 0
    if patience is not None and not has_val:
        raise ValueError("early stopping needs a validation set.")

    # --- scaling (fit on train only) ---
    # conv: per (variable, level) so each profile point is normalized on its own scale
    Xc_mean, Xc_std = Xc_train.mean(0), Xc_train.std(0)
    Xc_std = np.where(Xc_std == 0, 1.0, Xc_std)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    y_std = np.where(y_std == 0, 1.0, y_std)
    scaler = {"Xc_mean": Xc_mean, "Xc_std": Xc_std, "y_mean": y_mean, "y_std": y_std}
    if has_scalar:
        Xs_mean, Xs_std = Xs_train.mean(0), Xs_train.std(0)
        Xs_std = np.where(Xs_std == 0, 1.0, Xs_std)
        scaler.update({"Xs_mean": Xs_mean, "Xs_std": Xs_std})

    def prep(Xc, Xs, y):
        tc = torch.tensor((Xc - Xc_mean) / Xc_std, dtype=torch.float32)
        ty = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
        if has_scalar:
            ts = torch.tensor((Xs - Xs_mean) / Xs_std, dtype=torch.float32)
            return tc, ts, ty
        return tc, torch.zeros(len(Xc), 0), ty     # placeholder keeps the loader uniform

    tc, ts, ty = prep(Xc_train, Xs_train, y_train)
    loader = DataLoader(TensorDataset(tc, ts, ty), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    if has_val:
        vc, vs, vy = prep(Xc_val, Xs_val, y_val)
        vc, vs, vy = vc.to(device), vs.to(device), vy.to(device)

    model = ConvScalarNet(
        n_conv_vars=tc.shape[1], n1=tc.shape[2],
        n_scalar=(ts.shape[1] if has_scalar else 0),
        n_outputs=ty.shape[1], **model_args,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, since = np.inf, None, -1, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xc, xs, yb in loader:
            xc, xs, yb = xc.to(device), xs.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xc, xs if has_scalar else None)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xc)
        train_loss = running / len(tc)
        history["train"].append(train_loss)

        val_loss = None
        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(vc, vs if has_scalar else None), vy).item()
            history["val"].append(val_loss)
            if val_loss < best_val - min_delta:
                best_val, best_epoch, since = val_loss, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since += 1

        if verbose:
            msg = f"epoch {epoch:3d} | train {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}"
            print(msg)

        if patience is not None and since >= patience:
            if verbose:
                print(f"early stop at epoch {epoch}; best epoch {best_epoch} "
                      f"(val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        torch.save({"model_state": model.state_dict(), "model_args": model_args,
                    "n_conv_vars": tc.shape[1], "n1": tc.shape[2],
                    "n_scalar": (ts.shape[1] if has_scalar else 0),
                    "n_outputs": ty.shape[1], "scaler": scaler}, save_path)
        if verbose:
            print(f"saved to {save_path}")

    return model, scaler, history





def evaluate_conv_scalar(model, scaler, Xc_test, Xs_test, y_test, batch_size=4096):
    """Evaluate a trained conv-scalar model on a test set; metrics in physical units.
    Pass Xs_test=None if the model was trained without scalar inputs."""
    device = next(model.parameters()).device
    model.eval()

    has_scalar = Xs_test is not None and "Xs_mean" in scaler

    Xc_m, Xc_s = scaler["Xc_mean"], scaler["Xc_std"]
    ym,   ys   = scaler["y_mean"],  scaler["y_std"]
    Xc_n = (Xc_test - Xc_m) / Xc_s
    if has_scalar:
        Xs_n = (Xs_test - scaler["Xs_mean"]) / scaler["Xs_std"]

    preds = []
    with torch.no_grad():
        for i in range(0, len(Xc_n), batch_size):
            xc = torch.tensor(Xc_n[i:i + batch_size], dtype=torch.float32).to(device)
            xs = None
            if has_scalar:
                xs = torch.tensor(Xs_n[i:i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xc, xs).cpu().numpy())
    pred = np.concatenate(preds) * ys + ym          # undo target scaling -> real units

    err = pred - y_test
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))

    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_test - y_test.mean(axis=0)) ** 2, axis=0)
    r2_per_output = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    r2_overall = np.nanmean(r2_per_output)

    return {
        "mse": mse, "rmse": rmse, "mae": mae,
        "r2_overall": r2_overall, "r2_per_output": r2_per_output,
    }





def predict_conv_scalar(model, scaler, Xc, Xs=None, batch_size=4096):
    """Predict with a trained conv-scalar model. Returns predictions in physical units.
    Pass Xs=None if the model was trained without scalar inputs."""
    device = next(model.parameters()).device
    model.eval()

    has_scalar = Xs is not None and "Xs_mean" in scaler

    Xc_n = (Xc - scaler["Xc_mean"]) / scaler["Xc_std"]
    if has_scalar:
        Xs_n = (Xs - scaler["Xs_mean"]) / scaler["Xs_std"]

    preds = []
    with torch.no_grad():
        for i in range(0, len(Xc_n), batch_size):
            xc = torch.tensor(Xc_n[i:i + batch_size], dtype=torch.float32).to(device)
            xs = None
            if has_scalar:
                xs = torch.tensor(Xs_n[i:i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xc, xs).cpu().numpy())

    pred = np.concatenate(preds)
    return pred * scaler["y_std"] + scaler["y_mean"]      # back to physical units






class DoubleConv(nn.Module):
    """(conv -> [BN] -> ReLU) x2 — the standard U-Net block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, batchnorm=True, dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        for i in range(2):
            layers.append(nn.Conv1d(in_ch if i == 0 else out_ch, out_ch,
                                    kernel_size, padding=pad))
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        n_conv_vars=4, n1=58, n_scalar=0, n_outputs=58,
        base_channels=32,        # channels at the top level; doubles each depth
        depth=3,                 # number of down/up steps
        kernel_size=3,
        batchnorm=True,
        dropout=0.0,
        scalar_hidden=(32,),     # dense stack for the scalar embedding
        scalar_embed=16,         # width of the embedding injected at the bottleneck
    ):
        super().__init__()
        self.n1, self.depth, self.n_outputs = n1, depth, n_outputs

        # pad profile up to a length divisible by 2**depth so pooling/upsampling align
        self.n_padded = int(np.ceil(n1 / 2 ** depth) * 2 ** depth)
        self.pad_left = (self.n_padded - n1) // 2
        self.pad_right = self.n_padded - n1 - self.pad_left

        # --- encoder ---
        self.downs = nn.ModuleList()
        ch = n_conv_vars
        for d in range(depth):
            out_ch = base_channels * (2 ** d)
            self.downs.append(DoubleConv(ch, out_ch, kernel_size, batchnorm, dropout))
            ch = out_ch
        self.pool = nn.MaxPool1d(2)

        # --- bottleneck ---
        bott_ch = base_channels * (2 ** depth)
        self.bottleneck = DoubleConv(ch, bott_ch, kernel_size, batchnorm, dropout)

        # --- scalar conditioning, injected at the bottleneck ---
        self.n_scalar = n_scalar
        if n_scalar > 0:
            layers, prev = [], n_scalar
            for h in scalar_hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, scalar_embed))
            self.scalar_net = nn.Sequential(*layers)
            self.merge = nn.Conv1d(bott_ch + scalar_embed, bott_ch, 1)
        else:
            self.scalar_net = None

        # --- decoder ---
        self.ups, self.up_convs = nn.ModuleList(), nn.ModuleList()
        ch = bott_ch
        for d in reversed(range(depth)):
            skip_ch = base_channels * (2 ** d)
            self.ups.append(nn.ConvTranspose1d(ch, skip_ch, 2, stride=2))
            self.up_convs.append(DoubleConv(skip_ch * 2, skip_ch,
                                            kernel_size, batchnorm, dropout))
            ch = skip_ch

        self.out_conv = nn.Conv1d(ch, 1, 1)      # 1x1 conv -> one value per level

        # fallback if outputs aren't per-level (see notes below)
        self.final = nn.Linear(n1, n_outputs) if n_outputs != n1 else None

    def forward(self, x_conv, x_scalar=None):
        x = F.pad(x_conv, (self.pad_left, self.pad_right), mode="reflect")

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        if self.scalar_net is not None:
            emb = self.scalar_net(x_scalar)                        # (b, scalar_embed)
            emb = emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])    # broadcast along profile
            x = self.merge(torch.cat([x, emb], dim=1))

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], dim=1))

        x = self.out_conv(x).squeeze(1)                            # (b, n_padded)
        x = x[:, self.pad_left:self.pad_left + self.n1]            # crop back to n1
        return self.final(x) if self.final is not None else x



def _eval_loss(model, xc, xs, y, loss_fn, has_scalar, batch_size=256):
    """Validation loss computed in batches, weighted by batch size."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(xc), batch_size):
            xb = xc[i:i+batch_size]
            sb = xs[i:i+batch_size] if has_scalar else None
            yb = y[i:i+batch_size]
            loss = loss_fn(model(xb, sb), yb)
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n
    

def train_unet(
    Xc_train, Xs_train, y_train,
    Xc_val=None, Xs_val=None, y_val=None,
    model_args=None,
    epochs=100, lr=1e-3, weight_decay=0.0, batch_size=256,
    patience=None, min_delta=0.0, num_workers=0,
    save_path=None, verbose=True,
):
    model_args = model_args or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_val = Xc_val is not None
    has_scalar = Xs_train is not None and Xs_train.shape[1] > 0
    if patience is not None and not has_val:
        raise ValueError("early stopping needs a validation set.")

    # --- scaling: fit on train only ---
    Xc_mean, Xc_std = Xc_train.mean(0), Xc_train.std(0)
    Xc_std = np.where(Xc_std == 0, 1.0, Xc_std)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    y_std = np.where(y_std == 0, 1.0, y_std)
    scaler = {"Xc_mean": Xc_mean, "Xc_std": Xc_std, "y_mean": y_mean, "y_std": y_std}
    if has_scalar:
        Xs_mean, Xs_std = Xs_train.mean(0), Xs_train.std(0)
        Xs_std = np.where(Xs_std == 0, 1.0, Xs_std)
        scaler.update({"Xs_mean": Xs_mean, "Xs_std": Xs_std})

    def prep(Xc, Xs, y):
        tc = torch.tensor((Xc - Xc_mean) / Xc_std, dtype=torch.float32)
        ty = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
        if has_scalar:
            ts = torch.tensor((Xs - Xs_mean) / Xs_std, dtype=torch.float32)
            return tc, ts, ty
        return tc, torch.zeros(len(Xc), 0), ty

    tc, ts, ty = prep(Xc_train, Xs_train, y_train)
    loader = DataLoader(TensorDataset(tc, ts, ty), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    if has_val:
        vc, vs, vy = prep(Xc_val, Xs_val, y_val)
        vc, vs, vy = vc.to(device), vs.to(device), vy.to(device)

    model = UNet1D(
        n_conv_vars=tc.shape[1], n1=tc.shape[2],
        n_scalar=(ts.shape[1] if has_scalar else 0),
        n_outputs=ty.shape[1], **model_args,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, since = np.inf, None, -1, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xc, xs, yb in loader:
            xc, xs, yb = xc.to(device), xs.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xc, xs if has_scalar else None), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xc)
        train_loss = running / len(tc)
        history["train"].append(train_loss)

        val_loss = None
        if has_val:
            val_loss = _eval_loss(model, vc, vs, vy, loss_fn, has_scalar,
                                  batch_size=batch_size)
            history["val"].append(val_loss)

        if verbose:
            msg = f"epoch {epoch:3d} | train {train_loss:.4e}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4e}"
            print(msg)

        if patience is not None and since >= patience:
            if verbose:
                print(f"early stop at epoch {epoch}; best epoch {best_epoch} "
                      f"(val {best_val:.4e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        torch.save({"model_state": model.state_dict(), "model_args": model_args,
                    "n_conv_vars": tc.shape[1], "n1": tc.shape[2],
                    "n_scalar": (ts.shape[1] if has_scalar else 0),
                    "n_outputs": ty.shape[1], "scaler": scaler}, save_path)
        if verbose:
            print(f"saved to {save_path}")

    return model, scaler, history











class ProfileTransformer(nn.Module):
    def __init__(
        self,
        n_conv_vars=4, n1=58, n_scalar=0, n_outputs=58,
        d_model=64,              # token embedding width
        nhead=4,                 # attention heads
        num_layers=3,            # transformer encoder layers
        dim_feedforward=128,     # width of each layer's internal MLP
        conv_channels=(32,),     # conv front-end that builds tokens from the profile
        kernel_size=3,
        dropout=0.1,
        scalar_hidden=(32,),
    ):
        super().__init__()
        self.n1, self.n_scalar, self.n_outputs = n1, n_scalar, n_outputs

        # conv front-end: (b, n_conv_vars, n1) -> (b, d_model, n1), giving each
        # level local context before attention
        layers, in_ch = [], n_conv_vars
        for out_ch in conv_channels:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size,
                                 padding=kernel_size // 2), nn.ReLU()]
            in_ch = out_ch
        layers.append(nn.Conv1d(in_ch, d_model, 1))    # project to d_model
        self.tokenizer = nn.Sequential(*layers)

        # learned positional embedding — one per level, so the model knows height order
        self.pos_embed = nn.Parameter(torch.randn(1, n1, d_model) * 0.02)

        # scalar context token
        if n_scalar > 0:
            slayers, prev = [], n_scalar
            for h in scalar_hidden:
                slayers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            slayers.append(nn.Linear(prev, d_model))
            self.scalar_net = nn.Sequential(*slayers)
        else:
            self.scalar_net = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, 1)          # per-level scalar output
        # fallback if outputs aren't one-per-level
        self.final = nn.Linear(n1, n_outputs) if n_outputs != n1 else None

    def forward(self, x_conv, x_scalar=None):
        b = x_conv.shape[0]
        tokens = self.tokenizer(x_conv).transpose(1, 2)   # (b, n1, d_model)
        tokens = tokens + self.pos_embed

        if self.scalar_net is not None:
            ctx = self.scalar_net(x_scalar).unsqueeze(1)  # (b, 1, d_model)
            tokens = torch.cat([ctx, tokens], dim=1)      # prepend context token

        encoded = self.encoder(tokens)

        # drop the context token before producing per-level outputs
        if self.scalar_net is not None:
            encoded = encoded[:, 1:, :]

        out = self.out_proj(encoded).squeeze(-1)          # (b, n1)
        return self.final(out) if self.final is not None else out





def train_transformer(
    Xc_train, Xs_train, y_train,
    Xc_val=None, Xs_val=None, y_val=None,
    model_args=None,
    epochs=100, lr=1e-3, weight_decay=0.0, batch_size=256,
    patience=None, min_delta=0.0, num_workers=0,
    save_path=None, verbose=True,
):

    model_args = model_args or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_val = Xc_val is not None
    has_scalar = Xs_train is not None and Xs_train.shape[1] > 0
    if patience is not None and not has_val:
        raise ValueError("early stopping needs a validation set.")

    # --- scaling: fit on train only ---
    Xc_mean, Xc_std = Xc_train.mean(0), Xc_train.std(0)
    Xc_std = np.where(Xc_std == 0, 1.0, Xc_std)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    y_std = np.where(y_std == 0, 1.0, y_std)
    scaler = {"Xc_mean": Xc_mean, "Xc_std": Xc_std, "y_mean": y_mean, "y_std": y_std}
    if has_scalar:
        Xs_mean, Xs_std = Xs_train.mean(0), Xs_train.std(0)
        Xs_std = np.where(Xs_std == 0, 1.0, Xs_std)
        scaler.update({"Xs_mean": Xs_mean, "Xs_std": Xs_std})

    def prep(Xc, Xs, y):
        tc = torch.tensor((Xc - Xc_mean) / Xc_std, dtype=torch.float32)
        ty = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
        if has_scalar:
            ts = torch.tensor((Xs - Xs_mean) / Xs_std, dtype=torch.float32)
            return tc, ts, ty
        return tc, torch.zeros(len(Xc), 0), ty

    tc, ts, ty = prep(Xc_train, Xs_train, y_train)
    loader = DataLoader(TensorDataset(tc, ts, ty), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
    if has_val:
        vc, vs, vy = prep(Xc_val, Xs_val, y_val)
        vc, vs, vy = vc.to(device), vs.to(device), vy.to(device)

   
    
    # ... identical setup, scaling, and prep() as train_conv_scalar ...

    model = ProfileTransformer(
        n_conv_vars=tc.shape[1], n1=tc.shape[2],
        n_scalar=(ts.shape[1] if has_scalar else 0),
        n_outputs=ty.shape[1], **(model_args or {}),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, since = np.inf, None, -1, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xc, xs, yb in loader:
            xc, xs, yb = xc.to(device), xs.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xc, xs if has_scalar else None), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xc)
        train_loss = running / len(tc)
        history["train"].append(train_loss)

        val_loss = None
        if has_val:
            val_loss = _eval_loss(model, vc, vs, vy, loss_fn, has_scalar,
                                  batch_size=batch_size)
            history["val"].append(val_loss)

        if verbose:
            msg = f"epoch {epoch:3d} | train {train_loss:.4e}"
            if val_loss is not None:
                msg += f" | val {val_loss:.4e}"
            print(msg)

        if patience is not None and since >= patience:
            if verbose:
                print(f"early stop at epoch {epoch}; best epoch {best_epoch} "
                      f"(val {best_val:.4e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_path is not None:
        torch.save({"model_state": model.state_dict(), "model_args": model_args,
                    "n_conv_vars": tc.shape[1], "n1": tc.shape[2],
                    "n_scalar": (ts.shape[1] if has_scalar else 0),
                    "n_outputs": ty.shape[1], "scaler": scaler}, save_path)
        if verbose:
            print(f"saved to {save_path}")

    return model, scaler, history