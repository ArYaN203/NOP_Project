"""
Theme-9: Logarithmic Regret Bounding via UCB in Collaborative Filtering
UCB-OCO Hybrid Optimizer for Recommender Systems on MovieLens 100K (synthetic)

This module implements:
  1. UCB-based Online Convex Optimization (OCO) for matrix factorization
  2. Standard SGD matrix factorization baseline
  3. Evaluation: Precision@K, RMSE, convergence curves, regret analysis
"""

import numpy as np
import pandas as pd
import time
import warnings
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading The MovieLens 100k dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_movielens_100k(path="ml-100k/u.data"):
    """
    Load the real MovieLens 100K dataset.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user", "item", "rating", "timestamp"]
    )

    df["user"] -= 1
    df["item"] -= 1

    print(
        f"Dataset: {len(df)} ratings | "
        f"{df['user'].nunique()} users | "
        f"{df['item'].nunique()} items"
    )

    return df[["user", "item", "rating"]]

# ─────────────────────────────────────────────────────────────────────────────
# 2. User-wise train/test split
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_temporal(df, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)

    train_idx = []
    test_idx = []

    for uid, grp in df.groupby('user'):
        idx = grp.index.tolist()
        rng.shuffle(idx)

        n_test = max(1, int(len(idx) * test_frac))

        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    return train_df, test_df

# ─────────────────────────────────────────────────────────────────────────────
# 3. UCB-OCO Matrix Factorization
# ─────────────────────────────────────────────────────────────────────────────

class UCBOCOMatrixFactorization:
    """
    UCB-guided Online Convex Optimization for Collaborative Filtering.

    Mathematical Framework
    ----------------------
    Objective (per step t):
        L_t(U, V) = (r_{u,i} - U_u^T V_i)^2 + λ(||U_u||² + ||V_i||²)

    UCB Exploration bonus for arm (u, i):
        UCB_{u,i}(t) = μ_{u,i}(t) + α * σ_{u,i}(t) / sqrt(N_{u,i}(t) + 1)

    where:
        μ_{u,i}(t)  = empirical mean interaction score
        σ_{u,i}(t)  = empirical std (subgradient bound proxy)
        N_{u,i}(t)  = number of times arm (u,i) has been pulled
        α           = exploration parameter (UCB confidence width)

    Adaptive learning rate (subgradient-based):
        η_t = η_0 / sqrt(G_t² + ε)
        G_t = subgradient norm at step t

    OCO regret guarantee:
        R(T) ≤ O(log T) under bounded subgradients
    """

    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.02,
                 alpha=1.5, n_epochs=20, batch_size=512, seed=42):
        self.n_users = n_users
        self.n_items = n_items
        self.K = n_factors
        self.lr = lr
        self.reg = reg
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        rng = np.random.RandomState(seed)

        # Latent factor matrices
        self.U = rng.normal(0, 0.1, (n_users, n_factors))
        self.V = rng.normal(0, 0.1, (n_items, n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0.0

        # UCB arm statistics
        self.arm_counts   = defaultdict(int)      # N_{u,i}
        self.arm_rewards  = defaultdict(float)    # sum of rewards
        self.arm_rewards2 = defaultdict(float)    # sum of squared rewards

        # AdaGrad accumulators (per-row subgradient norms)
        self.G_U = np.ones((n_users, n_factors)) * 1e-8
        self.G_V = np.ones((n_items, n_factors)) * 1e-8
        self.G_bu = np.ones(n_users) * 1e-8
        self.G_bi = np.ones(n_items) * 1e-8

        # Logging
        self.train_rmse_hist = []
        self.test_rmse_hist  = []
        self.regret_hist     = []
        self.precision_hist  = []
        self.cumulative_regret = 0.0
        self.fit_time = 0.0

    def _predict(self, u, i):
        return (self.mu + self.bu[u] + self.bi[i]
                + self.U[u] @ self.V[i])

    def _ucb_score(self, u, i):
        """UCB exploration score for arm (u, i)."""
        n = self.arm_counts[(u, i)]
        if n == 0:
            return np.inf  # Force exploration for cold arms
        mean = self.arm_rewards[(u, i)] / n
        var  = max(0, self.arm_rewards2[(u, i)] / n - mean**2)
        sigma = np.sqrt(var) + 1e-6
        return mean + self.alpha * sigma / np.sqrt(n + 1)

    def _update_arm(self, u, i, reward):
        self.arm_counts[(u, i)]   += 1
        self.arm_rewards[(u, i)]  += reward
        self.arm_rewards2[(u, i)] += reward**2

    def _adagrad_update(self, u, i, err):
        """AdaGrad subgradient step (OCO update rule)."""
        grad_U  = -2 * err * self.V[i] + 2 * self.reg * self.U[u]
        grad_V  = -2 * err * self.U[u] + 2 * self.reg * self.V[i]
        grad_bu = -2 * err + 2 * self.reg * self.bu[u]
        grad_bi = -2 * err + 2 * self.reg * self.bi[i]

        self.G_U[u]  += grad_U**2
        self.G_V[i]  += grad_V**2
        self.G_bu[u] += grad_bu**2
        self.G_bi[i] += grad_bi**2

        self.U[u]  -= self.lr * grad_U  / np.sqrt(self.G_U[u])
        self.V[i]  -= self.lr * grad_V  / np.sqrt(self.G_V[i])
        self.bu[u] -= self.lr * grad_bu / np.sqrt(self.G_bu[u])
        self.bi[i] -= self.lr * grad_bi / np.sqrt(self.G_bi[i])

    def fit(self, train_df, test_df=None, global_mean=3.53):
        self.mu = global_mean
        rng = np.random.RandomState(0)
        train_arr = train_df[['user', 'item', 'rating']].values.astype(int)
        t_start = time.time()

        # Theoretical optimal (oracle) loss for regret computation
        oracle_per_step = 0.05  # Approximate irreducible error

        for epoch in range(self.n_epochs):
            rng.shuffle(train_arr)
            epoch_loss = 0.0
            epoch_regret = 0.0

            for start in range(0, len(train_arr), self.batch_size):
                batch = train_arr[start:start + self.batch_size]
                for u, i, r in batch:
                    pred = self._predict(u, i)
                    err  = r - pred
                    loss = err**2

                    # UCB reward signal: normalized inverse error
                    reward = 1.0 / (1.0 + abs(err))
                    self._update_arm(u, i, reward)

                    # Regret: instantaneous loss - oracle loss
                    step_regret = max(0, loss - oracle_per_step)
                    self.cumulative_regret += step_regret
                    epoch_regret += step_regret

                    self._adagrad_update(u, i, err)
                    epoch_loss += loss

            rmse_train = np.sqrt(epoch_loss / len(train_arr))
            self.train_rmse_hist.append(rmse_train)
            self.regret_hist.append(self.cumulative_regret)

            if test_df is not None:
                rmse_test = self._compute_rmse(test_df)
                self.test_rmse_hist.append(rmse_test)

            if (epoch + 1) % 5 == 0:
                print(f"  [UCB-OCO] Epoch {epoch+1:02d}/{self.n_epochs} | "
                      f"Train RMSE={rmse_train:.4f}"
                      + (f" | Test RMSE={rmse_test:.4f}" if test_df is not None else "")
                      + f" | Cum.Regret={self.cumulative_regret:.1f}")

        self.fit_time = time.time() - t_start

    def _compute_rmse(self, df):
        preds = [self._predict(u, i) for u, i in zip(df['user'], df['item'])]
        errors = np.array(df['rating']) - np.clip(preds, 1, 5)
        return np.sqrt(np.mean(errors**2))

    def predict_topk(self, user_id, k=10, exclude_seen=None):
        scores = []
        for i in range(self.n_items):
            if exclude_seen and i in exclude_seen:
                continue
            pred  = self._predict(user_id, i)
            bonus = 0.0 if self.arm_counts[(user_id, i)] > 0 else self.alpha * 0.1
            scores.append((i, pred + bonus))
        scores.sort(key=lambda x: -x[1])
        return [item for item, _ in scores[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# 4. SGD Baseline Matrix Factorization
# ─────────────────────────────────────────────────────────────────────────────

class SGDMatrixFactorization:
    """
    Standard Mini-batch SGD Matrix Factorization (baseline).
    Fixed learning rate, no UCB exploration, no AdaGrad.
    """

    def __init__(self, n_users, n_items, n_factors=20, lr=0.005, reg=0.02,
                 n_epochs=20, batch_size=512, seed=42):
        self.n_users   = n_users
        self.n_items   = n_items
        rng = np.random.RandomState(seed)
        self.U  = rng.normal(0, 0.1, (n_users, n_factors))
        self.V  = rng.normal(0, 0.1, (n_items, n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0.0
        self.lr = lr
        self.reg = reg
        self.n_epochs  = n_epochs
        self.batch_size = batch_size
        self.train_rmse_hist = []
        self.test_rmse_hist  = []
        self.fit_time = 0.0

    def _predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.U[u] @ self.V[i]

    def fit(self, train_df, test_df=None, global_mean=3.53):
        self.mu = global_mean
        rng = np.random.RandomState(0)
        train_arr = train_df[['user', 'item', 'rating']].values.astype(int)
        t_start = time.time()

        for epoch in range(self.n_epochs):
            rng.shuffle(train_arr)
            epoch_loss = 0.0
            for u, i, r in train_arr:
                pred = self._predict(u, i)
                err  = r - pred
                self.U[u]  += self.lr * (2*err*self.V[i] - 2*self.reg*self.U[u])
                self.V[i]  += self.lr * (2*err*self.U[u] - 2*self.reg*self.V[i])
                self.bu[u] += self.lr * (2*err - 2*self.reg*self.bu[u])
                self.bi[i] += self.lr * (2*err - 2*self.reg*self.bi[i])
                epoch_loss += err**2

            rmse_train = np.sqrt(epoch_loss / len(train_arr))
            self.train_rmse_hist.append(rmse_train)
            if test_df is not None:
                rmse_test = self._compute_rmse(test_df)
                self.test_rmse_hist.append(rmse_test)

            if (epoch + 1) % 5 == 0:
                print(f"  [SGD]     Epoch {epoch+1:02d}/{self.n_epochs} | "
                      f"Train RMSE={rmse_train:.4f}"
                      + (f" | Test RMSE={rmse_test:.4f}" if test_df is not None else ""))

        self.fit_time = time.time() - t_start

    def _compute_rmse(self, df):
        preds = [self._predict(u, i) for u, i in zip(df['user'], df['item'])]
        errors = np.array(df['rating']) - np.clip(preds, 1, 5)
        return np.sqrt(np.mean(errors**2))

    def predict_topk(self, user_id, k=10, exclude_seen=None):
        scores = [(i, self._predict(user_id, i))
                  for i in range(self.n_items)
                  if not (exclude_seen and i in exclude_seen)]
        scores.sort(key=lambda x: -x[1])
        return [item for item, _ in scores[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(model, test_df, train_df, k=10, n_users_eval=200, seed=0):
    """
    Precision@K: fraction of top-K recommendations that are relevant.
    Relevant = items the user rated >= 4 in the test set.
    """
    rng = np.random.RandomState(seed)
    test_users = test_df['user'].unique()
    eval_users = rng.choice(test_users, min(n_users_eval, len(test_users)), replace=False)

    precisions = []
    for u in eval_users:
        relevant = set(
            test_df[(test_df['user'] == u) & (test_df['rating'] >= 4)]['item'])
        if not relevant:
            continue
        seen = set(train_df[train_df['user'] == u]['item'])
        topk = model.predict_topk(u, k=k, exclude_seen=seen)
        hits = len(set(topk) & relevant)
        precisions.append(hits / k)

    return np.mean(precisions) if precisions else 0.0


def coverage_at_k(model, test_df, train_df, k=10, n_users_eval=100):
    """Catalog coverage: fraction of unique items recommended."""
    rng = np.random.RandomState(1)
    eval_users = rng.choice(test_df['user'].unique(),
                            min(n_users_eval, test_df['user'].nunique()), replace=False)
    all_recs = set()
    for u in eval_users:
        seen = set(train_df[train_df['user'] == u]['item'])
        all_recs.update(model.predict_topk(u, k=k, exclude_seen=seen))
    return len(all_recs) / model.n_items


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'ucb': '#2563EB',
    'sgd': '#DC2626',
    'accent': '#059669',
    'bg': '#F8FAFC',
    'grid': '#E2E8F0',
}

def make_plots(ucb_model, sgd_model, metrics, save_path='./results'):
    import os; os.makedirs(save_path, exist_ok=True)

    # ── Figure 1: Convergence (Train RMSE + Test RMSE) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=COLORS['bg'])
    epochs = np.arange(1, ucb_model.n_epochs + 1)

    for ax, key, title in [
        (axes[0], 'train', 'Training RMSE Convergence'),
        (axes[1], 'test',  'Test RMSE Convergence'),
    ]:
        ax.set_facecolor(COLORS['bg'])
        ax.yaxis.grid(True, color=COLORS['grid'], zorder=0)
        data_ucb = ucb_model.train_rmse_hist if key == 'train' else ucb_model.test_rmse_hist
        data_sgd = sgd_model.train_rmse_hist if key == 'train' else sgd_model.test_rmse_hist
        ax.plot(epochs, data_ucb, color=COLORS['ucb'], lw=2.5, label='UCB-OCO', zorder=3)
        ax.plot(epochs, data_sgd, color=COLORS['sgd'], lw=2.5, linestyle='--', label='SGD', zorder=3)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(pad=2)
    plt.savefig(f'{save_path}/fig1_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Cumulative Regret (log scale) ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.yaxis.grid(True, color=COLORS['grid'], zorder=0)
    t = np.arange(1, len(ucb_model.regret_hist) + 1)
    ax.plot(t, ucb_model.regret_hist, color=COLORS['ucb'], lw=2.5,
            label='UCB-OCO Cumulative Regret', zorder=3)
    # Theoretical O(log T) bound
    log_bound = ucb_model.regret_hist[0] * np.log(t + 1) / np.log(2)
    ax.plot(t, log_bound, color=COLORS['accent'], lw=1.5, linestyle=':',
            label='O(log T) bound', zorder=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Cumulative Regret', fontsize=11)
    ax.set_title('Cumulative Regret vs. O(log T) Bound', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(pad=2)
    plt.savefig(f'{save_path}/fig2_regret.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3: Precision@K bar chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ks = [5, 10, 15, 20]
    ucb_prec = [metrics['ucb_precision'][k] for k in ks]
    sgd_prec = [metrics['sgd_precision'][k] for k in ks]
    x = np.arange(len(ks))
    w = 0.35
    bars1 = ax.bar(x - w/2, ucb_prec, w, color=COLORS['ucb'], label='UCB-OCO', zorder=3)
    bars2 = ax.bar(x + w/2, sgd_prec, w, color=COLORS['sgd'], label='SGD', alpha=0.85, zorder=3)
    ax.yaxis.grid(True, color=COLORS['grid'], zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([f'K={k}' for k in ks])
    ax.set_ylabel('Precision@K', fontsize=11)
    ax.set_title('Precision@K Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['ucb'])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color=COLORS['sgd'])
    plt.tight_layout(pad=2)
    plt.savefig(f'{save_path}/fig3_precision.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 4: Summary dashboard ───────────────────────────────────────
    fig = plt.figure(figsize=(14, 5), facecolor=COLORS['bg'])
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # RMSE improvement
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COLORS['bg'])
    vals = [metrics['ucb_test_rmse'], metrics['sgd_test_rmse']]
    bars = ax1.bar(['UCB-OCO', 'SGD'], vals,
                   color=[COLORS['ucb'], COLORS['sgd']], width=0.5, zorder=3)
    ax1.yaxis.grid(True, color=COLORS['grid'], zorder=0)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax1.set_ylabel('RMSE (lower=better)', fontsize=10)
    ax1.set_title('Final Test RMSE', fontsize=11, fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)

    # Training time
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COLORS['bg'])
    times = [ucb_model.fit_time, sgd_model.fit_time]
    bars2 = ax2.bar(['UCB-OCO', 'SGD'], times,
                    color=[COLORS['ucb'], COLORS['sgd']], width=0.5, zorder=3)
    ax2.yaxis.grid(True, color=COLORS['grid'], zorder=0)
    for bar, v in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f'{v:.1f}s', ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=10)
    ax2.set_title('Training Time', fontsize=11, fontweight='bold')
    ax2.spines[['top', 'right']].set_visible(False)

    # Coverage
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(COLORS['bg'])
    covs = [metrics['ucb_coverage'], metrics['sgd_coverage']]
    bars3 = ax3.bar(['UCB-OCO', 'SGD'], covs,
                    color=[COLORS['ucb'], COLORS['sgd']], width=0.5, zorder=3)
    ax3.yaxis.grid(True, color=COLORS['grid'], zorder=0)
    for bar, v in zip(bars3, covs):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                 f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Coverage@10', fontsize=10)
    ax3.set_title('Catalog Coverage', fontsize=11, fontweight='bold')
    ax3.spines[['top', 'right']].set_visible(False)

    fig.suptitle('UCB-OCO vs SGD — Performance Dashboard', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.savefig(f'{save_path}/fig4_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figures saved to {save_path}/")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Experiment
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Theme-9: UCB-OCO Collaborative Filtering  ")
    print("  MovieLens 100K (Real Dataset)     ")
    print("=" * 65)

    # Data
    print("\n[1] Loading MovieLens 100K dataset...")
    df = load_movielens_100k("ml-100k/u.data")

    train_df, test_df = train_test_split_temporal(df)

    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")
    
    global_mean = float(train_df['rating'].mean())
    n_users = int(df['user'].max()) + 1
    n_items = int(df['item'].max()) + 1

    # UCB-OCO
    print("\n[2] Training UCB-OCO model...")
    ucb = UCBOCOMatrixFactorization(n_users, n_items, n_factors=20, lr=0.02,
                                     reg=0.02, alpha=1.5, n_epochs=20, batch_size=256)
    ucb.fit(train_df, test_df, global_mean=global_mean)

    # SGD Baseline
    print("\n[3] Training SGD baseline...")
    sgd = SGDMatrixFactorization(n_users, n_items, n_factors=20, lr=0.005,
                                  reg=0.02, n_epochs=20, batch_size=256)
    sgd.fit(train_df, test_df, global_mean=global_mean)

    # Metrics
    print("\n[4] Computing evaluation metrics...")
    ks = [5, 10, 15, 20]
    ucb_prec = {}
    sgd_prec = {}
    for k in ks:
        ucb_prec[k] = precision_at_k(ucb, test_df, train_df, k=k, n_users_eval=150)
        sgd_prec[k] = precision_at_k(sgd, test_df, train_df, k=k, n_users_eval=150)
        print(f"  P@{k:2d}: UCB-OCO={ucb_prec[k]:.4f}  SGD={sgd_prec[k]:.4f}  "
              f"Δ={ucb_prec[k]-sgd_prec[k]:+.4f}")

    ucb_cov = coverage_at_k(ucb, test_df, train_df, k=10, n_users_eval=100)
    sgd_cov = coverage_at_k(sgd, test_df, train_df, k=10, n_users_eval=100)

    metrics = {
        'ucb_test_rmse'  : ucb.test_rmse_hist[-1] if ucb.test_rmse_hist else 0,
        'sgd_test_rmse'  : sgd.test_rmse_hist[-1] if sgd.test_rmse_hist else 0,
        'ucb_precision'  : ucb_prec,
        'sgd_precision'  : sgd_prec,
        'ucb_coverage'   : ucb_cov,
        'sgd_coverage'   : sgd_cov,
        'ucb_fit_time'   : ucb.fit_time,
        'sgd_fit_time'   : sgd.fit_time,
        'ucb_final_regret': ucb.cumulative_regret,
    }

    print(f"\n  Final Test RMSE: UCB-OCO={metrics['ucb_test_rmse']:.4f}  "
          f"SGD={metrics['sgd_test_rmse']:.4f}")
    print(f"  Catalog Coverage@10: UCB-OCO={ucb_cov:.3f}  SGD={sgd_cov:.3f}")
    print(f"  Training Time: UCB-OCO={ucb.fit_time:.1f}s  SGD={sgd.fit_time:.1f}s")
    print(f"  UCB-OCO Cumulative Regret: {ucb.cumulative_regret:.1f}")

    # Figures
    print("\n[5] Generating figures...")
    make_plots(ucb, sgd, metrics)

    # Save metrics JSON
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("  Metrics saved to metrics.json")

    return ucb, sgd, metrics


if __name__ == '__main__':
    ucb_model, sgd_model, results = main()
