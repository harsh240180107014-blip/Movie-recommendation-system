"""
Movie Recommendation System - Data Mining Project
Dataset : MovieTweetings (IMDb ratings shared on Twitter) - ratings.dat, movies.dat
Methods : Popularity baseline, Content-Based (TF-IDF + cosine), Item-Based CF,
          SVD Matrix Factorization, K-Means user clustering
Run     : python movie_recommender.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

np.random.seed(42)
RES = {}

# ---------------------------------------------------------------- 1. LOAD
ratings = pd.read_csv("ratings.dat", sep="::", engine="python",
                      names=["user", "movie", "rating", "ts"])
movies = pd.read_csv("movies.dat", sep="::", engine="python",
                     names=["movie", "title", "genres"], encoding="utf-8")
RES["raw_ratings"] = len(ratings)
RES["raw_movies"] = len(movies)

# ------------------------------------------------------------ 2. PREPROCESS
# keep reasonably active users / popular movies (reduces sparsity & noise)
mc = ratings.movie.value_counts()
ratings = ratings[ratings.movie.isin(mc[mc >= 100].index)]
uc = ratings.user.value_counts()
ratings = ratings[ratings.user.isin(uc[uc >= 30].index)]
movies = movies[movies.movie.isin(ratings.movie.unique())].reset_index(drop=True)
movies["year"] = movies.title.str.extract(r"\((\d{4})\)")[0].astype(float)
movies["genres"] = movies.genres.fillna("Unknown")

users = ratings.user.unique()
uidx = {u: i for i, u in enumerate(users)}
midx = {m: i for i, m in enumerate(movies.movie)}
ratings = ratings[ratings.movie.isin(midx)].copy()
ratings["u"] = ratings.user.map(uidx)
ratings["m"] = ratings.movie.map(midx)
n_u, n_m = len(users), len(movies)
RES.update(users=n_u, movies=n_m, ratings=len(ratings),
           sparsity=round(100 * (1 - len(ratings) / (n_u * n_m)), 2),
           mean_rating=round(ratings.rating.mean(), 2))

# --------------------------------------------------------------------- 3. EDA
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ratings.rating.value_counts().sort_index().plot.bar(ax=ax[0], color="#2E5EAA")
ax[0].set(title="Rating distribution", xlabel="Rating", ylabel="Count")
g = movies.genres.str.split("|").explode().value_counts().head(10)
g.sort_values().plot.barh(ax=ax[1], color="#E07A5F")
ax[1].set(title="Top 10 genres (movies)")
ratings.groupby("u").size().clip(upper=500).plot.hist(bins=40, ax=ax[2], color="#3D9970")
ax[2].set(title="Ratings per user (clipped at 500)", xlabel="# ratings")
plt.tight_layout(); plt.savefig("fig_eda.png", dpi=150); plt.close()

# ------------------------------------------------------------ 4. TRAIN / TEST
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
R_tr = csr_matrix((train.rating, (train.u, train.m)), shape=(n_u, n_m))
gmean = train.rating.mean()
RES["train"], RES["test"] = len(train), len(test)

def rmse(pred):
    return float(np.sqrt(np.mean((pred - test.rating.values) ** 2)))

# ------------------------------------------------------- 5. MODELS (rating)
# 5.0 baseline: global mean
RES["rmse_global_mean"] = rmse(np.full(len(test), gmean))

# 5.1 baseline: user mean + movie bias
mm = train.groupby("m").rating.mean()
um = train.groupby("u").rating.mean()
bias_pred = (test.u.map(um).fillna(gmean) + test.m.map(mm).fillna(gmean) - gmean).clip(1, 10)
RES["rmse_bias"] = rmse(bias_pred.values)

# 5.2 Item-based collaborative filtering (cosine on mean-centred columns)
dense = np.zeros((n_u, n_m)); dense[train.u, train.m] = train.rating
mask = dense > 0
umean = np.where(mask.sum(1) > 0, dense.sum(1) / np.maximum(mask.sum(1), 1), gmean)
centered = np.where(mask, dense - umean[:, None], 0)
item_sim = cosine_similarity(centered.T)
np.fill_diagonal(item_sim, 0)

def item_cf_predict(u, m, k=30):
    rated = np.where(mask[u])[0]
    if len(rated) == 0:
        return gmean
    sims = item_sim[m, rated]
    top = np.argsort(-np.abs(sims))[:k]
    s, r = sims[top], rated[top]
    denom = np.abs(s).sum()
    if denom == 0:
        return umean[u]
    return umean[u] + (s * centered[u, r]).sum() / denom

sample = test.sample(min(20000, len(test)), random_state=1)   # speed
pred_cf = np.array([item_cf_predict(u, m) for u, m in zip(sample.u, sample.m)]).clip(1, 10)
RES["rmse_item_cf"] = float(np.sqrt(np.mean((pred_cf - sample.rating.values) ** 2)))
RES["rmse_bias_same_sample"] = float(np.sqrt(np.mean(
    ((sample.u.map(um).fillna(gmean) + sample.m.map(mm).fillna(gmean) - gmean).clip(1, 10)
     - sample.rating.values) ** 2)))

# 5.3 SVD matrix factorisation (on mean-centred matrix)
best = None
svd_curve = []
for k in [5, 10, 20, 40, 60]:
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z = svd.fit_transform(centered)
    recon = Z @ svd.components_ + umean[:, None]
    p = np.clip(recon[test.u.values, test.m.values], 1, 10)
    e = rmse(p); svd_curve.append((k, e))
    if best is None or e < best[1]:
        best = (k, e, recon)
RES["svd_curve"] = svd_curve
RES["svd_best_k"], RES["rmse_svd"] = best[0], best[1]
recon_best = best[2]

plt.figure(figsize=(5.5, 3.6))
plt.plot(*zip(*svd_curve), marker="o", color="#2E5EAA")
plt.title("SVD: RMSE vs latent factors"); plt.xlabel("k"); plt.ylabel("Test RMSE")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("fig_svd.png", dpi=150); plt.close()

# ------------------------------------------------- 6. TOP-N RANKING METRICS
# relevant = test rating >= 8 ; candidates = movies not seen in training
REL = 8
test_rel = test[test.rating >= REL].groupby("u").m.apply(set)
eval_users = test_rel.index.to_numpy()
eval_users = np.random.choice(eval_users, min(1500, len(eval_users)), replace=False)
K = 10

pop_score = np.asarray(R_tr.getnnz(axis=0), dtype=float)          # popularity
tf = TfidfVectorizer(token_pattern=r"[^|]+").fit_transform(movies.genres)
content_sim = cosine_similarity(tf)

def score_popularity(u): return pop_score.copy()
def score_content(u):
    liked = np.where(mask[u] & (dense[u] >= REL))[0]
    if len(liked) == 0: return pop_score.copy()
    return content_sim[liked].mean(0)
def score_item_cf(u):
    rated = np.where(mask[u])[0]
    w = centered[u, rated]
    return item_sim[:, rated] @ w
def score_svd(u): return recon_best[u]

def evaluate(fn):
    P, Rc, N = [], [], []
    for u in eval_users:
        sc = fn(u).astype(float); sc[mask[u]] = -np.inf
        top = np.argsort(-sc)[:K]
        rel = test_rel[u]
        hits = [1 if m in rel else 0 for m in top]
        P.append(sum(hits) / K); Rc.append(sum(hits) / len(rel))
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), K)))
        N.append(dcg / idcg)
    return dict(precision=float(np.mean(P)), recall=float(np.mean(Rc)), ndcg=float(np.mean(N)))

models = {"Popularity": score_popularity, "Content-Based": score_content,
          "Item-CF": score_item_cf, "SVD": score_svd}
RES["ranking"] = {n: evaluate(f) for n, f in models.items()}
RES["eval_users"] = int(len(eval_users)); RES["K"] = K

# ------------------------------------------------------- 7. USER CLUSTERING
Zc = TruncatedSVD(n_components=20, random_state=42).fit_transform(centered)
inertia, sil = [], []
from sklearn.metrics import silhouette_score
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Zc)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Zc, km.labels_, sample_size=3000, random_state=42))
kbest = 2 + int(np.argmax(sil))
km = KMeans(n_clusters=kbest, n_init=10, random_state=42).fit(Zc)
RES["kmeans_k"] = kbest
RES["silhouette"] = [round(float(s), 4) for s in sil]
cl = pd.Series(km.labels_, name="cluster")
RES["cluster_sizes"] = cl.value_counts().sort_index().tolist()
genre_rows = []
mg = movies.genres.str.split("|")
for c in range(kbest):
    members = np.where(km.labels_ == c)[0]
    liked = dense[members] >= REL
    cnt = {}
    for j in np.where(liked.any(0))[0]:
        for g_ in mg[j]:
            cnt[g_] = cnt.get(g_, 0) + liked[:, j].sum()
    top3 = sorted(cnt, key=cnt.get, reverse=True)[:3]
    genre_rows.append(top3)
RES["cluster_top_genres"] = genre_rows

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(range(2, 9), inertia, marker="o", color="#2E5EAA"); ax[0].set(title="Elbow (inertia)", xlabel="k")
ax[1].plot(range(2, 9), sil, marker="o", color="#E07A5F"); ax[1].set(title="Silhouette score", xlabel="k")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_kmeans.png", dpi=150); plt.close()

# model comparison chart
names = list(RES["ranking"]); vals = [RES["ranking"][n]["ndcg"] for n in names]
plt.figure(figsize=(5.5, 3.6))
plt.bar(names, vals, color=["#9AA5B1", "#E07A5F", "#3D9970", "#2E5EAA"])
plt.title(f"NDCG@{K} by model"); plt.ylabel("NDCG"); plt.tight_layout()
plt.savefig("fig_models.png", dpi=150); plt.close()

# --------------------------------------------------- 8. DEMO RECOMMENDATIONS
def recommend(u, fn=score_svd, n=5):
    sc = fn(u).astype(float); sc[mask[u]] = -np.inf
    return movies.title.iloc[np.argsort(-sc)[:n]].tolist()

demo_u = int(eval_users[0])
seen = movies.title.iloc[np.argsort(-dense[demo_u])[:5]].tolist()
RES["demo"] = dict(user=demo_u, liked=seen, svd=recommend(demo_u),
                   itemcf=recommend(demo_u, score_item_cf),
                   content=recommend(demo_u, score_content))

json.dump(RES, open("results.json", "w"), indent=2, default=str)
print(json.dumps(RES, indent=2, default=str))
