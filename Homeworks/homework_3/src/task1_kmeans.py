import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


def load_kmeans_dataset(data_dir: str = "../data"):
    data_path = Path(data_dir) / "data.csv"
    label_path = Path(data_dir) / "label.csv"

    df_X = pd.read_csv(data_path)
    df_y = pd.read_csv(label_path)

    X = df_X.values.astype(float)
    y = df_y.values.squeeze().astype(int)

    return X, y


# -----------------------------------------------------------
# Distance Functions
# -----------------------------------------------------------

def euclidean_distance(x: np.ndarray, c: np.ndarray) -> float:
    diff = x - c
    return np.sqrt(np.sum(diff ** 2))


def cosine_distance(x: np.ndarray, c: np.ndarray) -> float:
    dot = float(np.dot(x, c))
    norm_x = float(np.linalg.norm(x))
    norm_c = float(np.linalg.norm(c))

    if norm_x == 0.0 or norm_c == 0.0:
        return 1.0

    cos_sim = dot / (norm_x * norm_c)
    cos_sim = max(min(cos_sim, 1.0), -1.0)

    return 1.0 - cos_sim


def generalized_jaccard_distance(x: np.ndarray, c: np.ndarray) -> float:
    if np.any(x < 0) or np.any(c < 0):
        raise ValueError("Generalized Jaccard requires non-negative vectors.")

    min_sum = float(np.sum(np.minimum(x, c)))
    max_sum = float(np.sum(np.maximum(x, c)))

    if max_sum == 0.0:
        return 0.0

    jac_sim = min_sum / max_sum
    return 1.0 - jac_sim


# -----------------------------------------------------------
# K-Means Class
# -----------------------------------------------------------

class KMeansScratch:
    def __init__(self, K, distance_function, max_iter=500,
                 stop_rule="classic", stop_condition=None):
        self.K = K
        self.distance_function = distance_function
        self.max_iter = max_iter
        self.stop_rule = stop_rule          # Q1 + Q2 -> classic, Q3 -> q3
        self.stop_condition = stop_condition # Q4 options

        self.centroids_ = None
        self.labels_ = None
        self.sse_ = None
        self.n_iter_ = None
        self.fit_time_ = None

    def initialize_centroids(self, X):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.K, replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            distances = np.array([
                self.distance_function(X[i], centroids[k])
                for k in range(self.K)
            ])
            labels[i] = np.argmin(distances)

        return labels

    def update_centroids(self, X, labels):
        n_samples, n_features = X.shape
        new_centroids = np.zeros((self.K, n_features))

        for k in range(self.K):
            cluster_points = X[labels == k]

            if len(cluster_points) == 0:
                rand_idx = np.random.randint(0, n_samples)
                new_centroids[k] = X[rand_idx]
            else:
                new_centroids[k] = cluster_points.mean(axis=0)

        return new_centroids

    def compute_sse(self, X, labels, centroids):
        sse = 0.0
        for i in range(X.shape[0]):
            diff = X[i] - centroids[labels[i]]
            sse += np.sum(diff ** 2)
        return sse

    def fit(self, X):
        import time
        start = time.time()

        centroids = self.initialize_centroids(X)
        prev_sse = None

        for iteration in range(self.max_iter):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels)

            current_sse = self.compute_sse(X, labels, new_centroids)

            # -------------------------------------------------
            # Q1 & Q2 stopping rule (classic)
            # -------------------------------------------------
            if self.stop_rule == "classic" and self.stop_condition is None:
                if np.allclose(centroids, new_centroids):
                    centroids = new_centroids
                    break

            # -------------------------------------------------
            # Q3 unified stopping rule
            # -------------------------------------------------
            elif self.stop_rule == "q3" and self.stop_condition is None:
                stop_centroid_same = np.allclose(centroids, new_centroids)
                stop_sse_increase = (
                    prev_sse is not None and current_sse > prev_sse
                )
                if stop_centroid_same or stop_sse_increase:
                    centroids = new_centroids
                    break
                prev_sse = current_sse

            # -------------------------------------------------
            # Q4: stop when centroids don't change
            # -------------------------------------------------
            elif self.stop_condition == "centroid":
                if np.allclose(centroids, new_centroids):
                    centroids = new_centroids
                    break

            # -------------------------------------------------
            # Q4: stop when SSE increases
            # -------------------------------------------------
            elif self.stop_condition == "sse_increase":
                if prev_sse is not None and current_sse > prev_sse:
                    centroids = new_centroids
                    break
                prev_sse = current_sse

            # -------------------------------------------------
            # Q4: stop when max_iter reached (do nothing)
            # -------------------------------------------------
            elif self.stop_condition == "max_iter":
                pass

            centroids = new_centroids

        # Final assignments
        final_labels = self.assign_clusters(X, centroids)
        final_sse = self.compute_sse(X, final_labels, centroids)
        end = time.time()

        self.centroids_ = centroids
        self.labels_ = final_labels
        self.sse_ = final_sse
        self.n_iter_ = iteration + 1
        self.fit_time_ = end - start

        return self

    def predict(self, X):
        return self.assign_clusters(X, self.centroids_)


# -----------------------------------------------------------
# Accuracy Helper Functions
# -----------------------------------------------------------

def majority_vote_labels(true_labels, cluster_assignments, K):
    mapping = {}
    for k in range(K):
        indices = np.where(cluster_assignments == k)[0]
        if len(indices) == 0:
            mapping[k] = 0
        else:
            majority = Counter(true_labels[indices]).most_common(1)[0][0]
            mapping[k] = majority
    return mapping


def compute_accuracy(true_labels, cluster_assignments, mapping):
    predicted = np.array([mapping[c] for c in cluster_assignments])
    return (predicted == true_labels).mean()


# -----------------------------------------------------------
# MAIN (Q1 + Q2 + Q3 + Q4)
# -----------------------------------------------------------

if __name__ == "__main__":
    X, y = load_kmeans_dataset()
    unique_labels = np.unique(y)
    K = len(unique_labels)

    # -------------------------------------------------------
    # Q1 — SSE Comparison
    # -------------------------------------------------------
    print("\n======================")
    print("Q1: SSE COMPARISON")
    print("======================")

    model_euc = KMeansScratch(K, euclidean_distance, max_iter=50)
    model_euc.fit(X)
    print(f"Euclidean SSE: {model_euc.sse_}")

    model_cos = KMeansScratch(K, cosine_distance, max_iter=50)
    model_cos.fit(X)
    print(f"Cosine SSE:    {model_cos.sse_}")

    model_jac = KMeansScratch(K, generalized_jaccard_distance, max_iter=50)
    model_jac.fit(X)
    print(f"Jaccard SSE:   {model_jac.sse_}")

    # -------------------------------------------------------
    # Q2 — Accuracy Comparison
    # -------------------------------------------------------
    print("\n======================")
    print("Q2: ACCURACY COMPARISON")
    print("======================")

    acc_euc = compute_accuracy(y, model_euc.labels_,
                               majority_vote_labels(y, model_euc.labels_, K))
    print(f"Euclidean Accuracy: {acc_euc:.4f}")

    acc_cos = compute_accuracy(y, model_cos.labels_,
                               majority_vote_labels(y, model_cos.labels_, K))
    print(f"Cosine Accuracy:    {acc_cos:.4f}")

    acc_jac = compute_accuracy(y, model_jac.labels_,
                               majority_vote_labels(y, model_jac.labels_, K))
    print(f"Jaccard Accuracy:   {acc_jac:.4f}")

    # -------------------------------------------------------
    # Q3 — Iteration + Time Comparison
    # -------------------------------------------------------
    print("\n======================")
    print("Q3: ITERATION + TIME COMPARISON")
    print("======================")

    model_euc_q3 = KMeansScratch(K, euclidean_distance, max_iter=500, stop_rule="q3")
    model_euc_q3.fit(X)
    print(f"Euclidean: iterations={model_euc_q3.n_iter_}, time={model_euc_q3.fit_time_:.4f}s")

    model_cos_q3 = KMeansScratch(K, cosine_distance, max_iter=500, stop_rule="q3")
    model_cos_q3.fit(X)
    print(f"Cosine:    iterations={model_cos_q3.n_iter_}, time={model_cos_q3.fit_time_:.4f}s")

    model_jac_q3 = KMeansScratch(K, generalized_jaccard_distance, max_iter=500, stop_rule="q3")
    model_jac_q3.fit(X)
    print(f"Jaccard:   iterations={model_jac_q3.n_iter_}, time={model_jac_q3.fit_time_:.4f}s")

    # -------------------------------------------------------
    # Q4 — SSE under 3 Stopping Conditions
    # -------------------------------------------------------
    print("\n======================")
    print("Q4: SSE UNDER 3 STOPPING CONDITIONS")
    print("======================")

    stop_conditions = ["centroid", "sse_increase", "max_iter"]

    for cond in stop_conditions:
        print(f"\n--- Stop Condition: {cond} ---")

        model_euc_q4 = KMeansScratch(K, euclidean_distance, max_iter=100, stop_condition=cond)
        model_euc_q4.fit(X)
        print(f"Euclidean SSE ({cond}): {model_euc_q4.sse_}")

        model_cos_q4 = KMeansScratch(K, cosine_distance, max_iter=100, stop_condition=cond)
        model_cos_q4.fit(X)
        print(f"Cosine SSE ({cond}):    {model_cos_q4.sse_}")

        model_jac_q4 = KMeansScratch(K, generalized_jaccard_distance, max_iter=100, stop_condition=cond)
        model_jac_q4.fit(X)
        print(f"Jaccard SSE ({cond}):   {model_jac_q4.sse_}")
