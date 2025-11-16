import pandas as pd
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate


# -----------------------------------------------------------
# Load ratings_small.csv into a Surprise Dataset
# -----------------------------------------------------------

def load_ratings(data_path: str = "../data/ratings_small.csv"):
    """
    Load ratings_small.csv and convert to a Surprise Dataset.
    Columns: userId, movieId, rating.
    """

    df = pd.read_csv(data_path)
    df = df[["userId", "movieId", "rating"]]   # ignore timestamp

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df, reader)

    print(f"Loaded {df.shape[0]} ratings from ratings_small.csv")
    print(df.head())
    return data


# -----------------------------------------------------------
# Helper: evaluate an algorithm with 5-fold CV
# -----------------------------------------------------------

def evaluate_algorithm(algo, data, cv_folds: int = 5):
    """
    Run cross-validation and return mean RMSE and MAE.
    """
    results = cross_validate(
        algo,
        data,
        measures=["RMSE", "MAE"],
        cv=cv_folds,
        verbose=True,
        n_jobs=-1
    )
    mean_rmse = results["test_rmse"].mean()
    mean_mae = results["test_mae"].mean()
    return mean_rmse, mean_mae


# -----------------------------------------------------------
# Q2(c) – PMF, User-CF, Item-CF with cosine similarity
# -----------------------------------------------------------

def run_q2c(data):
    print("\n======================")
    print("Q2(c): 5-fold CV Results")
    print("======================")

    # PMF (SVD with no bias)
    pmf = SVD(biased=False, random_state=42)
    rmse_pmf, mae_pmf = evaluate_algorithm(pmf, data)
    print(f"\nPMF (SVD unbiased) -> RMSE={rmse_pmf:.4f}, MAE={mae_pmf:.4f}")

    # User-based CF (cosine)
    user_cf = KNNBasic(sim_options={"name": "cosine", "user_based": True})
    rmse_user, mae_user = evaluate_algorithm(user_cf, data)
    print(f"User-based CF (cosine) -> RMSE={rmse_user:.4f}, MAE={mae_user:.4f}")

    # Item-based CF (cosine)
    item_cf = KNNBasic(sim_options={"name": "cosine", "user_based": False})
    rmse_item, mae_item = evaluate_algorithm(item_cf, data)
    print(f"Item-based CF (cosine) -> RMSE={rmse_item:.4f}, MAE={mae_item:.4f}")

    print("\nFinal Summary (Q2c):")
    print(f"  PMF:      RMSE={rmse_pmf:.4f}, MAE={mae_pmf:.4f}")
    print(f"  User-CF:  RMSE={rmse_user:.4f}, MAE={mae_user:.4f}")
    print(f"  Item-CF:  RMSE={rmse_item:.4f}, MAE={mae_item:.4f}")

    return {
        "PMF": (rmse_pmf, mae_pmf),
        "User-CF": (rmse_user, mae_user),
        "Item-CF": (rmse_item, mae_item),
    }


# -----------------------------------------------------------
# Q2(e) – Similarity metrics: cosine, MSD, Pearson
# -----------------------------------------------------------

def run_q2e_similarity_experiments(data):
    """
    Examine how cosine, MSD, and Pearson similarities impact
    User-based and Item-based Collaborative Filtering.

    Produces:
      - q2e_user_rmse.png
      - q2e_user_mae.png
      - q2e_item_rmse.png
      - q2e_item_mae.png
    """
    similarity_measures = ["cosine", "msd", "pearson"]

    user_rmse = []
    user_mae = []
    item_rmse = []
    item_mae = []

    print("\n======================")
    print("Q2(e): Similarity Impact (User-based CF)")
    print("======================")

    for sim in similarity_measures:
        print(f"\nUser-CF with similarity = {sim}")
        algo_user = KNNBasic(sim_options={"name": sim, "user_based": True})
        rmse_u, mae_u = evaluate_algorithm(algo_user, data)
        print(f"User-CF ({sim}) -> RMSE={rmse_u:.4f}, MAE={mae_u:.4f}")
        user_rmse.append(rmse_u)
        user_mae.append(mae_u)

    print("\n======================")
    print("Q2(e): Similarity Impact (Item-based CF)")
    print("======================")

    for sim in similarity_measures:
        print(f"\nItem-CF with similarity = {sim}")
        algo_item = KNNBasic(sim_options={"name": sim, "user_based": False})
        rmse_i, mae_i = evaluate_algorithm(algo_item, data)
        print(f"Item-CF ({sim}) -> RMSE={rmse_i:.4f}, MAE={mae_i:.4f}")
        item_rmse.append(rmse_i)
        item_mae.append(mae_i)

    # --------- PLOTS (overwrite old PNGs — NO duplicates) ---------

    plt.figure()
    plt.plot(similarity_measures, user_rmse, marker="o")
    plt.title("Q2(e): User-based CF – RMSE vs Similarity")
    plt.xlabel("Similarity Metric")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig("q2e_user_rmse.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(similarity_measures, user_mae, marker="o")
    plt.title("Q2(e): User-based CF – MAE vs Similarity")
    plt.xlabel("Similarity Metric")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig("q2e_user_mae.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(similarity_measures, item_rmse, marker="o")
    plt.title("Q2(e): Item-based CF – RMSE vs Similarity")
    plt.xlabel("Similarity Metric")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig("q2e_item_rmse.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(similarity_measures, item_mae, marker="o")
    plt.title("Q2(e): Item-based CF – MAE vs Similarity")
    plt.xlabel("Similarity Metric")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig("q2e_item_mae.png", bbox_inches="tight")
    plt.close()

    return {
        "similarities": similarity_measures,
        "user_rmse": user_rmse,
        "user_mae": user_mae,
        "item_rmse": item_rmse,
        "item_mae": item_mae,
    }


# -----------------------------------------------------------
# Q2(f) – Effect of number of neighbors (k)
# -----------------------------------------------------------

def run_q2f_neighbors_experiments(data):
    """
    Evaluate RMSE for different values of k (neighbors)
    using MSD similarity for both User-based and Item-based CF.

    Produces:
      - q2f_user_rmse.png
      - q2f_item_rmse.png
    """

    neighbor_values = [5, 10, 20, 30, 40, 50]
    user_rmse = []
    item_rmse = []

    print("\n======================")
    print("Q2(f): Impact of k (User-based CF, MSD)")
    print("======================")

    for k in neighbor_values:
        print(f"\nUser-CF: k = {k}")
        algo_user = KNNBasic(
            k=k,
            sim_options={"name": "msd", "user_based": True}
        )
        rmse_u, _ = evaluate_algorithm(algo_user, data)
        user_rmse.append(rmse_u)
        print(f"User-CF (k={k}) -> RMSE={rmse_u:.4f}")

    print("\n======================")
    print("Q2(f): Impact of k (Item-based CF, MSD)")
    print("======================")

    for k in neighbor_values:
        print(f"\nItem-CF: k = {k}")
        algo_item = KNNBasic(
            k=k,
            sim_options={"name": "msd", "user_based": False}
        )
        rmse_i, _ = evaluate_algorithm(algo_item, data)
        item_rmse.append(rmse_i)
        print(f"Item-CF (k={k}) -> RMSE={rmse_i:.4f}")

    # --------- PLOTS (overwrite old PNGs — NO duplicates) ---------

    plt.figure()
    plt.plot(neighbor_values, user_rmse, marker="o")
    plt.title("Q2(f): User-based CF – RMSE vs k (MSD)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig("q2f_user_rmse.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(neighbor_values, item_rmse, marker="o")
    plt.title("Q2(f): Item-based CF – RMSE vs k (MSD)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig("q2f_item_rmse.png", bbox_inches="tight")
    plt.close()

    return neighbor_values, user_rmse, item_rmse


# -----------------------------------------------------------
# MAIN: Q2(c), Q2(e), Q2(f)
# -----------------------------------------------------------

def main():
    data = load_ratings()

    _ = run_q2c(data)
    _ = run_q2e_similarity_experiments(data)
    _ = run_q2f_neighbors_experiments(data)


if __name__ == "__main__":
    main()
