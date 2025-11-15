import matplotlib.pyplot as plt

# SSE values
sse_euclidean = 25430232913.993008
sse_cosine    = 25580053055.769646
sse_jaccard   = 25482911827.89959

metrics = ["Euclidean", "Cosine", "Jaccard"]
sse_values = [sse_euclidean, sse_cosine, sse_jaccard]

plt.figure(figsize=(9, 5))

plt.plot(metrics, sse_values, marker='o', linewidth=3, markersize=10, color="#4A90E2")

# Add labels on each point
for metric, sse in zip(metrics, sse_values):
    plt.text(metric, sse, f"{sse:.2e}", ha='center', va='bottom', fontsize=10)

plt.title("Q1: SSE Comparison Across Distance Metrics")
plt.ylabel("SSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("q1_sse_plot_line.png", dpi=300)
plt.show()

print("Saved as q1_sse_plot_line.png")
