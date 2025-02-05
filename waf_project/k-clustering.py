#1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#2. Load Data
data_file = "data/processed_loan_data.csv"
loan_df = pd.read_csv(data_file, low_memory=True)

#3. Data Preprocessing
features = [
    "LOAN_AMOUNT",
    "EMPLOYEE_COUNT",
    "IS_LOW_DOC",
    "CHARGE_OFF_AMOUNT",
    "UNRATE",
    "PERCENTAGE_EXPOSURE"
]
X = loan_df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#4. Determine Optimal Clusters Using the Elbow Method
kmeans_model = KMeans(random_state=5)
visualizer = KElbowVisualizer(kmeans_model, k=(2, 10))
visualizer.fit(X_scaled)
visualizer.show()  # Displays the elbow plot

optimal_k = visualizer.elbow_value_
print(f"\nOptimal number of clusters: {optimal_k}")

#5. Apply K-Means Clustering & PCA for Visualization
kmeans = KMeans(n_clusters=optimal_k, random_state=5)
loan_df["Cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=loan_df["Cluster"], cmap="viridis", alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Cluster Visualization of Loan Applications")
plt.colorbar(scatter, label="Cluster")
plt.show()

#6. Display Sample Data from Each Cluster
sample_borrowers = loan_df.groupby("Cluster").first().reset_index()
print("\nSample borrowers from each cluster:")
print(sample_borrowers.head())

#7. Cluster Summary and Visualization
cluster_summary = loan_df.groupby("Cluster").agg({
    "LOAN_AMOUNT": "mean",
    "PERCENTAGE_EXPOSURE": "mean",
    "CHARGE_OFF_AMOUNT": "mean",
    "EMPLOYEE_COUNT": "mean"
}).reset_index()

print("\nCluster Summary Statistics:")
print(cluster_summary.head(1000))

# Plot summary statistics for each cluster
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

sns.barplot(ax=axs[0, 0], x="Cluster", y="LOAN_AMOUNT", data=cluster_summary, palette="viridis")
axs[0, 0].set_title("Avg Loan Amount by Cluster")
axs[0, 0].set_ylabel("Average Loan Amount ($)")

sns.barplot(ax=axs[0, 1], x="Cluster", y="PERCENTAGE_EXPOSURE", data=cluster_summary, palette="coolwarm")
axs[0, 1].set_title("Avg Exposure Percentage by Cluster")
axs[0, 1].set_ylabel("Average Exposure (%)")

sns.barplot(ax=axs[1, 0], x="Cluster", y="CHARGE_OFF_AMOUNT", data=cluster_summary, palette="magma")
axs[1, 0].set_title("Avg Charge-Off Amount by Cluster")
axs[1, 0].set_ylabel("Average Charge-Off Amount ($)")

sns.barplot(ax=axs[1, 1], x="Cluster", y="EMPLOYEE_COUNT", data=cluster_summary, palette="Blues")
axs[1, 1].set_title("Avg Employee Count by Cluster")
axs[1, 1].set_ylabel("Average Employee Count")

plt.tight_layout()
plt.show()
