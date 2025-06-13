from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

def perform_clustering(features_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_scaled)
    ch_score = calinski_harabasz_score(features_scaled, labels)
    db_score = davies_bouldin_score(features_scaled, labels)
    return labels, ch_score, db_score
