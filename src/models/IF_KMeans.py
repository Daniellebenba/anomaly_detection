##### IForest-KMeans #####
    
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

class IF_KMeans(IsolationForest):
    
    def __init__(self,  *args, **kwargs):
        super(IF_KMeans, self).__init__(*args, **kwargs)
        self.KMeans_model = KMeans(n_clusters=2)
        
        
    def fit(self, X):
        print('Fitting the Isolation Forest model...')
        super().fit(X)
        self.anomaly_scores = -super().score_samples(X)  # sklearn Opposite of the anomaly score defined in the original 
        print('Fitting K-Means on the Anomaly Scores...')
        self.KMeans_model.fit(self.anomaly_scores.reshape(-1, 1))
        
        
    def predict(self, X):
        _pred = -super().score_samples(X)
        return self.KMeans_model.predict(_pred.reshape(-1, 1))
    
