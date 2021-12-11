############# ALGORITHM CLASSES ##############


# TODO: 
# -implement classes to the algorithms below 
# - add w&b loggings as well
# - implement experiemnt framework

# We implement the following algorithms classes:
# 1. IF : we use sklearn implementation
# 2. IF-Kmeans : based on the paper: https://arxiv.org/pdf/2104.13190.pdf
# 3. Random Cut Forest : https://github.com/kLabUM/rrcf
# 4. AE with Entity Embeddings : based on the paper: https://arxiv.org/pdf/1910.02203.pdf


##### Isolation Forest #####

# use sklearn


##### IForest-KMeans #####
    
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

class IF_KMeans(IsolationForest):
    
    def __init__(self,  *args, **kwargs):
        super(IF_KMeans, self).__init()
#         self.IF_model = IsolationForest(*args, **kwargs)
        self.KMeans_model = KMeans(n_clusters=2)
        
        
    def fit(self, X):
        print('Fitting the Isolation Forest model...')
        super.fit(X)
        self.anomaly_scores = -super().score_samples(X)  # sklearn Opposite of the anomaly score defined in the original 
        print('Fitting K-Means on the Anomaly Scores...')
        self.KMeans_model.fit(self.anomaly_scores.reshape(-1, 1))
        
        
    def predict(self, X):
        _pred = -super().score_samples(X)
        return model.KMeans_model.predict(_pred.reshape(-1, 1))
    

##### AE - Entity Embeddings #####    

