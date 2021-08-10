
def _detect_time_annotation(adata):

    time_column = [i for i in adata.obs.columns if "time" in i][0]

    return time_column

def _standardize_time_column(adata):
    
    current_time_column = _detect_time_annotation(adata)
    adata.obs['time'] = adata.obs[current_time_column]
    
def _standardize_time(adata):
    
    _standardize_time_column(adata)
    adata.obs.time = adata.obs.time / adata.obs.time.max()

def _preprocess(adata, return_adata=False):

    """Scale data for PCA and subsequent neural diffeq learning. Typically executed before."""

    from sklearn import preprocessing

    try:
        X_train = adata.X.toarray()
    except:
        X_train = adata.X

    scaler = preprocessing.StandardScaler().fit(X_train)
    adata.X = scaler.transform(X_train)
    
    _standardize_time(adata)
    
    if return_adata:
        return adata