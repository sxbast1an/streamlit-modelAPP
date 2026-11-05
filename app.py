import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

# Configuración de página
st.set_page_config(page_title="Streamlit Unsupervised Models App", layout="wide")
st.title("Activity: Streamlit Unsupervised Models App")

# 1. DATA INPUT
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # 2. FEATURE SELECTION & PREPROCESSING
    st.sidebar.header("2. Feature Selection & Preprocessing")
    all_columns = df.columns.tolist()
    selected_cols = st.sidebar.multiselect("Select columns to include", all_columns, default=all_columns)

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns to proceed.")
        st.stop()

    df_selected = df[selected_cols].copy()

    # Preprocesamiento
    num_cols = df_selected.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_selected.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    X_preprocessed = preprocessor.fit_transform(df_selected)

    # Selección de modo: Manual vs PCA
    feature_mode = st.sidebar.radio("Select feature mode:", ["Manual Selection", "PCA Reduction"])

    if feature_mode == "Manual Selection":
        if X_preprocessed.shape[1] < 2:
            st.error("At least 2 features are required after encoding.")
            st.stop()
        X_final = X_preprocessed
    else:  # PCA Reduction
        max_comp = min(X_preprocessed.shape[0], X_preprocessed.shape[1])
        if max_comp < 2:
            st.error("Dataset must have at least 2 features for PCA.")
            st.stop()
            
        n_components = st.sidebar.number_input(
            "PCA Components (at least 2):", 
            min_value=2, 
            max_value=max_comp, 
            value=2, 
            step=1
        )
        pca = PCA(n_components=n_components)
        X_final = pca.fit_transform(X_preprocessed)

    # 3. MODEL SELECTION & HYPERPARAMETERS
    st.sidebar.header("3. Model Selection & Hyperparameters")
    algo_options = ["K-Means", "DBSCAN", "Hierarchical Clustering"]
    if HAS_KMEDOIDS:
        algo_options.insert(1, "K-Medoids")
    
    algorithm = st.sidebar.selectbox("Select a clustering algorithm:", algo_options)
    model = None

    if algorithm == "K-Means":
        n_clusters = st.sidebar.slider("n_clusters", min_value=2, max_value=15, value=3)
        model = KMeans(n_clusters=n_clusters, random_state=42)

    elif algorithm == "K-Medoids":
        n_clusters = st.sidebar.slider("n_clusters", min_value=2, max_value=15, value=3)
        metric = st.sidebar.selectbox("metric", ["euclidean", "manhattan", "cosine"])
        model = KMedoids(n_clusters=n_clusters, metric=metric, random_state=42)

    elif algorithm == "DBSCAN":
        eps = st.sidebar.number_input("eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        min_samples = st.sidebar.slider("min_samples", min_value=1, max_value=20, value=5)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    elif algorithm == "Hierarchical Clustering":
        n_clusters = st.sidebar.slider("n_clusters", min_value=2, max_value=15, value=3)
        linkage = st.sidebar.selectbox("linkage", ["ward", "complete", "average", "single"])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    # 4. OUTPUT & VISUALIZATION & EVALUATION
    if model is not None:
        labels = model.fit_predict(X_final)
        
        # Display cluster labels & Show clustered dataset
        df_clustered = df.copy()
        df_clustered['Cluster'] = labels.astype(str)

        st.subheader("Clustered Dataset Preview")
        st.dataframe(df_clustered.head())

        # Visualization
        st.subheader("Visualization")
        if X_final.shape[1] > 2:
            st.info("More than 2 features selected → applying PCA to 2D for plot.")
            pca_vis = PCA(n_components=2)
            X_plot = pca_vis.fit_transform(X_final)
            xlabel, ylabel = "PCA 1", "PCA 2"
        else:
            X_plot = X_final
            xlabel, ylabel = "Feature 1", "Feature 2"

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=labels, palette="tab10", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Clustering Results ({algorithm})")
        st.pyplot(fig)

        # Evaluation
        st.subheader("Evaluation Metrics")
        unique_labels = set(labels) - {-1}
        if len(unique_labels) > 1:
            valid_mask = labels != -1
            if np.sum(valid_mask) > 0 and len(set(labels[valid_mask])) > 1:
                sil_score = silhouette_score(X_final[valid_mask], labels[valid_mask])
                db_score = davies_bouldin_score(X_final[valid_mask], labels[valid_mask])
                
                c1, c2 = st.columns(2)
                c1.metric("Silhouette Score", f"{sil_score:.4f}")
                c2.metric("Davies-Bouldin Index", f"{db_score:.4f}")
            else:
                st.warning("Not enough clusters to compute metrics.")
        else:
            st.warning("Metrics require at least 2 valid clusters.")

else:
    st.info("Please upload a CSV file in the sidebar to get started.")