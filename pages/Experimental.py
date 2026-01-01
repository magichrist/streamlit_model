import streamlit as st
from sklearn import datasets
from matplotlib import pyplot as plt
import seaborn as sb
import sklearn
from sklearn.model_selection import train_test_split
import pickle
import time
import sklearn.cluster
import sklearn.naive_bayes

try:
    from yellowbrick.cluster import KElbowVisualizer
except Exception as e:
    st.error("Please install yellowbrick")

sb.set()
keys = st.session_state.keys()
if any(i not in keys for i in
       ["n_samples", "no_of_saving", "data_preprocessing",
        "n_features", "centers_std", "seed", "xcenters", "lock_seed", "xdata", "xy", "model"]):
    st.session_state.n_samples = 20
    st.session_state.n_features = 3
    st.session_state.xcenters = 2
    st.session_state.pred_center = 2
    st.session_state.xdata = None
    st.session_state.xy = None
    st.session_state.model = "KMeans"
    st.session_state.centers = 2
    st.session_state.centers_std = 1
    st.session_state.seed = 42
    st.session_state.lock_seed = True
    st.session_state.data_preprocessing = None
    st.session_state.no_of_saving = 0

st.cache_resource(show_spinner="Loading normalizer")


def fixer_preprocessing():
    try:
        fixer = getattr(sklearn.preprocessing, st.session_state.data_preprocessing)
        return fixer
    except Exception as e:
        st.error(e)


st.cache_data(show_spinner="Creating data ...")


def generate_df():
    n_samples = st.session_state.n_samples
    n_features = st.session_state.n_features
    centers = st.session_state.centers
    centers_std = st.session_state.centers_std
    data, y, centers = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=int(centers),
                                           shuffle=True, return_centers=True, cluster_std=float(centers_std),
                                           random_state=st.session_state.seed if st.session_state.lock_seed else None)
    if st.session_state.data_preprocessing:
        fixer = fixer_preprocessing()
        if fixer:
            f = fixer()
            data = f.fit_transform(data)
            centers = f.transform(centers)
    return data, y, centers


st.cache_data()


def poof(data, y, centers):
    fig, ax = plt.subplots()
    ax.set_title(
        f"Generated {st.session_state.n_samples}dp features:{st.session_state.n_features}\ncenters:{st.session_state.centers}")
    ax.scatter(data[:, :1], data[:, 1:2], c=y, cmap="inferno")
    ax.scatter(centers.T[0], centers.T[1])
    return fig


if "first_time" not in st.session_state:
    st.session_state.first_time = False
    st.session_state.xdata, st.session_state.xy, st.session_state.xcenters = generate_df()

st.cache_resource(show_spinner="Loading model ...")


def model_train():
    data = st.session_state.xdata
    y = st.session_state.xy
    model_input = st.session_state.model
    if model_input:
        if model_input == "KNeighborsClassifier":
            final_model = getattr(sklearn.neighbors, model_input)(n_neighbors=st.session_state.pred_center)
        elif model_input == "CategoricalNB" or model_input == "MultinomialNB":
            final_model = getattr(sklearn.naive_bayes, model_input)()
        elif model_input == "KMeans":
            final_model = getattr(sklearn.cluster, model_input)(n_clusters=st.session_state.pred_center)
        elif model_input == "DecisionTreeClassifier":
            final_model = getattr(sklearn.tree, model_input)()
        else:
            st.error("Unknow model or bad parameter!")
    x_train, x_test, y_train, y_test = train_test_split(data, y, shuffle=True, test_size=0.3)
    m = final_model
    m.fit(x_train, y=y_train)
    preds = m.predict(x_test)
    if model_input == "KNeighborsClassifier":
        center = m.classes_
    elif model_input == "CategoricalNB" or model_input == "MultinomialNB":
        center = m.classes_
    elif model_input == "KMeans":
        center = m.cluster_centers_
    elif model_input == "DecisionTreeClassifier":
        center = m.classes_
    col1, col2 = st.columns([0.75, 0.25])
    col1.code(sklearn.metrics.classification_report(y_test, preds))
    if col2.button("KElbow"):
        try:
            visualizer = KElbowVisualizer(m, k=(1, 20))
            visualizer.fit(x_train)
            fig = plt.gcf()
            fig.set_size_inches(15, 10)
            fig.tight_layout()
            visualizer.show(outpath=None)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.clf()
        except Exception as e:
            st.error("KElbow failed.")
    col2.download_button("Download model", data=pickle.dumps(m), file_name=f"{model_input}_{time.time()}.pkl",
                         mime="application/octet-stream")
    return x_test, preds, center


Tab1, Tab2 = st.tabs(["Generation", "Model"])
with Tab1:
    st.header("Data Generation", divider="grey", help="Generate Data for model training")
    with st.form(key="form_data"):
        col1, col2, col3 = st.columns(3)
        col1.number_input(label="n_samples", step=10, min_value=100,
                          max_value=3000, key="n_samples")
        col2.number_input(label="n_features", step=1, min_value=2,
                          max_value=20, key="n_features")
        col3.number_input(label="centers", step=1, min_value=2,
                          max_value=10, key="centers")
        col1.slider(label="cluster_std", step=0.2, min_value=0.5,
                    max_value=9.0, key="centers_std")
        col2.selectbox(label="Preprocessing", placeholder="Select a type",
                       options=[None, "RobustScaler", "StandardScaler", "QuantileTransformer", "PowerTransformer",
                                "Normalizer"],
                       key="data_preprocessing")
        col3.number_input("Seed", step=1, min_value=1, key="seed", disabled=not st.session_state.lock_seed)
        col3.checkbox("Use Seed", key="lock_seed")

        if st.form_submit_button(type="secondary"):
            st.session_state.xdata, st.session_state.xy, st.session_state.xcenters = generate_df()
    if st.session_state.xdata is not None and st.session_state.xy is not None:
        st.pyplot(fig=poof(st.session_state.xdata, st.session_state.xy, st.session_state.xcenters))

with Tab2:
    st.header("Model Training", divider="grey", help="Generate Data for model training")
    with st.form(key="model_form"):
        col1, col2, col3 = st.columns(3)
        col1.selectbox(label="Model",
                       options=["KNeighborsClassifier",
                                "KMeans", "CategoricalNB", "MultinomialNB",
                                "DecisionTreeClassifier"], key="model")
        col2.number_input("pred_centers", key="pred_center", value=2)

        st.form_submit_button(type="secondary")
    if st.session_state.xdata is not None and st.session_state.xy is not None:
        st.pyplot(fig=poof(*model_train()))

st.sidebar.button("clear cache", on_click=lambda: st.cache_data.clear())
