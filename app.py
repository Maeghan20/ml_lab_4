
# ============================================================
# MAI511-2 - ADVANCED MACHINE LEARNING
# LAB EXERCISE 4
# COMPARATIVE ANALYSIS OF PCA, LDA AND ICA
# Dataset: Breast Cancer Wisconsin
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCA, LDA and ICA | Advanced ML Lab",
    page_icon="🧬",
    layout="wide"
)

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 15px;
        color: #808b99;
        margin-bottom: 20px;
    }

    .section-note {
        padding: 15px 18px;
        border-left: 4px solid #4776a8;
        border-radius: 6px;
        background-color: rgba(90, 135, 180, 0.12);
        margin: 15px 0 20px 0;
    }

    .section-note p {
        margin-bottom: 5px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(
        dataset.target,
        name="Diagnosis"
    )

    return X, y


X, y = load_dataset()

class_names = {
    0: "Malignant",
    1: "Benign"
}

y_labels = y.map(class_names)


# ============================================================
# 3. GLOBAL PREPROCESSING
# ============================================================

# Standardisation is performed globally so that X_scaled
# is available in every navigation section.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def section_intro(objective, explanation):

    st.markdown(
        f"""
        <div class="section-note">
            <p><b>OBJECTIVE</b></p>
            <p>{objective}</p>
            <p>{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_dimensions(original, reduced):

    col1, col2 = st.columns(2)

    col1.metric(
        "Original Features",
        original
    )

    col2.metric(
        "Reduced Components",
        reduced
    )

    st.write(
        f"Original dimensions: {X.shape}"
    )

    st.write(
        f"Reduced dimensions: {reduced_data_shape(reduced)}"
    )


def reduced_data_shape(number_of_components):

    return (X.shape[0], number_of_components)


def plot_scatter(
    transformed_data,
    x_column,
    y_column,
    title,
    xlabel,
    ylabel
):

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {
        "Malignant": "tomato",
        "Benign": "royalblue"
    }

    for label in ["Malignant", "Benign"]:

        mask = (y_labels == label).to_numpy()

        ax.scatter(
            transformed_data[mask, x_column],
            transformed_data[mask, y_column],
            label=label,
            color=colors[label],
            alpha=0.7,
            s=35
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Diagnosis")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("ADVANCED MACHINE LEARNING")
st.sidebar.caption("MAI511-2 | Lab Exercise 4")

st.sidebar.divider()

st.sidebar.subheader("Experiment Navigation")

section = st.sidebar.radio(
    "Select a section",
    [
        "1. Dataset Loading",
        "2. Data Preprocessing",
        "3. Principal Component Analysis (PCA)",
        "4. Linear Discriminant Analysis (LDA)",
        "5. Independent Component Analysis (ICA)",
        "6. Comparative Analysis",
        "7. Self-Learning Extension",
        "8. Conclusion"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("""
**Dataset:** Breast Cancer Wisconsin

**Techniques:**
- PCA
- LDA
- ICA

**Language:** Python
""")

st.sidebar.divider()

st.sidebar.caption(
    "Dimensionality Reduction Techniques"
)


# ============================================================
# 6. MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Comparative Analysis of PCA, LDA and ICA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">MAI511-2 | Advanced Machine Learning | Lab Exercise 4</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SECTION 1: DATASET LOADING
# ============================================================

if section == "1. Dataset Loading":

    st.header("1. Dataset Loading and Understanding")

    section_intro(
        "Load and understand the Breast Cancer Wisconsin dataset "
        "available in the Scikit-learn library.",
        "The dataset contains numerical measurements of cell nuclei "
        "obtained from breast mass samples. These measurements are "
        "used to distinguish between malignant and benign cases."
    )

    st.subheader("1.1 Loading the Dataset")

    st.write("""
    The Breast Cancer Wisconsin dataset is loaded using
    Scikit-learn's built-in `load_breast_cancer()` function.

    The dataset contains numerical measurements describing
    the characteristics of cell nuclei.
    """)

    st.code("""
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

y = pd.Series(dataset.target)
    """, language="python")

    st.subheader("1.2 Dataset Dimensions")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", X.shape[0])
    c2.metric("Numerical Features", X.shape[1])
    c3.metric("Target Classes", y.nunique())

    st.success(
        f"Original dataset dimensions: {X.shape[0]} × {X.shape[1]}"
    )

    st.subheader("1.3 Understanding the Target Classes")

    st.write("""
    The dataset contains two target classes:

    - **Malignant (0):** Represents cancerous tumour samples.
    - **Benign (1):** Represents non-cancerous tumour samples.
    """)

    class_distribution = (
        y_labels.value_counts()
        .rename_axis("Diagnosis")
        .reset_index(name="Number of Records")
    )

    st.dataframe(
        class_distribution,
        width="stretch",
        hide_index=True
    )

    st.subheader("1.4 Sample Dataset Records")

    st.write("""
    The following table displays the first five observations
    and their corresponding numerical feature values.
    """)

    sample = X.head().copy()

    sample["Diagnosis"] = y_labels.head().values

    st.dataframe(
        sample,
        width="stretch"
    )

    st.subheader("1.5 Feature Information")

    st.write("""
    Each observation contains 30 numerical features.
    These features describe properties such as radius,
    texture, perimeter, area, smoothness and symmetry.
    """)

    feature_info = pd.DataFrame({
        "Feature": X.columns,
        "Data Type": X.dtypes.astype(str).values,
        "Mean": X.mean().round(3).values,
        "Standard Deviation": X.std().round(3).values
    })

    st.dataframe(
        feature_info,
        width="stretch",
        hide_index=True
    )

    st.info("""
    **Observation:** The dataset contains 569 observations,
    30 numerical input features and two diagnosis classes.
    Since the features have different numerical scales,
    standardisation is required before applying dimensionality
    reduction techniques.
    """)


# ============================================================
# SECTION 2: DATA PREPROCESSING
# ============================================================

elif section == "2. Data Preprocessing":

    st.header("2. Data Preprocessing")

    section_intro(
        "Prepare the dataset before applying dimensionality "
        "reduction techniques.",
        "Preprocessing includes separating the input features "
        "and target labels, checking missing values and "
        "standardising numerical features."
    )

    st.subheader("2.1 Separating Features and Target")

    st.code("""
X = dataset.data
y = dataset.target
    """, language="python")

    st.write("""
    - X represents the independent input features.
    - y represents the dependent target variable.

    The features are used for dimensionality reduction,
    while the target labels are additionally used by LDA.
    """)

    st.subheader("2.2 Checking Missing Values")

    missing_values = X.isnull().sum()

    missing_table = pd.DataFrame({
        "Feature": X.columns,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_table,
        width="stretch",
        hide_index=True
    )

    total_missing = int(missing_values.sum())

    if total_missing == 0:
        st.success("No missing values were found.")
    else:
        st.warning(
            f"{total_missing} missing values were detected."
        )

    st.subheader("2.3 Feature Standardisation")

    st.write("""
    Standardisation converts numerical features into a
    common scale.

    StandardScaler transforms each feature so that its
    mean is approximately zero and its standard deviation
    is approximately one.

    This is important because features such as area and
    perimeter have much larger numerical values than
    features such as smoothness and symmetry.

    Without standardisation, features with larger scales
    could dominate the dimensionality reduction process.
    """)

    st.code("""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
    """, language="python")

    st.subheader("2.4 Before and After Standardisation")

    selected_feature = st.selectbox(
        "Select a feature to examine",
        X.columns
    )

    before_mean = X[selected_feature].mean()
    before_std = X[selected_feature].std()

    after_mean = scaled_df[selected_feature].mean()
    after_std = scaled_df[selected_feature].std(ddof=0)

    stats = pd.DataFrame({
        "Measurement": [
            "Mean",
            "Standard Deviation"
        ],
        "Before Standardisation": [
            round(before_mean, 4),
            round(before_std, 4)
        ],
        "After Standardisation": [
            round(after_mean, 4),
            round(after_std, 4)
        ]
    })

    st.dataframe(
        stats,
        width="stretch",
        hide_index=True
    )

    st.subheader("2.5 Standardised Dataset")

    st.write(
        "First five records after standardisation:"
    )

    st.dataframe(
        scaled_df.head(),
        width="stretch"
    )

    st.success(
        f"Standardisation completed. New dimensions: {X_scaled.shape}"
    )

    st.info("""
    **Observation:** The standardised dataset retains the
    same number of observations and features. However,
    all numerical features are now on comparable scales.
    """)

    st.caption(
        "Note: Standardisation is performed globally here for "
        "the prescribed exploratory dimensionality reduction. "
        "The separate classification experiment uses training-only "
        "fitting to prevent data leakage."
    )


# ============================================================
# SECTION 3: PRINCIPAL COMPONENT ANALYSIS
# ============================================================

elif section == "3. Principal Component Analysis (PCA)":

    st.header("3. Principal Component Analysis (PCA)")

    section_intro(
        "Apply PCA to reduce the dimensionality of the "
        "standardised dataset while retaining maximum variance.",
        "PCA is an unsupervised technique. It does not use "
        "the diagnosis labels during component extraction."
    )

    st.subheader("3.1 Understanding PCA")

    st.write("""
    Principal Component Analysis is a dimensionality reduction
    technique that transforms the original correlated features
    into new uncorrelated variables called principal components.

    The first principal component captures the maximum possible
    variance in the dataset.

    Each subsequent component captures the maximum remaining
    variance while remaining orthogonal to the previous components.
    """)

    st.write("**Primary objective: Maximum variance retention.**")

    st.subheader("3.2 Applying PCA")

    st.write("""
    PCA is applied to the standardised dataset.

    Select a variance threshold to determine how many
    components are required to retain the desired amount
    of information.
    """)

    variance_target = st.slider(
        "Select desired cumulative variance",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f"
    )

    pca = PCA(
        n_components=variance_target,
        random_state=42
    )

    X_pca = pca.fit_transform(X_scaled)

    st.code("""
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
    """, language="python")

    st.subheader("3.3 Reduced Dataset Dimensions")

    c1, c2, c3 = st.columns(3)

    c1.metric("Original Features", X.shape[1])

    c2.metric(
        "PCA Components",
        X_pca.shape[1]
    )

    c3.metric(
        "Variance Retained",
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    st.write(
        f"Original dimensions: {X.shape}"
    )

    st.write(
        f"Reduced dimensions: {X_pca.shape}"
    )

    st.subheader("3.4 Explained Variance Ratio")

    st.write("""
    Explained variance ratio represents the proportion
    of total variance captured by each principal component.

    A higher explained variance ratio indicates that
    the component captures more variation from the dataset.
    """)

    variance_df = pd.DataFrame({
        "Component": [
            f"PC{i+1}" for i in range(X_pca.shape[1])
        ],
        "Explained Variance Ratio": (
            pca.explained_variance_ratio_
        ),
        "Percentage": (
            pca.explained_variance_ratio_ * 100
        )
    })

    st.dataframe(
        variance_df.round(4),
        width="stretch",
        hide_index=True
    )

    st.subheader("3.5 Cumulative Explained Variance")

    pca_all = PCA()
    pca_all.fit(X_scaled)

    cumulative_variance = np.cumsum(
        pca_all.explained_variance_ratio_
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        range(1, 31),
        cumulative_variance,
        marker="o",
        linewidth=2
    )

    ax.axhline(
        y=variance_target,
        linestyle="--",
        label=f"{variance_target:.0%} variance target"
    )

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    components_needed = (
        np.argmax(cumulative_variance >= variance_target) + 1
    )

    st.success(
        f"{components_needed} components are required to "
        f"retain at least {variance_target:.0%} variance."
    )

    st.subheader("3.6 PCA Scatter Plot")

    st.write("""
    The first two principal components are used to
    visualise the transformed dataset in two dimensions.

    The diagnosis labels are used only to colour the
    observations for interpretation. PCA does not
    use these labels during transformation.
    """)

    pca_2d = PCA(n_components=2)

    X_pca_2d = pca_2d.fit_transform(X_scaled)

    plot_scatter(
        X_pca_2d,
        0,
        1,
        "PCA - First Two Principal Components",
        "Principal Component 1",
        "Principal Component 2"
    )

    st.info(f"""
    **Observation:** The selected PCA representation uses
    {X_pca.shape[1]} components instead of the original
    30 features and retains {pca.explained_variance_ratio_.sum():.2%}
    of the total variance.
    """)


# ============================================================
# SECTION 4: LINEAR DISCRIMINANT ANALYSIS
# ============================================================

elif section == "4. Linear Discriminant Analysis (LDA)":

    st.header("4. Linear Discriminant Analysis (LDA)")

    section_intro(
        "Project the dataset into a lower-dimensional space "
        "while considering the diagnosis class labels.",
        "LDA is a supervised technique that attempts to maximise "
        "between-class separation while minimising within-class "
        "variation."
    )

    st.subheader("4.1 Understanding LDA")

    st.write("""
    Linear Discriminant Analysis is a supervised dimensionality
    reduction technique.

    Unlike PCA, LDA uses the target class labels during
    the transformation process.

    LDA attempts to:

    - Maximise the distance between different class means.
    - Minimise the variation within each class.

    This produces a representation intended to distinguish
    between different classes.
    """)

    st.write("**Primary objective: Maximum class separation.**")

    st.subheader("4.2 Applying LDA")

    st.write("""
    The dataset contains two target classes:

    - Malignant
    - Benign

    The maximum number of LDA components is calculated using:

    Number of classes - 1

    Therefore, the maximum number of components is:

    2 - 1 = 1
    """)

    st.code("""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)

X_lda = lda.fit_transform(X_scaled, y)
    """, language="python")

    lda = LinearDiscriminantAnalysis(
        n_components=1
    )

    X_lda = lda.fit_transform(
        X_scaled,
        y
    )

    st.subheader("4.3 Reduced Dataset Dimensions")

    c1, c2, c3 = st.columns(3)

    c1.metric("Original Features", X.shape[1])

    c2.metric("Target Classes", y.nunique())

    c3.metric("LDA Components", X_lda.shape[1])

    st.write(
        f"Original dimensions: {X.shape}"
    )

    st.write(
        f"Reduced dimensions: {X_lda.shape}"
    )

    st.subheader("4.4 LDA Class Separation")

    st.write("""
    The following visualisation displays the distribution
    of malignant and benign observations along the
    discriminant direction.

    Greater separation between the two distributions
    indicates better class discrimination.
    """)

    rng = np.random.default_rng(42)

    jitter = rng.normal(
        0,
        0.025,
        size=len(y)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for value, label, color in [
        (0, "Malignant", "tomato"),
        (1, "Benign", "royalblue")
    ]:

        mask = (y.to_numpy() == value)

        ax.scatter(
            X_lda[mask, 0],
            jitter[mask],
            label=label,
            color=color,
            alpha=0.65,
            s=35
        )

    ax.set_xlabel("Linear Discriminant Component 1")
    ax.set_ylabel("Display jitter (not an LDA feature)")
    ax.set_yticks([])
    ax.set_title("LDA - Class Separation")
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4.5 Interpretation")

    st.info("""
    **Observation:** LDA transforms the original 30 features
    into one discriminant component. It uses the diagnosis
    labels to find a direction that maximises class separation.

    A two-dimensional LDA plot is not possible for this
    binary dataset because only one discriminant component
    can be extracted.
    """)


# ============================================================
# SECTION 5: INDEPENDENT COMPONENT ANALYSIS
# ============================================================

elif section == "5. Independent Component Analysis (ICA)":

    st.header("5. Independent Component Analysis (ICA)")

    section_intro(
        "Extract statistically independent components "
        "from the standardised dataset.",
        "ICA is an unsupervised technique that attempts to "
        "identify components that are statistically independent "
        "of one another."
    )

    st.subheader("5.1 Understanding ICA")

    st.write("""
    Independent Component Analysis is a feature extraction
    technique that attempts to separate observed data into
    statistically independent components.

    It is based on the idea that observed variables may
    contain combinations of underlying independent sources.

    ICA is commonly used in signal processing, source
    separation and EEG analysis.
    """)

    st.write("**Primary objective: Statistical independence.**")

    st.subheader("5.2 Applying ICA")

    st.write("""
    Two components are selected initially to visualise
    the transformed dataset in two dimensions.

    The number of components can be changed below.
    """)

    n_ica = st.slider(
        "Select number of independent components",
        min_value=2,
        max_value=10,
        value=2
    )

    st.code("""
from sklearn.decomposition import FastICA

ica = FastICA(
    n_components=2,
    random_state=42,
    max_iter=2000,
    tol=0.001,
    whiten="unit-variance"
)

X_ica = ica.fit_transform(X_scaled)
    """, language="python")

    ica = FastICA(
        n_components=n_ica,
        random_state=42,
        max_iter=2000,
        tol=0.001,
        whiten="unit-variance"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        X_ica = ica.fit_transform(X_scaled)

    st.subheader("5.3 Extracted Component Dimensions")

    c1, c2, c3 = st.columns(3)

    c1.metric("Original Features", X.shape[1])

    c2.metric(
        "Independent Components",
        X_ica.shape[1]
    )

    c3.metric(
        "Iterations Used",
        ica.n_iter_
    )

    st.write(
        f"Original dimensions: {X.shape}"
    )

    st.write(
        f"Reduced dimensions: {X_ica.shape}"
    )

    if ica.n_iter_ >= 2000:
        st.warning(
            "ICA reached the iteration limit. "
            "Convergence may require a higher max_iter."
        )
    else:
        st.success(
            "ICA completed within the specified iteration limit."
        )

    st.subheader("5.4 ICA Scatter Plot")

    st.write("""
    The first two independent components are visualised
    to examine the transformed dataset.

    The original diagnosis labels are used only to colour
    the observations for interpretation.
    """)

    plot_scatter(
        X_ica,
        0,
        1,
        "ICA - First Two Independent Components",
        "Independent Component 1",
        "Independent Component 2"
    )

    st.subheader("5.5 Examining Independent Components")

    st.write("""
    ICA components are combinations of the original features.

    The component weights indicate how the original features
    contribute to each extracted independent component.

    The following tables display the eight features with
    the largest absolute weights for each of the first
    two components.
    """)

    for i in range(2):

        weights = pd.DataFrame({
            "Feature": X.columns,
            "Component Weight": ica.components_[i],
            "Absolute Weight": np.abs(ica.components_[i])
        })

        weights = weights.sort_values(
            "Absolute Weight",
            ascending=False
        ).head(8)

        st.write(
            f"**Independent Component {i+1}**"
        )

        st.dataframe(
            weights.round(4),
            width="stretch",
            hide_index=True
        )

    st.info("""
    **Observation:** ICA extracts statistically independent
    components from the original feature space. Unlike PCA,
    the components are not ranked according to explained
    variance. Their signs and ordering are not inherently fixed.
    """)


# ============================================================
# SECTION 6: COMPARATIVE ANALYSIS
# ============================================================

elif section == "6. Comparative Analysis":

    st.header("6. Comparative Analysis of PCA, LDA and ICA")

    section_intro(
        "Compare PCA, LDA and ICA based on their learning "
        "approaches, objectives, use of class labels and outputs.",
        "The comparison helps identify which technique is "
        "appropriate for different dimensionality reduction "
        "and feature extraction problems."
    )

    st.subheader("6.1 Comparison of Techniques")

    comparison = pd.DataFrame({

        "Parameter": [
            "Learning Approach",
            "Uses Class Labels",
            "Primary Objective",
            "Main Output",
            "Common Applications"
        ],

        "PCA": [
            "Unsupervised",
            "No",
            "Maximum Variance",
            "Principal Components",
            "Data Compression"
        ],

        "LDA": [
            "Supervised",
            "Yes",
            "Maximum Class Separation",
            "Discriminant Components",
            "Classification"
        ],

        "ICA": [
            "Unsupervised",
            "No",
            "Statistical Independence",
            "Independent Components",
            "Signal Processing"
        ]

    })

    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True
    )

    st.subheader("6.2 Comparison of Extracted Components")

    pca_95 = PCA(n_components=0.95)

    X_pca_95 = pca_95.fit_transform(X_scaled)

    lda_compare = LinearDiscriminantAnalysis(
        n_components=1
    )

    X_lda_compare = lda_compare.fit_transform(
        X_scaled,
        y
    )

    ica_compare = FastICA(
        n_components=2,
        random_state=42,
        max_iter=2000,
        tol=0.001,
        whiten="unit-variance"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        X_ica_compare = ica_compare.fit_transform(X_scaled)

    dimensions = pd.DataFrame({

        "Technique": ["PCA", "LDA", "ICA"],

        "Original Features": [30, 30, 30],

        "Reduced Components": [
            X_pca_95.shape[1],
            X_lda_compare.shape[1],
            X_ica_compare.shape[1]
        ],

        "Uses Class Labels": [
            "No",
            "Yes",
            "No"
        ]

    })

    st.dataframe(
        dimensions,
        width="stretch",
        hide_index=True
    )

    st.caption("""
    PCA retains at least 95% variance, LDA extracts the
    maximum possible one component for two classes, and
    ICA extracts two components for comparison.
    """)

    st.subheader("6.3 Understanding the Differences")

    with st.expander("Why does PCA focus on maximum variance?"):

        st.write("""
        PCA is an unsupervised technique and does not receive
        target labels.

        It identifies directions containing the maximum
        overall variation in the dataset.

        However, maximum variance does not necessarily
        indicate maximum class separation.
        """)

    with st.expander("How does LDA use class labels?"):

        st.write("""
        LDA uses the known target labels to calculate
        between-class and within-class variation.

        It searches for directions that maximise the
        separation between different classes.
        """)

    with st.expander("Why does ICA identify independent components?"):

        st.write("""
        ICA attempts to represent observed data using
        statistically independent components.

        It is useful when observed measurements contain
        combinations of underlying independent sources.
        """)

    st.subheader("6.4 Which Technique is More Suitable?")

    st.write("""
    For this dataset, LDA is directly suited to the objective
    of class separation because it uses the malignant and
    benign labels during transformation.

    PCA is useful when the goal is to reduce the number
    of features while retaining most of the total variance.

    ICA is useful when the objective is to identify
    statistically independent patterns in the data.

    Therefore, the most suitable technique depends on
    the objective of the machine learning task.
    """)

    st.info("""
    **Conclusion:** LDA is designed specifically for
    supervised class separation, PCA focuses on variance
    preservation, and ICA focuses on statistical independence.
    """)


# ============================================================
# SECTION 7: SELF-LEARNING EXTENSION
# ============================================================

elif section == "7. Self-Learning Extension":

    st.header("7. Self-Learning Extension")

    section_intro(
        "Evaluate whether dimensionality reduction preserves "
        "information useful for a classification task.",
        "This additional experiment compares Logistic Regression "
        "using the original features and the reduced representations "
        "generated by PCA, LDA and ICA."
    )

    st.subheader("7.1 Objective")

    st.write("""
    The prescribed experiment compares PCA, LDA and ICA
    through their mathematical objectives and visualisations.

    As an additional self-learning experiment, we investigate
    whether the reduced representations retain enough
    information for predicting the diagnosis.

    Logistic Regression is used to classify observations
    as malignant or benign.
    """)

    st.subheader("7.2 Experimental Setup")

    st.write("""
    Four feature representations are compared:

    1. Original dataset containing 30 features.
    2. PCA representation retaining 95% variance.
    3. LDA representation containing one component.
    4. ICA representation containing two components.

    The dataset is split into 80% training data and
    20% testing data using stratified sampling.
    """)

    st.write("""
    **Evaluation metrics:**

    - **Accuracy:** Proportion of correctly classified observations.
    - **Weighted F1-score:** Combines precision and recall while
      accounting for the number of observations in each class.
    """)

    st.subheader("7.3 Classification Implementation")

    st.write("""
    Scikit-learn pipelines are used so that standardisation
    and dimensionality reduction are fitted using training
    data only.

    This prevents information from the test set leaking
    into the training process.
    """)

    st.code("""
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Logistic Regression using:
# 1. Original features
# 2. PCA-reduced features
# 3. LDA-reduced features
# 4. ICA-reduced features
    """, language="python")

    st.warning(
        "Click below to run the classification experiment."
    )

    if st.button(
        "Run Classification Experiment",
        type="primary"
    ):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        models = {

            "Original Features": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=5000,
                    random_state=42
                ))
            ]),

            "PCA (95% Variance)": Pipeline([
                ("scaler", StandardScaler()),
                ("reduction", PCA(
                    n_components=0.95,
                    random_state=42
                )),
                ("classifier", LogisticRegression(
                    max_iter=5000,
                    random_state=42
                ))
            ]),

            "LDA (1 Component)": Pipeline([
                ("scaler", StandardScaler()),
                ("reduction", LinearDiscriminantAnalysis(
                    n_components=1
                )),
                ("classifier", LogisticRegression(
                    max_iter=5000,
                    random_state=42
                ))
            ]),

            "ICA (2 Components)": Pipeline([
                ("scaler", StandardScaler()),
                ("reduction", FastICA(
                    n_components=2,
                    random_state=42,
                    max_iter=3000,
                    tol=0.001,
                    whiten="unit-variance"
                )),
                ("classifier", LogisticRegression(
                    max_iter=5000,
                    random_state=42
                ))
            ])

        }

        results = []

        progress = st.progress(0)

        for index, (name, model) in enumerate(models.items()):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)

                model.fit(
                    X_train,
                    y_train
                )

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted"
            )

            results.append({

                "Representation": name,

                "Accuracy": accuracy,

                "Weighted F1-Score": f1

            })

            progress.progress(
                (index + 1) / len(models)
            )

        results_df = pd.DataFrame(results)

        st.success(
            "Classification experiment completed successfully."
        )

        st.subheader("7.4 Performance Comparison")

        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.4f}",
                "Weighted F1-Score": "{:.4f}"
            }),
            width="stretch",
            hide_index=True
        )

        st.subheader("7.5 Performance Visualisation")

        fig, ax = plt.subplots(figsize=(10, 5))

        results_df.set_index(
            "Representation"
        )[[
            "Accuracy",
            "Weighted F1-Score"
        ]].plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Feature Representation")
        ax.set_ylabel("Evaluation Score")
        ax.set_title("Classification Performance Comparison")
        ax.set_ylim(0, 1.05)
        ax.legend()

        plt.xticks(rotation=15)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("7.6 Best Performing Representation")

        best_result = results_df.loc[
            results_df["Accuracy"].idxmax()
        ]

        st.success(
            f"Highest test accuracy: "
            f"{best_result['Representation']} "
            f"({best_result['Accuracy']:.4f})"
        )

        st.write("""
        **Interpretation:**

        The results show how different feature representations
        affect classification performance.

        Dimensionality reduction may improve performance
        by removing redundant information. However, it may
        also discard useful predictive information.

        The original dataset acts as a baseline for comparing
        the reduced representations.
        """)

        st.caption("""
        This experiment uses one stratified train/test split.
        Results are educational comparisons and should not
        be interpreted as evidence of clinical performance.
        """)


# ============================================================
# SECTION 8: CONCLUSION
# ============================================================

elif section == "8. Conclusion":

    st.header("8. Conclusion and Learning Outcomes")

    section_intro(
        "Summarise the implementation and understand the "
        "differences between PCA, LDA and ICA.",
        "The experiment demonstrates how different mathematical "
        "objectives produce different lower-dimensional "
        "representations of the same dataset."
    )

    st.subheader("8.1 Experimental Summary")

    summary = pd.DataFrame({

        "Stage": [
            "Dataset Loading",
            "Data Preprocessing",
            "PCA",
            "LDA",
            "ICA",
            "Comparative Analysis",
            "Self-Learning"
        ],

        "Implementation": [
            "Loaded Breast Cancer Wisconsin dataset",
            "Checked missing values and standardised features",
            "Reduced dimensions using maximum variance",
            "Projected data using class labels",
            "Extracted independent components",
            "Compared objectives and outputs",
            "Evaluated classification performance"
        ]

    })

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True
    )

    st.subheader("8.2 Final Observations")

    st.markdown("""
    **Principal Component Analysis (PCA)**
    - Unsupervised technique.
    - Preserves maximum variance.
    - Useful for dimensionality reduction and data compression.

    **Linear Discriminant Analysis (LDA)**
    - Supervised technique.
    - Uses class labels to maximise class separation.
    - Limited to one component for this binary dataset.

    **Independent Component Analysis (ICA)**
    - Unsupervised technique.
    - Identifies statistically independent components.
    - Commonly used in signal processing and source separation.
    """)

    st.subheader("8.3 Conclusion")

    st.write("""
    In this experiment, PCA, LDA and ICA were implemented
    and compared using the Breast Cancer Wisconsin dataset.

    The original dataset was standardised before applying
    the three dimensionality reduction techniques.

    PCA reduced the feature space while retaining maximum
    variance. LDA used class labels to identify a projection
    that improves separation between malignant and benign
    observations. ICA extracted statistically independent
    components from the original features.

    The three techniques were compared based on their
    objectives, component counts and visualisation outcomes.

    An additional Logistic Regression experiment was
    performed to evaluate the usefulness of the reduced
    representations for classification.

    The experiment demonstrates that each technique
    has a different objective, and the appropriate
    technique depends on the machine learning problem.
    """)

    st.subheader("8.4 Learning Outcomes")

    st.write("""
    After completing this experiment, the following
    concepts were understood:

    1. Dimensionality reduction and its importance.
    2. Feature standardisation.
    3. Explained variance using PCA.
    4. Class separation using LDA.
    5. Statistical independence using ICA.
    6. Comparison of dimensionality reduction techniques.
    7. Evaluation of reduced features using classification.
    """)

    st.success(
        "Lab Exercise 4 - Implementation and Analysis Completed."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MAI511-2 | Advanced Machine Learning | "
    "Lab Exercise 4"
)

st.caption(
    "Comparative Analysis of PCA, LDA and ICA"
)

st.caption(
    "Educational implementation using Scikit-learn. "
    "Not intended for medical diagnosis."
)
