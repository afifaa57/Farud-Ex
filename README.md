# ===========================================================
# Intelligent Detection Dashboard
# Modules:
# - Email Spam Detection
# - Fake News Detection
# - Transaction Fraud Analysis
# ===========================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import io, base64, warnings
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ===========================================================
# Utility Functions
# ===========================================================
def fig_to_base64(fig):
    """Convert a Matplotlib figure into a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


# ===========================================================
# Fraud Detection Model
# ===========================================================
def create_sample_fraud_data(n_samples=30000, random_state=42):
    """Generate synthetic transaction data with realistic fraud patterns."""
    np.random.seed(random_state)
    df = pd.DataFrame({
        "transaction_amount": np.random.uniform(1, 2000, n_samples),
        "transaction_time_hour": np.random.randint(0, 24, n_samples),
        "location_type": np.random.choice(["online", "in-store", "atm"], n_samples, p=[0.5, 0.35, 0.15]),
        "merchant_category": np.random.choice(
            ["retail", "travel", "utility", "entertainment", "other"], n_samples
        ),
        "card_present": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        "num_prev_transactions_24hr": np.random.randint(0, 60, n_samples),
        "avg_prev_amount_24hr": np.random.uniform(1, 800, n_samples),
        "is_international": np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
        "distance_from_home_km": np.random.uniform(0, 3000, n_samples),
        "ip_country_match_card_country": np.random.choice([0, 1], n_samples, p=[0.12, 0.88]),
    })

    # Weighted logic to simulate fraud risk patterns
    risk_score = (
        (df["transaction_amount"] > 800).astype(int) * 0.18 +
        (df["transaction_time_hour"].isin([0, 1, 2, 3, 4])).astype(int) * 0.08 +
        (df["location_type"] == "online").astype(int) * 0.12 +
        (df["is_international"] == 1).astype(int) * 0.18 +
        (df["ip_country_match_card_country"] == 0).astype(int) * 0.14 +
        (df["distance_from_home_km"] > 500).astype(int) * 0.09 +
        ((df["transaction_amount"] / (df["avg_prev_amount_24hr"] + 1)) > 3).astype(int) * 0.12 +
        (df["card_present"] == 0).astype(int) * 0.05 +
        (df["num_prev_transactions_24hr"] > 20).astype(int) * 0.05
    )

    df["is_fraud"] = ((risk_score + np.random.uniform(0, 0.25, n_samples)) > 0.45).astype(int)
    return df


def train_fraud_model(random_state=42):
    """Train an XGBoost model on the synthetic fraud dataset."""
    df = create_sample_fraud_data(30000, random_state)
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    categorical_features = ["location_type", "merchant_category"]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=random_state,
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    model_pipeline.fit(X_train, y_train)
    return model_pipeline, X_train, df


@st.cache_resource
def get_fraud_model():
    """Cache model and preprocessor to avoid reloading."""
    model_pipeline, X_train, df = train_fraud_model()
    xgb_model = model_pipeline.named_steps["classifier"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    return model_pipeline, xgb_model, preprocessor, X_train, df


# ===========================================================
# Fraud Detection Interface
# ===========================================================
def fraud_app():
    st.header("Transaction Fraud Detection")

    model_pipeline, xgb_model, feature_preprocessor, X_train_raw, df = get_fraud_model()
    feature_names = list(feature_preprocessor.get_feature_names_out())

    st.write("Enter transaction details manually or upload a dataset for analysis:")

    tab1, tab2 = st.tabs(["Manual Entry", "Upload File"])

    with tab1:
        transaction_amount = st.number_input("Transaction Amount ($)", 1.0, 5000.0, 150.0)
        transaction_time_hour = st.slider("Transaction Hour (0–23)", 0, 23, 10)
        location_type = st.selectbox("Location Type", ["online", "in-store", "atm"])
        merchant_category = st.selectbox("Merchant Category", ["retail", "travel", "utility", "entertainment", "other"])
        card_present = st.checkbox("Card Present", value=True)
        num_prev_transactions_24hr = st.number_input("Previous Transactions (24hr)", 0, 100, 2)
        avg_prev_amount_24hr = st.number_input("Average Amount (24hr)", 1.0, 5000.0, 80.0)
        is_international = st.checkbox("International Transaction", value=False)
        distance_from_home_km = st.number_input("Distance from Home (km)", 0.0, 10000.0, 10.0)
        ip_country_match_card_country = st.checkbox("IP Country Matches Card Country", value=True)

        new_tx = pd.DataFrame([{
            "transaction_amount": transaction_amount,
            "transaction_time_hour": transaction_time_hour,
            "location_type": location_type,
            "merchant_category": merchant_category,
            "card_present": int(card_present),
            "num_prev_transactions_24hr": num_prev_transactions_24hr,
            "avg_prev_amount_24hr": avg_prev_amount_24hr,
            "is_international": int(is_international),
            "distance_from_home_km": distance_from_home_km,
            "ip_country_match_card_country": int(ip_country_match_card_country)
        }])

        if st.button("Analyze Transaction", key="manual"):
            prob = model_pipeline.predict_proba(new_tx)[0, 1]
            pred = int(prob > 0.5)
            st.metric("Fraud Probability", f"{prob:.2%}")
            if pred == 1:
                st.error("Potentially fraudulent transaction detected.")
            else:
                st.success("Transaction appears legitimate.")

            st.markdown("---")
            st.subheader("Explanation and Insights")

            try:
                X = feature_preprocessor.transform(new_tx)
                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer(X)
                shap.plots.waterfall(shap_values[0], show=False)
                shap_img = fig_to_base64(plt.gcf())
                st.image(f"data:image/png;base64,{shap_img}", caption="Feature Impact (SHAP)")
            except Exception as e:
                st.error(f"SHAP Visualization Error: {e}")

            try:
                X_train_transformed = feature_preprocessor.transform(X_train_raw)
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_transformed,
                    feature_names=feature_names,
                    class_names=["Legit", "Fraud"],
                    discretize_continuous=True
                )
                X_new = feature_preprocessor.transform(new_tx)
                exp = lime_explainer.explain_instance(X_new[0], xgb_model.predict_proba, num_features=8)
                fig = exp.as_pyplot_figure()
                lime_img = fig_to_base64(fig)
                st.image(f"data:image/png;base64,{lime_img}", caption="Local Explanation (LIME)")
            except Exception as e:
                st.error(f"LIME Visualization Error: {e}")

    with tab2:
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df_uploaded = pd.read_csv(uploaded)
                else:
                    df_uploaded = pd.read_json(uploaded)
                st.write(f"Loaded {df_uploaded.shape[0]} records.")
                if st.button("Run Analysis", key="file"):
                    probs = model_pipeline.predict_proba(df_uploaded)[:, 1]
                    df_uploaded["fraud_probability"] = probs
                    df_uploaded["prediction"] = (probs > 0.5).astype(int)
                    st.dataframe(df_uploaded)
            except Exception as e:
                st.error(f"File Processing Error: {e}")


# ===========================================================
# Email Spam and Fake News Detection
# ===========================================================
@st.cache_resource
def create_text_models():
    """Train simple models for email spam and fake news detection."""
    def generate_text_data(n, pos_kw, neg_kw, pos_frac, context):
        texts, labels = [], []
        for _ in range(n):
            if np.random.rand() < pos_frac:
                k = np.random.choice(pos_kw, size=np.random.randint(1, 3))
                texts.append(" ".join(k) + f" urgent {context} now!")
                labels.append(1)
            else:
                k = np.random.choice(neg_kw, size=np.random.randint(1, 3))
                texts.append(" ".join(k) + f" regular {context}.")
                labels.append(0)
        return texts, labels

    spam_pos = ["buy", "free", "winner", "click", "offer"]
    spam_neg = ["meeting", "report", "invoice", "project"]
    fake_pos = ["shocking", "conspiracy", "miracle", "exposed", "secret"]
    fake_neg = ["study", "official", "research", "survey"]

    datasets = {
        "email": generate_text_data(4000, spam_pos, spam_neg, 0.15, "email"),
        "news": generate_text_data(4000, fake_pos, fake_neg, 0.1, "article"),
    }

    def train_pipeline(texts, labels):
        X, y = np.array(texts), np.array(labels)
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=5000),
            LogisticRegression(solver="liblinear", max_iter=1000)
        )
        pipe.fit(X, y)
        return pipe

    return {
        "email": train_pipeline(*datasets["email"]),
        "news": train_pipeline(*datasets["news"]),
    }


text_models = create_text_models()
lime_explainers = {
    "email": LimeTextExplainer(class_names=["legit", "spam"]),
    "news": LimeTextExplainer(class_names=["legit", "fake"]),
}


def email_app():
    st.header("Email Spam Detection")
    model = text_models["email"]
    text = st.text_area("Paste email text:", value="Congratulations! You have won a FREE iPhone. Click here!")
    if st.button("Analyze"):
        prob = model.predict_proba([text])[0, 1]
        st.metric("Spam Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.error("Spam detected.")
        else:
            st.success("Email appears legitimate.")

        exp = lime_explainers["email"].explain_instance(text, model.predict_proba, num_features=10)
        components.html(exp.as_html(), height=400)


def news_app():
    st.header("Fake News Detection")
    model = text_models["news"]
    text = st.text_area("Paste article text:", value="Shocking! Scientists EXPOSE secret cure they don’t want you to know.")
    if st.button("Analyze"):
        prob = model.predict_proba([text])[0, 1]
        st.metric("Fake Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.error("Likely fake news.")
        else:
            st.success("Article appears credible.")

        exp = lime_explainers["news"].explain_instance(text, model.predict_proba, num_features=10)
        components.html(exp.as_html(), height=400)


# ===========================================================
# Dashboard Layout
# ===========================================================
st.set_page_config(page_title="Intelligent Detection Dashboard", layout="wide")
st.title("Unified Detection Dashboard")

choice = st.sidebar.radio("Select Module:", ["Fraud Detection", "Email Spam Detector", "Fake News Detector"])

if choice == "Fraud Detection":
    fraud_app()
elif choice == "Email Spam Detector":
    email_app()
else:
    news_app()

st.markdown("---")
st.caption("Developed using Streamlit, XGBoost, SHAP, LIME, and Scikit-learn")
