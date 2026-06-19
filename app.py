import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ImportError:
    StemmerFactory = None
    StopWordRemoverFactory = None


APP_TITLE = "Prediksi Sentimen Saham IDX80 Berbasis Artikel Berita"
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "lstm_sentiment_model.keras"
TOKENIZER_PATH = ARTIFACT_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"
CONFIG_PATH = ARTIFACT_DIR / "config.json"

LABEL_NORMALIZATION = {
    "naik": "Positif",
    "positif": "Positif",
    "positive": "Positif",
    "turun": "Negatif",
    "negatif": "Negatif",
    "negative": "Negatif",
    "netral": "Netral",
    "neutral": "Netral",
}

IDX80_TICKERS = [
    "AADI", "ACES", "ADMR", "ADRO", "AKRA", "AMMN", "ANTM", "ARTO", "ASII",
    "BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRIS", "BRMS", "BRPT", "BSDE",
    "CMRY", "CPIN", "CTRA", "DSSA", "EMTK", "ESSA", "EXCL", "GOTO", "HEAL",
    "ICBP", "INCO", "INDF", "INKP", "INTP", "ISAT", "ITMG", "JPFA", "JSMR",
    "KLBF", "MAPA", "MAPI", "MBMA", "MDKA", "MEDC", "MIKA", "MNCN", "PGAS",
    "PGEO", "PTBA", "SIDO", "SMGR", "SMRA", "TLKM", "TOWR", "UNTR", "UNVR",
]


@st.cache_resource
def get_text_preprocessors():
    remover = None
    stemmer = None

    if StopWordRemoverFactory is None:
        return remover, stemmer

    remover = StopWordRemoverFactory().create_stop_word_remover()

    if StemmerFactory is not None:
        stemmer = StemmerFactory().create_stemmer()

    return remover, stemmer


def clean_text(text: str) -> str:
    remover, stemmer = get_text_preprocessors()
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remover is not None:
        text = remover.remove(text)

    if stemmer is not None:
        text = stemmer.stem(text)

    return text


def normalize_sentiment_label(label: str) -> str:
    normalized = str(label).strip().lower()
    return LABEL_NORMALIZATION.get(normalized, str(label).strip().title())


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    return data


def combine_article_text(data: pd.DataFrame, title_column: str, body_column: str) -> pd.Series:
    title = data[title_column].fillna("").astype(str)
    body = data[body_column].fillna("").astype(str)
    return (title + " " + body).str.strip()


def build_model(max_words: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(max_len,)),
            Embedding(max_words, 128),
            LSTM(128),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def prepare_training_data(
    data: pd.DataFrame,
    title_column: str,
    body_column: str,
    label_column: str,
    max_words: int,
    max_len: int,
):
    data = data.dropna(subset=[label_column]).copy()
    data["sentiment_standard"] = data[label_column].apply(normalize_sentiment_label)
    data["text"] = combine_article_text(data, title_column, body_column)
    data["clean_text"] = data["text"].apply(clean_text)
    data = data[data["clean_text"].str.len() > 0]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["sentiment_standard"].astype(str))

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data["clean_text"])

    sequences = tokenizer.texts_to_sequences(data["clean_text"])
    features = pad_sequences(sequences, maxlen=max_len)

    return features, labels, tokenizer, label_encoder, data


def save_artifacts(model, tokenizer, label_encoder, config: dict) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    with TOKENIZER_PATH.open("wb") as file:
        pickle.dump(tokenizer, file)

    with LABEL_ENCODER_PATH.open("wb") as file:
        pickle.dump(label_encoder, file)

    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


@st.cache_resource
def load_artifacts():
    if not all(path.exists() for path in [MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH, CONFIG_PATH]):
        return None

    model = load_model(MODEL_PATH)

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    with LABEL_ENCODER_PATH.open("rb") as file:
        label_encoder = pickle.load(file)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return model, tokenizer, label_encoder, config


def predict_sentiment(text: str, model, tokenizer, label_encoder, max_len: int) -> dict:
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len)
    probabilities = model.predict(padded, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    label = label_encoder.inverse_transform([class_index])[0]

    return {
        "clean_text": cleaned,
        "label": label,
        "confidence": float(probabilities[class_index]),
        "probabilities": {
            label_encoder.inverse_transform([index])[0]: float(score)
            for index, score in enumerate(probabilities)
        },
    }


def detect_tickers(text: str) -> list[str]:
    upper_text = str(text).upper()
    return [ticker for ticker in IDX80_TICKERS if re.search(rf"\b{ticker}\b", upper_text)]


def render_training_tab():
    st.subheader("Latih Model LSTM")
    st.write("Upload dataset CSV dengan minimal kolom judul berita, isi artikel, dan label sentimen.")

    uploaded_file = st.file_uploader("Dataset CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Contoh kolom dari notebook: `judul`, `isi_berita`, `sentiment`.")
        return

    encoding = st.selectbox("Encoding CSV", ["utf-8", "latin1", "cp1252"], index=1)

    try:
        data = normalize_columns(pd.read_csv(uploaded_file, encoding=encoding))
    except Exception as exc:
        st.error(f"CSV gagal dibaca: {exc}")
        return

    st.dataframe(data.head(), use_container_width=True)

    columns = list(data.columns)
    title_column = st.selectbox("Kolom judul", columns, index=columns.index("judul") if "judul" in columns else 0)
    body_column = st.selectbox(
        "Kolom isi berita",
        columns,
        index=columns.index("isi_berita") if "isi_berita" in columns else 0,
    )
    label_column = st.selectbox(
        "Kolom label sentimen",
        columns,
        index=columns.index("sentiment") if "sentiment" in columns else 0,
    )

    left, right, third = st.columns(3)
    with left:
        max_words = st.number_input("Jumlah kata maksimum", min_value=1000, max_value=50000, value=10000, step=1000)
    with right:
        max_len = st.number_input("Panjang sequence", min_value=20, max_value=500, value=100, step=10)
    with third:
        epochs = st.number_input("Epoch", min_value=1, max_value=100, value=10, step=1)

    batch_size = st.slider("Batch size", min_value=8, max_value=128, value=32, step=8)
    test_size = st.slider("Proporsi data uji", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    if not st.button("Latih dan Simpan Model", type="primary"):
        return

    with st.spinner("Menyiapkan data dan melatih model LSTM..."):
        features, labels, tokenizer, label_encoder, prepared_data = prepare_training_data(
            data,
            title_column,
            body_column,
            label_column,
            int(max_words),
            int(max_len),
        )

        stratify = labels if pd.Series(labels).value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=float(test_size),
            random_state=42,
            stratify=stratify,
        )

        model = build_model(int(max_words), int(max_len), len(label_encoder.classes_))
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=0,
        )
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        save_artifacts(
            model,
            tokenizer,
            label_encoder,
            {
                "max_words": int(max_words),
                "max_len": int(max_len),
                "classes": label_encoder.classes_.tolist(),
                "title_column": title_column,
                "body_column": body_column,
                "label_column": label_column,
                "standard_label_column": "sentiment_standard",
                "training_rows": int(len(prepared_data)),
                "test_accuracy": float(accuracy),
                "lstm_units": 128,
                "embedding_dim": 128,
                "preprocessing": [
                    "case_folding",
                    "cleansing",
                    "stopword_removal",
                    "stemming",
                    "tokenizing",
                    "padding",
                ],
            },
        )
        load_artifacts.clear()

    st.success("Model berhasil dilatih dan disimpan.")
    st.metric("Akurasi data uji", f"{accuracy:.2%}")
    st.line_chart(pd.DataFrame(history.history)[["loss", "val_loss"]])


def render_prediction_tab():
    st.subheader("Prediksi Sentimen Artikel")
    artifacts = load_artifacts()

    if artifacts is None:
        st.warning("Belum ada model tersimpan. Latih model terlebih dahulu di tab `Training`.")
        return

    model, tokenizer, label_encoder, config = artifacts
    max_len = int(config["max_len"])

    article_title = st.text_input("Judul artikel")
    article_body = st.text_area("Isi artikel berita", height=220)
    selected_ticker = st.selectbox("Kode saham IDX80 terkait", ["Otomatis dari teks"] + IDX80_TICKERS)

    if st.button("Prediksi Sentimen", type="primary"):
        article_text = f"{article_title} {article_body}".strip()
        if not article_text:
            st.error("Masukkan judul atau isi artikel terlebih dahulu.")
            return

        result = predict_sentiment(article_text, model, tokenizer, label_encoder, max_len)
        detected = detect_tickers(article_text)
        tickers = detected if selected_ticker == "Otomatis dari teks" else [selected_ticker]

        left, right = st.columns(2)
        left.metric("Sentimen", result["label"])
        right.metric("Confidence", f"{result['confidence']:.2%}")

        if tickers:
            st.caption("Saham IDX80 terdeteksi: " + ", ".join(tickers))
        else:
            st.caption("Tidak ada kode saham IDX80 yang terdeteksi otomatis dari teks.")

        probability_data = pd.DataFrame(
            {
                "Sentimen": list(result["probabilities"].keys()),
                "Probabilitas": list(result["probabilities"].values()),
            }
        ).sort_values("Probabilitas", ascending=False)
        st.bar_chart(probability_data, x="Sentimen", y="Probabilitas")

        with st.expander("Teks setelah preprocessing"):
            st.write(result["clean_text"])


def render_batch_tab():
    st.subheader("Prediksi Batch CSV")
    artifacts = load_artifacts()

    if artifacts is None:
        st.warning("Belum ada model tersimpan. Latih model terlebih dahulu di tab `Training`.")
        return

    model, tokenizer, label_encoder, config = artifacts
    max_len = int(config["max_len"])

    uploaded_file = st.file_uploader("Upload CSV artikel untuk diprediksi", type=["csv"], key="batch_csv")
    if uploaded_file is None:
        st.info("CSV batch sebaiknya memiliki kolom `judul` dan `isi_berita`.")
        return

    encoding = st.selectbox("Encoding CSV batch", ["utf-8", "latin1", "cp1252"], index=1)

    try:
        data = normalize_columns(pd.read_csv(uploaded_file, encoding=encoding))
    except Exception as exc:
        st.error(f"CSV gagal dibaca: {exc}")
        return

    columns = list(data.columns)
    title_column = st.selectbox("Kolom judul batch", columns, index=columns.index("judul") if "judul" in columns else 0)
    body_column = st.selectbox(
        "Kolom isi batch",
        columns,
        index=columns.index("isi_berita") if "isi_berita" in columns else 0,
    )

    if not st.button("Prediksi Semua Artikel"):
        return

    with st.spinner("Memproses prediksi batch..."):
        predictions = []
        for _, row in data.iterrows():
            text = f"{row.get(title_column, '')} {row.get(body_column, '')}".strip()
            result = predict_sentiment(text, model, tokenizer, label_encoder, max_len)
            predictions.append(
                {
                    "prediksi_sentimen": result["label"],
                    "confidence": result["confidence"],
                    "ticker_idx80_terdeteksi": ", ".join(detect_tickers(text)),
                }
            )

        result_data = pd.concat([data.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)

    st.dataframe(result_data, use_container_width=True)
    st.download_button(
        "Download Hasil Prediksi",
        data=result_data.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_sentimen_idx80.csv",
        mime="text/csv",
    )


def render_model_info():
    st.subheader("Informasi Model")
    artifacts = load_artifacts()

    if artifacts is None:
        st.info("Model belum tersedia.")
        return

    _, _, _, config = artifacts
    st.json(config)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title(APP_TITLE)
    st.caption("Metode: preprocessing teks berita, tokenisasi, padding sequence, dan klasifikasi sentimen dengan LSTM.")

    tabs = st.tabs(["Prediksi", "Training", "Batch CSV", "Informasi Model"])
    with tabs[0]:
        render_prediction_tab()
    with tabs[1]:
        render_training_tab()
    with tabs[2]:
        render_batch_tab()
    with tabs[3]:
        render_model_info()


if __name__ == "__main__":
    main()
