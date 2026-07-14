import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
DEFAULT_BATCH_DATASET = Path("idx80_3000.csv")

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

    if StopWordRemoverFactory is not None:
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


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).replace("\ufeff", "").strip().lower() for column in data.columns]
    return data


def detect_tickers(text: str) -> list[str]:
    upper_text = str(text).upper()
    return [ticker for ticker in IDX80_TICKERS if re.search(rf"\b{ticker}\b", upper_text)]


@st.cache_resource
def load_artifacts():
    required_paths = [MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH, CONFIG_PATH]
    if not all(path.exists() for path in required_paths):
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


def predict_sentiments(texts: list[str], model, tokenizer, label_encoder, max_len: int) -> pd.DataFrame:
    cleaned_texts = [clean_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    probabilities = model.predict(padded, verbose=0)
    class_indexes = np.argmax(probabilities, axis=1)
    labels = label_encoder.inverse_transform(class_indexes)
    confidence = probabilities[np.arange(len(class_indexes)), class_indexes]

    return pd.DataFrame(
        {
            "prediksi_sentimen": labels,
            "confidence": confidence,
            "clean_text": cleaned_texts,
        }
    )


def render_prediction_tab():
    st.subheader("Prediksi Sentimen Artikel")
    artifacts = load_artifacts()

    if artifacts is None:
        st.warning("Belum ada model tersimpan. Jalankan `python train_model.py` terlebih dahulu.")
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
        st.warning("Belum ada model tersimpan. Jalankan `python train_model.py` terlebih dahulu.")
        return

    model, tokenizer, label_encoder, config = artifacts
    max_len = int(config["max_len"])

    source_options = ["Upload CSV"]
    if DEFAULT_BATCH_DATASET.exists():
        source_options.insert(0, "Gunakan dataset bawaan idx80_3000.csv")

    source = st.radio("Sumber data batch", source_options, horizontal=True)
    encoding = st.selectbox("Encoding CSV batch", ["utf-8-sig", "utf-8", "latin1", "cp1252"], index=0)

    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV artikel untuk diprediksi", type=["csv"], key="batch_csv")
        if uploaded_file is None:
            st.info("CSV batch sebaiknya memiliki kolom `judul` dan `isi_berita`.")
            return

        try:
            data = normalize_columns(pd.read_csv(uploaded_file, encoding=encoding))
        except Exception as exc:
            st.error(f"CSV gagal dibaca: {exc}")
            return
    else:
        try:
            data = normalize_columns(pd.read_csv(DEFAULT_BATCH_DATASET, encoding=encoding))
        except Exception as exc:
            st.error(f"Dataset bawaan gagal dibaca: {exc}")
            return

    columns = list(data.columns)
    st.caption(f"Total data batch: {len(data):,} baris")
    st.dataframe(data.head(10), use_container_width=True)

    title_column = st.selectbox("Kolom judul batch", columns, index=columns.index("judul") if "judul" in columns else 0)
    body_column = st.selectbox(
        "Kolom isi batch",
        columns,
        index=columns.index("isi_berita") if "isi_berita" in columns else 0,
    )
    row_limit = st.number_input(
        "Jumlah baris yang diprediksi",
        min_value=1,
        max_value=int(len(data)),
        value=int(min(len(data), 3000)),
        step=50,
    )

    if not st.button("Prediksi Semua Artikel", type="primary"):
        return

    with st.spinner("Memproses prediksi batch..."):
        batch_data = data.head(int(row_limit)).copy()
        texts = [
            f"{row.get(title_column, '')} {row.get(body_column, '')}".strip()
            for _, row in batch_data.iterrows()
        ]
        predictions = predict_sentiments(texts, model, tokenizer, label_encoder, max_len)
        predictions["ticker_idx80_terdeteksi"] = [", ".join(detect_tickers(text)) for text in texts]
        result_data = pd.concat([batch_data.reset_index(drop=True), predictions], axis=1)

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
    st.caption(
        "Aplikasi hanya memuat model tersimpan dan melakukan prediksi. "
        "Training dilakukan terpisah melalui `train_model.py`."
    )

    tabs = st.tabs(["Prediksi", "Informasi Model"])
    with tabs[0]:
        prediction_tabs = st.tabs(["Artikel Tunggal", "Batch CSV"])
        with prediction_tabs[0]:
            render_prediction_tab()
        with prediction_tabs[1]:
            render_batch_tab()
    with tabs[1]:
        render_model_info()


if __name__ == "__main__":
    main()
