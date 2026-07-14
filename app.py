import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ImportError:
    StemmerFactory = None
    StopWordRemoverFactory = None


APP_TITLE = "Prediksi Sentimen Saham IDX80 Berbasis Artikel Berita"
ARTIFACT_DIR = Path("artifacts")
MODEL_WEIGHTS_PATH = ARTIFACT_DIR / "lstm_weights.npz"
TOKENIZER_JSON_PATH = ARTIFACT_DIR / "tokenizer.json"
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
    required_paths = [MODEL_WEIGHTS_PATH, TOKENIZER_JSON_PATH, CONFIG_PATH]
    if not all(path.exists() for path in required_paths):
        return None

    weights = np.load(MODEL_WEIGHTS_PATH)
    tokenizer = SimpleTokenizer(json.loads(TOKENIZER_JSON_PATH.read_text(encoding="utf-8")))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    classes = config["classes"]
    model = NumpyLSTMClassifier(weights)
    return model, tokenizer, classes, config


class SimpleTokenizer:
    def __init__(self, tokenizer_data: dict):
        self.word_index = tokenizer_data["word_index"]
        self.num_words = tokenizer_data.get("num_words")
        self.oov_token = tokenizer_data.get("oov_token")
        self.oov_index = self.word_index.get(self.oov_token) if self.oov_token else None

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        return [self.text_to_sequence(text) for text in texts]

    def text_to_sequence(self, text: str) -> list[int]:
        sequence = []
        for word in str(text).split():
            index = self.word_index.get(word)
            if index is None:
                if self.oov_index is not None:
                    sequence.append(self.oov_index)
                continue
            if self.num_words is not None and index >= self.num_words:
                if self.oov_index is not None:
                    sequence.append(self.oov_index)
                continue
            sequence.append(index)
        return sequence


class NumpyLSTMClassifier:
    def __init__(self, weights):
        self.embedding = weights["embedding"].astype(np.float32)
        self.lstm_kernel = weights["lstm_kernel"].astype(np.float32)
        self.lstm_recurrent_kernel = weights["lstm_recurrent_kernel"].astype(np.float32)
        self.lstm_bias = weights["lstm_bias"].astype(np.float32)
        self.dense_kernel = weights["dense_kernel"].astype(np.float32)
        self.dense_bias = weights["dense_bias"].astype(np.float32)
        self.output_kernel = weights["output_kernel"].astype(np.float32)
        self.output_bias = weights["output_bias"].astype(np.float32)
        self.units = self.lstm_recurrent_kernel.shape[0]

    def predict(self, padded_sequences: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        embedded = self.embedding[padded_sequences]
        batch_size = embedded.shape[0]
        hidden_state = np.zeros((batch_size, self.units), dtype=np.float32)
        cell_state = np.zeros((batch_size, self.units), dtype=np.float32)

        for timestep in range(embedded.shape[1]):
            gates = (
                embedded[:, timestep, :] @ self.lstm_kernel
                + hidden_state @ self.lstm_recurrent_kernel
                + self.lstm_bias
            )
            input_gate, forget_gate, candidate, output_gate = np.split(gates, 4, axis=1)
            input_gate = sigmoid(input_gate)
            forget_gate = sigmoid(forget_gate)
            candidate = np.tanh(candidate)
            output_gate = sigmoid(output_gate)
            cell_state = forget_gate * cell_state + input_gate * candidate
            hidden_state = output_gate * np.tanh(cell_state)

        dense = np.maximum(hidden_state @ self.dense_kernel + self.dense_bias, 0)
        logits = dense @ self.output_kernel + self.output_bias
        return softmax(logits)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60, 60)))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def pad_text_sequences(sequences: list[list[int]], max_len: int) -> np.ndarray:
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for row_index, sequence in enumerate(sequences):
        truncated = sequence[-max_len:]
        if truncated:
            padded[row_index, -len(truncated):] = truncated
    return padded


def predict_sentiment(text: str, model, tokenizer, classes: list[str], max_len: int) -> dict:
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_text_sequences(sequence, max_len)
    probabilities = model.predict(padded, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    label = classes[class_index]

    return {
        "clean_text": cleaned,
        "label": label,
        "confidence": float(probabilities[class_index]),
        "probabilities": {
            classes[index]: float(score)
            for index, score in enumerate(probabilities)
        },
    }


def predict_sentiments(texts: list[str], model, tokenizer, classes: list[str], max_len: int) -> pd.DataFrame:
    cleaned_texts = [clean_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_text_sequences(sequences, max_len)
    probabilities = model.predict(padded, verbose=0)
    class_indexes = np.argmax(probabilities, axis=1)
    labels = [classes[index] for index in class_indexes]
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

    if not all(path.exists() for path in [MODEL_WEIGHTS_PATH, TOKENIZER_JSON_PATH, CONFIG_PATH]):
        st.warning("Belum ada model tersimpan. Jalankan `python train_model.py` terlebih dahulu.")
        return

    article_title = st.text_input("Judul artikel")
    article_body = st.text_area("Isi artikel berita", height=220)
    selected_ticker = st.selectbox("Kode saham IDX80 terkait", ["Otomatis dari teks"] + IDX80_TICKERS)

    if st.button("Prediksi Sentimen", type="primary"):
        article_text = f"{article_title} {article_body}".strip()
        if not article_text:
            st.error("Masukkan judul atau isi artikel terlebih dahulu.")
            return

        with st.spinner("Memuat model dan memprediksi sentimen..."):
            artifacts = load_artifacts()
            if artifacts is None:
                st.error("Artefak model belum lengkap.")
                return
            model, tokenizer, classes, config = artifacts
            result = predict_sentiment(article_text, model, tokenizer, classes, int(config["max_len"]))

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

    if not all(path.exists() for path in [MODEL_WEIGHTS_PATH, TOKENIZER_JSON_PATH, CONFIG_PATH]):
        st.warning("Belum ada model tersimpan. Jalankan `python train_model.py` terlebih dahulu.")
        return

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
        value=int(min(len(data), 100)),
        step=50,
    )

    if not st.button("Prediksi Semua Artikel", type="primary"):
        return

    with st.spinner("Memproses prediksi batch..."):
        artifacts = load_artifacts()
        if artifacts is None:
            st.error("Artefak model belum lengkap.")
            return
        model, tokenizer, classes, config = artifacts
        batch_data = data.head(int(row_limit)).copy()
        texts = [
            f"{row.get(title_column, '')} {row.get(body_column, '')}".strip()
            for _, row in batch_data.iterrows()
        ]
        predictions = predict_sentiments(texts, model, tokenizer, classes, int(config["max_len"]))
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

    if not CONFIG_PATH.exists():
        st.info("Model belum tersedia.")
        return

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
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
