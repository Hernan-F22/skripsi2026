import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ImportError:
    StemmerFactory = None
    StopWordRemoverFactory = None


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "lstm_sentiment_model.keras"
TOKENIZER_PATH = ARTIFACT_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"
CONFIG_PATH = ARTIFACT_DIR / "config.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

TITLE_COLUMN = "judul"
BODY_COLUMN = "isi_berita"
LABEL_COLUMN = "sentiment"
STANDARD_LABEL_COLUMN = "sentiment_standard"

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


def normalize_sentiment_label(label: str) -> str:
    normalized = str(label).strip().lower()
    return LABEL_NORMALIZATION.get(normalized, str(label).strip().title())


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).replace("\ufeff", "").strip().lower() for column in data.columns]
    return data


def combine_article_text(data: pd.DataFrame, title_column: str, body_column: str) -> pd.Series:
    title = data[title_column].fillna("").astype(str)
    body = data[body_column].fillna("").astype(str)
    return (title + " " + body).str.strip()


def create_text_preprocessors():
    remover = None
    stemmer = None

    if StopWordRemoverFactory is not None:
        remover = StopWordRemoverFactory().create_stop_word_remover()

    if StemmerFactory is not None:
        stemmer = StemmerFactory().create_stemmer()

    return remover, stemmer


def clean_text(text: str, remover=None, stemmer=None) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remover is not None:
        text = remover.remove(text)

    if stemmer is not None:
        text = stemmer.stem(text)

    return text


def read_dataset(path: Path, encoding: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        data = pd.read_csv(path, encoding="latin1")

    data = normalize_columns(data)
    required_columns = {TITLE_COLUMN, BODY_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Kolom wajib belum ada di dataset: {sorted(missing_columns)}")

    return data


def prepare_training_data(data: pd.DataFrame, max_words: int, max_len: int):
    remover, stemmer = create_text_preprocessors()

    data = data.dropna(subset=[LABEL_COLUMN]).copy()
    data[STANDARD_LABEL_COLUMN] = data[LABEL_COLUMN].apply(normalize_sentiment_label)
    data["text"] = combine_article_text(data, TITLE_COLUMN, BODY_COLUMN)
    data["clean_text"] = data["text"].apply(lambda text: clean_text(text, remover, stemmer))
    data = data[data["clean_text"].str.len() > 0].reset_index(drop=True)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data[STANDARD_LABEL_COLUMN].astype(str))

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(data["clean_text"])

    sequences = tokenizer.texts_to_sequences(data["clean_text"])
    features = pad_sequences(sequences, maxlen=max_len)

    return features, labels, tokenizer, label_encoder, data


def build_model(max_words: int, max_len: int, num_classes: int, embedding_dim: int, lstm_units: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(max_len,)),
            Embedding(max_words, embedding_dim),
            LSTM(lstm_units),
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


def save_artifacts(model, tokenizer, label_encoder, config: dict, metrics: dict) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    model.save(MODEL_PATH)

    with TOKENIZER_PATH.open("wb") as file:
        pickle.dump(tokenizer, file)

    with LABEL_ENCODER_PATH.open("wb") as file:
        pickle.dump(label_encoder, file)

    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train(args):
    data = read_dataset(Path(args.dataset), args.encoding)
    features, labels, tokenizer, label_encoder, prepared_data = prepare_training_data(
        data,
        args.max_words,
        args.max_len,
    )

    stratify = labels if pd.Series(labels).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = build_model(
        args.max_words,
        args.max_len,
        len(label_encoder.classes_),
        args.embedding_dim,
        args.lstm_units,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, predictions).tolist()

    config = {
        "dataset": args.dataset,
        "max_words": args.max_words,
        "max_len": args.max_len,
        "classes": label_encoder.classes_.tolist(),
        "title_column": TITLE_COLUMN,
        "body_column": BODY_COLUMN,
        "label_column": LABEL_COLUMN,
        "standard_label_column": STANDARD_LABEL_COLUMN,
        "training_rows": int(len(prepared_data)),
        "test_accuracy": float(accuracy),
        "lstm_units": args.lstm_units,
        "embedding_dim": args.embedding_dim,
        "preprocessing": [
            "case_folding",
            "cleansing",
            "stopword_removal",
            "stemming",
            "tokenizing",
            "padding",
        ],
    }
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }

    save_artifacts(model, tokenizer, label_encoder, config, metrics)

    print(f"Akurasi data uji: {accuracy:.4f}")
    print(f"Model tersimpan di: {MODEL_PATH}")
    print(f"Tokenizer tersimpan di: {TOKENIZER_PATH}")
    print(f"Label encoder tersimpan di: {LABEL_ENCODER_PATH}")
    print(f"Config tersimpan di: {CONFIG_PATH}")
    print(f"Metrics tersimpan di: {METRICS_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training model LSTM sentimen saham IDX80.")
    parser.add_argument("--dataset", default="idx80_3000.csv", help="Path dataset CSV training.")
    parser.add_argument("--encoding", default="utf-8-sig", help="Encoding dataset CSV.")
    parser.add_argument("--max-words", type=int, default=10000, help="Jumlah kata maksimum tokenizer.")
    parser.add_argument("--max-len", type=int, default=100, help="Panjang sequence setelah padding.")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch maksimum.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporsi data uji.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state split data.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Dimensi embedding.")
    parser.add_argument("--lstm-units", type=int, default=128, help="Jumlah unit LSTM.")
    parser.add_argument("--patience", type=int, default=3, help="Patience EarlyStopping.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
