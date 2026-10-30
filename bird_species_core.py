"""
Bird species from perceptual attributes — random forest (Eckroth Packt Ch. 2).

CUB-200-2011 attribute matrix pivoted to 312 binary features per image; 200 species labels.
Train on first 8,000 shuffled rows; evaluate on remainder (book defaults).
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

TRAIN_ROWS = 8000
RANDOM_STATE = 0
N_ESTIMATORS = 100
MAX_FEATURES = 50
CV_FOLDS = 5
NUM_ATTRIBUTES = 312


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "CUB_200_2011"


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "web" / "data" / "cache"


def _attribute_labels_path(data_dir: Path) -> Path:
    return data_dir / "attributes" / "image_attribute_labels.txt"


def ensure_data_present(data_dir: Path | None = None) -> Path:
    root = data_dir or default_data_dir()
    needed = [
        root / "classes.txt",
        root / "image_class_labels.txt",
        _attribute_labels_path(root),
    ]
    missing = [p for p in needed if not p.is_file()]
    if missing:
        hint = (
            "Run: python scripts/fetch_cub_metadata.py "
            f"(missing {missing[0].name})"
        )
        raise FileNotFoundError(hint)
    return root


def load_class_names(data_dir: Path) -> list[str]:
    birds = pd.read_csv(
        data_dir / "classes.txt",
        sep=r"\s+",
        header=None,
        usecols=[1],
        names=["birdname"],
    )
    return birds["birdname"].astype(str).tolist()


def load_feature_matrix(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    imgatt = pd.read_csv(
        _attribute_labels_path(data_dir),
        sep=r"\s+",
        header=None,
        on_bad_lines="skip",
        usecols=[0, 1, 2],
        names=["imgid", "attid", "present"],
    )
    imgatt2 = imgatt.pivot(index="imgid", columns="attid", values="present")
    imgatt2 = imgatt2.reindex(columns=range(1, NUM_ATTRIBUTES + 1), fill_value=0)

    imglabels = pd.read_csv(
        data_dir / "image_class_labels.txt",
        sep=" ",
        header=None,
        names=["imgid", "label"],
    ).set_index("imgid")

    df = imgatt2.join(imglabels)
    df = df.sample(frac=1, random_state=RANDOM_STATE)

    df_att = df.iloc[:, :NUM_ATTRIBUTES]
    df_label = df.iloc[:, NUM_ATTRIBUTES].squeeze()
    if isinstance(df_label, pd.DataFrame):
        df_label = df_label.iloc[:, 0]

    return df_att, df_label


@dataclass
class BirdSpeciesModel:
    classifier: RandomForestClassifier
    class_names: list[str]
    holdout_accuracy: float
    cv_mean: float
    cv_std: float
    train_size: int
    test_size: int
    test_image_ids: list[int]
    test_labels: list[int]
    feature_columns: list[int]

    def predict_image(self, image_id: int) -> dict[str, Any]:
        # Re-load row from stored test set only for demo IDs
        if image_id not in self._test_id_to_index:
            raise ValueError("Unknown image id for this demo holdout set")
        row = self._test_features.loc[image_id]
        proba = self.classifier.predict_proba([row.values])[0]
        classes = self.classifier.classes_
        order = np.argsort(proba)[::-1][:5]
        top = [
            {
                "class_id": int(classes[i]),
                "species": self.class_names[int(classes[i]) - 1],
                "probability": float(proba[i]),
            }
            for i in order
        ]
        true_id = int(self._test_id_to_label[image_id])
        pred_id = int(classes[np.argmax(proba)])
        return {
            "image_id": image_id,
            "true_class_id": true_id,
            "true_species": self.class_names[true_id - 1],
            "predicted_class_id": pred_id,
            "predicted_species": self.class_names[pred_id - 1],
            "correct": pred_id == true_id,
            "top_predictions": top,
        }

    def attach_test_frame(self, test_att: pd.DataFrame, test_label: pd.Series) -> None:
        self._test_features = test_att
        self._test_id_to_index = {int(i): idx for idx, i in enumerate(test_att.index)}
        self._test_id_to_label = {int(i): int(test_label.loc[i]) for i in test_att.index}


def train_model(data_dir: Path | None = None) -> BirdSpeciesModel:
    root = ensure_data_present(data_dir)
    class_names = load_class_names(root)
    df_att, df_label = load_feature_matrix(root)

    train_att = df_att.iloc[:TRAIN_ROWS]
    train_label = df_label.iloc[:TRAIN_ROWS]
    test_att = df_att.iloc[TRAIN_ROWS:]
    test_label = df_label.iloc[TRAIN_ROWS:]

    clf = RandomForestClassifier(
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
    )
    clf.fit(train_att, train_label)

    holdout = float(clf.score(test_att, test_label))
    scores = cross_val_score(clf, train_att, train_label, cv=CV_FOLDS)
    cv_mean = float(scores.mean())
    cv_std = float(scores.std() * 2)

    test_ids = [int(i) for i in test_att.index.tolist()]
    model = BirdSpeciesModel(
        classifier=clf,
        class_names=class_names,
        holdout_accuracy=holdout,
        cv_mean=cv_mean,
        cv_std=cv_std,
        train_size=len(train_att),
        test_size=len(test_att),
        test_image_ids=test_ids,
        test_labels=[int(test_label.loc[i]) for i in test_att.index],
        feature_columns=list(train_att.columns),
    )
    model.attach_test_frame(test_att, test_label)
    return model


def sample_demo_birds(model: BirdSpeciesModel, count: int = 8) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_STATE)
    picks = rng.sample(model.test_image_ids, k=min(count, len(model.test_image_ids)))
    out: list[dict[str, Any]] = []
    for img_id in picks:
        label_id = int(model._test_id_to_label[img_id])
        out.append(
            {
                "image_id": img_id,
                "species": model.class_names[label_id - 1],
            }
        )
    return out


def get_or_train_model(
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> BirdSpeciesModel:
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / "bird_rf_model.pkl"
    if cache_file.is_file():
        with cache_file.open("rb") as fh:
            model = pickle.load(fh)
        if isinstance(model, BirdSpeciesModel):
            return model

    model = train_model(data_dir)
    with cache_file.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return model
