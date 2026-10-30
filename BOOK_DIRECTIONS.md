# Book directions — Bird species identifier

**Catalog #1168** · Joshua Eckroth, *Python Artificial Intelligence Projects for Beginners*, **Chapter 2** — bird species from attributes / random forest

- Packt repo: [Chapter02/bird-identifier.py](https://github.com/PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners/tree/master/Chapter02)
- Local reference: [`reference/bird-identifier.py`](./reference/bird-identifier.py)
- Shared logic: [`bird_species_core.py`](./bird_species_core.py)
- Dataset: CUB-200-2011 **attribute** files (no images) under `data/CUB_200_2011/`

## What the book asks you to build

Using the Caltech-UCSD Birds **attribute labels**:

1. Pivot `image_attribute_labels.txt` to one row per image, 312 binary columns.
2. Join `image_class_labels.txt` (200 species).
3. Shuffle; train on **first 8,000** rows, test on the rest.
4. `RandomForestClassifier(max_features=50, n_estimators=100, random_state=0)`.
5. Report holdout accuracy, confusion matrix (optional), and compare tree/SVM; tune with CV grid.

## Web version (`web/` — FastAPI, port **3031**)

| Book step | Web equivalent |
|-----------|----------------|
| Prepare matrix + train RF | `bird_species_core.get_or_train_model` (cached) |
| Evaluate | `GET /api/status` (holdout + 5-fold CV on train split) |
| Classify one bird | `POST /api/predict` with CUB **image id** from holdout set |
| Explore | Sample chips + top-5 species probabilities |

Fetch metadata: `python scripts/fetch_cub_metadata.py`
