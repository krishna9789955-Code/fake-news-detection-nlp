# Dataset Documentation: TruthLens.NLP

## 1. Overview
The machine learning pipeline trains directly from the CSV files located in the `original_data/` directory:
- **`original_data/True.csv`**: 21,417 raw real news articles (news wire reporting).
- **`original_data/Fake.csv`**: 23,481 raw fake news articles (misinformation, hoaxes, clickbait).
- **Total Raw Records:** 44,898 articles.

### Curated & Deduplicated Dataset
After eliminating empty/short text fragments and near-duplicate articles, the dataset is synchronized into `data/news_dataset.csv`:
- **Total Valid Curated Samples:** **38,783 articles**
- **Real News (Class 0):** 20,893 articles (53.9%)
- **Fake News (Class 1):** 17,890 articles (46.1%)
- **Split Ratio:** 80% Training (31,026 articles), 20% Held-Out Testing (7,757 articles)

---

## 2. CSV Schema & Columns

### Source Files (`original_data/Fake.csv` & `original_data/True.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `title` | Text (string) | The news headline or report title |
| `text` | Text (string) | The full article body text |
| `subject` | Text (string) | Category/subject classification |
| `date` | Text (string) | Publication date string |

### Curated Target File (`data/news_dataset.csv`)
| Column | Data Type | Description |
| :--- | :--- | :--- |
| `title` | Text (string) | Preprocessed news headline (datelines stripped) |
| `text` | Text (string) | Preprocessed article body (publisher shortcuts stripped) |
| `label` | Integer (`0` or `1`) | Ground-truth class (`0` = Likely Real News, `1` = Likely Fake News) |

---

## 3. Data Cleaning & Leakage Prevention Safeguards

Standard public datasets (such as the Kaggle / ISOT dataset) suffer from severe **source-specific data leakage**:
- **The Wire Shortcut Problem:** Almost all real news articles originate from news agencies like Reuters and start with datelines such as `WASHINGTON (Reuters) -` or `GENEVA (Reuters) -`.
- **The Shortcut Vulnerability:** A naive NLP model easily learns that the token `"reuters"` indicates 100% Real News, inflating accuracy on test splits while failing completely on general unbranded articles.

### Mitigations Applied in This Project:
1. **Publisher Dateline Purging:** Stripped regex patterns matching city/wire datelines (e.g. `[CITY] (Reuters) -`).
2. **Wire Mention Removal:** Stripped standalone occurrences of wire services (`reuters`, `associated press`, `ap`, `afp`).
3. **Exact & Near-Duplicate Removal:** 6,098 duplicated records purged.
4. **Content Length Threshold:** Any headline/article pair with fewer than 40 characters excluded.
5. **Leakage Verification Audit:** `evaluate_model.py` audits the final TF-IDF vocabulary to ensure no publisher shortcuts exist in feature vectors.

---

## 4. Retraining on the Original Data Files
To train or refresh the pipeline directly from `original_data/Fake.csv` and `original_data/True.csv`:
```bash
python train_model.py
```
To run an independent audit:
```bash
python evaluate_model.py
```
