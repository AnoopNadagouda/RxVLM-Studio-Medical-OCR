# Raw Workspace Datasets Audit & Taxonomy Catalog

This report presents a clean, comprehensive taxonomy catalog of all dataset subdirectories found in `DataDump/`. All existing `composite_medocr` and legacy POC artifacts were excluded from this classification.

---

## Dataset Taxonomy Summary

| Category / Bucket | Description | Member Directories | Total Sample Volume / Format |
| :--- | :--- | :--- | :--- |
| **Bucket A: Clean Word-Level Pairs** | Isolated handwritten word cropped images with exact ground-truth label CSVs | `kaggle_bd_rx`, `rxhandbd` | ~10,000+ clean word-image pairs |
| **Bucket B: Structured Datasets** | Prescription scans with bounding boxes, nested JSONs, multi-item CSVs, or PyArrow HF datasets | `kalashsh`, `Prescription_Dataset(600)`, `annotations`, `chaithanyakota`, `chinmays`, `tavishm` | Multi-field JSONs, CSV bounding boxes, Hugging Face arrow tables |
| **Bucket C: Unlabeled / Raw Scans** | Full prescription page scans, raw clinic dumps, and unannotated image sets | `chhavimohitkar`, `avi-kai`, `eka`, `mehaksingal` | Full document PNG/JPG scans, zip archives |
| **Bucket D: Helper Corpora** | External drug lexicons, medicine name lists, and text vocabularies | `corpora`, `med_names.txt`, `medicines.json`, `layer1_vocabulary.csv`, `spottings.csv` | Text corpora, CSV lexicons, JSON databases |

---

## Detailed Directory Catalog

### Bucket A: Clean Word-Level Pairs
- **`kaggle_bd_rx/`**: Kaggle Bangladesh Doctor's Handwritten Word Dataset.
  - Structure: `Train_Set/`, `Test_Set/`, `Train_Label.csv`, `Test_Labels.csv`.
  - Format: Isolated handwritten word PNG images with `(Filename, Identity)` metadata.
  - Utilization: Primary input source for synthetic multi-word prescription line generation.
- **`rxhandbd/`**: Doctor's Handwritten Prescription BD Dataset (Expanded).
  - Structure: `Doctor’s Handwritten Prescription BD dataset/` containing `Training/training_words`, `training_labels.csv`, `Validation`, `Testing`.
  - Format: Word-level image crops paired with transcription labels in CSV.
  - Utilization: Primary input source for synthetic multi-word prescription line generation.

---

### Bucket B: Structured Datasets
- **`kalashsh/`**: Crop/word annotations from Kalash dataset.
  - Structure: `images/`, `train.csv`, `validate.csv`, `test.csv`.
- **`Prescription_Dataset(600)/`**: Structured dataset of 600 full prescription pages.
  - Structure: `images/`, `Prescription Dataset.csv`, `manifest_rx600.jsonl`.
- **`annotations/`**: Bounding box & key-value JSON annotations.
  - Structure: 12 detailed JSON schema files (`00a3f5f...json`, etc.) mapping bounding boxes to text fields.
- **`chaithanyakota/`**: Teammate dataset dump with prescription images and comma-separated medicine lists.
  - Structure: `data/`, `images/`, `chaithanyakota.csv`.
- **`chinmays/`**: Chinmay's processed prescription dataset and Hugging Face export.
  - Structure: `chinmays_rx/`, `chinmays_rx_hf/`, `manifest_chinmays_test.jsonl`.
- **`tavishm/`**: Tavish's dataset in Hugging Face Datasets PyArrow binary format.
  - Structure: `tavishm/tavishm/train/data-00000-of-00001.arrow`.

---

### Bucket C: Unlabeled / Raw Scans
- **`chhavimohitkar/`**: Raw prescription page scans (`Sample1.png` - `Sample30.png`) and raw snippet crops (`data1.0.png` - `data30.7.png`).
- **`avi-kai/`**: Full prescription document scans in `Prescription_Content/`.
- **`eka/`**: Eka clinic raw prescription dumps in `eka_test/`, `mirage_test/`, `mirage_train/`.
- **`mehaksingal/`**: Raw illegible prescription images and `illegible-medical-prescription-images-dataset.zip`.

---

### Bucket D: Helper Corpora & Lexicons
- **`corpora/`**: Comprehensive drug lexicons and medicine name collections (`indian_drug_corpus.txt`, `medical_corpus.txt`, `bd_focused_corpus.txt`, `all_india_drugbank/`, `az-medicine-dataset-of-india/`).
- **`med_names.txt`**: 8.3 MB plain text list of medicine names.
- **`medicines.json`**: 14.6 MB JSON database of medicines.
- **`layer1_vocabulary.csv`**: 57.8 MB medical vocabulary dictionary.
- **`spottings.csv`**: 228 KB lexicon lookup CSV.
