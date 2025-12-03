# Construction Accident Report Analysis — NLP, ML & LLM for Root Cause Classification

This project builds an AI-assisted pipeline for analyzing construction accident reports.
It extracts hazards, predicts root causes using classical ML, and leverages a transformer LLM to generate human-readable explanations.
SBERT embeddings are used for semantic similarity between incidents.

# Project Structure

├── notebook.ipynb         # Main analysis notebook (NLP, ML, LLM pipeline)

├── rca_model.py           # Python script with model functions / utilities

├── rca_eval_with_llm.csv  # Evaluation results (ML + LLM)

├── rca_similarity.csv     # Semantic similarity scores using SBERT

├── README.md              # This file


# Dataset

A small synthetic dataset of 8 construction accident reports was created for proof-of-concept testing.

Each entry contains:

- report_id

- text (accident description)

- root_cause (manual label)

- Root cause categories include:

- Housekeeping Failure

- Communication Failure

- Procedural Failure

- Engineering Control Failure

- Supervision Failure

- PPE Non-Compliance

- PPE Provision Failure

**Note**: Dataset size is intentionally small. Accuracy is NOT the objective.
The goal is to demonstrate a full working pipeline.

# NLP Feature Extraction

Using spaCy, each accident description is broken down into:

- **hazards** → extracted from nouns

- **actions** → extracted from verbs

- **conditions** → extracted from adjectives

This step shows how unstructured text can be converted to structured safety signals.

# Classical Machine Learning Model
**Vectorization**
- TfidfVectorizer()

  Transforms accident text into numerical vectors.

**Model**
- LogisticRegression(max_iter=200)

**Train/Test Split**

- 70% training

- 30% testing

- random_state=42

**Performance**

The classification report shows very low performance, which is expected because:

- Only 8 samples

- Many labels appear only once

- High class imbalance

- The ML section exists to show the pipeline, not accuracy.

# Large Language Model (LLM)
**Model Used**

HuggingFace sequence-to-sequence / generative transformer model using:

`tokenizer_llm = AutoTokenizer.from_pretrained(model_name)`

`model_llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)`


# Function

llm_root_cause_and_explanation(report_text) performs:

- Reads accident text

- Selects one root cause category

- Produces a 1–2 sentence explanation

- Enforces a strict output format:

`ROOT_CAUSE:`

`EXPLANATION:`

**Example result**

`ROOT_CAUSE: PPE Non-Compliance`

`EXPLANATION: No explanation provided.`


(The fallback parser says “No explanation provided” if the LLM doesn't format properly.)

# Embedding Model (SBERT)

**Model loaded:**

`all-MiniLM-L6-v2`


**Used for:**

- Sentence embeddings

- Semantic similarity between incidents

- Generating the similarity matrix exported as:

`rca_similarity.csv`

# Evaluation Outputs

My notebook saved two results files:

`eval_df.to_csv("rca_eval_with_llm.csv", index=False)`

`sim_df.to_csv("rca_similarity.csv", index=False)`

# Files Produced

`rca_eval_with_llm.csv` → LLM predictions + explanations

`rca_similarity.csv` → SBERT similarity scores between reports

# Project Purpose 

This project demonstrates how NLP, classical machine learning, and LLMs can be combined to analyze construction accident reports, extract hazards, predict root causes, and generate interpretable explanations.

# Future Improvements

- Use a larger dataset (500–3,000 real reports)

- Fine-tune LLM for construction safety domain

- Improve prompt formatting

- Add visualization dashboard

- Build a retrieval-based system using embeddings
