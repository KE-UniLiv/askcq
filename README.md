# AskCQ
## A Comparative Study of Competency Question Elicitation Methods from Ontology Requirements

This repository accompanies research investigating different approaches for formulating Competency Questions (CQs) in ontology engineering. It provides the **AskCQ dataset**, Python code, and a suite of Jupyter notebooks for a comprehensive computational analysis of CQs generated through manual, template-based, and Large Language Model (LLM)-based methods. Competency Questions (CQs) are crucial in the ontology engineering lifecycle for eliciting requirements, scoping the ontology, and guiding its design, validation, and reuse. However, formulating effective CQs can be challenging. This work addresses this by investigating three main CQ formulation approaches: manual formulation by experienced ontology engineers, semi-automated formulation via instantiation of CQ patterns (from Ren et al., 2014), and automated formulation using LLMs (specifically GPT4.1 and Gemini 2.5 Pro). A core contribution is **AskCQ**, the first multi-annotator dataset of CQs generated from identical source material (a user story in cultural heritage) using these distinct elicitation approaches. The project involves a systematic comparative analysis of the generated CQs, using qualitative multi-annotator evaluation and quantitative assessment of features like suitability, ambiguity, relevance, readability, and complexity, alongside an analysis of semantic diversity and overlap using sentence embeddings. Key findings highlight that human-formulated CQs generally score highest in suitability and readability with lower complexity, while LLM-based approaches, though capable of generating relevant CQs, tend to produce more complex and less readable questions without further refinement.

## The AskCQ Dataset

The AskCQ dataset is a novel multi-annotator resource designed for studying Competency Questions. All CQs were elicited from a single, detailed user story developed in the context of a cultural heritage project. This story involves a music archivist and a collection curator, focusing on their information needs regarding a museum's collection. LLMs were prompted without specific definitions, numbers, or desired properties of CQs to avoid bias in their generation. All generated CQs were assigned unique identifiers and anonymized with respect to their generation method for unbiased evaluation during the study.

Key properties of the AskCQ dataset are summarized below:

| Property              | Description                                                                                                                               |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **Total CQs** | 204                                                                                                                                       |
| **Source Material** | A single user story from a cultural heritage project (focus: museum collection data management for music archivist & collection curator). |
| **Generation Sets** | 5 distinct sets based on elicitation method:                                                                                              |
|                       | - **HA-1 (44 CQs):** Manual (Human Annotator 1: 20+ yrs OE experience)                                                                  |
|                       | - **HA-2 (54 CQs):** Manual (Human Annotator 2: 5 yrs OE experience in domain)                                                            |
|                       | - **Pattern (38 CQs):** Template-based (instantiated from Ren et al., 2014 by engineer with 5+ yrs requirements eng. experience)        |
|                       | - **GPT (26 CQs):** LLM-generated (GPT 4.1)                                                                                           |
|                       | - **Gemini (42 CQs):** LLM-generated (Gemini 2.5 Pro)                                                                                       |
| **Anonymization** | CQs anonymized by generation method for evaluation.                                                                                       |
| **License** | CC BY 4.0                                                                                                                                 |
| **File Location** | `data/askcq_dataset.csv` (The user story is also realeased in `data/bme_us1.md`).                                                          |

Ensure `askcq_dataset.csv` is present in the `data/` directory before running any analysis notebooks.

## CQ Feature Analysis

The notebooks and modules facilitate the extraction and analysis of several CQ features. These features were chosen to provide a multi-faceted view of CQ quality and characteristics, as investigated in the paper.

| Feature                 | Description & Methodology                                                                                                                                                                  | Tools/Libraries Used      |
| :---------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------ |
| **Suitability** | Expert evaluation by three independent OE experts (accept/reject based on relevance and suitability for OE tasks).                                                                         | Manual Annotation         |
| **Ambiguity** | Binary label derived from expert annotator comments indicating lack of clarity or difficulty in understanding the CQ's scope/intent.                                                         | Manual Analysis           |
| **Relevance** | LLM-assessed (Gemini) alignment with user story requirements on a 4-point Likert scale (explicit, inferable, contextual, or additional). Prompt engineering validated on a subset.        | LLM       |
| **Readability** | Computationally assessed ease of understanding. Key indices reported: Flesch-Kincaid Grade Level (FKGL) and Dale-Chall Readability Score (DCR). Interpreted as comparative indicators.    | `textstat`                |
| **Complexity ($c_0$)** | **Length:** Number of characters in the CQ.                                                                                                                                                | Python standard library   |
| **Complexity ($c_1$)** | **Requirement Complexity:** Quantifies emerging ontological primitives (Concepts, Properties, Relationships, Filters, Cardinality, Aggregation). Extracted via LLM (Gemini) and scored. | LLM      |
| **Complexity ($c_2$)** | **Linguistic Complexity:** Surface NLP features (Noun Phrases, Verbs, Prepositions, Conjunctions, Modifiers, Interrogative Structure). Extracted via NLP and scored.                     | `spaCy`                   |
| **Complexity ($c_3$)** | **Syntactic Complexity:** Grammatical structure from dependency parsing (Node Count, Tree Depth, specific Dependency Relations). Extracted via NLP and scored.                           | `spaCy`                   |
| **Semantic Analysis** | Utilizes Sentence-BERT embeddings to quantify semantic overlap (centroid cosine similarity, coverage analysis, bidirectional coverage) between CQ sets. Threshold $\tau=0.75$.                 | `sentence-transformers`   |

## Repository Structure

-   `askcq/`
    -   `agreement.py`, `complexity.py`, `embedding.py`, `utils.py`: Core Python modules for data processing and analysis.
    -   `prompts.py`: Includes all the prompts and system roles used in the LLM-based experiments (CQ generation, relevance assessment, complexity feature extraction).
    -   `config.py`: provides the configuration used to prompt all the LLMs (GPT and Gemini models).
    -   `cq_generation.ipynb`: LLM-based CQ generation from the user story.
    -   `overview.ipynb`: expert analysis overview, CQ feature exploration, and general data analysis.
    -   `cq_readability.ipynb`: computational readability analysis, including correlation analysis of different readability indices.
    -   `cq_relevance.ipynb`: relevance scoring and rationale extraction using LLMs.
    -   `cq_complexity.ipynb`: complexity feature extraction (requirement, linguistic, syntactic) and analysis from CQs.
    -   `cq_embeddings.ipynb`: embedding-based semantic analysis of CQ sets.
-   `data/`: Contains all data provided to or generated by the experiments, including:
    -   `askcq_dataset.csv`: The primary dataset.
    -   `bme_us1.md`: the user story driving the elicitation.
    -   Derived metrics from various analyses.
-   `plots/`: Output figures and plots generated by the notebooks.

## Reproducibility Instructions

### 1. Environment Setup

-   **Python Version:** Python 3.11+ is recommended.
-   **Dependencies:** Install the required packages using:
    ```sh
    pip install -r requirements.txt
    ```
    If `requirements.txt` encounters issues, ensure at least the following are installed (versions might need checking for full compatibility):
    ```
    pandas numpy matplotlib seaborn scikit-learn textstat readability openai google-generativeai tqdm spacy sentence-transformers
    ```
    You will also need to download spaCy models, e.g.:
    ```sh
    python -m spacy download en_core_web_sm
    ```
-   **API Keys:** For experiments involving LLMs (CQ generation, relevance scoring, requirement complexity extraction), you will need API keys for OpenAI (GPT) and Google AI Studio (Gemini).
    Create an `api_config.yml` file within the `askcq/` directory with your API keys. Follow this example format:

    ```yml
    gemini:
      key: YOUR_GEMINI_API_KEY

    openai:
      key: YOUR_OPENAI_API_KEY
    ```
    *LLM Configuration Note:* The paper specifies that LLMs were prompted with temperature=0, frequency_penalty=0, presence_penalty=0, and seed=46 to maximize reproducibility. These settings are generally reflected in `config.py`.

### 2. Running the Notebooks

-   Each notebook is designed to be self-contained and can typically be run independently, though some may depend on outputs from others if data is passed through CSV files.
-   **Step 1:** Open the desired Jupyter notebook (e.g., `askcq/cq_readability.ipynb`) in a Jupyter environment (like Jupyter Lab, Jupyter Notebook, or VS Code).
-   **Step 2:** Ensure the `askcq_dataset.csv` file is located in the `data/` directory.
-   **Step 3:** Run all cells sequentially.
-   **Step 4:** Outputs such as plots, tables, and derived CSV data files will be saved in the appropriate folders (e.g., `plots/`, `data/`).

**Important Note on API Usage:** Some notebooks will make calls to OpenAI and Google Cloud APIs. Ensure your `api_config.yml` is correctly set up. The `utils.py` module likely contains functions like `get_key` to securely access these keys. Be mindful of potential costs associated with API calls.

## How to Cite

If you use the AskCQ dataset or the methodologies and findings from this research in your work, please cite the original paper. (*Bibliographic details are omitted here to preserve anonymity for review processes.*)

```bibtex
@article{AnonymousYEARAskCQ,
  title={A Comparative Study of Competency Question Elicitation Methods from Ontology Requirements},
  author={Anonymous Author(s)},
  journal={Submitted for review},
  year={202X},
  note={Details omitted for double-blind review..}
}