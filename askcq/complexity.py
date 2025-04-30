"""
Competency Question Complexity Analysis Module
================================================
This module provides functionality to analyze the complexity of competency
questions (CQs) using linguistic and syntactic features. The approaches that
are implemented include: (1) ontology primitives extraction, (2) linguistic
complexity analysis, and (3) syntactic complexity analysis.
"""
import re
import os
import json
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

import spacy
from pydantic import BaseModel, Field, ValidationError

# --- spaCy Model Loading ---

# Choose spaCy model (ensure it's downloaded)
# Options: "en_core_web_sm", "en_core_web_md", "en_core_web_lg"
SPACY_MODEL_NAME = "en_core_web_sm"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    NLP = spacy.load(SPACY_MODEL_NAME)
    print(f"Successfully loaded spaCy model '{SPACY_MODEL_NAME}'")
except OSError:
    print(f"Error: spaCy model '{SPACY_MODEL_NAME}' not found.")
    print(f"Please download it: python -m spacy download {SPACY_MODEL_NAME}")
    exit() # Or handle gracefully depending on application context

# -------------------------------------

class CQAnalysis(BaseModel):
    """
    Represents the analysis of a Competency Question (CQ)
    identifying key ontological primitives.
    """
    concepts: List[str] = Field(
        description="List of distinct fundamental entity types or classes mentioned or clearly implied by the question (e.g., 'Item', 'Artist', 'Event', 'MultimediaFile', 'Genre', 'Period', 'Publication'). Use singular form, CamelCase.",
        default_factory=list
    )
    properties: List[str] = Field(
        description="List of attributes or data properties associated with the concepts (e.g., 'name', 'title', 'description', 'caption', 'format', 'resolution', 'duration', 'copyrightStatus'). Use camelCase.",
        default_factory=list
    )
    relationships: List[str] = Field(
        description="List of named relationship types connecting concepts (e.g., 'isPartOf', 'relatedTo', 'hasGenre', 'belongsToPeriod', 'hasImage', 'hasAudio', 'hasVideo', 'associatedArtist', 'usedBy', 'producedBy', 'ownedBy', 'involvedInWork', 'usedDuringPerformance', 'involvedInEvent', 'featuredInPublication'). Use camelCase.",
        default_factory=list
    )
    filters: List[str] = Field(
        description="List of specific constraints, conditions, or filtering criteria mentioned (e.g., 'main textual description', 'primary image', 'specific genre', 'specific period', 'significant historical events', 'significant publication'). Describe the filter.",
        default_factory=list
    )
    cardinality_hint: str = Field(
        description="Indication of the expected result cardinality based on the question's phrasing ('single', 'multiple', 'existence_check'). Default to 'single' if not obvious.",
        # default="single" # Default to single if not obvious
    )
    aggregation_hint: str = Field(
        description="Type of aggregation implied, if any ('count', 'sum', 'average', 'none', etc.). Default to 'none' if not applicable.",
        # default="none"
    )
    rationale: str = Field(
        description="Brief step-by-step rationale explaining how the primitives were derived from the question.",
        # default=""
    )

# -------------------------------------
# 2. LLM Prompt Generation
# -------------------------------------
def generate_complexity_prompt(cq: str) -> str:
    """Generates the prompt for the LLM."""

    prompt = f"""
    You are an expert ontology engineer analyzing competency questions (CQs) to identify the underlying ontological primitives required to answer them. Your task is to analyze the given CQ and extract the relevant concepts, properties, relationships, filters, cardinality hints, and aggregation hints.

    Focus on the *minimal* set of primitives essential to fulfill the requirement expressed in the question. Assume an ideal knowledge graph exists.

    **Competency Question:**
    "{cq}"

    **Instructions:**
    1.  **Identify Concepts:** List the distinct real-world entity types (classes) involved (e.g., Item, Artist, Event, MultimediaFile). Use singular form, CamelCase.
    2.  **Identify Properties:** List the attributes or data characteristics of those concepts needed (e.g., name, description, duration, format). Use camelCase.
    3.  **Identify Relationships:** List the connections *between concepts* required (e.g., isPartOf, relatedTo, usedBy, associatedArtist). Use camelCase. Do not list relationships between a concept and its literal property.
    4.  **Identify Filters:** List any specific conditions or constraints applied (e.g., 'primary image', 'specific genre').
    5.  **Determine Cardinality Hint:** Indicate if the question asks for one result ('single'), potentially many results ('multiple'), or just confirmation of existence ('existence_check').
    6.  **Determine Aggregation Hint:** Indicate if the question implies counting or other aggregations ('count', 'none', etc.).
    7.  **Provide Rationale:** Briefly explain your reasoning.
    """
    
    return prompt

    
def calculate_complexity_score(analysis: CQAnalysis) -> float:
    """
    Calculates a complexity score based on the extracted primitives.
    """
    score = 0.0

    # Weights (these can be tuned based on empirical analysis or domain knowledge)
    concept_weight = 1.0
    property_weight = 0.8
    relationship_weight = 2.0  # Relationships often imply joins/paths -> higher complexity
    filter_weight = 1.5  # Filters add query complexity
    cardinality_multiple_bonus = 1.0
    cardinality_existence_bonus = 0.5 # Less complex than retrieving data, but more than simple property access
    aggregation_bonus = 2.5  # Aggregations (count, sum, etc.) are usually more complex

    score += len(analysis.concepts) * concept_weight
    score += len(analysis.properties) * property_weight
    score += len(analysis.relationships) * relationship_weight
    score += len(analysis.filters) * filter_weight

    if analysis.cardinality_hint == 'multiple':
        score += cardinality_multiple_bonus
    elif analysis.cardinality_hint == 'existence_check':
        score += cardinality_existence_bonus
        # Optional Tweak: Reduce impact of properties/relationships for simple existence checks?
        # score -= (len(analysis.properties) * property_weight * 0.5)
        # score -= (len(analysis.relationships) * relationship_weight * 0.5)

    if analysis.aggregation_hint != 'none':
        score += aggregation_bonus
    # Ensure score is not negative if using tweaks
    score = max(0.0, score)
    return round(score, 2)

# -------------------------------------------------
# --- Linguistic Feature Extraction and Scoring ---
# -------------------------------------------------

def get_question_type(doc: spacy.tokens.Doc) -> str:
    """Determines the type of question based on the first few tokens."""
    if not doc:
        return 'OTHER'

    first_token = doc[0]
    first_token_lower = first_token.lower_

    # Check for "How many"
    if first_token_lower == "how" and len(doc) > 1 and doc[1].lower_ == "many":
        return 'HOW_MANY'

    # Check for other WH-words
    wh_words = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
    if first_token_lower in wh_words or first_token.tag_ in ['WDT', 'WP', 'WP$', 'WRB']:
        return 'WH'

    # Check for auxiliary verbs indicating a boolean question
    if first_token.pos_ == 'AUX' or first_token_lower in {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had", "can", "could", "will", "would", "shall", "should", "may", "might"}:
         return 'BOOL' # Changed from AUX_BOOL for clarity

    return 'OTHER' # Imperative ("Give me...") or other structures


def analyse_linguistic_complexity(cq: str, nlp: spacy.language.Language) -> Tuple[float, Dict[str, Any]]:
    """
    Analyzes a CQ using spaCy to extract linguistic features and calculate score.

    Args:
        cq: The Competency Question string.
        nlp: The loaded spaCy Language object.

    Returns:
        A tuple containing (complexity score, dictionary of extracted features).
    """
    # Heuristic weights for scoring (tune based on empirical results)
    WEIGHTS = {
        'noun_phrase': 1.0,
        'verb': 0.5,
        'preposition': 0.8, # Proxy for relationships / PPs
        'conjunction': 1.2, # Combining clauses/criteria often adds complexity
        'modifier': 0.6,    # Adjectives/Adverbs often signal filters
        'q_type_wh': 0.5,   # Base bonus for standard WH questions
        'q_type_bool': 0.2, # Existence checks might be slightly simpler
        'q_type_how_many': 2.0, # Aggregation is often more complex
        'q_type_other': 0.0
    }

    if not cq:
        return 0.0, {"error": "Empty question"}

    # Preprocess lightly: strip whitespace and trailing question mark
    clean_cq = cq.strip().rstrip('?')
    doc = nlp(clean_cq)

    features = {}

    # 1. Number of Noun Phrases (Chunks)
    features['num_noun_phrases'] = len(list(doc.noun_chunks))
    # 2. Number of Verbs (includes auxiliaries)
    features['num_verbs'] = sum(1 for token in doc if token.pos_ == 'VERB' or token.pos_ == 'AUX')
    # 3. Number of Prepositional Phrases (approximated by counting prepositions)
    features['num_prepositions'] = sum(1 for token in doc if token.pos_ == 'ADP')
    # 4. Number of Conjunctions (Coordinating: 'and', 'or', etc.)
    features['num_conjunctions'] = sum(1 for token in doc if token.pos_ == 'CCONJ')
    # 5. Number of Modifiers (Adjectives and Adverbs)
    features['num_modifiers'] = sum(1 for token in doc if token.pos_ in ['ADJ', 'ADV'])
    # 6. Question Type
    features['question_type'] = get_question_type(doc)

    # --- Calculate Score ---
    score = 0.0
    score += features['num_noun_phrases'] * WEIGHTS['noun_phrase']
    score += features['num_verbs'] * WEIGHTS['verb']
    score += features['num_prepositions'] * WEIGHTS['preposition']
    score += features['num_conjunctions'] * WEIGHTS['conjunction']
    score += features['num_modifiers'] * WEIGHTS['modifier']

    # Add bonus based on question type
    if features['question_type'] == 'WH':
        score += WEIGHTS['q_type_wh']
    elif features['question_type'] == 'BOOL':
        score += WEIGHTS['q_type_bool']
    elif features['question_type'] == 'HOW_MANY':
        score += WEIGHTS['q_type_how_many']
    else: # Includes imperative 'Give me...' which often implies multiple results
         score += WEIGHTS['q_type_other'] # Could assign a specific weight for imperatives if needed

    # Add base complexity for having words at all
    if len(doc) > 0 :
        score += 0.1 # Small base score

    return round(score, 2), features


# ------------------------------------
# --- Syntactic Feature Extraction ---
# ------------------------------------

def calculate_tree_depth(doc: spacy.tokens.Doc) -> int:
    """Calculates the maximum depth of the dependency tree."""
    max_depth = 0
    for token in doc:
        current_depth = 0
        current_token = token
        # Trace path to root (token.head == token indicates root)
        # Add a safeguard for potential cycles, though unlikely
        visited_count = 0
        max_visit = len(doc) * 2

        while current_token.head != current_token and visited_count < max_visit:
            current_token = current_token.head
            current_depth += 1
            visited_count += 1
        if visited_count >= max_visit:
             print(f"Warning: Potential cycle detected or very deep tree for token '{token.text}' in sentence: {doc.text}")

        max_depth = max(max_depth, current_depth)
    return max_depth

def count_relevant_dependencies(doc: spacy.tokens.Doc, relevant_deps_set: Set[str]) -> Dict[str, int]:
    """Counts occurrences of specified dependency relations in the doc."""
    dep_counts = defaultdict(int)
    for token in doc:
        if token.dep_ in relevant_deps_set:
            dep_counts[token.dep_] += 1
    return dict(dep_counts) # Return as standard dict


def analyse_syntactic_complexity(cq: str, nlp: spacy.language.Language) -> Tuple[float, Dict[str, Any]]:
    """
    Analyzes a CQ using spaCy to extract syntactic features and calculate score.

    Args:
        cq: The Competency Question string.
        nlp: The loaded spaCy Language object.

    Returns:
        A tuple containing (complexity score, dictionary of extracted metrics).
    """
    # Heuristic weights for scoring (tune based on empirical results)
    # Higher weights mean these features contribute more to the complexity score
    WEIGHTS = {
        'node_count': 0.1,  # Raw length contributes a little
        'tree_depth': 0.8,  # Nested structures (depth) often indicate complexity
        'relevant_deps_total': 0.6 # Number of key syntactic relations matters
    }
    # Define which dependency relations are considered "relevant" for complexity
    # This set can be adjusted based on linguistic intuition or empirical analysis
    RELEVANT_DEPS: Set[str] = {
        # Core grammatical relations often involving entities/arguments
        'nsubj', 'nsubjpass', 'dobj', 'iobj', 'csubj', 'csubjpass', 'pobj',
        # Complements indicating structure
        'attr', 'acomp', 'xcomp', 'ccomp', 'pcomp',
        # Modifiers often indicating properties, filters, or related concepts
        'amod', 'advmod', 'prep', 'acl', 'relcl', 'npadvmod',
        # Relations indicating coordination or agency
        'conj', 'cc', 'agent',
        # Auxiliaries and particles can sometimes add nuance, optional
        # 'aux', 'auxpass', 'prt',
        # Prepositional objects/complements also captured by 'pobj'/'pcomp' above
    }
    if not cq:
        return 0.0, {"error": "Empty question"}

    # Preprocess lightly
    clean_cq = cq.strip().rstrip('?')
    doc = nlp(clean_cq)

    metrics = {}

    # 1. Node Count (Number of tokens)
    metrics['node_count'] = len(doc)
    # 2. Tree Depth
    metrics['tree_depth'] = calculate_tree_depth(doc)
    # 3. Relevant Dependency Counts
    relevant_dep_counts = count_relevant_dependencies(doc, RELEVANT_DEPS)
    metrics['relevant_dep_counts'] = relevant_dep_counts
    metrics['total_relevant_deps'] = sum(relevant_dep_counts.values())

    # --- Calculate Score ---
    score = 0.0
    score += metrics['node_count'] * WEIGHTS['node_count']
    score += metrics['tree_depth'] * WEIGHTS['tree_depth']
    score += metrics['total_relevant_deps'] * WEIGHTS['relevant_deps_total']

    # Add base complexity
    if metrics['node_count'] > 0 :
        score += 0.1

    return round(score, 2), metrics
