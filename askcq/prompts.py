


SYSTEM_ROLE_GEN = "You are an ontology engineer who is tasked to formulate requirements in the form of competency question given a set of persona descriptions and user stories."

SYSTEM_ROLE_COMP = " You are an expert ontology engineer analyzing competency questions (CQs) to identify the underlying ontological primitives required to answer them. Your task is to analyze the given CQ and extract the relevant concepts, properties, relationships, filters, cardinality hints, and aggregation hints. You will also provide a brief explanation of your reasoning for each extracted element."

PROMPT_COMP = """
Focus on the *minimal* set of primitives essential to fulfill the requirement expressed in the question.

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

