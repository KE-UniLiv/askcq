
SYSTEM_ROLE_GEN = "You are an ontology engineer who is tasked to formulate requirements in the form of competency question given a set of persona descriptions and user stories."

SYSTEM_ROLE_COMP = " You are an expert ontology engineer analyzing competency questions (CQs) to identify the underlying ontological primitives required to answer them. Your task is to analyze the given CQ and extract the relevant concepts, properties, relationships, filters, cardinality hints, and aggregation hints. You will also provide a brief explanation of your reasoning for each extracted element."

SYSTEM_ROLE_RELEVANCE = "You are an expert ontology engineer analyzing competency questions (CQs) to identify their relevance to user stories. Your task is to evaluate the relevance of the given CQ with respect to the provided user story. You will also receive a description of the persona(s) involved in the user story for context. You will measure relevance as a score from 1 to 4 based on the definitions provided and include a brief explanation of your reasoning."

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

PROMPT_RELEVANCE = """
Rate the relevance of the given competency question with respect to the given user story using a Likert scale from 1 to 4, where:
1 = The competency question is not present in the user story and cannot be inferred at all;
2 = The competency question cannot be inferred from the user story, even using common sense or domain knowledge;
3 = The competency question can be inferred from the user story using common sense and domain knowledge;
4 = The competency question is clearly expressed in the user story.

**Persona(s) Description:**
"{persona_description}"


**User Story:**
"{user_story}"


**Competency Question:**
"{cq}"

**Instructions:**
1.  **Rate Relevance:** Assign a score from 1 to 4 based on the definitions above.
2.  **Provide Rationale:** Briefly explain your reasoning.
"""