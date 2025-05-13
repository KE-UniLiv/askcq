

import pandas as pd
import numpy as np
# If using statsmodels version 0.14 or later:
try:
    from statsmodels.stats.inter_rater import fleiss_kappa
# For older versions (e.g., 0.13), the path might be different:
except ImportError:
     # Attempt older import path if the newer one fails
     try:
        from statsmodels.stats.inter_rater import aggregate_raters
        # Older versions often calculated kappa via aggregate_raters output
        # We will define a wrapper function later if needed.
        # For now, just note that the direct fleiss_kappa function is preferred.
        print("Warning: Using older statsmodels import structure for inter-rater agreement.")
     except ImportError:
        print("Error: Could not import inter-rater agreement functions from statsmodels.")
        print("Please ensure statsmodels is installed (`pip install statsmodels`).")
        # Define dummy functions or raise error if statsmodels isn't essential
        # For this code, we'll assume it's needed and let potential errors propagate.
        pass

# --- Helper Function ---
def score_to_counts(score):
    """
    Converts the aggregated score (-3, -1, 1, 3) from 3 raters
    into counts of [accepts, rejects].
    """
    if score == 3:
        return [3, 0]
    elif score == 1:
        return [2, 1]
    elif score == -1:
        return [1, 2]
    elif score == -3:
        return [0, 3]
    else:
        # Handle unexpected scores if necessary
        # print(f"Warning: Unexpected score {score} encountered.")
        return [np.nan, np.nan] # Or raise an error

# --- Calculation Function ---
def calculate_fleiss_kappa_from_scores(df, group_by_col=None, score_col='score', set_mapping=None):
    """
    Calculates Fleiss' Kappa for inter-annotator agreement with 3 raters,
    based on aggregated scores. Can calculate overall or grouped by a column.

    Args:
        df (pd.DataFrame): DataFrame containing the scores and grouping column.
        group_by_col (str, optional): Column name to group by (e.g., 'set').
                                     If None, calculates overall kappa. Defaults to None.
        score_col (str): Name of the column with scores (-3, -1, 1, 3). Defaults to 'score'.
        set_mapping (dict, optional): Dictionary mapping group IDs to names. Defaults to None.

    Returns:
        dict or float: If group_by_col is provided, returns a dictionary mapping
                       group names (or IDs if no mapping) to kappa values.
                       If group_by_col is None, returns a single float kappa value.
                       Returns np.nan for groups with insufficient data.
    """
    results = {}

    if group_by_col:
        grouped = df.groupby(group_by_col)
        for name, group in grouped:
            # Apply the transformation to get the N x k table
            # Drop rows where score_to_counts returned NaN (due to invalid scores)
            counts_table = group[score_col].apply(score_to_counts).tolist()
            counts_table = [item for item in counts_table if not np.isnan(item).any()]

            # Ensure we have data and it's meaningful
            if len(counts_table) < 2: # Need at least 2 subjects for kappa
                 print(f"Warning: Skipping group '{name}' due to insufficient data (less than 2 valid CQs).")
                 kappa = np.nan
            else:
                try:
                    # Use direct fleiss_kappa if available (statsmodels >= 0.14)
                    if 'fleiss_kappa' in globals():
                         kappa = fleiss_kappa(counts_table)
                    # Fallback for older versions (less direct, might need aggregate_raters)
                    # This part might need adjustment based on the exact older version behavior
                    elif 'aggregate_raters' in globals():
                         # aggregate_raters returns data in a different format,
                         # often used with methods like fleiss_kappa_alt or similar.
                         # This is a simplification; check statsmodels docs for your version.
                         # For demonstration, we'll just mark it as potentially needing review.
                         print(f"Warning: Direct fleiss_kappa not found. Calculation for group '{name}' might require adaptation for older statsmodels.")
                         # Example using aggregate_raters if it directly yields kappa (unlikely)
                         # agg_data, _ = aggregate_raters(np.array(counts_table))
                         # kappa = some_kappa_function(agg_data) # Placeholder
                         kappa = np.nan # Defaulting to NaN as direct calculation differs
                    else:
                        print("Error: No suitable kappa function found in statsmodels.")
                        kappa = np.nan

                except Exception as e:
                    print(f"Error calculating Fleiss' Kappa for group '{name}': {e}")
                    kappa = np.nan

            group_name = set_mapping.get(name, name) if set_mapping else name
            results[group_name] = kappa
        return results
    else:
        # Calculate overall kappa
        counts_table = df[score_col].apply(score_to_counts).tolist()
        counts_table = [item for item in counts_table if not np.isnan(item).any()]

        if len(counts_table) < 2:
            print("Warning: Insufficient data for overall kappa calculation (less than 2 valid CQs).")
            return np.nan
        try:
             # Use direct fleiss_kappa if available (statsmodels >= 0.14)
             if 'fleiss_kappa' in globals():
                  kappa = fleiss_kappa(counts_table)
             # Fallback logic (simplified)
             elif 'aggregate_raters' in globals():
                  print("Warning: Direct fleiss_kappa not found. Overall calculation might require adaptation for older statsmodels.")
                  kappa = np.nan # Placeholder
             else:
                 print("Error: No suitable kappa function found in statsmodels.")
                 kappa = np.nan

             return kappa
        except Exception as e:
            print(f"Error calculating overall Fleiss' Kappa: {e}")
            return np.nan


# --- Example Usage ---
# Create a sample DataFrame (replace with your actual cq_df)
data = {
    'cq': range(15),
    'set': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    'score': [3, 1, -1, 3, 3, -3, 1, -1, -3, 3, 1, 1, -1, -3, -3],
    'comment': [''] * 15,
    'ambiguity': [False] * 15
}


