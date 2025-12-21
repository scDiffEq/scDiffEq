import pandas as pd
import re

def filter_df(df, regex, include=[True], axis=1):
    """
    Filters a pandas dataframe using regular expressions based on the provided words.

    Parameters:
    df (pd.DataFrame): The input dataframe to be filtered.
    words (list): A list of words to be included or excluded in the filter.
    include (bool): If True, include only columns/rows that match the words; if False, exclude columns/rows that match the words. Default is True.
    axis (int): The axis along which to filter. 0 for rows, 1 for columns. Default is 1 (columns).

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    
    
    _df = df.copy()
    
    
    for regex_pattern, include_pattern in zip(regex, include):
        # If exclude is chosen, modify regex pattern to exclude the words
        if not include_pattern:            
            # Join words with '|' as a regex pattern to match any of the words
            # regex_pattern = '|'.join(regex_pattern) # for list of lists - unlikely use-case.
            # only works if not a list... 
            # can think about how to include this or if it's even necessary sometime later on....
            
            regex_pattern = f"^(?!.*(?:{regex_pattern})).*$"
        
        # Filter the dataframe using the regex pattern
        _df = _df.filter(regex=regex_pattern, axis=axis)
    return _df

def _filter_df(df, words, include=True, axis=1):
    """
    Filters a pandas dataframe using regular expressions based on the provided words.

    Parameters:
    df (pd.DataFrame): The input dataframe to be filtered.
    words (list): A list of words to be included or excluded in the filter.
    include (bool): If True, include only columns/rows that match the words; if False, exclude columns/rows that match the words. Default is True.
    axis (int): The axis along which to filter. 0 for rows, 1 for columns. Default is 1 (columns).

    Returns:
    pd.DataFrame: The filtered dataframe.
    """

    # Join words with '|' as a regex pattern to match any of the words
    regex_pattern = '|'.join(words)

    # If exclude is chosen, modify regex pattern to exclude the words
    if not include:
        regex_pattern = f"^(?!.*(?:{regex_pattern})).*$"

    # Filter the dataframe using the regex pattern
    filtered_df = df.filter(regex=regex_pattern, axis=axis)

    return filtered_df
