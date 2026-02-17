from importlib import resources

def load_bikeshare():
    """
    Load the bikeshare dataset.

    This dataset is used in the examples and tests.

    Returns
    -------
    file-like object
        A file-like object containing the bikeshare data in CSV format.
        You can read this with your preferred CSV reader (e.g., pandas.read_csv).
    
    Examples
    --------
    
    try:
        import pandas as pd
        with load_bikeshare() as f:
            bikeshare = pd.read_csv(f)
        print(bikeshare.head())
    except ImportError:
        print("pandas is not installed. Please install it to run this example.")

    """
    return resources.files('scatter_smooth').joinpath('data/bikeshare.csv').open('r')
