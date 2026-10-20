import io
from importlib import resources

def load_bikeshare():
    """
    Load the bikeshare dataset.

    This dataset is used in the examples and tests. It is the ISLR2 version
    (also ISLP's `Bikeshare`) of the UCI Bike Sharing data: Fanaee-T, H. and
    Gama, J. (2013), "Event labeling combining ensemble detectors and
    background knowledge", *Progress in Artificial Intelligence*, 2, 113-127,
    https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset, licensed
    under CC BY 4.0.

    Returns
    -------
    io.StringIO
        The bikeshare data in CSV format, read into memory (the file is
        closed). You can read this with your preferred CSV reader (e.g.,
        pandas.read_csv).

    Examples
    --------

    try:
        import pandas as pd
        bikeshare = pd.read_csv(load_bikeshare())
        print(bikeshare.head())
    except ImportError:
        print("pandas is not installed. Please install it to run this example.")

    """
    return io.StringIO(resources.files('scatter_smooth').joinpath('data/bikeshare.csv')
                       .read_text())
