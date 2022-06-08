import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from hime.linear_model import LinearRegression


@pytest.fixture
def df_test():
    """test dataset from seaborn"""
    return sns.load_dataset("iris")


class TestLinearRegression:
    """Testing class to test the LinearRegression class."""

    def test_if_dataframe_not_affected(self, df_test):
        """Check if the function leaves the data frame the same."""

        target = "species_cat_codes"
        df_original = df_test.assign(
            **{target: lambda x: x["species"].astype("category").cat.codes}
        )
        features_list = df_original.select_dtypes(float).columns.tolist()


        assert df_original.drop(target, axis=1).equals(df_test)

