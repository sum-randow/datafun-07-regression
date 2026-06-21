"""app_penguins_case.py - Project script (example).

Author: Denise Case
Date: 2026-06

Purpose:
    - simple linear regression on a dataset with a clear linear relationship
    - serving as the "good fit" baseline to compare against a messier case
    - choosing a feature (x) and a target (y) based on EDA findings
    - fitting a straight line with scikit-learn (with an optional numpy check)
    - reading off the slope and intercept
    - computing fitted values and residuals
    - examining R-squared and RMSE
    - making a prediction for a chosen feature value
    - charting the data, the fitted line, and the residuals

Data Source:
- Palmer Penguins, via Seaborn's built-in datasets (sns.load_dataset)
- Original: Horst AM, Hill AP, Gorman KB (2020). palmerpenguins.

Assumptions:
- The data contains columns like:
  species, island, bill_length_mm, bill_depth_mm,
  flipper_length_mm, body_mass_g, sex

Terminal command to run this file from the root project folder:

uv run python -m datafun.app_penguins_case

OBS:
  Don't edit this file - it should remain a working example.
  Copy it, rename it, and modify your copy.

  This script does NOT decide for you whether a straight line is a good
  description of the data. It fits the line and computes the numbers an
  analyst uses to make that call (residuals, R-squared, RMSE) - and then
  it shows them to you. Whether the relationship is "linear enough" is a
  judgment you make after looking. Doing the analysis is how you find out.

  This example was chosen because the relationship is expected to be a
  clean straight line. Compare its residual plot to a messier dataset's:
  here the residuals should scatter randomly around zero, with no pattern.
"""


# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging  # for type hinting only
from typing import Final  # for type hinting

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Type hint for Axes object (basic plot type returned by Seaborn)
# A seaborn plot is a set of axes. Set title, labels, etc. on the axes.
# A figure can contain multiple axes (plots)
# from matplotlib.figure import Figure

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("P07", level="DEBUG")
log_header(LOG, "P07")

# === Section 1c. Global Constants and Configuration ===

# CUSTOM: These are dataset-specific constants
# used in multiple places in the code.
# Inspect or explore the dataset (during EDA) to determine the columns needed.

# CUSTOM: Data set name (a Seaborn built-in dataset, loaded by name)
DATASET_NAME: Final[str] = "penguins"

# ==========================================================
# ANALYST CHOICE:
# Linear regression models one numeric TARGET (y) as a straight-line
# function of one numeric FEATURE (x):  y = slope * x + intercept
#
# This pairing (flipper length -> body mass) is expected to be close to a
# straight line: bigger penguins tend to have both longer flippers and
# greater mass. We still do not assume it - we fit the line and look.
# ==========================================================

# CUSTOM: One numeric feature (the predictor, plotted on the x-axis)
FEATURE_COL: Final[str] = "flipper_length_mm"

# CUSTOM: One numeric target (the response, plotted on the y-axis)
TARGET_COL: Final[str] = "body_mass_g"

# CUSTOM: Assign readable labels for the charted variables.
FEATURE_LABEL: Final[str] = "Flipper length (mm)"
TARGET_LABEL: Final[str] = "Body mass (g)"

# CUSTOM: A single feature value to predict the target for, as an example.
# Pick a value inside (or near) the range of the data you observed in EDA.
EXAMPLE_FEATURE_VALUE: Final[float] = 200.0  # example flipper length in mm

# === Section 1d. Pandas Configuration for Display ===

# Pandas display configuration (helps in notebooks)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load a dataset into a DataFrame.

    This function loads the Palmer Penguins dataset, which ships with
    Seaborn. It is loaded by name, so there is no CSV file to manage.

    Arguments: None

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    LOG.info(f"Loading dataset: {DATASET_NAME}")
    df: pd.DataFrame = sns.load_dataset(DATASET_NAME)
    count_of_rows: int = df.shape[0]
    count_of_columns: int = df.shape[1]
    LOG.info(f"Loaded: {count_of_rows} rows, {count_of_columns} columns")

    return df


# === Section 3. Prepare a Modeling View (Feature + Target Only) ===


def make_model_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view containing only the rows we can model.

    Strategy:
    - Keep the original DataFrame unchanged
    - Drop rows missing the feature OR the target

    WHY: A regression cannot use a row that is missing either the x value
    or the y value. We remove those rows up front so the model sees only
    complete (x, y) pairs.

    Arguments:
        df: The original DataFrame.

    Returns:
        pd.DataFrame: A cleaned view with no missing feature/target values.
    """
    LOG.info("Creating modeling view (dropping rows missing feature or target)")

    # The two columns we require to be non-missing.
    # FEATURE_COL and TARGET_COL are single strings, so wrap them in a list.
    cols_required: list[str] = [FEATURE_COL, TARGET_COL]
    LOG.debug(f"Columns required to be non-missing: {cols_required}")

    # dropna(subset=...) only looks at the specified columns, not the whole row.
    # .copy() creates a new DataFrame so we don't accidentally modify the original.
    df_model: pd.DataFrame = df.dropna(subset=cols_required).copy()

    # Report what was kept and what was dropped
    count_original: int = df.shape[0]
    count_model: int = df_model.shape[0]
    count_dropped: int = count_original - count_model

    LOG.info(f"Original rows: {count_original}")
    LOG.info(f"Model rows:    {count_model}")
    LOG.info(f"Rows dropped:  {count_dropped}")

    return df_model


# === Section 4. Build the Feature Matrix X and Target Vector y ===


def build_x_and_y(df_model: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build X (feature matrix) and y (target vector) for scikit-learn.

    WHY: scikit-learn expects two different shapes:
    - X must be 2-D, with shape (n_rows, n_features). Even with one
      feature, it must be (n_rows, 1) - a column, not a flat list.
    - y is 1-D, with shape (n_rows,).

    The double-bracket df[[FEATURE_COL]] returns a DataFrame (2-D),
    while the single-bracket df[TARGET_COL] returns a Series (1-D).
    Converting each to a NumPy array gives the shapes sklearn wants.

    Arguments:
        df_model: The cleaned modeling view.

    Returns:
        tuple[np.ndarray, np.ndarray]: (X with shape (n, 1), y with shape (n,))
    """
    LOG.info("Building feature matrix X and target vector y")

    # Double brackets -> DataFrame -> 2-D array of shape (n, 1)
    X: np.ndarray = df_model[[FEATURE_COL]].to_numpy()

    # Single brackets -> Series -> 1-D array of shape (n,)
    y: np.ndarray = df_model[TARGET_COL].to_numpy()

    LOG.debug(f"X shape: {X.shape}  (rows, features)")
    LOG.debug(f"y shape: {y.shape}  (rows,)")

    return X, y


# === Section 5. Fit a Straight Line ===


def fit_line(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Fit a straight line y = slope * x + intercept using scikit-learn.

    WHY: LinearRegression() is the standard tool. The pattern you learn
    here - create a model, call .fit() to train it, then read its results
    and call .predict() - is the SAME pattern used for every other
    scikit-learn model (other regressions, classification, and beyond).
    Learning it once carries over.

    The model exposes its learned parameters after fitting:
    - .coef_      the slope (an array, one value per feature)
    - .intercept_ the intercept (a single number)

    Arguments:
        X: Feature matrix of shape (n, 1).
        y: Target vector of shape (n,).

    Returns:
        LinearRegression: The fitted scikit-learn model.
    """
    LOG.info("Fitting a linear regression (scikit-learn)")

    # Create the model object, then fit it to the data.
    model = LinearRegression()
    model.fit(X, y)

    # coef_ is an array (one slope per feature); we have one feature.
    slope: float = float(model.coef_[0])
    intercept: float = float(model.intercept_)
    LOG.debug(f"  slope:     {slope:.6g}")
    LOG.debug(f"  intercept: {intercept:.6g}")

    LOG.info("Fitted line:")
    LOG.info(f"  {TARGET_COL} = {slope:.6g} * {FEATURE_COL} + {intercept:.6g}")

    # OPTIONAL sanity check - the library is not magic.
    # numpy can fit the same straight line directly: degree 1 returns
    # [slope, intercept]. Uncomment to confirm the values match.
    #   np_slope, np_intercept = np.polyfit(X.ravel(), y, 1)
    #   LOG.debug(f"  numpy check -> slope {np_slope:.6g}, intercept {np_intercept:.6g}")

    return model


# === Section 6. Predict (Fitted Values and an Example Value) ===


def predict(model: LinearRegression, X: np.ndarray) -> np.ndarray:
    """Compute fitted values and predict for one example feature value.

    WHY: The fitted values (y_hat) are what the line says y "should" be
    for each observed x. We will compare them to the real y later to see
    how far off the line is. We also show a single prediction for a chosen
    feature value as an example of using the model.

    Arguments:
        model: The fitted scikit-learn model.
        X: Feature matrix of shape (n, 1).

    Returns:
        np.ndarray: Fitted values y_hat of shape (n,).
    """
    LOG.info("Computing fitted values for every observed row")
    y_hat: np.ndarray = model.predict(X)

    LOG.info(f"Predicting {TARGET_LABEL} for one example {FEATURE_LABEL} value")

    # The model expects a 2-D input of shape (n, 1), even for one value.
    X_example: np.ndarray = np.array([[EXAMPLE_FEATURE_VALUE]])
    y_example: float = float(model.predict(X_example)[0])

    LOG.debug(f"  example {FEATURE_COL}: {EXAMPLE_FEATURE_VALUE:.6g}")
    LOG.debug(f"  predicted {TARGET_COL}: {y_example:.6g}")

    return y_hat


# === Section 7. Examine the Fit (Residuals, R-squared, RMSE) ===


def examine_fit(
    model: LinearRegression, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray
) -> np.ndarray:
    """Compute the numbers used to judge a linear fit.

    WHY: A line can be fit to ANY pair of numeric columns. These numbers
    are how an analyst decides whether the line is a reasonable description
    of the data:

    - residual = actual y - fitted y_hat (one per row)
        How far each point sits above (+) or below (-) the line.
    - R-squared: the fraction of the variation in y the line accounts for.
        Ranges roughly 0 to 1. Higher means the line explains more.
    - RMSE: root mean squared error, in the same units as y.
        The typical size of a residual.

    This function computes and reports these. It does NOT declare the fit
    "good" or "bad" - you read the numbers and the residual plot (Section 8)
    and make that call yourself.

    Arguments:
        model: The fitted scikit-learn model.
        X: Feature matrix of shape (n, 1).
        y: Actual target values, shape (n,).
        y_hat: Fitted target values, shape (n,).

    Returns:
        np.ndarray: Residuals of shape (n,).
    """
    LOG.info("Computing residuals (actual - fitted)")
    residuals: np.ndarray = y - y_hat

    # R-squared straight from the model (sklearn's .score is R-squared for
    # regression). Equivalent to comparing the line to a flat mean line.
    r_squared: float = model.score(X, y)

    # RMSE: square the residuals, average them, take the square root.
    rmse: float = float(np.sqrt(np.mean(residuals**2)))

    LOG.info("Fit numbers (for you to interpret):")
    LOG.debug(f"  R-squared: {r_squared:.4f}")
    LOG.debug(f"  RMSE:      {rmse:.6g}  (in units of {TARGET_LABEL})")
    LOG.debug(f"  residual min:  {float(np.min(residuals)):.6g}")
    LOG.debug(f"  residual max:  {float(np.max(residuals)):.6g}")
    LOG.debug(f"  residual mean: {float(np.mean(residuals)):.6g}")

    LOG.info("""
CUSTOM: Update these notes and use Markdown cells to narrate what you see.

How to read these (this is YOUR judgment, not the script's):

 - R-squared near 1: the line accounts for most of the variation in y.
   R-squared near 0: the line accounts for almost none.
 - RMSE: the typical distance between a point and the line, in y's units.
 - Residual plot (Section 8): if a straight line fits well, residuals
   scatter randomly around 0 with no pattern. A curve, a funnel (spread
   that grows or shrinks), or clusters are signs a straight line is not
   the right description - which is itself a useful finding.

There is no threshold that decides this for you. Look at the numbers and
the plots together and write down what you conclude.
""")

    return residuals


# === Section 8. Create Visualizations ===


def make_plots(
    df_model: pd.DataFrame, y_hat: np.ndarray, residuals: np.ndarray
) -> None:
    """Create notebook-friendly plots for the regression.

    Arguments:
        df_model: Cleaned modeling view.
        y_hat: Fitted values, shape (n,).
        residuals: Residuals (actual - fitted), shape (n,).

    Returns:
        None

    WHY: The fitted-line plot shows whether the line tracks the points.
    The residual plot shows whether what's left over has a pattern.
    Together they show whether a straight line is a fair description.

    Common charts here:
    1. A scatter of feature vs target with the fitted line drawn on top.
    2. A residual plot: residuals vs the feature, with a line at zero.
    """
    feature_values: np.ndarray = df_model[FEATURE_COL].to_numpy()
    target_values: np.ndarray = df_model[TARGET_COL].to_numpy()

    LOG.info("---- Creating Scatter Plot with Fitted Line ----------")
    LOG.info(f"----   Set x to {FEATURE_LABEL} -----------------------")
    LOG.info(f"----   Set y to {TARGET_LABEL} -------------------------")

    # Open a fresh blank canvas before a new chart
    plt.figure()

    # The observed points
    scatter_plt: Axes = sns.scatterplot(
        x=feature_values,
        y=target_values,
    )

    # The fitted line. Sort by x so the line is drawn left to right.
    order: np.ndarray = np.argsort(feature_values)
    scatter_plt.plot(feature_values[order], y_hat[order])

    scatter_plt.set_xlabel(FEATURE_LABEL)
    scatter_plt.set_ylabel(TARGET_LABEL)
    scatter_plt.set_title(f"{FEATURE_LABEL} vs {TARGET_LABEL} with fitted line")

    # IN NOTEBOOK: SHOW AS YOU GO
    #      plt.show() displays the current chart and closes it
    #      Call this before starting a new chart
    #      or next chart will be drawn on top of this one
    # IN SCRIPT: WAIT TO SHOW TILL THE END
    #      Do not call plt.show() here - let figures accumulate
    #      so all charts display together with sequential Figure numbers.
    #      plt.show() is called once at the end of main()
    # plt.show()

    LOG.info("------ Creating Residual Plot --------------------------")
    LOG.info(f"------   Set x to {FEATURE_LABEL} ----------------------")
    LOG.info("------   Set y to the residual (actual - fitted) -------")

    # Open a fresh blank canvas before a new chart
    plt.figure()

    residual_plt: Axes = sns.scatterplot(
        x=feature_values,
        y=residuals,
    )

    # A reference line at residual = 0. Points scattered randomly around
    # this line (no pattern) is what a good straight-line fit looks like.
    residual_plt.axhline(0)

    residual_plt.set_xlabel(FEATURE_LABEL)
    residual_plt.set_ylabel(f"Residual ({TARGET_LABEL})")
    residual_plt.set_title(f"Residuals vs {FEATURE_LABEL}")

    # IN NOTEBOOK: SHOW AS YOU GO
    #      plt.show() displays the current chart and closes it
    # IN SCRIPT: WAIT TO SHOW TILL THE END
    #      Do not call plt.show() here - plt.show() is called once at end.
    # plt.show()


# === Section 9. Summary and Next Steps ===


def summarize(
    df: pd.DataFrame, df_model: pd.DataFrame, model: LinearRegression
) -> None:
    """Log a brief summary of the model and what to examine next.

    WHY: Regression is not the final answer. The summary records what was
    fit and points to what the analyst still has to decide.

    Arguments:
        df: The original DataFrame.
        df_model: The cleaned modeling view.
        model: The fitted scikit-learn model.

    Returns:
        None
    """
    slope: float = float(model.coef_[0])
    intercept: float = float(model.intercept_)

    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Feature (x): {FEATURE_COL}")
    LOG.info(f"Target  (y): {TARGET_COL}")

    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Model rows:    {df_model.shape[0]}")

    LOG.info("Fitted line:")
    LOG.info(f"  {TARGET_COL} = {slope:.6g} * {FEATURE_COL} + {intercept:.6g}")

    LOG.info("======================")
    LOG.info("Review the fit numbers (R-squared, RMSE). ")
    LOG.info("Look at the fitted-line plot and the residual plot. ")
    LOG.info("Decide if a `straight line` is a fair description. ")
    LOG.info("If the residuals DO show a pattern,")
    LOG.info("then a straight line is NOT a good description. ")
    LOG.info("If the residuals DO NOT show a pattern,")
    LOG.info("then a straight line MIGHT be a good description. ")
    LOG.info("Either way, the findings may be valuable.")
    LOG.info("======================")
    LOG.info("Repeat with a different feature, or a transformed feature, ")
    LOG.info("to investigate other options.")
    LOG.info("======================")
    LOG.info("Include instructions and specifics in your README.md file.")
    LOG.info("Write up your narrative on your docs/index.md file.")
    LOG.info("Include your next step suggestions for further analysis or modeling.")
    LOG.info("======================")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the linear regression workflow."""
    log_header(LOG, "REGRESSION")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    LOG.info(f"--- Section 2: Load dataset: {DATASET_NAME} ---")
    df = load_data()

    LOG.info("--- Section 3: Prepare a modeling view (feature + target) ---")
    df_model = make_model_view(df)

    LOG.info("--- Section 4: Build feature matrix X and target vector y ---")
    X, y = build_x_and_y(df_model)

    LOG.info("--- Section 5: Fit a straight line (scikit-learn) ---")
    model = fit_line(X, y)

    LOG.info("--- Section 6: Predict fitted values and an example value ---")
    y_hat = predict(model, X)

    LOG.info("--- Section 7: Examine the fit (residuals, R-squared, RMSE) ---")
    residuals = examine_fit(model, X, y, y_hat)

    LOG.info("--- Section 8: Charts ---")
    make_plots(df_model, y_hat, residuals)

    LOG.info("--- Section 9: Summary and next steps ---")
    summarize(df, df_model, model)

    LOG.info(
        "----- in a script, call plt.show() once at the end to display all charts -----"
    )
    LOG.info(
        "----- in a script, close the chart windows (with the close button) to continue  -----"
    )
    plt.show()

    LOG.info("Linear regression workflow complete")
    LOG.info("IMPORTANT: This script creates chart windows.")
    LOG.info(
        "Close any chart windows and terminate this process with CTRL+c as needed."
    )
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

# WHY: Only call main() when running this file directly as a script.
# This is standard Python boilerplate.

if __name__ == "__main__":
    main()
