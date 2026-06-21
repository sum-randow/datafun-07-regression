# Glossary

> Key terms for predictive analytics, machine learning, and linear regression.

Expand the VS Code **Outline** view (below the navigator on the right)
to see this file organization at-a-glance.

## 1. Predictive Analytics

### analytics

The practice of examining data to support decisions.
It is commonly described in three stages, each answering a different question.

- **Descriptive analytics:** What happened? (summaries, charts, EDA)
- **Predictive analytics:** What is likely to happen? (models that estimate unknown values)
- **Prescriptive analytics:** What **should we do about it**? (optimization, recommended actions)

EDA projects answer the descriptive question.
Regression begins the predictive one.

### predictive analytics

Using patterns learned from existing data to estimate values that are unknown,
missing, or not yet observed.
A predictive method takes known inputs and produces an estimate of an output.

Linear regression is one of the oldest and simplest predictive methods.
It is the natural next step after EDA:
EDA finds a relationship worth examining,
and prediction puts a number on it.

### prediction

An estimate of an unknown output, produced by a fitted model from known inputs.
A prediction is not a fact.
It is the model's best guess given what it learned,
and it carries error.

```python
# Predict CO2 emissions for one example GDP value
y_example = model.predict([[1.0e12]])
```

### model

A simplified, mathematical description of a relationship in data.
A model has a fixed *form* (here, a straight line) and adjustable *parameters*
(for example, a **slope** and an **intercept**) that are determined
by **fitting the model to data**.

> "All models are wrong, but some are useful." — George Box.
> A model is a deliberate simplification; the question is whether it is useful.

### generative AI

Systems that produce new content (text, images, code) rather than a single number
may be called *generative AI*.

Generative AI rests, in part, on predictive analytics:
a large language model works by repeatedly **predicting**
the most likely next token given the tokens so far.

The core idea is the same one introduced by linear regression:
learn parameters from data, then use them to predict.

What changes is scale and form, not the underlying premise.
Section 6 traces that connection.

## 2. Machine Learning

### machine learning

A set of methods that learn the parameters of a model directly from data,
rather than having a person specify them by hand.
"Learning" here means adjusting parameters to fit observed examples.

**Linear regression** is the common entry point to **machine learning**:
the slope and intercept are *learned* from the data,
not chosen by the analyst.

### training (fitting)

The process of adjusting a model's parameters
so the model matches the data as closely as possible.
"Training," "fitting," and "learning the parameters" are the
same idea.

```python
model = LinearRegression()
model.fit(X, y)   # training: find the best slope and intercept
```

### supervised learning

Machine learning where each training example includes both inputs
and the correct output.
The model learns to map inputs to outputs by seeing many labeled pairs.

Linear regression is supervised:
every row supplies both the selected feature (`gdp`)
and the known target (`co2`).

### unsupervised learning

Machine learning where examples have inputs but no known output.
The model finds structure on its own (for example, grouping similar rows).
Clustering is the common case.
(Not used in this project.)

### regression

A supervised task whose output is a **continuous number**
(an amount, a price, a measurement).
Predicting `co2` emissions is regression because the answer is a quantity.

### classification

A supervised task whose output is a **category or label**
(spam or not, species A/B/C).
The counterpart to regression.
Predicting penguin species would be classification.

### feature

An **input variable** used to make a prediction.
Also called a predictor or independent variable;
it goes on the x-axis.
In this script the feature is `gdp`.

### target

The output variable the model is trying to predict.
Also called the response, label, or dependent variable;
it goes on the y-axis.
In this script the target is `co2`.

### parameter (weight)

A number inside the model that is learned during training.
In linear regression there are two parameters:
the slope and the intercept.
In larger models there can be billions,
and they are usually called *weights*.

## 3. Linear Regression

### linear regression

A model that describes the target as a straight-line function of the feature:

```text
target = slope * feature + intercept
```

It assumes the relationship is a straight line.
Whether that assumption holds for a given dataset is something you test by
looking at the residuals and the fit — not something the method guarantees.

### slope (coefficient)

How much the target changes for a one-unit increase in the feature.
The slope is the learned parameter `model.coef_`.
A larger magnitude means the feature has a stronger effect on the target.

```python
slope = float(model.coef_[0])   # one slope per feature
```

In the fitted line `co2 = 3.21582e-10 * gdp + 308.446`,
the slope is the multiplier on `gdp`.

### intercept

The model's predicted target value when the feature is zero.
It is where the fitted line crosses the y-axis,
and it is the learned parameter `model.intercept_`.

```python
intercept = float(model.intercept_)
```

In `co2 = 3.21582e-10 * gdp + 308.446`, the intercept is `308.446`.

### fitted line (regression equation)

The specific straight line produced by training,
with the slope and intercept filled in with their learned values.
It is the model's summary of the relationship.

### ordinary least squares (OLS)

The standard method for fitting a regression line.
It chooses the slope and intercept that make the **sum of the squared residuals**
as small as possible.
Squaring keeps positive and negative misses from canceling and penalizes large
misses more heavily.
Both `numpy.polyfit(x, y, 1)` and scikit-learn's `LinearRegression` use OLS.

### fitted value (y-hat)

The target value the line predicts for an observed feature value, written `y_hat`.
Comparing each fitted value to the actual value is how we measure error.

```python
y_hat = model.predict(X)   # one fitted value per row
```

## 4. Evaluating the Fit

### residual

The difference between an actual value and its fitted value, for one row.

```text
residual = actual_y - fitted_y_hat
```

A positive residual sits above the line, a negative one below.
Residuals are the raw material for judging a fit:
if a straight line is a fair description, the residuals scatter randomly around
zero with no pattern.

### error (loss)

A single number summarizing how far the model's predictions are from the truth.
Training minimizes a loss; evaluation reports one.
RMSE below is one such measure.

### R-squared

The fraction of the variation in the target that the model accounts for,
on a scale from roughly 0 to 1.
Near 1, the line explains most of the variation; near 0, almost none.
It is what scikit-learn's `model.score(X, y)` returns for a regressor.

```text
R-squared = 1 - (sum of squared residuals) / (total variation in y)
```

R-squared describes how much the line explains, not whether a line is the
*right shape*. A patterned residual plot can accompany a high R-squared.

### RMSE (root mean squared error)

The typical size of a residual, in the same units as the target.

```python
rmse = float(np.sqrt(np.mean(residuals**2)))
```

Smaller is closer. Because it shares the target's units,
RMSE is often easier to interpret than R-squared.

### overfitting

When a model matches its training data so closely that it captures noise
rather than the real pattern, and then predicts poorly on new data.
A straight line rarely overfits; flexible models (neural networks) easily can.

### generalization

How well a model predicts data it did not train on.
Good generalization is the actual goal of predictive modeling;
fitting the training data is only a means to it.

### train/test split

Holding back part of the data during training so it can be used to check
generalization honestly.
This script fits on all rows for simplicity and does not split,
but splitting is standard practice once a model is taken seriously.

## 5. Terms in the App

### feature matrix (X)

The inputs arranged as a 2-D table with shape `(n_rows, n_features)`,
because scikit-learn always expects a column, even for a single feature.
Double brackets return a DataFrame, which becomes a 2-D array.

```python
X = df_model[[FEATURE_COL]].to_numpy()   # shape (n, 1)
```

### target vector (y)

The outputs arranged as a 1-D list with shape `(n_rows,)`.
Single brackets return a Series, which becomes a 1-D array.

```python
y = df_model[TARGET_COL].to_numpy()      # shape (n,)
```

### shape

The dimensions of an array, given as a tuple.
`(308, 1)` means 308 rows and 1 column; `(308,)` means a flat list of 308 values.
Matching shapes to what a tool expects is half of getting machine learning code
to run.

### reshape / ravel

Changing an array's shape without changing its values.
`reshape(-1, 1)` turns a flat list into a single column;
`ravel()` flattens a column back into a list.

```python
x_flat = X.ravel()                 # (n, 1) -> (n,) for numpy.polyfit
X_one = np.array([[1.0e12]])       # a single (1, 1) input for predict
```

### numpy.polyfit

A NumPy function that fits a polynomial to data.
With degree `1` it fits a straight line and returns `[slope, intercept]`.
The script uses it as the "by hand" fit to confirm scikit-learn's result.

```python
slope, intercept = np.polyfit(x_flat, y, 1)
```

### LinearRegression (scikit-learn)

The standard library object for fitting a linear regression by OLS.
You create it, call `.fit()` to train, then use `.coef_`, `.intercept_`,
`.predict()`, and `.score()`.

### coef\_ / intercept\_

The learned parameters, available after `.fit()`.
The trailing underscore is a scikit-learn convention meaning
"set by fitting, not by you."
`coef_` is an array (one slope per feature); `intercept_` is a single number.

### .predict()

Applies the fitted model to inputs and returns predicted target values.
Used both for the fitted values (`predict(X)`) and for a new example
(`predict([[value]])`).

### .score()

For a regressor, returns the R-squared of the model on the data you pass in.

```python
r_squared = model.score(X, y)
```

### modeling view

A copy of the data reduced to the rows a model can actually use:
rows missing the feature or the target are dropped first.
This is why the run reported 350 original rows but 308 model rows.

```python
df_model = df.dropna(subset=[FEATURE_COL, TARGET_COL]).copy()
```

## 6. From Linear Regression to Generative AI

### weights and bias

The general names for a model's learned parameters.
The slope generalizes to a vector of **weights** (one per input),
and the intercept generalizes to a **bias**.
A linear regression with two parameters and a language model with billions
are described by the same vocabulary.

### loss function

A formula that scores how wrong a model's predictions are,
which training works to minimize.
OLS minimizes squared error; classifiers and language models minimize other
losses (such as cross-entropy). The principle is identical.

### gradient descent

The general procedure for minimizing a loss:
nudge the parameters a little at a time in the direction that reduces error,
and repeat.
OLS has a direct formula and does not need it,
but every large model is trained this way.

### neural network

A model built by stacking many simple linear pieces with nonlinear steps
between them.
A single linear unit with no nonlinearity *is* a linear regression;
stacking and bending them is what lets the family represent curves, images,
and language.

### next-token prediction

The task a language model is trained on:
given the text so far, predict a probability for every possible next token,
then pick or sample one.
Generating a paragraph is this single prediction repeated many times.

### where linear regression fits

Linear regression is the smallest member of one continuous family of
predictive models:

```text
linear regression   ->  one feature, one weight, a straight line
multiple regression ->  many features, many weights, a flat plane
logistic regression ->  same machinery, output is a category
neural network      ->  many stacked weighted sums plus nonlinearity
transformer / LLM   ->  a very large network that predicts the next token
```

Every step up adds parameters and flexibility,
but the premise never changes:
learn parameters from data by reducing a loss,
then use them to predict.
Understanding the slope and intercept in this script is understanding,
in miniature, the engine underneath generative AI.
