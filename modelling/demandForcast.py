import pandas as pd
import numpy as np
import os 
from dotenv import load_dotenv
from pathlib import Path 
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
import pickle

# Load environment variables
cur_file_path = Path(__file__).resolve()
env_path = cur_file_path.parent.parent / '.env'
load_dotenv(env_path)
DB_URL = os.getenv("DATABASE_URL")

# Load the model input table from Supabase
def read_model_from_supabase(table):
    try:
        engine = create_engine(DB_URL)
        print(f"Load '{table}' from Supabase !!")
        model_df = pd.read_sql(f"SELECT * FROM {table}", engine)
        model_df.proj_type = model_df.proj_type.fillna('Other')
        return model_df
    except Exception as e:
        print("Failed to read table from Supabase")

# write random forest reg res to excel file
def writeSkuPrice2Excel(df, file_path):
    # prepare default tab for model input  
    def_df = None
    def_df = df.groupby(["sku_pk"], as_index = False, observed = False).agg(
        price_floor = ('price', 'min'),
        price_ceil = ('price', 'max')
    )
    def_df["new price"] = [np.nan] * def_df.shape[0]

    # xlsx not exist --> create new sheet and create xlsx
    if not os.path.exists(file_path):

        # subset sku with high price adj frequency (wait for new price assignment)
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as wrtr:
            def_df.to_excel(excel_writer = wrtr, sheet_name = "sku price adj (high freq)", index = False)

# Define adjusted R^2
def adjusted_R2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    n = y_pred.size
    return 1 - (1 - r2) * (n - 1) / (n - 22 - 1)

# Define SMAPE (symmetric mean absolute percentage error)
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_values = np.where(denominator == 0, 0, np.abs(y_true - y_pred) / denominator)
    return 100 * np.mean(smape_values)  # Return as percentage

# Generate mapping table for SKU as input
def SkuInputMapping(df):
    df["sku_pk"] = df["sku_pk"].astype("category")
    skuInput = {idx: category for idx, category in enumerate(df.sku_pk.cat.categories)}
    stratifiedLabels = df.sku_pk.cat.codes
    # print(f"Cat field conversion to numeric val:\n{stratifiedLabels}\n")

    # Stats exploration
    sku_pks, freq = np.unique(df.sku_pk, return_counts=True)
    mue, median, m = np.mean(freq), np.median(freq), stats.mode(freq)[0]
    # print(f"Unique list of pks:\n{sku_pks}\nFreq:\n{freq}\n")
    print(f"Mean = {mue};\tMedian = {median};\tMode = {m}")
    return skuInput, stratifiedLabels

# Perform stratified k-fold to evaluate the model
def rf_fine_tune(df, stratifiedLabels):
    # Prepare for model training
    df = df.copy()
    df = df.drop('sku_pk', axis=1)
    cat_features = ['sku_cat', 'prod_cd', 'proj_type', 'cust_cat', 'cust_tier']
    X = df[[col for col in df.columns if col != "quantity"]]
    y = df["quantity"].values.flatten()

    # Define categorical feature transformation
    cat_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine all numeric/categorical transformations
    preprocessor = ColumnTransformer([
        ("cat", cat_transformer, cat_features)
    ])

    # Define RF regressor model (use all 23 features)
    rf_model = RandomForestRegressor(random_state=17, max_features=1.0)
    rf_pip = Pipeline([
        ("prep", preprocessor),
        ("rfr", rf_model)
    ])

    # RF regressor hyperparameters (tailored to dataset properties)
    param_grid = {
        "rfr__n_estimators": [50, 100],
        "rfr__max_depth": [10, 20, 35, None],  # Prevent overfitting
        "rfr__min_samples_leaf": [5, 7, 10, 15, 20, 25],  # Ensure sufficient transactions
    }

    # Define custom scorers
    adj_r2_eval = make_scorer(adjusted_R2, greater_is_better=True)
    smape_eval = make_scorer(smape, greater_is_better=False)

    # Perform stratified k-fold cross-validation
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfGridCv = GridSearchCV(
        rf_pip, param_grid,
        cv=strat_cv.split(X, stratifiedLabels),
        scoring={"Adjusted_R2": adj_r2_eval, "SMAPE": smape_eval},
        refit="SMAPE",  # Refit the best model based on SMAPE
        n_jobs=-1, verbose=0
    )
    rfGridCv.fit(X, y)
    return rfGridCv

# xgboost model for demand forcast
def xgboost_fine_tune(df, stratifiedLabels):
    # Prepare for model training
    df = df.copy()
    df = df.drop('sku_pk', axis=1)
    cat_features = ['sku_cat', 'prod_cd', 'proj_type', 'cust_cat', 'cust_tier']
    X = df[[col for col in df.columns if col != "quantity"]]
    y = df["quantity"].values.flatten()

    # Define categorical feature transformation
    cat_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine all numeric/categorical transformations
    preprocessor = ColumnTransformer([
        ("cat", cat_transformer, cat_features)
    ])

    # Define RF regressor model (use all 23 features)
    xgb_model = xgb.XGBRegressor(random_state=17, objective='reg:squarederror')
    rf_pip = Pipeline([
        ("prep", preprocessor),
        ("xgb", xgb_model)
    ])

    # RF regressor hyperparameters (tailored to dataset properties)
    param_grid = {
        "xgb__n_estimators": [50, 100],  
        "xgb__max_depth": [10, 20, 35], 
        "xgb__learning_rate": [0.01, 0.1, 0.2], 
        "xgb__subsample": [0.8, 1.0]           # control how % of rows used for each weak tree
    }

    # Define custom scorers
    adj_r2_eval = make_scorer(adjusted_R2, greater_is_better=True)
    smape_eval = make_scorer(smape, greater_is_better=False)

    # Perform stratified k-fold cross-validation
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgbGridCv = GridSearchCV(
        rf_pip, param_grid,
        cv=strat_cv.split(X, stratifiedLabels),
        scoring={"Adjusted_R2": adj_r2_eval, "SMAPE": smape_eval},
        refit="SMAPE",  # Refit the best model based on SMAPE
        n_jobs=-1, verbose=0
    )
    xgbGridCv.fit(X, y)
    return xgbGridCv

# mode func for pd.DataFrame manipulation
def mode(x):
    modes = x.mode()        # call mode on pd.Series
    return modes.iloc[0] if not modes.empty else np.nan  

# Integrate newPrice xlsx and aggregated train model --> as test dataset
# predict the model with this new test dataset
def demand_forcast(myModel, input_df, newPrice_df):
    test_df = input_df.groupby(["sku_pk"], as_index = False, observed = False).agg({
        'sku_cat': mode,
        'prod_cd': mode,
        'proj_type': mode,
        'price': 'mean',
        # 'quantity': 'mean',
        'cust_cat': mode,
        'cust_tier': mode,
        'avg_p': 'median',
        'avg_q': 'median',
        'p_std': 'median',
        'q_std': 'median',
        '2023_avg_p': 'mean',
        '2023_avg_q': 'mean',
        '2024_avg_p': 'mean',
        '2024_avg_q': 'mean',
        '2025_avg_p': 'mean',
        '2025_avg_q': 'mean',
        'ped': 'median'
    })
    test_df.price = [np.nan] * test_df.shape[0]

    test_df = test_df.merge(newPrice_df, on = "sku_pk", how = "left")
    test_df.price = test_df["new price"]
    test_df.drop(inplace = True, columns = ["new price"])

    quantity_forcast = myModel.predict(test_df)
    return quantity_forcast, test_df

# Test section with pickle
if __name__ == "__main__":
    # Load data for 35 SKU, price_model pairs with frequent price adjustments and abundant transactions
    tree_input_df = read_model_from_supabase('tree_model_input')

    X = tree_input_df[[col for col in tree_input_df.columns if col != "quantity"]]
    rfr = None
    if os.path.exists("rfr.pkl"):
        with open("rfr.pkl", "rb") as fp:
            rfr = pickle.load(fp)
    pred_y = rfr.predict(X)
    act_y = tree_input_df["quantity"].values.flatten()
    print(pred_y)
    print(act_y)
    print(f"RFR has a Symmetic Mean Abs Percent Err of {smape(act_y, pred_y)}%")
