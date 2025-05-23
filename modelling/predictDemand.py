import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle                               # Save the trained model
from modelling.advAnalyticsModel import AdvAnalyticsModel
from shared.azureDBWriter import *

class DemandForecast (AdvAnalyticsModel):
    def __init__(self):
        super().__init__("sp_demandForecastInput", "demandForecastInput")

    # Define preprocessor pipeline
    # Concatenate (sku, '@', Region) as predPK (before running this pipline)
    # Generate PED / coefficient of variance metrics
    # Perform inputDf split --> acquire predictors and response variable
    def preprocess_pip(self, TestDf = None, **kwargs):
        df = None
        # Preprocess the Training dataset
        if TestDf is None:
            df = self.inputDf.copy()
        
        # Preprocess the Test dataset
        else:
            df = TestDf.copy()

        # Transfrom time-related column --> numeric year + categorical quarter + categorical weekend falg
        df.bill_dt = pd.to_datetime(df.bill_dt, errors = 'coerce')
        df['year'] = df.bill_dt.dt.year
        df['quarter'] = df.bill_dt.dt.quarter.astype('category')           # explictly cast quarter as categorical field (don't imply order scale)
        df['weekend_flag'] = df.bill_dt.dt.dayofweek >= 5                  # convert day of week to 0/1 binaries 
        df.drop(columns=['bill_dt'], axis= 1, inplace= True)

        # Transform raw SQL input --> Generate price elasticity (PED) + coefficient of variance (CV) metrics for price-demand forecast
        df['coefvar_p'] = df['stdP'] / df['mueP']
        df['coefvar_q'] = df['stdQ'] / df['mueQ']
        df.drop(columns=['stdP','stdQ','mueP', 'mueQ'], axis= 1, inplace= True)

        # identify cat / num cols
        catCols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numCols = list(set(df.columns.tolist()) - set(catCols))
        numCols.remove('quantity')

        # modelling pipeline
        num_trans = Pipeline(
            steps= [
                ('scaler', StandardScaler())
            ]
        )
        cat_trans = Pipeline(
            steps= [
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_trans, numCols),
                ('cat', cat_trans, catCols)
            ]
        )

        # Only mount the preprocessor pipeline (No actual transformation)
        self._processor = preprocessor
        self._X, self._y = df.loc[:, df.columns != 'quantity'], df['quantity']
        self.logger.info(f"[AZURE] preprocess_pip (demand Forecast) Mounted !\n")
    
    # XGBoost Regressor to train the dateset
    def train_XGBoostRegressor(self):
        # define training pipeline
        xgb_pipeline = Pipeline([
                ('preprocessor', self._processor),
                ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
            ])
        
        # hyper param tunning
        param_grid = {
            'xgb__n_estimators': [300,500,800],
            'xgb__max_depth': [6,8,10],
            'xgb__learning_rate': [0.01, 0.005],
            'xgb__reg_lambda': [0, 1,2]
        }

        # define custom scorers
        adj_r2_eval = make_scorer(self._adjusted_R2, greater_is_better=True)
        smape_eval = make_scorer(self._smape, greater_is_better=False)

        # Perform stratified k-fold cross-validation
        kfcv = KFold(n_splits=5, shuffle=True, random_state=42)
        xgbGridCv = GridSearchCV(
            xgb_pipeline, 
            param_grid,
            cv = kfcv,
            scoring={"Adjusted_R2": adj_r2_eval, "SMAPE": smape_eval},
            refit="SMAPE",                                                      # use custom SMAPE to evaluate the model
            n_jobs=1, verbose=0
        ).fit(self._X, self._y)

        # store the optimal model
        self._trainer = xgbGridCv.best_estimator_

        self.logger.info(f"[AZURE] train_XGBoostRegressor's best SMAPE: {xgbGridCv.best_score_:.4f}")
        self.logger.info(f"[AZURE] train_XGBoostRegressor's best SMAPE: {xgbGridCv.best_params_}")

    # Generate Demand Forecast Input: (sku, customer, region, price)
    # Prompt users to enter an adjusted price with historical price stats based on (sku, customer, region)
    # Output format: sku, customer, cust tier, region, price (min), price (median), price(max), price (std)
    def generate_forecast_prompt (self):
        df = self.inputDf.copy()
        # print(df.columns.tolist())
        inputHstRaw = df[['predPk', 'price']]
        inputHst = inputHstRaw.groupby(by = ['predPk'], as_index=False)\
                    .agg({
                        'price': ['min', 'median', 'max', 'std', 'count']
                    })
        # rename the columns
        newCols = ['predPk', 'price (min)', 'price (median)', 'price (max)', 'price (std)', 'num orders (5 years)']
        inputHst.columns = newCols
        inputHst.loc[:,'sku'] = inputHst['predPk'].apply(lambda x: x.split("@")[0])
        inputHst.loc[:,'state'] = inputHst['predPk'].apply(lambda x: x.split("@")[1])
        inputHst = inputHst[['sku','state'] + newCols]

        # append price adjustment col
        inputHst.loc[:, "price adjusted"] = [np.nan] * inputHst.shape[0]

    # Send Model Input Prompt / Demand Forecast as attachments to email
    def monthly_demandForecast(self):

        # prepare for email sent
        # myDBWriter = AzureDBWriter(None, None)
        # attDfs, attNames = [], []
        # inputHst = self.generate_forecast_prompt()
        # attDfs.append(inputHst)
        # attNames.append("Forecast Prompt.xlsx")

        # # send out attachments (ask for adjusted price input)
        # myDBWriter.auto_send_email(outputDfs=attDfs, attachmentNames=attNames
        #                            , emailSubject="Monthly Demand Forecast [NO-REPLY]", recipients=['andrew.chen@enerlites.com'])

        # Preprocess the inputDf
        self.preprocess_pip()
        self.train_XGBoostRegressor()

