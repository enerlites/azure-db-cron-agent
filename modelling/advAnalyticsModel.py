'''
Defined an Abstract Parent Class For both (custTier Model & demandForecast Model)

Provide Generic class method & fields 
    for better code usability
'''
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, r2_score
from abc import ABC, abstractmethod
import os 
from dotenv import load_dotenv
from sqlalchemy import create_engine
import urllib
import logging
from shared.azureDBWriter import *                          # import packages and modules

class AdvAnalyticsModel(ABC):
    # Class Constructor --> execute a T-SQL Procedure  --> Read the model input table from Azure db
    def __init__(self, ProcedureName, InputTableName):
        load_dotenv()           # load the .env vars
        self.__ODBC_18_CONN = urllib.parse.quote_plus(
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server=tcp:{os.getenv('DB_SERVER')},1433;"
            f"Database=enerlitesDB;"
            f"Uid=sqladmin;"
            f"Pwd={os.getenv('DB_PASS')};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )
        self.__DB_CONN = f"mssql+pyodbc:///?odbc_connect={self.__ODBC_18_CONN}"
        self._AZ_ENGINE = create_engine(self.__DB_CONN)
        self._SQLQuery = f'SELECT * FROM modeling.{InputTableName};'
        with self._AZ_ENGINE.connect() as conn:
            # conn.execution_options(isolation_level="AUTOCOMMIT")\
            #     .execute(text(f"EXEC modeling.{ProcedureName};"))   # execute the T-SQL procedure before input table load
            self.inputDf = pd.read_sql(self._SQLQuery, conn)         # store modeling input (read from db)

        # set up class level logger in Azure Prod env
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # modelling fields
        self._X, self._y = None, None                   # store model predictors and response
        self._processor = None                          # store current preprocessor
        self._trainer = None                            # store current trainer model (pickle)

    # Load Model Output Back 2 Azure Database
    def loadModelResBack2DB(self, outputDf, schema, tableName):
        with self._AZ_ENGINE.connect() as conn:
            outputDf.to_sql(tableName, con = conn, schema=schema, if_exists = "replace", index =False)
        self.logger.info(f"[AZURE] loadModelResBack2DB LOAD {outputDf.shape} 2 \'{tableName}\'!\n")

    # Force each model to overwrite this preprocess_pip func
    @abstractmethod
    def preprocess_pip(self, **kwargs):
        TestDf = kwargs.get('TestDf', None)
        pass

    # Custom SMAPE Metric (symmetic mean absolute percentage err)
    def _smape(self, y_true, y_pred):
        deno = (np.array(y_true) + np.array(y_pred)) / 2.0
        numer = np.abs(y_true - y_pred) 
        percentErr = np.where(deno == 0 ,0, numer / deno)
        return np.mean(percentErr) * 100
    
    # Custom Adjusted R^2 Metric (adjusted R square)
    def _adjusted_R2(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        n = self.inputDf.shape[0]
        return 1 - (1 - r2) * (n - 1) / (n - self.inputDf.shape[1] - 1)
