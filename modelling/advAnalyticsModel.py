'''
Defined an Abstract Parent Class For both (custTier Model & demandForecast Model)

Provide Generic class method & fields 
    for better code usability
'''
import pandas as pd
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
            conn.execution_options(isolation_level="AUTOCOMMIT")\
                .execute(text(f"EXEC modeling.{ProcedureName};"))   # execute the T-SQL procedure before input table load
            self.inputDf = pd.read_sql(self._SQLQuery, conn)         # store modeling input (read from db)

        # set up class level logger in Azure Prod env
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    # Load Model Output Back 2 Azure Database
    def loadModelResBack2DB(self, outputDf, schema, tableName):
        with self._AZ_ENGINE.connect() as conn:
            outputDf.to_sql(tableName, con = conn, schema=schema, if_exists = "replace", index =False)
        self.logger.info(f"[AZURE] loadModelResBack2DB LOAD {outputDf.shape} 2 \'{tableName}\'!\n")

    # Force each subclass to overwrite this func
    @abstractmethod
    def preprocess_pip(self):
        pass
