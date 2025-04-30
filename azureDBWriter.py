import os
import msal
import requests
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import urllib.parse
import schedule 
import time 
from datetime import datetime
from IPython.display import display
import pytz
import re

# DB class for Azure SQL db functions
class AzureDBWriter():
    def __init__(self, df, tableCols):
        load_dotenv()           # load the .env vars
        self.DB_CONN = f"mssql+pyodbc://sqladmin:{urllib.parse.quote_plus(os.getenv('DB_PASS'))}@{os.getenv('DB_SERVER')}:1433/enerlitesDB?driver=ODBC+Driver+17+for+SQL+Server&encrypt=yes"
        self.myDf = df 
        self.myCols = tableCols
        
    # Preprocess oceanAir Inventory in Python Memory
    # Dedup based on --> inv_eval_dt
    def oceanAir_Inv_preprocess(self):
        # parse from object to date
        inv_eval_dt = pd.to_datetime(self.myDf.columns[3], format = 'mixed').date()
        # pandas manipulation for time reconciliation 
        dt_df = pd.DataFrame({"inv_eval_dt": [inv_eval_dt] * self.myDf.shape[0]})
        data = self.myDf.iloc[2:, :len(self.myCols) - 2]
        self.myDf = pd.concat([dt_df, data], axis = 1)
        numeric_cols = self.myDf.columns[6:-1]
        self.myDf[numeric_cols] = self.myDf[numeric_cols].astype('Int64')

        # Access table from azure db
        engine = create_engine(self.DB_CONN)
        query = "SELECT distinct inv_eval_dt FROM landing.googleDrive_ocean_air_inv_fct;"
        trg_df = pd.read_sql(query, engine)
        engine.dispose()

        # dedup logic below 
        if trg_df.shape[0] != 0:     # db table is not empty  
            trg_dt = trg_df.inv_eval_dt.unique().tolist()
            src_dt = self.myDf.inv_eval_dt.unique().tolist()
            if bool(set(trg_dt) & set(src_dt)):         # duplicate found
                self.myDf = pd.DataFrame()
                return 
    
    def __items_sold_hst_clean(self):
        df = self.myDf.copy()
        # drop bad records from netsuite
        # (sku, bill_num, sys_dt) should never be null
        df = df.dropna(subset=['sku','bill_num','sys_dt'])

        # Transform records pulled from ERP
        df.sys_dt = df.sys_dt.apply(lambda x: x.replace(" am", "").replace(" pm", "") if isinstance(x, str) else x)
        df.onboard_dt = df.onboard_dt.apply(lambda x: x.replace(" am", "").replace(" pm", "") if isinstance(x,str) else x)
        df.quote_num = df.quote_num.apply(lambda x: x.replace("Sales Order #", "") if isinstance(x, str) else x)
        df["sys_dt"] = pd.to_datetime(df['sys_dt'], format='%m/%d/%Y %H:%M')
        df["onboard_dt"] = pd.to_datetime(df['onboard_dt'], format='%m/%d/%Y %H:%M')

        # Pandas treat col with mixed NaN and Object type as of Object type
        # must check the value is not float NaN 
        df["state_cd"] = df.state_cd.apply(lambda x: 
                                           "NY" if isinstance(x, str) and x.lower().startswith("n.y.") 
                                           else "FL" if isinstance(x, str) and x.lower().startswith("fl")
                                           else "CA" if isinstance(x, str) and x.lower().startswith("ca")
                                           else x
                                           )
        df["discount"] = df.discount.fillna(0.0).abs()
        df["sku"] = df.sku.apply(lambda s: s[0].lower() + s[1:] if "Combo" in s else s)
        # Group sku_cat into reliable Category in home electronic manufacturer
        df["sku_cat"] = df.sku_cat.apply(lambda s: 
                                            'Electrical Device' if s in ['Wiring Devices', 'GFCI', 'NEMA', 'Combination Devices', 'USB']
                                            else 'Smart Home' if s in ['Automation', 'Room Control', 'EV', 'Data-Com Devices']
                                            else 'Switch' if s in ['Switches', 'Dimmers', 'Fan Speed Control']
                                            else 'Sensors' if s in ['Sensors', 'In Wall Sensors', 'Humidity Sensors']
                                            else 'Energy Mangement' if s in ['Plug Load', 'Timers']
                                            else 'Specialty Accessories' if s in ['Locking Devices', 'Weatherproof Covers', 'Floor Box', 'Wall Plates', 'RV']
                                            else 'Other' if isinstance(s, float)
                                            else 'Other'
                                        )

        # Aggregate by (bill_num, sys_dt, sku) --> for dedup logic
        groupCols = ['bill_num', 'sys_dt', 'sku']
        dedup_df = df.groupby(by=groupCols, as_index = False).agg(
            quantity=("quantity", "sum"),
            amt=("amt", "sum")
        )

        # Given that netsuite contain duplicates given (bill_num, sys_dt, sku)
        new_df = pd.merge(dedup_df, df, on = groupCols, how="inner")\
                        .drop(columns=["quantity_y", "amt_y"], axis = 1)\
                        .rename(columns= {"quantity_x": "quantity", "amt_x": "amt"}).drop_duplicates(subset=groupCols)
        # display(new_df.head(5))
        new_df = new_df[["customer","bill_num", "quote_num", "sys_dt"
                        , "sku", "quantity", "amt", "price_model", "prod_cd"
                        , "sku_cat", "state_cd", "proj_type", "onboard_dt", "cust_cat", "discount"]]
        
        # Eliminate bad transaction records
        new_df = new_df[new_df.quantity != 0]
        print(f"[DEBUG] __items_sold_hst_clean PROCESSED df\n{df.shape}\n")

        self.myDf = new_df

    # Preprocess NetSuite csv files in Python Memory 
    # Dedup based on (bill_num, sys_dt, sku)
    def netsuite_items_sold_hst_preprocess(self):
        # Do the clean and tranformation first
        self.__items_sold_hst_clean()
        engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})

        # Query PK out of the Azure db
        query = "SELECT distinct bill_num, sys_dt, sku FROM landing.erp_items_sold_history;"
        trg_df = pd.read_sql(query, engine)
        src_df = self.myDf.copy()
        src_df["pk"] = src_df.bill_num.astype(str) + '@' + src_df.sys_dt.astype(str) + '@' + src_df.sku.astype(str)
        engine.dispose()

        if trg_df.shape[0] != 0:     # db table is not empty  
            trg_df["pk"] = trg_df.bill_num.astype(str) + '@' + trg_df.sys_dt.astype(str) + '@' + trg_df.sku.astype(str)
            # Remove duplicate from src_df that exists in trg_df
            src_df = src_df[~src_df.pk.isin(trg_df.pk)]
            src_df = src_df.drop(columns=['pk'], axis=1)
            self.myDf = src_df
            print(f"[DEBUG] netsuite_items_sold_hst_preprocess (Memory Dedup) CLEANs df of shape {self.myDf.shape}\n")
            return 
        print(f"[DEBUG] netsuite_items_sold_hst_preprocess (Azure empty) CLEANs df of shape {self.myDf.shape}\n")

    def email_comp_price_alerts (self, insertionDf, updateDf):
        mydf = pd.concat([insertionDf, updateDf], axis = 0)
        pass


    # Preprocess Competitor Agent Web xlsx file --> perform upsert on pandas dataframe and azure db
    # PK ~ ('release_dt','state_cd','en_sku','comp_sku','distr_typ')
    def comp_agent_web_upsert_preprocess(self):
        PK_COLS = ['release_dt','state_cd','en_sku','comp_sku','distr_typ']

        if self.myDf.empty:       # empty in memory dataframe 
            return 
        # Clean the primary key columns (not nullable + no leading trailing whitespace)
        self.myDf = self.myDf.dropna(subset=PK_COLS)
        for PK in PK_COLS:
            self.myDf.loc[:,PK] = self.myDf[PK].apply(lambda x: x.strip() if isinstance(x,str) else x)

        # pandas normalization and standardization 
        self.myDf.loc[:,"comp_sku"] = self.myDf.comp_sku.apply(
                lambda x: re.search(r'(?<=:)\s*(.*)', x).group(1)
                if isinstance(x, str) and ':' in x else x
            )
        self.myDf.loc[:,"state_cd"] = self.myDf.state_cd.apply(
                lambda x: "FL" if x == "Florida"
                          else "OR" if x == "Oregon"
                          else "UT" if x == "Utah"
                          else "CR" if x == "Costa Rica"
                          else x.upper()
            )
        self.myDf.loc[:,"release_dt"] = pd.to_datetime(self.myDf.release_dt, format = "mixed", errors="coerce").dt.date
        self.myDf.loc[:,"mnf"] = self.myDf.mnf.apply(lambda x: x.capitalize() if isinstance(x, str) else x)
        self.myDf.loc[:,"distr_typ"] = self.myDf.distr_typ.str.capitalize()
        self.myDf.loc[:,"distr"] = self.myDf.distr\
            .apply(lambda mystr: ' '.join([word.capitalize() for word in mystr.split(' ')]) if isinstance(mystr, str) else mystr)
        self.myDf.loc[:,["en_sku", "comp_sku"]] = self.myDf[["en_sku", "comp_sku"]].astype("str")       # remember to cast to str instead of obj type

        # dedup pandas dataframe
        self.myDf = self.myDf.drop_duplicates(subset = PK_COLS, keep = 'last')

        try:
            # Insertion logic based on non-duplicate PK
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            query = "SELECT distinct release_dt,state_cd,en_sku,comp_sku,distr_typ,mnf_stk_price FROM landing.en_comp_sku_fct;"
            trg_df = pd.read_sql(query, engine)
            trg_df['release_dt'] = pd.to_datetime(trg_df.release_dt, format = "%Y-%m-%d", errors="coerce")

            src_df = self.myDf.copy()
            leftMergeDf = src_df.merge(trg_df, on = PK_COLS, how = 'left', indicator = True)
            insertionDf = leftMergeDf[leftMergeDf['_merge'] == 'left_only'].drop(columns = ['_merge', 'mnf_stk_price_y'], axis = 1)
            print(f"[DEBUG] comp_agent_web_upsert_preprocess INSERTION GETS {insertionDf.shape}\n")

            if insertionDf.shape[0] == 0:
                self.myDf = pd.DataFrame()
            else:
                self.myDf = insertionDf

            # Update logic based on duplicate pk
            updateDf = leftMergeDf[
                                    (leftMergeDf["_merge"] == "both") 
                                    & (leftMergeDf["mnf_stk_price_x"] != leftMergeDf["mnf_stk_price_y"])
                                    & ~(pd.isna(leftMergeDf.mnf_stk_price_x))
                                  ]\
                .drop(columns = ["_merge"], axis = 1)
            
            print(f"[DEBUG] comp_agent_web_upsert_preprocess UPDATE gets df:\n{updateDf.shape}\n")
            
            if updateDf.shape[0] == 0:
                return 
            updateDf["sys_dt"] = pd.to_datetime('now')
            with engine.begin() as conn:  # Ensures commit/rollback
                for _, row in updateDf.iterrows():
                    # Use parameterized SQL to avoid SQL injection and type issues
                    updt_stmt = text("""
                        UPDATE landing.en_comp_sku_fct 
                        SET mnf_stk_price = :mnf_stk_price,
                            mnf = :mnf,
                            quantity = :quantity,
                            mnf_desc = :mnf_desc,
                            rep_name = :rep_name,
                            sys_dt = :sys_dt
                        WHERE release_dt = :release_dt
                            AND state_cd = :state_cd
                            AND en_sku = :en_sku
                            AND comp_sku = :comp_sku
                            AND distr_typ = :distr_typ;
                    """)
                    conn.execute(updt_stmt, {
                        'mnf_stk_price': None if pd.isna(row.get('mnf_stk_price')) else row.get('mnf_stk_price'),
                        'mnf': None if pd.isna(row.get('mnf')) else row.get('mnf'),
                        'quantity': None if pd.isna(row.get('quantity')) else row.get('quantity'),
                        'mnf_desc': None if pd.isna(row.get('mnf_desc')) else row.get('mnf_desc'),
                        'rep_name': None if pd.isna(row.get('rep_name')) else row.get('rep_name'),
                        'sys_dt': row['sys_dt'],
                        'release_dt': row['release_dt'],
                        'state_cd': row['state_cd'],
                        'en_sku': row['en_sku'],
                        'comp_sku': row['comp_sku'],
                        'distr_typ': row['distr_typ']
                    })
        except SQLAlchemyError as sqlerr: 
            print(f"[DEBUG] comp_agent_web_preprocess GETS Azure DB err: {sqlerr}\n")
        finally:
            engine.dispose()
        
    # commit flatFile 2 azure db 
    def flatFile2db (self, schema, table):
        engine = create_engine(self.DB_CONN)
        try:
            # empty df abort load job
            if self.myDf.empty:
                return 
    
            df = self.myDf.copy()
            tableCols = self.myCols
            # append getdate() datetim2 
            df['cur_ts'] = pd.to_datetime('now')

            '''Below section for data cleaning prior to db load'''
            if "promo dt" in df.columns:
                df["promo dt"] = pd.to_datetime(df["promo dt"],format="mixed",errors='coerce') 
            # handle manual input err
            if "Promotion Reason" in df.columns:
                df["Promotion Reason"] = df["Promotion Reason"].apply(lambda x: 'Discontinued' if x == 'Disontinued' else x)
            if "promo category" in df.columns:
                df["promo category"] = df["promo category"].apply(lambda x: 'Discontinued' if x == 'Discontinued item' else x)   
            # drop Nan in promo base xlsx 
            if 'sku' in df.columns:
                df = df.dropna(subset = ['sku'])

            # persist df name with that of defined in ssms   
            df.columns = tableCols
            self.myDf = df
            print(f"[DEBUG] flatFile2db GETs df of shape {self.myDf.shape}\n")
            
            batch_size = 500
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                batch.to_sql(
                    name=table,
                    con=engine,
                    schema=schema,
                    if_exists="append",
                    index=False,
                    method= None,
                    chunksize=batch_size
                )
            print(f"Successfully wrote {len(df)} rows to {table}")
            
        except Exception as e:
            print(f"Error writing to database: {str(e)}")
        finally:
            engine.dispose()