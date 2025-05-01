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
from preprocessENPricing import *               # Internal SKU Pricing Related Preprocessing Funcs 

# DB class for Azure SQL db functions
class AzureDBWriter():
    def __init__(self, df, tableCols):
        load_dotenv()           # load the .env vars
        self.DB_CONN = f"mssql+pyodbc://sqladmin:{urllib.parse.quote_plus(os.getenv('DB_PASS'))}@{os.getenv('DB_SERVER')}:1433/enerlitesDB?driver=ODBC+Driver+17+for+SQL+Server&encrypt=yes"
        self.myDf = df 
        self.myCols = tableCols
    
    # Transform dataframe w.r.t. azure db ddl 
    def __transform_df_wrt_azuredb(self, PK_COLS):
        # print(f"[DEBUG] __transform_df_wrt_azuredb GETS {self.myDf}\n")
        if self.myDf.empty:
            return 
        
        cleaned_df = self.myDf.copy() 
        dfCols = cleaned_df.columns.tolist()
        cleaned_df = cleaned_df.dropna(subset =PK_COLS)
        for pk in PK_COLS:
            cleaned_df.loc[:, pk] = cleaned_df[pk].apply(lambda x: x.strip() if isinstance(x, str) else x)

        # conditional cleaning based on pandas column
        # below for en_comp_sku_fct preprocessing
        if "comp_sku" in dfCols:
            cleaned_df.loc[:,"comp_sku"] = cleaned_df.comp_sku.apply(
                lambda x: re.search(r'(?<=:)\s*(.*)', x).group(1)
                if isinstance(x, str) and ':' in x else x
            )
        if "state_cd" in dfCols:
            cleaned_df.loc[:,"state_cd"] = cleaned_df.state_cd.apply(
                    lambda x: "FL" if x == "Florida"
                            else "OR" if x == "Oregon"
                            else "UT" if x == "Utah"
                            else "CR" if x == "Costa Rica"
                            else x.upper()
                )
        if "release_dt" in dfCols:
            cleaned_df.loc[:,"release_dt"] = pd.to_datetime(cleaned_df.release_dt, format = "mixed", errors="coerce").dt.date
        if "mnf" in dfCols:
            cleaned_df.loc[:,"mnf"] = cleaned_df.mnf.apply(lambda x: x.capitalize() if isinstance(x, str) else x)
        if "distr_typ" in dfCols:
            cleaned_df.loc[:,"distr_typ"] = cleaned_df.distr_typ.str.capitalize()
        if "distr" in dfCols:
            cleaned_df.loc[:,"distr"] = cleaned_df.distr\
                .apply(lambda mystr: ' '.join([word.capitalize() for word in mystr.split(' ')]) if isinstance(mystr, str) else mystr)
        if "en_sku" in dfCols and "comp_sku" in dfCols:
            cleaned_df.loc[:,["en_sku", "comp_sku"]] = cleaned_df[["en_sku", "comp_sku"]].astype("str")       # remember to cast to str instead of obj type

        # Below for sku_master_dim_hst preprocessing 
        if "Model No" in dfCols:
            cleaned_df.loc[:, "Model No"] = cleaned_df.loc[:, "Model No"].apply(lambda x: str(x).strip())

        # Deduplicate pandas in memory
        cleaned_df = cleaned_df.drop_duplicates(subset = PK_COLS, keep = 'last')
        self.myDf = cleaned_df

    # Define Python ETL Update and Insert Logic 
    # Read records in Azure db --> Generate Insertion df & Update df (respectively)
    def __trigger_upsert_df_wrt_azuredb (self, SQLQuery, PK_COLS):
        try:
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            trg_df = pd.read_sql(SQLQuery, engine)
            trgCols = trg_df.columns.tolist()
            src_df = self.myDf.copy()

            # Cast to Proper Python Type
            if "release_dt" in trgCols:
                trg_df['release_dt'] = pd.to_datetime(trg_df.release_dt, format = "%Y-%m-%d", errors="coerce")
            if "src_updt_dt" in trgCols:
                trg_df['src_updt_dt'] = pd.to_datetime(trg_df.src_updt_dt, format = "%Y-%m-%d", errors="coerce")
            
            leftMergeDf = src_df.merge(trg_df, on = PK_COLS, how = 'left', indicator = True)
            if "mnf_stk_price" in trgCols:
                insertionDf = leftMergeDf[leftMergeDf['_merge'] == 'left_only'].drop(columns = ['_merge', 'mnf_stk_price_y'], axis = 1)
                # Update logic based on duplicate pk
                updateDf = leftMergeDf[
                                        (leftMergeDf["_merge"] == "both") 
                                        & (leftMergeDf["mnf_stk_price_x"] != leftMergeDf["mnf_stk_price_y"])
                                        & ~(pd.isna(leftMergeDf.mnf_stk_price_x))
                                    ]\
                    .drop(columns = ["_merge"], axis = 1)
            elif "src_updt_dt" in trgCols:
                insertionDf = leftMergeDf[leftMergeDf['_merge'] == 'left_only'].drop(columns = ['_merge', 'src_updt_dt_y'], axis = 1)
                # Update logic based on duplicate pk
                updateDf = leftMergeDf[
                                        (leftMergeDf["_merge"] == "both") 
                                        & (leftMergeDf["src_updt_dt_x"] != leftMergeDf["src_updt_dt_y"])
                                        & ~(pd.isna(leftMergeDf.src_updt_dt_y))
                                    ]\
                    .drop(columns = ["_merge"], axis = 1)
            print(f"[DEBUG] __trigger_upsert_df_wrt_azuredb INSERTION GETS {insertionDf.shape}\n")
            print(f"[DEBUG] __trigger_upsert_df_wrt_azuredb UPDATE GETS {updateDf.shape}\n")

            if insertionDf.shape[0] == 0:
                self.myDf = pd.DataFrame()
            else:
                self.myDf = insertionDf
            
            if updateDf.shape[0] == 0:
                return pd.DataFrame()
            return updateDf
        except SQLAlchemyError as sqlerr: 
            print(f"[DEBUG] comp_agent_web_preprocess GETS Azure DB err: {sqlerr}\n")
        finally:
            engine.dispose()

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

        # Transform pandas df wrt database settings
        self.__transform_df_wrt_azuredb(PK_COLS)

        try:
            # Insertion logic based on non-duplicate PK
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            SQLQuery = "SELECT distinct release_dt,state_cd,en_sku,comp_sku,distr_typ,mnf_stk_price FROM landing.en_comp_sku_fct;"
            updateDf = self.__trigger_upsert_df_wrt_azuredb(SQLQuery, PK_COLS)
            if updateDf.empty:
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

    # Preprocess sku_master_dim_hst xlsx file --> perform upsert on pandas dataframe and azure db
    # PK ~ ('model_no','price_model','lower_bucket','upper_bucket')
    def sku_master_dim_hst_preprocess(self):
        PK_COLS = ['model_no','price_model','lower_bucket','upper_bucket']
        df = self.myDf.copy()
        # update unit price wrt pallet
        price_updt_df = update_sku_master_unitprice_wrt_pallet(df)
        # convert to tabular form
        pivoted_df = pivot_sku_master_price_conds(price_updt_df)
        self.myDf = pivoted_df
        
        # Transform pandas df w.r.t. azure ddl
        self.__transform_df_wrt_azuredb(PK_COLS)



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