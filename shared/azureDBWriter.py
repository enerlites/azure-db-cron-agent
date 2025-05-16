import os
import pandas as pd
import numpy as np
import io
import base64
import requests
from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import urllib.parse
from datetime import datetime
import pytz
import re
from shared.preprocessENPricing import *               # Internal SKU Pricing Related Preprocessing Funcs 
import logging                                         # log errs in production env
from shared.OneDriveFlatFileReader import *            # import class from shared module

# DB class for Azure SQL db functions
class AzureDBWriter():
    def __init__(self, df, tableCols):
        load_dotenv()           # load the .env vars

        ODBC_18_CONN = urllib.parse.quote_plus(
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server=tcp:{os.getenv('DB_SERVER')},1433;"
            f"Database=enerlitesDB;"
            f"Uid=sqladmin;"
            f"Pwd={os.getenv('DB_PASS')};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )
        self.DB_CONN = f"mssql+pyodbc:///?odbc_connect={ODBC_18_CONN}"
        # self.DB_CONN = f"mssql+pyodbc://sqladmin:{urllib.parse.quote_plus(os.getenv('DB_PASS'))}@{os.getenv('DB_SERVER')}:1433/enerlitesDB?driver=ODBC+Driver+17+for+SQL+Server&encrypt=yes"
        self.myDf = df 
        self.myCols = tableCols

        # set up class level logger in Azure Prod env
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
    
    # Transform dataframe w.r.t. azure db ddl 
    def __transform_df_wrt_azuredb(self, PK_COLS):
        if self.myDf.empty:
            return 
        
        cleaned_df = self.myDf.copy() 
        cleaned_df.columns = self.myCols[:-1]                   # Intentionally omit the last auto-generated timestamp
        dfCols = cleaned_df.columns.tolist()
        cleaned_df = cleaned_df.dropna(subset =PK_COLS)
        for pk in PK_COLS:
            cleaned_df.loc[:, pk] = cleaned_df[pk].apply(lambda x: re.sub(r'[\u00A0\u200B\uFEFF\t\n\r]+', '', x).strip() if isinstance(x, str) else x)

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
                            else x.upper() if isinstance(x,str)
                            else x
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
        if "model_no" in dfCols:
            cleaned_df.loc[:, "model_no"] = cleaned_df.loc[:, "model_no"].apply(lambda x: str(x).strip()).astype(str)
        if "src_updt_dt" in dfCols:
            cleaned_df.loc[:,"src_updt_dt"] = pd.to_datetime(cleaned_df.src_updt_dt, format = "mixed", errors="coerce").dt.date
        
        # Below for oneDrive_promo_sku_base preprocessing
        if "category" in dfCols:
            cleaned_df.loc[:,"category"] = cleaned_df.loc[:,"category"].apply(lambda x: 'Discontinued' if x == 'Discontinued item' else x)
        if "promo_reason" in dfCols:
            cleaned_df.loc[:,"promo_reason"] = cleaned_df.loc[:,"promo_reason"].apply(lambda x: 'Discontinued' if x == 'Disontinued' else x)

        # Below for erp_quote_hst_fct preprocessing 
        if "quote_dt" in dfCols:
            cleaned_df.loc[:,"quote_dt"] = pd.to_datetime(cleaned_df.quote_dt, format = "mixed", errors="coerce").dt.date
        if "quantity" in dfCols:
            cleaned_df.quantity = cleaned_df.quantity.apply(lambda val: round(val,0) if not pd.isna(val) else val)
            cleaned_df.quantity = cleaned_df['quantity'].astype('Int64')           # Int64 accept Null values in Python 
        if "customer" in dfCols:
            cleaned_df.customer = cleaned_df.customer.apply(lambda x: " ".join(x.split(" ")[1:]) if isinstance(x, str) else x)
            cleaned_df.customer = cleaned_df.customer.apply(lambda x: x.split(" : ",1)[1] 
                                                            if isinstance(x, str) and " : " in x else x)
        # Deduplicate pandas in memory
        cleaned_df = cleaned_df.drop_duplicates(subset = PK_COLS, keep = 'last')
        self.myDf = cleaned_df

    # Define Python ETL Update and Insert Logic 
    # Read records in Azure db --> Generate Insertion df & Update df (respectively)
    def __trigger_upsert_df_wrt_azuredb (self, SQLQuery, PK_COLS):
        try:
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            trg_df = pd.read_sql(SQLQuery, engine)
            srcCols = self.myCols
            src_df = self.myDf.copy()
            insertionDf, updateDf = pd.DataFrame(), pd.DataFrame()

            # Cast to Proper Python Type
            if "release_dt" in srcCols:
                trg_df['release_dt'] = pd.to_datetime(trg_df.release_dt, format = "%Y-%m-%d", errors="coerce")
            if "src_updt_dt" in srcCols:
                trg_df['src_updt_dt'] = pd.to_datetime(trg_df.src_updt_dt, format = "%Y-%m-%d", errors="coerce")

            leftMergeDf = src_df.merge(trg_df, on = PK_COLS, how = 'left', indicator = True)
            # update logic for en_comp_sku_fct table
            if "mnf_stk_price" in srcCols:
                insertionDf = leftMergeDf[leftMergeDf['_merge'] == 'left_only']\
                    .drop(columns = ['_merge', 'mnf_stk_price_y'], axis = 1)\
                        .rename(columns= {"mnf_stk_price_x": "mnf_stk_price"})
                # Update logic based on duplicate pk
                updateDf = leftMergeDf[
                                        (leftMergeDf["_merge"] == "both") 
                                        & (leftMergeDf["mnf_stk_price_x"] != leftMergeDf["mnf_stk_price_y"])
                                        & ~(pd.isna(leftMergeDf.mnf_stk_price_x))
                                    ]\
                    .drop(columns = ["_merge"], axis = 1)
            # update logic for sku_master_dim_hst table
            elif "src_updt_dt" in srcCols:
                insertionDf = leftMergeDf[leftMergeDf['_merge'] == 'left_only'].drop(columns = ['_merge', 'src_updt_dt_y'], axis = 1)
                # Update logic based on duplicate pk
                updateDf = leftMergeDf[
                                        (leftMergeDf["_merge"] == "both") 
                                        & (leftMergeDf["src_updt_dt_x"] != leftMergeDf["src_updt_dt_y"])
                                        & ~(pd.isna(leftMergeDf.src_updt_dt_y))
                                    ]\
                    .drop(columns = ["_merge"], axis = 1)
            # update logic for quote_hst_fct table
            elif "row_stat" in srcCols:
                insertionDf = leftMergeDf[leftMergeDf['_merge'] == 'left_only'].drop(columns = ['_merge', 'row_stat_y'], axis = 1)
                # Update logic based on duplicate pk
                updateDf = leftMergeDf[
                                        (leftMergeDf["_merge"] == "both") 
                                        & (leftMergeDf["row_stat_x"] != leftMergeDf["row_stat_y"])
                                        & ~(pd.isna(leftMergeDf.row_stat_y))
                                    ]\
                    .drop(columns = ["_merge"], axis = 1)
            self.logger.info(f"[DEBUG] __trigger_upsert_df_wrt_azuredb INSERTION GETS {insertionDf.shape}\n")
            self.logger.info(f"[DEBUG] __trigger_upsert_df_wrt_azuredb UPDATE GETS {updateDf.shape}\n")

            if insertionDf.shape[0] == 0:
                self.myDf = pd.DataFrame()
            else:
                self.myDf = insertionDf

            # Return both insertionDf and updateDf
            return insertionDf, updateDf
        except SQLAlchemyError as sqlerr: 
            self.logger.info(f"[DEBUG] __trigger_upsert_df_wrt_azuredb GETS Azure DB err: {sqlerr}\n")
        finally:
            engine.dispose()
    
    # Preprocess oneDrive_hst_promo_sku in Python Memory
    def hst_promo_sku_preprocess(self):
        PK_COLS = ['month_st', 'month_ed', 'sku']
        self.__transform_df_wrt_azuredb(PK_COLS)

    # Preprocess oceanAir Inventory in Python Memory
    # Dedup based on --> inv_eval_dt
    def oceanAir_Inv_preprocess(self):
        # parse from object to date
        inv_eval_dt = pd.to_datetime(self.myDf.columns[3], format = 'mixed').date()
        # pandas manipulation for time reconciliation 
        dt_df = pd.DataFrame({"inv_eval_dt": [inv_eval_dt] * self.myDf.shape[0]})
        data = self.myDf.iloc[2:, :len(self.myCols) - 2]
        self.myDf = pd.concat([dt_df, data], axis = 1)
        self.myDf = self.myDf.dropna(subset=self.myDf.columns.tolist()[4:8], how = 'all')
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
        itemSoldHstPK = ['bill_num','sys_dt', 'sku']

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
        dedup_df = df.groupby(by=itemSoldHstPK, as_index = False).agg(
            quantity=("quantity", "sum"),
            amt=("amt", "sum")
        )

        # Given that netsuite contain duplicates given (bill_num, sys_dt, sku)
        new_df = pd.merge(dedup_df, df, on = itemSoldHstPK, how="inner")\
                        .drop(columns=["quantity_y", "amt_y"], axis = 1)\
                        .rename(columns= {"quantity_x": "quantity", "amt_x": "amt"}).drop_duplicates(subset=itemSoldHstPK)
        # display(new_df.head(5))
        new_df = new_df[["customer","bill_num", "quote_num", "sys_dt"
                        , "sku", "quantity", "amt", "price_model", "prod_cd"
                        , "sku_cat", "state_cd", "proj_type", "onboard_dt", "cust_cat", "discount"]]
        
        # Eliminate bad transaction records
        new_df = new_df[new_df.quantity != 0]
        self.logger.info(f"[DEBUG] __items_sold_hst_clean PROCESSED df\n{df.shape}\n")

        self.myDf = new_df
        self.__transform_df_wrt_azuredb(itemSoldHstPK)

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
            self.logger.info(f"[DEBUG] netsuite_items_sold_hst_preprocess (Memory Dedup) CLEANs df of shape {self.myDf.shape}\n")
            return 
        self.logger.info(f"[DEBUG] netsuite_items_sold_hst_preprocess (Azure empty) CLEANs df of shape {self.myDf.shape}\n")

    # Preprocess quotesByItemList csv file --> perform upsert on pandas dataframe and azure db
    # PK ~ ('quote_num', 'model_no')
    def netsuite_quoteByItem_preprocess(self):
        PK_COLS = ['quote_num', 'model_no', 'price_model']

        # Transform pandas df wrt database settings
        self.__transform_df_wrt_azuredb(PK_COLS)
        if self.myDf.empty:
            return 

        try:
            # Insertion logic based on non-duplicate PK
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            SQLQuery = "SELECT distinct quote_num, model_no, price_model, row_stat FROM landing.erp_quote_hst_fct;"
            _, updateDf = self.__trigger_upsert_df_wrt_azuredb(SQLQuery, PK_COLS)
            if updateDf.empty:
                return 
            update_ts = datetime.now(pytz.timezone('America/Los_Angeles'))
            with engine.begin() as conn:  # Ensures commit/rollback
                for _, row in updateDf.iterrows():
                    # Use parameterized SQL to avoid SQL injection and type issues
                    updt_stmt = text("""
                        UPDATE landing.erp_quote_hst_fct 
                        SET 
                            quantity = :quantity,
                            distr_typ = :distr_typ,
                            trans_stat = :trans_stat,
                            row_stat = :row_stat,
                            data_dt = :data_dt
                        WHERE
                            model_no = :model_no
                            AND price_model = :price_model
                            AND quote_num = :quote_num;
                    """)
                    conn.execute(updt_stmt, {
                        'quantity': None if pd.isna(row.get('quantity')) else row.get('quantity'),
                        'distr_typ': None if pd.isna(row.get('distr_typ')) else row.get('distr_typ'),
                        'trans_stat': None if pd.isna(row.get('trans_stat')) else row.get('trans_stat'),
                        'row_stat': None if pd.isna(row.get('row_stat')) else row.get('row_stat'),
                        'data_dt': update_ts,
                        'model_no': row['model_no'],
                        'price_model': row['price_model'],
                        'quote_num': row['quote_num']
                    })

        except SQLAlchemyError as sqlerr: 
            self.logger.error(f"[DEBUG] comp_agent_web_upsert_preprocess GETS Azure DB err: {sqlerr}\n")
        finally:
            engine.dispose()
    
    # Prepare Pricing Alerts for Competitor Agent Web Entries
    # new insertion / update records --> Trigger this pricing alerts
    def __get_pricing_alerts (self, insertionDf, updateDf, threshold):
        stgDf = pd.concat([insertionDf, updateDf.iloc[:, :-1]], axis = 0)
        compPricing = stgDf[['release_dt','state_cd','mnf_stk_price','en_sku','comp_sku','mnf','distr_typ']]

        # below is T-SQL query for most up-to-date internal price 
        enSQLQuery = """
        SELECT 
            * 
        FROM (
            SELECT 
                *, 
                ROW_NUMBER() OVER(PARTITION BY model_no ORDER BY src_updt_dt DESC) AS idx
            FROM (
                SELECT DISTINCT 
                    COALESCE(src_updt_dt, CAST('2000-01-01' AS DATE)) AS src_updt_dt,
                    model_no, 
                    cat, 
                    prod_cd,
                    unit_cost, 
                    unit_price, 
                    price_model
                FROM landing.sku_master_dim_hst
            ) A
        ) B 
        WHERE idx = 1;
        """
        if compPricing.empty:           # No new competitor pricing records
            return pd.DataFrame()
        try:
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            enPricing = pd.read_sql(enSQLQuery, engine)
            # perform inner join 
            mergeDf = pd.merge(enPricing, compPricing, left_on = 'model_no', right_on= 'en_sku', how ='inner')
            
            # check if there is a matching records with EN internal item pricing catalog
            if not mergeDf.empty:
                mergeDf = mergeDf.drop(columns=['model_no'], axis=1, errors='ignore')
                mergeDf.loc[:,"priceVar"] = (mergeDf.unit_price - mergeDf.mnf_stk_price).abs() / mergeDf.unit_price.replace(0, np.nan)
                mergeDf = mergeDf.dropna(subset = ['priceVar'])

                # filter out records where price difference is above certain threshold
                alertDf = mergeDf.loc[mergeDf.priceVar >= threshold, :]
                alertDf.priceVar = alertDf.priceVar.apply(lambda x: str(round(x*100,1)) + "%")

                # subset and prepare the alert records with 
                alertDf = alertDf[['price_model', 'en_sku', 'comp_sku', 'mnf', 'unit_price', 'mnf_stk_price', 'priceVar','distr_typ','release_dt', 'state_cd']]
                alertDf.columns = ['Price Model', "EN Model No", "Competitor Model No","Manufacturer","EN Price", "Competitor Price", "Price Variance Ratio", "Channel", "Competitor Release Date", "State"]
                print(f"[DEBUG] __get_pricing_alerts with records of shape {alertDf.shape}\n")
                
                return alertDf
            else: 
                return pd.DataFrame()


        except SQLAlchemyError as sqlerr: 
            self.logger.error(f"[DEBUG] get_pricing_alert_records GETS Azure DB err: {sqlerr}\n")
            print(f"[DEBUG] get_pricing_alert_records GETS Azure DB err: {sqlerr}\n")
        finally:
            engine.dispose()
    
    # func that send email with multi attachment to list of recipients
    # outputDfs       --> list of attachments to be converted to base64
    # attachmentNames --> name of attachments
    def auto_send_email(self, outputDfs, attachmentNames, emailSubject, sender = 'andrew.chen@enerlites.com', recipients = ['andrew.chen@enerlites.com','angel@Enerlites.com']):
        if all(df.empty for df in outputDfs): 
            return 

        ATTCHMENTS = []
        for idx, df in enumerate(outputDfs):
            if df.empty:
                continue 

            # Create Excel as base64
            excel_io = io.BytesIO()
            df.to_excel(excel_io, index=False, engine = 'openpyxl')
            excel_io.seek(0)
            excel_base64 = base64.b64encode(excel_io.read()).decode('utf-8')
            
            # append to attchment
            ATTCHMENTS.append({
                        "@odata.type": "#microsoft.graph.fileAttachment",
                        "name": f"{attachmentNames[idx]}",
                        "contentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "contentBytes": excel_base64
                    })

        myApp = OneDriveFlatFileReader(sender)
        ACCESSTOKEN = myApp._OneDriveFlatFileReader__get_access_token()

        LIST_OF_RECEPTS = [{"emailAddress": {"address": r}} for r in recipients]

        # build email payload 
        PAYLOAD = {
            "message": {
                "subject": emailSubject,
                "body": {
                    "contentType": "Text",
                    "content": f"Hi Teams,\n\nPlease feel free to view the attached excel sheet.\n\nWarm Regards,\n{sender.split('@enerlites')[0]}"
                },
                "toRecipients": LIST_OF_RECEPTS,
                "attachments": ATTCHMENTS
            },
            "saveToSentItems": "true"
        }

        # Send the email using Microsoft Graph
        try:
            headers = {
                'Authorization': f'Bearer {ACCESSTOKEN}',
                'Content-Type': 'application/json'
            }
            send_url = f"https://graph.microsoft.com/v1.0/users/{sender}/sendMail"
            res = requests.post(send_url, headers=headers, json=PAYLOAD)

            if res.status_code == 202:
                self.logger.info(f'[AZURE] auto_send_email SENT 2 {recipients} on {datetime.now()}')
            else:
                self.logger.error(f'[AZURE] auto_send_email FAILED with: {res.status_code} - {res.text}')

        except Exception as e:
            self.logger.error(f'[AZURE] auto_send_email EXCEPTION: {e}')


    # Preprocess Competitor Agent Web xlsx file --> perform upsert on pandas dataframe and azure db
    # PK ~ ('release_dt','state_cd','en_sku','comp_sku','distr_typ')
    def comp_agent_web_upsert_preprocess(self):
        PK_COLS = ['release_dt','state_cd','en_sku','comp_sku','distr_typ']

        # Transform pandas df wrt database settings
        self.__transform_df_wrt_azuredb(PK_COLS)
        if self.myDf.empty:
            return 

        try:
            # Insertion logic based on non-duplicate PK
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            SQLQuery = "SELECT distinct release_dt,state_cd,en_sku,comp_sku,distr_typ,mnf_stk_price FROM landing.en_comp_sku_fct;"
            insertionDf, updateDf = self.__trigger_upsert_df_wrt_azuredb(SQLQuery, PK_COLS)
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
            self.logger.error(f"[DEBUG] comp_agent_web_upsert_preprocess GETS Azure DB err: {sqlerr}\n")
        finally:
            engine.dispose()
            # send alerts email if necessary
            alertDf = self.__get_pricing_alerts(insertionDf, updateDf, 0.1)

            # test to only send to andrew.chen@enerlites.com
            attName = f"{datetime.now().strftime('%Y-%m-%d')} Pricing Alerts"
            self.auto_send_email(outputDfs=[alertDf],emailSubject="Competitor Pricing Alerts [NO-REPLY]", attachmentNames=[attName])
            

    # Preprocess sku_master_dim_hst xlsx file --> perform upsert on pandas dataframe and azure db
    # PK ~ ('model_no','price_model','lower_bucket','upper_bucket')
    def sku_master_dim_hst_preprocess(self):
        PK_COLS = ['model_no','price_model','lower_bucket','upper_bucket']
        df = self.myDf.copy()
        # update unit price wrt pallet
        price_updt_df = update_sku_master_unitprice_wrt_pallet(df)
        price_updt_df.loc[:,'src_updt_dt'] = pd.to_datetime(price_updt_df.loc[:,'src_updt_dt'], format='mixed', errors= 'coerce').dt.date
        # convert to tabular form
        pivoted_df = pivot_sku_master_price_conds(price_updt_df)
        self.myDf = pivoted_df
        
        # Transform pandas df w.r.t. azure ddl
        self.__transform_df_wrt_azuredb(PK_COLS)
        if self.myDf.empty:
            return 

        # Write Insertion df to memory and return update df
        try:
            # Insertion logic based on non-duplicate PK
            engine = create_engine(self.DB_CONN, connect_args={"timeout": 30})
            SQLQuery = "SELECT distinct model_no,price_model,lower_bucket,upper_bucket, src_updt_dt FROM landing.sku_master_dim_hst;"
            _, updateDf = self.__trigger_upsert_df_wrt_azuredb(SQLQuery, PK_COLS)
            if updateDf.empty:
                return 
            CUR_TS = pd.to_datetime('now')
            with engine.begin() as conn:  # Ensures commit/rollback
                for _, row in updateDf.iterrows():
                    # Use parameterized SQL to avoid SQL injection and type issues
                    updt_stmt = text("""
                        UPDATE landing.sku_master_dim_hst 
                        SET src_updt_dt = :src_updt_dt,
                            descript = :descript,
                            prod_cd = :prod_cd,
                            unit_cost = :unit_cost,
                            unit_price = :unit_price,
                            data_dt = :data_dt
                        WHERE model_no = :model_no
                            AND price_model = :price_model
                            AND lower_bucket = :lower_bucket
                            AND upper_bucket = :upper_bucket;
                    """)
                    conn.execute(updt_stmt, {
                        'src_updt_dt': None if pd.isna(row.get('src_updt_dt')) else row.get('src_updt_dt'),
                        'descript': None if pd.isna(row.get('descript')) else row.get('descript'),
                        'prod_cd': None if pd.isna(row.get('prod_cd')) else row.get('prod_cd'),
                        'unit_cost': None if pd.isna(row.get('unit_cost')) else row.get('unit_cost'),
                        'unit_price': None if pd.isna(row.get('unit_price')) else row.get('unit_price'),
                        'data_dt': CUR_TS,
                        'model_no': row['model_no'],
                        'price_model': row['price_model'],
                        'lower_bucket': row['lower_bucket'],
                        'upper_bucket': row['upper_bucket']
                    })
        except SQLAlchemyError as sqlerr: 
            self.logger.error(f"[DEBUG] sku_master_dim_hst_preprocess GETS Azure DB err: {sqlerr}\n")
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
            # append local LA datetime as cur_ts field 
            df['cur_ts'] = datetime.now(pytz.timezone('America/Los_Angeles'))

            # persist df name with that of defined in ssms   
            df.columns = tableCols
            self.myDf = df
            
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
            self.logger.info(f"[Azure] Upserted {df.shape[0]} rows to {schema}.{table} !\n")
            
        except Exception as e:
            self.logger.error(f"[Azure] Error writing to database: {str(e)} !\n", exc_info=True)
        finally:
            engine.dispose()