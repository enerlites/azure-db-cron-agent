'''
Below Script 
a) requires Application Permissions (API permission) --> Run as background service without signed-in user
b) configure the oneDrive with approprite file management
c) intend to be deployed over Azure Function App 

All configuration vars could be found from: https://portal.azure.com/?quickstart=true#home

'''
import os
import msal
import requests
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from urllib.parse import quote
from sqlalchemy import create_engine
import urllib.parse
import schedule 
import time 
from datetime import datetime
from IPython.display import display
import pytz

# OneDrive class for all oneDrive functionalities
class OneDriveFlatFileReader:
    def __init__(self, corporateEmail):
        load_dotenv()
        self.client_id = os.getenv("AZ_CLI_ID")
        self.client_secret = os.getenv("AZ_CLI_SECRET")
        self.tenant_id = os.getenv("AZ_TENANT_ID")
        self.user_principal = corporateEmail
        self.base_graph_url = "https://graph.microsoft.com/v1.0"
    
    # get the Azure access token
    def __get_access_token(self):
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=authority,
            client_credential=self.client_secret
        )

        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])

        if "access_token" in result:
            return result["access_token"]
        else:
            error_details = result.get("error_description", "No error description provided")
            raise Exception(f"Authentication failed: {result.get('error')} - {error_details}")
    
    # get principal user's driver_id
    def __get_drive_id(self, access_token):
        url = f"{self.base_graph_url}/users/{self.user_principal}/drive"
        
        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        try:
            # List all available drives (including personal OneDrive)
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                print(f"\n{self.user_principal} exists with drive id = \'{response.json()['id']}\'\n")
                return response.json()["id"]
            elif response.status_code == 404:
                raise Exception("OneDrive not found. It may not be provisioned yet.")
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get drive info: {str(e)}")
    
    # Get file id with (access_token, driver_id, folderName)
    def __get_fileDownload_url(self, access_token, driver_id, folderName, fileName):
        oneDriveBaseURL = f"{self.base_graph_url}/drives/{driver_id}"
        FolderURL = f"{oneDriveBaseURL}/root/children"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            res = requests.get(FolderURL, headers= headers, timeout= 30)
            items = res.json().get('value', [])         # return a list of python dict
            folderId = None
            
            # found foldername first within the oneDrive root dir
            for item in items:
                if item['name'] == folderName:          # return specified folder id
                    # print(f"\n{folderName} folder found in OneDrive !\n")
                    folderId = item["id"]
                    FileURL = f"{oneDriveBaseURL}/items/{folderId}/children"
                    res = requests.get(FileURL, headers = headers, timeout= 30)
                    fileItems = res.json().get('value', [])
                    
                    for file in fileItems:
                        if file['name'] == fileName:
                            return file['@microsoft.graph.downloadUrl']
            print(f"Given {fileName} not found in {folderName} folder !")         
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error searching for folder/file: {str(e)}")
            return None
    
    # read dataframe from ms download link        
    def __url2df(self, download_url, access_token, sheet_name=None, mode = 'xlsx'):
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        try:
            res = requests.get(download_url, headers= headers, timeout=30)
            BinaryData = BytesIO(res.content)
            if mode == 'xlsx':            
                df = pd.read_excel(
                    BinaryData,
                    sheet_name=sheet_name,
                    engine='openpyxl'
                )
                print(f"[DEBUG] __url2df GETS df of shape {df.shape}\n")
                return df
            elif mode == 'csv':
                df = pd.read_csv(
                    BinaryData, low_memory= False
                )
                print(f"[DEBUG] __url2df GETS df of shape {df.shape}\n")
                return df 
            else: 
                return None
        except requests.exceptions.RequestException as e:
            raise Exception(f"File download failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Excel parsing failed: {str(e)}")

    def read_excel_from_onedrive(self, folderName, fileName, sheet_name=None):
        # driver function that coordinates all private / public class functions
        try:
            access_token = self.__get_access_token()
            drive_id = self.__get_drive_id(access_token)
            download_url = self.__get_fileDownload_url(access_token,drive_id,folderName,fileName)
            return self.__url2df(download_url, access_token, sheet_name, mode='xlsx')

        except Exception as e:
            raise Exception(f"{str(e)}")
    
    # Given a folderName, grab a list of file donwload url based on timestamp
    def __get_fileDownload_urls_on_ts (self, access_token, driver_id, folderName):
        oneDriveBaseURL = f"{self.base_graph_url}/drives/{driver_id}"
        FolderURL = f"{oneDriveBaseURL}/root/children"
        headers = {"Authorization": f"Bearer {access_token}"}

        # convert ts to UTC 
        local_ts = datetime.now(pytz.UTC)
        local_year, local_month = local_ts.year, local_ts.month
        
        try:
            res = requests.get(FolderURL, headers= headers, timeout= 30)
            items = res.json().get('value', [])
            folderId = None
            monthly_downloadFileUrl = []                # A list of donwload url that were modified this month
            
            # Iterate over folder items
            for item in items:
                if item['name'] == folderName:          # return specified folder id
                    folderId = item["id"]
                    FileURL = f"{oneDriveBaseURL}/items/{folderId}/children"
                    res = requests.get(FileURL, headers = headers, timeout= 30)
                    fileItems = res.json().get('value', []) 
                    
                    # Iterate over files
                    for file in fileItems:
                        last_modified_ts = datetime.strptime(file['lastModifiedDateTime'], "%Y-%m-%dT%H:%M:%SZ")
                        last_modified_y, last_modified_m = last_modified_ts.year, last_modified_ts.month
                        # matching year and month 
                        if local_year == last_modified_y and local_month == last_modified_m:
                            print(f"[DEBUG] __get_fileDownload_urls_on_ts GETS \'{file['name']}\'\n")
                            monthly_downloadFileUrl.append(file["@microsoft.graph.downloadUrl"])
                        
            return monthly_downloadFileUrl

        except requests.exceptions.RequestException as e:
            print(f"Error searching for folder/file: {str(e)}")
            return None

    def read_current_month_csvs_from_onedrive(self, folderName):
        try:
            access_token = self.__get_access_token()
            drive_id = self.__get_drive_id(access_token)
            download_urls = self.__get_fileDownload_urls_on_ts(access_token,drive_id,folderName)
            dfs = [self.__url2df(url, access_token, mode='csv') for url in download_urls]
            return dfs

        except Exception as e:
            raise Exception(f"{str(e)}")
        
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
        engine = create_engine(self.DB_CONN)

        # dedup logic below 
        query = "SELECT distinct inv_eval_dt FROM landing.googleDrive_ocean_air_inv_fct;"
        trg_df = pd.read_sql(query, engine)
        engine.dispose()

        if trg_df.shape[0] != 0:     # db table is not empty  
            trg_dt = pd.read_sql(query, engine).inv_eval_dt.unique().tolist()
            src_dt = self.myDf.inv_eval_dt.unique().tolist()
            if bool(set(trg_dt) & set(src_dt)):         # duplicate found
                self.myDf = pd.DataFrame()
                return 

        # pandas manipulation for time reconciliation 
        dt_df = pd.DataFrame({"dt": [inv_eval_dt] * self.myDf.shape[0]})
        data = self.myDf.iloc[2:, :len(self.myCols) - 2]
        self.myDf = pd.concat([dt_df, data], axis = 1)
        numeric_cols = self.myDf.columns[6:-1]
        self.myDf[numeric_cols] = self.myDf[numeric_cols].astype('Int64')
    
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
        
    # commit flatFile 2 azure db 
    def flatFile2db (self, schema, table):
        print(f"flatFile2db func was called with data of shape {self.myDf.shape}\n")
        engine = create_engine(self.DB_CONN)
        try:
            if self.myDf.shape[0] == 0:          # empty dataframe --> No db load
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

# Define a monthly promotion sku job (run on 12:30 AM on 15th of each month)
def monthly_promotion_brochure_job():
    try:
        # create an instance to read from andrew.chen@enerlites.com
        oneDriveReader = OneDriveFlatFileReader("andrew.chen@enerlites.com")
        
        # Define file management related fields
        files = ['Promotion Data.xlsx', 'Ocean_Air in Transit List.xlsx']
        sku_baseCols = ['sku','color','category','promo_reason','descrip','moq','socal', 'ofs','free_sku','feb_sales','inv_quantity','inv_level', 'photo_url', 'sys_dt']
        sku_hstCols = ['promo_dt','promo_cat','sku','sys_dt']
        oceanAirInvCols = [
            "inv_eval_dt",
            "co_cd",
            "inv_level",
            "sku",
            "asin_num",
            "sku_cat",
            "en_last_120_outbound",
            "en_last_90_outbound",
            "en_last_60_outbound",
            "en_last_30_outbound",
            "tg_last_120_outbound",
            "tg_last_90_outbound",
            "tg_last_60_outbound",
            "tg_last_30_outbound",
            "ca_instock_quantity",
            "il_instock_quantity",
            "lda_instock_quantity",
            "tg_instock_quantity",
            "sys_dt"
        ]
        
        # load potential sku base first
        sku_base_df = oneDriveReader.read_excel_from_onedrive(
            "sku promotion",
            files[0],
            sheet_name='potential_skus'
        )
        sku_base_db = AzureDBWriter(sku_base_df,sku_baseCols)
        sku_base_db.flatFile2db('landing', 'oneDrive_promo_sku_base')

        # load hst sku df second
        hst_sku_df = oneDriveReader.read_excel_from_onedrive(
            "sku promotion",
            files[0],
            sheet_name='past sku promo'
        )
        hst_sku_db = AzureDBWriter(hst_sku_df,sku_hstCols)
        hst_sku_db.flatFile2db('landing', 'oneDrive_hst_promo_sku')

        oceanAirInv_df = oneDriveReader.read_excel_from_onedrive(
            "sku promotion",
            files[1],
            sheet_name='Friday Inventory TGEN'
        )
        oceanAirInv_db = AzureDBWriter(oceanAirInv_df,oceanAirInvCols)
        oceanAirInv_db.oceanAir_Inv_preprocess()
        oceanAirInv_db.flatFile2db('landing', 'googleDrive_ocean_air_inv_fct')

        print(f">>>>>>>>>>>>>>>>>>>> monthly_promotion_brochure_auto_job() executed at {datetime.now()} <<<<<<<<<<<<<<<<<<<<<<<<\n")
        
    except Exception as e:
        print(f"{str(e)}")    

# Define monthly netsuite ERP csv to db (ONLY accept .csv file over OneDrive)
def monthly_netsuite_erp_job():
    try:
        # init an oneDrive account
        oneDriveReader = OneDriveFlatFileReader("andrew.chen@enerlites.com")
        ns_erp_dfs = oneDriveReader.read_current_month_csvs_from_onedrive("NetSuite ERP") 

        # define ddl fields
        erp_items_sold_history_cols = [
            "customer",
            "bill_num",
            "quote_num",
            "sys_dt",
            "sku",
            "quantity",
            "amt",
            "price_model",
            "prod_cd",
            "sku_cat",
            "state_cd",
            "proj_type",
            "onboard_dt",
            "cust_cat",
            "discount",
            "data_dt"
        ]

        # for each netsuite erp df --> call preprocess --> write2db
        for ns_df in ns_erp_dfs:
            nsWriter = AzureDBWriter(ns_df, erp_items_sold_history_cols)
            nsWriter.netsuite_items_sold_hst_preprocess()
            nsWriter.flatFile2db('landing','erp_items_sold_history')

    except Exception as e:
        print(f"{str(e)}")
        
           
# Test Section 
if __name__ == "__main__":
    # Test Once
    # monthly_promotion_brochure_job()
    monthly_netsuite_erp_job()

    # exec 2 jobs on 15th at 12:30 am
    schedule.every().day.at("00:30").do(lambda: monthly_promotion_brochure_job() if datetime.now().day == 15 else None)
    schedule.every().day.at("00:30").do(lambda: monthly_netsuite_erp_job() if datetime.now().day == 15 else None)

    print("========================== Azure DB Cron Agent Started (Dev) ================================")
    while True:
        schedule.run_pending()
        time.sleep(60)  # wait for each 1 minute