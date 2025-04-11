'''
Below Script requires Application Permissions (API permission) --> Run as background service without signed-in user

Read directly from Andrew Chen's OneDrive Shared Folder for different projects
'''
import os
import msal
import requests
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from urllib.parse import quote
# from IPython import display
from sqlalchemy import create_engine
import urllib.parse
import schedule 
import time 
from datetime import datetime

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
                print(f"\n{self.user_principal} exists with drive id = \'{response.json()["id"]}\'\n")
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
                    
                    for item in fileItems:
                        if item['name'] == fileName:
                            return item['@microsoft.graph.downloadUrl']
            print(f"Given {fileName} not found in {folderName} folder !")         
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error searching for folder/file: {str(e)}")
            return None
    
    # read dataframe from ms download link        
    def __url2df(self, download_url, access_token, sheet_name=None):
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        try:
            res = requests.get(download_url, headers= headers, timeout=30)
            print(f"[DEBUG] Response length: {len(res.content)} bytes")
            excelData = BytesIO(res.content)
                        
            df = pd.read_excel(
                excelData,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            print(f"[DEBUG] DataFrame created with shape: {df.shape}")
            return df
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
            return self.__url2df(download_url, access_token, sheet_name)

        except Exception as e:
            raise Exception(f"{str(e)}")
        
# DB class for Azure SQL db functions
class AzureDBWriter():
    def __init__(self, df, tableCols):
        load_dotenv()           # load the .env vars
        self.DB_CONN = f"mssql+pyodbc://sqladmin:{urllib.parse.quote_plus(os.getenv("DB_PASS"))}@{os.getenv("DB_SERVER")}:1433/enerlitesDB?driver=ODBC+Driver+17+for+SQL+Server&encrypt=yes"
        self.myDf = df 
        self.myCols = tableCols
        
    # OceanAir Inventory google xlsx sheet preprocessing (inplace operation)
    # Skip the first 3 rows and only read in len(tableCols) -1 cols
    def oceanAir_Inv_preprocess(self):
        self.myDf = self.myDf.iloc[2:, :len(self.myCols) - 1]
        numeric_cols = self.myDf.columns[5:-1]
        self.myDf[numeric_cols] = self.myDf[numeric_cols].astype('Int64')
        return self
        
    # commit flatFile 2 azure db 
    def flatFile2db (self, schema, table):
        engine = create_engine(self.DB_CONN)
        try:
            df = self.myDf.copy()
            tableCols = self.myCols
            # append getdate() datetim2 
            df['sys_dt'] = pd.to_datetime('now')

            '''Below section for data cleaning prior to db load'''
            if "promo dt" in df.columns:
                df["promo dt"] = pd.to_datetime(df["promo dt"],format="mixed",errors='coerce') 
            # handle manual input err
            elif "Promotion Reason" in df.columns:
                df["Promotion Reason"] = df["Promotion Reason"].apply(lambda x: 'Discontinued' if x == 'Disontinued' else x)
            elif "promo category" in df.columns:
                df["promo category"] = df["promo category"].apply(lambda x: 'Discontinued' if x == 'Discontinued item' else x)   
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
            print(f"Successfully wrote {len(df)} rows to {table}")
            
        except Exception as e:
            print(f"Error writing to database: {str(e)}")
        finally:
            engine.dispose()

# Define a monthly scheduler that run the cron procedures at 12 AM on 15th of each month
def monthly_promotion_brochure_job():
    try:
        # create an instance to read from andrew.chen@enerlites.com
        oneDriveReader = OneDriveFlatFileReader("andrew.chen@enerlites.com")
        
        # Define file management related fields
        folderPath = "sku promotion"
        files = ['Promotion Data.xlsx', 'Ocean_Air in Transit List.xlsx']
        sku_baseCols = ['sku','category','promo_reason','descrip','moq','socal', 'ofs','free_sku','feb_sales','inv_quantity','inv_level', 'photo_url', 'sys_dt']
        sku_hstCols = ['promo_dt','promo_cat','sku','sys_dt']
        oceanAirInvCols = [
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
        
        # load potential sku from OneDrive first
        sku_base_df = oneDriveReader.read_excel_from_onedrive(
            folderPath,
            files[0],
            sheet_name='potential_skus'
        )
        hst_sku_df = oneDriveReader.read_excel_from_onedrive(
            folderPath,
            files[0],
            sheet_name='past sku promo'
        )
        oceanAirInv_df = oneDriveReader.read_excel_from_onedrive(
            folderPath,
            files[1],
            sheet_name='Friday Inventory TGEN'
        )
        
        # load 2 respective tables
        sku_base_db = AzureDBWriter(sku_base_df,sku_baseCols)
        sku_base_db.flatFile2db('landing', 'oneDrive_promo_sku_base')
        hst_sku_db = AzureDBWriter(hst_sku_df,sku_hstCols)
        hst_sku_db.flatFile2db('landing', 'oneDrive_hst_promo_sku')
        
        oceanAirInv_db = AzureDBWriter(oceanAirInv_df,oceanAirInvCols)
        oceanAirInv_db.oceanAir_Inv_preprocess()
        oceanAirInv_db.flatFile2db('landing', 'googleDrive_ocean_air_inv_fct')
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"monthly_promotion_brochure_auto_job() executed at {datetime.now()} !\n")
        
    except Exception as e:
        print(f"{str(e)}")    
        
           
# Test Section 
if __name__ == "__main__":
    # # exec once
    # monthly_promotion_brochure_job()
    
    # exec this job on 15th at 12:30 am
    schedule.every().day.at("00:30").do(lambda: monthly_promotion_brochure_job() if datetime.now().day == 15 else None)

    print("========================== Executing Monthly SKU Promotion Task ================================")
    while True:
        schedule.run_pending()
        time.sleep(60)  # wait for each 1 minute