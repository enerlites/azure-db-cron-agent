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
    # 5th argument --> used for time_eval_level 
    def _get_single_file_dw_url(self, access_token, driver_id, folderName, fileName, time_eval_level = None):
        oneDriveBaseURL = f"{self.base_graph_url}/drives/{driver_id}"
        FolderURL = f"{oneDriveBaseURL}/root/children"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        try:
            res = requests.get(FolderURL, headers= headers, timeout= 30)
            items = res.json().get('value', [])         # return a list of python dict
            folderId = None

            # convert ts to UTC 
            local_ts = datetime.now(pytz.UTC)
            local_year, local_month, local_day = local_ts.year, local_ts.month, local_ts.day
            
            # found foldername first within the oneDrive root dir
            for item in items:
                if item['name'] == folderName:          # return specified folder id
                    # print(f"\n{folderName} folder found in OneDrive !\n")
                    folderId = item["id"]
                    FileURL = f"{oneDriveBaseURL}/items/{folderId}/children"
                    res = requests.get(FileURL, headers = headers, timeout= 30)
                    fileItems = res.json().get('value', [])
                    
                    for file in fileItems:
                        # no time restriction on this file (just read) 
                        if file['name'] == fileName and not time_eval_level:
                            return file['@microsoft.graph.downloadUrl']
                        # align file modification ts w.r.t day 
                        elif file['name'] == fileName:
                            modified_ts = datetime.strptime(file['lastModifiedDateTime'], "%Y-%m-%dT%H:%M:%SZ")
                            modified_y, modified_m, modified_d = modified_ts.year, modified_ts.month, modified_ts.day

                            if modified_y == local_year and modified_m == local_month and modified_d == local_day and time_eval_level == 'day':
                                return file['@microsoft.graph.downloadUrl']
                            elif modified_y == local_year and modified_m == local_month and time_eval_level == 'month':
                                return file['@microsoft.graph.downloadUrl']
            print(f"\"{fileName}\" either not exists in \"{folderName}\" folder or not modified ! (Skip db load)\n")         
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
            if download_url:    # obtain valid download url
                res = requests.get(download_url, headers= headers, timeout=30)
                BinaryData = BytesIO(res.content)
                if mode == 'xlsx':            
                    df = pd.read_excel(
                        BinaryData,
                        sheet_name=sheet_name,
                        engine='openpyxl'
                    )
                    print(f"[DEBUG] __url2df (xlsx) GETS {df.shape}\n")
                    return df
                elif mode == 'csv':
                    df = pd.read_csv(
                        BinaryData, low_memory= False
                    )
                    print(f"[DEBUG] __url2df (csv) GETS {df.shape}\n")
                    return df 
                else: 
                    print(f"[DEBUG] __url2df unsupported {mode}")
                    return pd.DataFrame()
            else:       # download url not available (Not Exists / Time Constraint Not Satisfied)
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            raise Exception(f"File download failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Excel parsing failed: {str(e)}")

    def read_excel_from_onedrive(self, folderName, fileName, sheet_name=None, time_eval_level= None):
        # driver function that coordinates all private / public class functions
        try:
            access_token = self.__get_access_token()
            drive_id = self.__get_drive_id(access_token)
            download_url = self._get_single_file_dw_url(access_token,drive_id,folderName,fileName, time_eval_level)
            return self.__url2df(download_url, access_token, sheet_name, mode='xlsx')

        except Exception as e:
            raise Exception(f"{str(e)}")
    
    '''
    Grab a list of ms downloadable url within time eval level (4th argument)
        eg. when time_eval_level = 'month'          --> generate file download urls only when a file is modified in current month
                 time_eval_level = 'day'            --> generate file download urls only when a file is modified on current day
    '''
    def __get_multi_files_dw_urls (self, access_token, driver_id, folderName, time_eval_level = None):
        oneDriveBaseURL = f"{self.base_graph_url}/drives/{driver_id}"
        FolderURL = f"{oneDriveBaseURL}/root/children"
        headers = {"Authorization": f"Bearer {access_token}"}

        # convert ts to UTC 
        local_ts = datetime.now(pytz.UTC)
        local_year, local_month, local_day = local_ts.year, local_ts.month, local_ts.day
        
        try:
            res = requests.get(FolderURL, headers= headers, timeout= 30)
            items = res.json().get('value', [])
            folderId = None
            dwFileUrls = []                # A list of donwload url that were modified this month
            
            # Iterate over folder items
            for item in items:
                if item['name'] == folderName:          # return specified folder id
                    folderId = item["id"]
                    FileURL = f"{oneDriveBaseURL}/items/{folderId}/children"
                    res = requests.get(FileURL, headers = headers, timeout= 30)
                    fileItems = res.json().get('value', []) 
                    
                    # Iterate over files
                    for file in fileItems:
                        modified_ts = datetime.strptime(file['lastModifiedDateTime'], "%Y-%m-%dT%H:%M:%SZ")
                        modified_y, modified_m, modified_d = modified_ts.year, modified_ts.month, modified_ts.day
                        # check monthly basis
                        if local_year == modified_y and local_month == modified_m and time_eval_level == 'month':
                            print(f"[DEBUG] __get_multi_files_dw_urls GETS \'{file['name']}\'\n")
                            dwFileUrls.append(file["@microsoft.graph.downloadUrl"])
                        
                        # check daily basis
                        elif local_year == modified_y and local_month == modified_m and local_day == modified_d and time_eval_level == 'day':
                            print(f"[DEBUG] __get_multi_files_dw_urls GETS \'{file['name']}\'\n")
                            dwFileUrls.append(file["@microsoft.graph.downloadUrl"])
                        
            return dwFileUrls

        except requests.exceptions.RequestException as e:
            print(f"Error searching for folder/file: {str(e)}")
            return None

    def read_csv_from_oneDrive(self, folderName, time_eval_level = 'month'):
        try:
            access_token = self.__get_access_token()
            drive_id = self.__get_drive_id(access_token)
            download_urls = self.__get_multi_files_dw_urls(access_token,drive_id,folderName, time_eval_level)
            dfs = [self.__url2df(url, access_token, mode='csv') for url in download_urls]
            return dfs

        except Exception as e:
            raise Exception(f"{str(e)}")