'''
Below Script 
a) requires Application Permissions (API permission) --> Run as background service without signed-in user
b) configure the oneDrive with approprite file management
c) intend to be deployed over Azure Function App 

All configuration vars could be found from: https://portal.azure.com/?quickstart=true#home

'''
from shared.OneDriveFlatFileReader import *
from shared.azureDBWriter import *

# Define a monthly promotion sku job (run on 12:30 AM on 15th of each month)
def monthly_promotion_brochure_job():
    try:
        # create an instance to read from andrew.chen@enerlites.com
        oneDriveReader = OneDriveFlatFileReader("andrew.chen@enerlites.com")
        
        # Define file management related fields
        files = ['Promotion Data.xlsx', 'Ocean_Air in Transit List.xlsx']
        sku_baseCols = ['sku','color','category','promo_reason','descrip','moq','socal', 'ofs','free_sku','feb_sales','inv_quantity','inv_level', 'photo_url', 'sys_dt']
        sku_hstCols = [
            "month_st",
            "month_ed",
            "sku",
            "promo_reason",
            "category",
            "descrip",
            "color",
            "moq",
            "socal",
            "ofs",
            "free_sku",
            "Qty On Hand (monthly)",
            "Qty sold (last 3 month)",
            "Months To Sell",
            "Actual Monthly Qty Sold (Netsuite)",
            "photo_url",
            "sys_dt"
        ]
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
        ns_erp_dfs = oneDriveReader.read_csv_from_oneDrive("NetSuite ERP", "month") 

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

# Define daily cron job for importing and updating en_comp_sku_fct table in Azure DB 
def daily_comp_pricing_job():
    try:
        oneDriveReader = OneDriveFlatFileReader("andrew.chen@enerlites.com")
        comp_pricing_df = oneDriveReader.read_excel_from_onedrive("price benchmarking", "en_comp_sku_fct.xlsx", "master", "day")

        # define ddl fields
        en_comp_sku_fct_cols = [
            "release_dt",
            "state_cd",
            "mnf_stk_price",
            "en_sku",
            "comp_sku",
            "quantity",
            "mnf",
            "distr",
            "mnf_desc",
            "distr_typ",
            "rep_name",
            "sys_dt"
        ]

        compWriter = AzureDBWriter(comp_pricing_df, en_comp_sku_fct_cols)
        compWriter.comp_agent_web_upsert_preprocess()
        compWriter.flatFile2db('landing', 'en_comp_sku_fct')

    except Exception as e:
        print(f"{str(e)}")

# Define monthly cron job for Enerlites Internal Pricing to Azure database
def monthly_en_internal_pricing_job():
    try:
        oneDriveReader = OneDriveFlatFileReader("andrew.chen@enerlites.com")
        en_sku_pricing_df = oneDriveReader.read_excel_from_onedrive("price benchmarking", "sku_master_dim_hst.xlsx", "master price", "month")

        # define ddl fields
        en_sku_pricing_cols = [
            "src_updt_dt",
            "model_no",
            "descript",
            "cat",
            "prod_cd",
            "supply_chain_lvl",
            "prod_lvl",
            "unit_cost",
            "comp_pricing_model_num",
            "unit_price",
            "price_model",
            "lower_bucket",
            "upper_bucket",
            "data_dt"
        ]

        enPricingWriter = AzureDBWriter(en_sku_pricing_df, en_sku_pricing_cols)
        enPricingWriter.sku_master_dim_hst_preprocess()
        enPricingWriter.flatFile2db('landing', 'sku_master_dim_hst')

    except Exception as e:
        print(f"{str(e)}")
           
# Test Section 
if __name__ == "__main__":
    # Test Once
    daily_comp_pricing_job()
    # monthly_promotion_brochure_job()
    # monthly_netsuite_erp_job()
    monthly_en_internal_pricing_job()

    # Schedule 2 jobs on 15th of each month at 12:30 am
    schedule.every().day.at("00:30").do(lambda: monthly_promotion_brochure_job() if datetime.now().day == 15 else None)
    schedule.every().day.at("01:00").do(lambda: monthly_netsuite_erp_job() if datetime.now().day == 15 else None)
    # Schedule 1 job on 23:00 AM of each day
    schedule.every().day.at("23:00").do(daily_comp_pricing_job)

    print("========================== Azure DB Cron Agent Started (Dev) ================================")
    while True:
        schedule.run_pending()
        time.sleep(60)  # wait for each 1 minute