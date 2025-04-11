import logging
import azure.functions as func
from azureApp import *          # sku promo module 

'''
auto oneDrive azure read / write in monthly basis 
'''
def monthly_sku_promo_task(sku_promo_db_cron: func.TimerRequest) -> None:
    # Log that the function is starting
    logging.info(f'========================= Monthly db sku promo cron job started ! =========================\n')
    
    try:
        # call func from module
        monthly_promotion_brochure_job()
        
    except Exception as e:
        # Log any errors
        logging.error(f'>>>>>>>>>>>>>>>>>>>>Error in Monthly db sku promo cron job <<<<<<<<<<<<<<<<<<<<<\n {str(e)}')
        raise
    
'''
Automate potential ERP reports to Azure
'''
def monthly_erp_task(erp_db_cron: func.TimerRequest) -> None:
    pass