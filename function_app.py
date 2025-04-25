import logging
import azure.functions as func
from azureApp import *          # sku promo module 

app = func.FunctionApp()

'''
auto oneDrive azure read / write in monthly basis 
Execute cron trigger on 12:30 AM UTC on 15th of each month
'''
@app.function_name(name="monthlySkuPromoTask")
@app.schedule(schedule="0 30 0 15 * *", arg_name="monthlyPromoCron", run_on_startup=True, use_monitor=True)
def monthly_sku_promo_task(monthlyPromoCron: func.TimerRequest) -> None:
    # Log that the function is starting
    logging.info(f'========================= Invoke monthly_promotion_brochure_job() ! =========================\n')
    
    try:
        # call func from module
        monthly_promotion_brochure_job()
        
    except Exception as e:
        # Log any errors
        logging.error(f'>>>>>>>>>>>>>>>>>>>> Error in monthly_promotion_brochure_job() <<<<<<<<<<<<<<<<<<<<<\n {str(e)}')
        raise
    
'''
Automate potential ERP reports to Azure
Execute cron trigger on 1:00 AM UTC on 15th of each month
'''
@app.function_name(name="monthlyERPTask")
@app.schedule(schedule="0 0 1 15 * *", arg_name="monthlyERPCron", run_on_startup=True, use_monitor=True)
def monthly_erp_task(monthlyERPCron: func.TimerRequest) -> None:
    # Log that the function is starting
    logging.info(f'========================= Invoke monthly_netsuite_erp_job() ! =========================\n')
    
    try:
        # call func from module
        monthly_netsuite_erp_job()
        
    except Exception as e:
        # Log any errors
        logging.error(f'>>>>>>>>>>>>>>>>>>>> Error in monthly_netsuite_erp_job() <<<<<<<<<<<<<<<<<<<<<\n {str(e)}')
        raise