import logging
import azure.functions as func
from azureApp import *          # sku promo module 

app = func.FunctionApp()

'''
auto oneDrive azure read / write in monthly basis 
Execute cron trigger on 12:30 AM PST on 15th of each month
'''
@app.function_name(name="monthlySkuPromoTask")
@app.schedule(schedule="0 30 8 15 * *", arg_name="monthlyPromoCron", run_on_startup=True, use_monitor=True)
def monthly_sku_promo_task(monthlyPromoCron: func.TimerRequest) -> None:
    logging.info(f'========================= Invoke monthly_promotion_brochure_job() ! =========================\n')
    
    try:
        monthly_promotion_brochure_job()
        
    except Exception as e:
        logging.error(f'>>>>>>>>>>>>>>>>>>>> Error in monthly_promotion_brochure_job() <<<<<<<<<<<<<<<<<<<<<\n {str(e)}')
        raise
    
'''
Automate potential ERP reports to Azure
Execute cron trigger on 1:00 AM PST on 15th of each month
'''
@app.function_name(name="monthlyERPTask")
@app.schedule(schedule="0 0 9 15 * *", arg_name="monthlyERPCron", run_on_startup=True, use_monitor=True)
def monthly_erp_task(monthlyERPCron: func.TimerRequest) -> None:
    logging.info(f'========================= Invoke monthly_netsuite_erp_job() ! =========================\n')
    
    try:
        monthly_netsuite_erp_job()
        
    except Exception as e:
        logging.error(f'>>>>>>>>>>>>>>>>>>>> Error in monthly_netsuite_erp_job() <<<<<<<<<<<<<<<<<<<<<\n {str(e)}')
        raise

'''
Automate competitor pricing to Azure
Execute cron trigger on 11:30 PM PST on every day
'''
@app.function_name(name="dailyCompPricingTask")
@app.schedule(schedule="30 7 * * *", arg_name="dailyCompPricingCron", run_on_startup=True, use_monitor=True)
def daily_comp_pricing_task(dailyCompPricingCron: func.TimerRequest) -> None:
    logging.info(f'========================= Invoke daily daily_comp_pricing_job() ! =========================\n')
    
    try:
        daily_comp_pricing_job()
    except Exception as e:
        logging.error(f'>>>>>>>>>>>>>>>>>>>> Error in daily_comp_pricing_job() <<<<<<<<<<<<<<<<<<<<<\n {str(e)}')
        raise
