import logging
import azure.functions as func
from azureApp import *

app = func.FunctionApp()

@app.function_name(name="monthlySkuPromoTask")
@app.schedule(schedule="0 30 8 15 * *", arg_name="monthlyPromoCron", run_on_startup=True, use_monitor=True)
def monthly_sku_promo_task(monthlyPromoCron: func.TimerRequest) -> None:
    logging.info("Running: monthly_promotion_brochure_job()")
    monthly_promotion_brochure_job()

@app.function_name(name="monthlyERPTask")
@app.schedule(schedule="0 0 9 15 * *", arg_name="monthlyERPCron", run_on_startup=True, use_monitor=True)
def monthly_erp_task(monthlyERPCron: func.TimerRequest) -> None:
    logging.info("Running: monthly_netsuite_erp_job()")
    monthly_netsuite_erp_job()

@app.function_name(name="dailyCompPricingTask")
@app.schedule(schedule="30 7 * * *", arg_name="dailyCompPricingCron", run_on_startup=True, use_monitor=True)
def daily_comp_pricing_task(dailyCompPricingCron: func.TimerRequest) -> None:
    logging.info("Running: daily_comp_pricing_job()")
    daily_comp_pricing_job()
