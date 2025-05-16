import logging
import azure.functions as func
from shared import daily_comp_pricing_job

def main(dailyCompPricingCron: func.TimerRequest) -> None:
    logging.info("Running: daily_comp_pricing_job()")
    daily_comp_pricing_job()