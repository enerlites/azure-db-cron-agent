import logging
import azure.functions as func
from shared import *

def main(monthlySkuPricingCron: func.TimerRequest) -> None:
    logging.info("Running: monthly_en_internal_pricing_job()")
    monthly_en_internal_pricing_job()