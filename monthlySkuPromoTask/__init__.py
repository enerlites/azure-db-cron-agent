import logging
import azure.functions as func
from shared import *

def main(monthlyPromoCron: func.TimerRequest) -> None:
    logging.info("Running: monthly_promotion_brochure_job()")
    monthly_promotion_brochure_job()