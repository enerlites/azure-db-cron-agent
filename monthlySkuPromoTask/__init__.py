import logging
import azure.functions as func
from ..shared.azureApp import monthly_promotion_brochure_job

def main(monthlyPromoCron: func.TimerRequest) -> None:
    logging.info("Running: monthly_promotion_brochure_job()")
    monthly_promotion_brochure_job()