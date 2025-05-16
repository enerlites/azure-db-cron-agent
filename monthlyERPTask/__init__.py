import logging
import azure.functions as func
from shared import *

def main(monthlyERPCron: func.TimerRequest) -> None:
    logging.info("Running: monthly_netsuite_erp_job()")
    monthly_netsuite_erp_job()