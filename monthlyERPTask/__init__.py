import logging
import azure.functions as func
from ..shared.azureApp import monthly_netsuite_erp_job

def main(monthlyERPCron: func.TimerRequest) -> None:
    logging.info("Running: monthly_netsuite_erp_job()")
    monthly_netsuite_erp_job()