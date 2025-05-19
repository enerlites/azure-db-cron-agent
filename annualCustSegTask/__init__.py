import logging
import azure.functions as func
from modelling import *

# define the Annual Customer Segmentation Clustering Task
def main(monthlyCustSegTask: func.TimerRequest) -> None:
    logging.info("Running: monthly_customer_segmentation()")
    myClustering = CustTierClustering()
    myClustering.monthly_customer_segmentation()