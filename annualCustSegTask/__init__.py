import logging
import azure.functions as func
from ..modelling.custTier import *

# define the Annual Customer Segmentation Clustering Task
def main(annualCustSegTask: func.TimerRequest) -> None:
    logging.info("Running: annual_customer_segmentation()")
    myClustering = CustTierClustering()
    myClustering.annual_customer_segmentation()