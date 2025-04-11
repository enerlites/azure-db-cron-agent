'''
Code entry for each Azure func
'''
from function_app import *      # import all funcs defined in module

def main(sku_promo_db_cron: dict) -> None:
    monthly_sku_promo_task(sku_promo_db_cron)