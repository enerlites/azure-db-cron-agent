import pandas as pd
# import math
# from IPython.display import display
import re
from datetime import datetime

# Helper functions for preprocessing Enerlites Internal Price xlsx file
def pm_regex_detection(unstructured_str):
    pm_regex = r"^([a-zA-Z- ]+)\d+"            # regex starts with a-z or - any number of times
    pm_res = re.match(pm_regex, unstructured_str.strip())
    if pm_res:
        return pm_res.group(1).strip()
    else:
        return unstructured_str.strip()     # If not socal / national price format --> just use entire colname
    
def order_regex_detection_lower(unstructured_str):
    bucket_regex1 = r"^[a-zA-Z- ]+\d+\s?\(\s?(\d+)-(\d+)\)"          # regex for range of orders
    bucket_regex2 = r"^[a-zA-Z- ]+\d+\s?(\d+)"
    res1 = re.match(bucket_regex1, unstructured_str.strip())
    res2 = re.match(bucket_regex2, unstructured_str.strip())
    if res1:
        return int(res1.group(1).strip())
    elif res2:
        return int(res2.group(1).strip())
    else:
        return int(0)

def order_regex_detection_upper(unstructured_str):
    bucket_regex1 = r"^[a-zA-Z- ]+\d+\s?\(\s?(\d+)-(\d+)\)"          # regex for range of orders
    bucket_regex2 = r"^[a-zA-Z- ]+\d+\s?(\d+)"
    res1 = re.match(bucket_regex1, unstructured_str.strip())
    res2 = re.match(bucket_regex2, unstructured_str.strip())
    if res1:
        return int(res1.group(2).strip())
    elif res2:
        return int(99999999)
    else:
        return int(99999999)
    
# tranform unstructured cleaned master excel sheet --> db loadable tabular input 
def pivot_sku_master_price_conds(df):
    subset_cols1 = ['src_updt_dt','Model No','Description','Class','Industry'
                   ,'Supply Chain Level','Product Level','EN-cost'
                   ,'Competitor Pricing/Model Number']          # 9 fixed columns 
    subset_cols2 = ['Model No'] + [col for col in df.columns 
                                   if (col.startswith("Out-of-state ") 
                                       or col.startswith("SoCal Project ")
                                       or col == "Electrical Marketplace Price"
                                       or col == "National SPA Price"
                                       or col == "Sun Valley Pricing"
                                       or col == "South CA SPA"
                                       or col == "Standard Price"
                                       or col == "CED Price"
                                       or col == "GraybaR Price"
                                       or col == "REXEL SPA Price"
                                       or col == "Wesco Pricing - no rebates"
                                       or col == "Platt Standard"
                                       or col == "CES Price"
                                       or col == "National Price"
                                       ) 
                                   and col != "Out-of-state Pallet price"
                                   ]
    
    new_df1 = df[subset_cols1]
    new_df2 = df[subset_cols2]
    new_df1.columns, new_df2.columns = new_df1.columns.str.strip(), new_df2.columns.str.strip()
    # Rename the price model for new_df2
    new_df2 = new_df2.rename(columns = {"Sun Valley Pricing": "Sun Valley Price"
                                        , "South CA SPA": "SPA South CA Price", "CED Price": "CED"
                                        , "Wesco Pricing - no rebates": "Wesco"})
    # Unpivot multiple columns to rows
    unpivoted_df = new_df2.melt(
        id_vars = ["Model No"],
        var_name = "pm_quantity_bucket",
        value_name = "unit_price",
    ).reset_index(drop = True)

    # add Price Model, quantity buckets cols after unpivot
    unpivoted_df['pm'] = unpivoted_df.pm_quantity_bucket.apply(
        pm_regex_detection
    )
    unpivoted_df['lower'] = unpivoted_df.pm_quantity_bucket.apply(
        order_regex_detection_lower
    )
    unpivoted_df['upper'] = unpivoted_df.pm_quantity_bucket.apply(
        order_regex_detection_upper
    )

    unpivoted_df = unpivoted_df.drop(['pm_quantity_bucket'], axis = 1)

    # display(unpivoted_df.loc[unpivoted_df['model no'] == 7701, :])

    # Perform pd.DataFrame left join 
    new_df = new_df1.merge(unpivoted_df, on ='Model No', how='left')
    return new_df

def update_sku_master_unitprice_wrt_pallet(df):
    # out-of-state match regex
    outofstate_regex = r"^Out-of-state Project\s?\d+\s?\(\s?\d+\s?-\s?\d+\s?\)|^Out-of-state Project\s?\d+\s?\d+|^Out-of-state Pallet price$"
    # socal match regex
    socal_regex = r"^SoCal Project\s?\d+\s?\(\s?\d+-\s?\d+\s?\)|^SoCal Project\s?\d+\s?\d+|^SoCal Pallet price$"

    # out of state / socal pallet range
    pallet_range_regex1 = r"\(\s?(\d+)\s?-\s?(\d+)\s?\)"
    pallet_range_regex2 = r"^(?:Out-of-state Project|SoCal Project)\s?\d+\s?(\d+)"
    
    new_df = df.copy() 
    
    # Update the unit price based on pallet ordered
    for row_idx, rows in df.iterrows():
        pallet_quantity = float(rows["Pallet Quantity"])
        new_rate, mode, update_soCal, update_oos = None, -1, False, False

        # Iterate through each column
        for col_idx, colname in enumerate(df.columns):
            # skip col headers that don't abide by socal and out-of-state format
            if not re.match(socal_regex, colname) and not re.match(outofstate_regex, colname):
                continue

            range_res1 = re.search(pallet_range_regex1, colname)
            lower, upper = 0, 99999999
            if range_res1:                              # extract lower & upper <-- regex format 1
                lower = float(range_res1.group(1))       
                upper = float(range_res1.group(2))      
            else:                                       # extract lower & upper <-- regex format 2
                range_res2 = re.search(pallet_range_regex2, colname)
                if range_res2:
                    lower = float(range_res2.group(1))
            # update mode and update flags accordingly
            if colname[:5] == 'Socal':
                mode, update_oos = 0, False
            elif colname[:12] == 'Out-of-state':
                mode, update_soCal = 1, False
            else: mode, update_oos, update_soCal = -1, False, False

            # update the rate (first)
            if mode != -1 and (update_soCal or update_oos):
                new_df.iloc[row_idx, col_idx] = new_rate
            # prepare rate update
            elif (lower < pallet_quantity and upper > pallet_quantity) and mode != -1:
                new_rate = rows[colname]
                if mode == 0:
                    update_soCal = True 
                elif mode == 1:
                    update_oos = True 
            else: 
                continue
    return new_df