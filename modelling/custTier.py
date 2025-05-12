'''
Latent Class Analysis for Customer Tier classification 

Bus Objectives: Group Customers into Customer Tier (Tier with price sensitivity attributes)

a) customer loyalty metrics:
    day of tenure
    order portfolio (diverse sku kinds)
b) purchase attribute (by product category)
    order freq
    avg purchase / order
    time between consecutive orders
c) discount factor 
    unit discount

Models: 
a) K-means (hard assignment)                            --> cluster assignment via closeness (solid assignment)    
    --> each predictors should be 
b) Gaussian Mixture model (soft assignment)             --> soft clustering with probability (rough assignment)
    --> assignment boundary is not clear && predictors has certain multicollinearity
'''
import pandas as pd
import numpy as np
import openpyxl
import os 
from dotenv import load_dotenv
from pathlib import Path 
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from scipy.stats import shapiro
from k_means_constrained import KMeansConstrained           # use constrained k-means to force at least 10 customers in each tier
import re


# get the absolute path of pwd
cur_file_path = Path(__file__).resolve() 
env_path = cur_file_path.parent.parent / '.env'

load_dotenv(env_path)
DB_URL = os.getenv("DATABASE_URL")

# Write multi dataframe to a single excel's tab
# delete tab if exists, else create 
def writeCustTier2excel(df, file_path, tab):
    try:
        myexcel = openpyxl.load_workbook(file_path)
        # delete tab if exists
        if tab in myexcel.sheetnames:
            del myexcel[tab]
        # save changes
        myexcel.save(file_path)
        myexcel.close()

        # Append new sheet_name to existing excel
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as wrtr:
            df.to_excel(excel_writer = wrtr, sheet_name = tab, index = False)
    except:
        # Create this excel sheet with sheet_name
        with pd.ExcelWriter(file_path, engine="openpyxl", mode = "w") as wrtr:
            df.to_excel(excel_writer = wrtr, sheet_name = tab, index = False)

# write the iter customer dictionary to text file
def write2txt (custTier_lookup, fp, mode, model_name):
    # write the dictinary to a text file
    with open(fp, mode) as outFile:
        outFile.write(f"Model {model_name}:\n\n")
        for c in custTier_lookup:
            customers = custTier_lookup[c]
            outFile.write(f"Tier{c} ({len(customers)} customers):\n[\n")
            for cust in customers[:-1]:
                outFile.write(f"{cust};\t")
            outFile.write(f"{customers[-1]}\n]\n\n")
        outFile.write(f"\n\n\n")

# Read input table from Supabase
def read_custTier_from_supabase ():
    try:

        engine = create_engine(DB_URL)
        print("Load \'customer_tier_input\' from Supabase !!")
        cust_df = pd.read_sql("SELECT * FROM quotedb.customer_tier_input", engine)
        cust_df.cust_cat = cust_df.cust_cat.fillna('NaN')
        return cust_df

    except Exception as e:
        print("Failed to read table from Supabase")

# Load the cust Tier result back to Supabase
def load_df_2_supbase(df, tableName):
    engine = create_engine(DB_URL)
    df.to_sql(tableName, con = engine, schema="model", if_exists = "replace", index =False)
    print(f"{tableName} loaded 2 Supabase !\n")

# Define model pipeline before model invocation
def preprocess_pip(cust_df):
    cols = cust_df.columns
    cus_series = cust_df.customer
    cust_df = cust_df.drop(['customer'], axis = 1)

    # Define cat & num Pipelines
    num_cols = [c for c in cols if c != "cust_cat" and c != "customer"]
    cat_cols = ['cust_cat']
    num_trans = Pipeline(
        steps= [
            ('scaler', StandardScaler())
        ]
    )
    cat_trans = Pipeline(
        steps= [
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_trans, num_cols),
            ('cat', cat_trans, cat_cols)
        ]
    )

    return cus_series, preprocessor         # return customer and preprocessor as pipeline object

# Model 1: K-means with sihoutte score (determine hard assignment )
#   --> Input: loyalty / purchase / discount / intrinstic feature / combine with product category
# Cons: bad when clusters assignment are loose && predictors are related
# retunr: df, dictionary
def k_means_fine_tune(cust_df):
    cus_series, preprocessor = preprocess_pip(cust_df)
    
    # Below section for fine tunning the k-means
    silscore_stats = []
    labs_stats = []
    k_vals = range(3,11,1)
    for k in k_vals:
        pip = Pipeline(
            [
                ('Prep', preprocessor),
                ('kmeans', KMeansConstrained(n_clusters=k, size_min = 15, random_state=42, n_init=17))
            ]
        )
        # write result back for plot
        res = pip.fit(cust_df)
        labs = res.named_steps['kmeans'].labels_
        processed_df = pd.DataFrame(res.named_steps['Prep'].fit_transform(cust_df))
        score = silhouette_score(processed_df, labs)
        silscore_stats.append(score)
        labs_stats.append(labs)
    
    # Pick the best K based on sihouette score (highest)
    k_idx = np.argmax(np.array(silscore_stats))
    opt_k = k_vals[k_idx]
    print(f"\nk-means model: optimal k = {opt_k} with max silhouette score = {np.max(silscore_stats)}\n\n")
    # grab the label for those customers
    opt_labs = np.array(labs_stats[k_idx])
    custTier_output = pd.concat([cus_series, pd.Series(opt_labs)], axis = 1)
    custTier_output.columns = ['customer', 'cust tier']
    custTier_output = pd.concat([custTier_output, cust_df.iloc[:,1:]], axis= 1)
    # writeCustTier2excel(custTier_output, 'custTier.xlsx', "customer tier (kmeans)")

    # Write into dictionary
    custTier_lookup = {}
    for label_class in range(opt_k):
        custTier_lookup[label_class] = cus_series.iloc[np.where(opt_labs == label_class)[0]].to_list()
    silh_score = np.max(silscore_stats)
    # write2txt (custTier_lookup, "custTier.txt", "w", "k-means", silh_score)
    return custTier_output, custTier_lookup

# Model 2: Guassian Mixture Modelling (soft assignment)
# pros: better handle multi-collinearity
def gmm_fine_tune(cust_df):
    cus_series, preprocessor = preprocess_pip(cust_df)

    # Below section for fine tunning the gmm 
    silscore_stats = []
    labs_stats = []
    k_vals = range(3,11,1)
    for k in k_vals:
        pip = Pipeline(
            [
                ('Prep', preprocessor),
                ('gmm', GaussianMixture(n_components=k, random_state=42, n_init=17))
            ]
        )
        # write result back for plot
        res = pip.fit(cust_df)
        processed_df = pd.DataFrame(res.named_steps['Prep'].fit_transform(cust_df))
        labs = res.named_steps['gmm'].predict(processed_df)
        score = silhouette_score(processed_df, labs)
        silscore_stats.append(score)
        labs_stats.append(labs)

    # Pick the best K based on sihouette score (highest)
    k_idx = np.argmax(np.array(silscore_stats))
    opt_k = k_vals[k_idx]
    print(f"\ngmm model: optimal k = {opt_k} with max silhouette score = {np.max(silscore_stats)}\n\n")
    # grab the label for those customers
    opt_labs = np.array(labs_stats[k_idx])
    custTier_output = pd.concat([cus_series, pd.Series(opt_labs)], axis = 1)
    custTier_output.columns = ['customer', 'cust tier']
    custTier_output = pd.concat([custTier_output, cust_df.iloc[:,1:]], axis= 1)
    # writeCustTier2excel(custTier_output, 'custTier.xlsx', "customer tier (gmm)")

    # Write into dictionary
    custTier_lookup = {}
    for label_class in range(opt_k):
        custTier_lookup[label_class] = cus_series.iloc[np.where(opt_labs == label_class)[0]].to_list()

    # write the dictinary to a text file
    silh_score = np.max(silscore_stats)
    # write2txt (custTier_lookup, "custTier.txt", "a", "gmm", silh_score)
    return custTier_output, custTier_lookup

# Design rule-based matching to map tier with clustering label
# 3 levels of metrics: clv (cust and product category) + customer loyalty + discount factor ()
# rule-based mapping logic: 
#           70% prod purchase metrics;  20% loyalty metrics;    10% discount metrics
def rule_based_tier_mapping(tier_profile_df, rule_percent_chain):
    tier_assignments = ['Platium', 'Gold', 'Silver', 'Bronze']
    prod_cat_clv_regex = r"^(?!day_)(?!.*_discprice)[a-zA-Z0-9_]* \(mue\)$"                     # clv regex (exclude loyalty and discount factor)
    cust_loyalty_regex = r"^day_[a-zA-Z0-9_]* \(mue\)$"                                         # loyalty regex
    prod_cat_disc_regex = r".*_discprice \(mue\)$"                                              # discount factor regex
    mycols = tier_profile_df.columns

    # extracted relevant cols for different metric evaluation
    clv_cols = ["CustTier"] + [col for col in mycols if re.match(prod_cat_clv_regex, col)]
    loy_cols = ["CustTier"] + [col for col in mycols if re.match(cust_loyalty_regex, col)]
    disc_cols = ["CustTier"] + [col for col in mycols if re.match(prod_cat_disc_regex, col)]

    # subset relevant df for final weighted sum (create a separate copy)
    clv_df, loy_df, disc_df = tier_profile_df[clv_cols].copy(), tier_profile_df[loy_cols].copy(), tier_profile_df[disc_cols].copy()
    # Rank metric col with different logic
    clv_df.iloc[:,1:] = clv_df.iloc[:, 1:].rank(method = "min", ascending = True)
    loy_df.iloc[:,[1]] = loy_df.iloc[:, [1]].rank(method = "min", ascending = True)
    loy_df.iloc[:,[2]] = loy_df.iloc[:, [2]].rank(method = "min", ascending = False)        # day bet purchases
    disc_df.iloc[:,1:] = disc_df.iloc[:, 1:].rank(method = "min", ascending = True)

    # col wise aggregation and set CustTier as index
    clv_agg = clv_df.set_index("CustTier").sum(axis = 1).rank(method = 'min') * rule_percent_chain[0]
    loy_agg = loy_df.set_index("CustTier").sum(axis = 1).rank(method = 'min') * rule_percent_chain[1]
    disc_agg = disc_df.set_index("CustTier").sum(axis = 1).rank(method = 'min') * rule_percent_chain[2]

    # aggregate all metrics for each customer
    cust_score = clv_agg + loy_agg + disc_agg
    cust_score = cust_score.rank(method = 'dense', ascending = False).astype(int).map(lambda tier: tier_assignments[tier-1]).to_dict()
    cust_score = {int(k): v for k, v in cust_score.items()}
    return cust_score

# Get 95 Confidence Interval for each numeric columns: 
# Check whether this column has enough observation
def get_conf_interval_numeric_col (tier_df, col):
    vals = tier_df.loc[:,col].values
    mue, std= np.mean(vals), np.std(vals)
    std_err = std / np.sqrt(len(vals))
    normality_flag = None

    ci = None
    statistics = None
    if len(vals) >= 100:                            # Apply z test for 95 % confidence interval 
        z_critical = stats.norm.ppf(0.5 + 0.95/2)
        ci = (mue - z_critical * std_err, mue + z_critical * std_err)
        statistics = (mue, f"[{ci[0]} - {ci[1]}]")
    elif len(vals) >= 10:
        # check for normality 
        _, p_val = shapiro(vals)
        normality_flag = 1 if p_val > 0.05 else 0 
        if normality_flag == 1:                     # Apply t test for 95 % confidence interval 
            t_critical = stats.t.ppf(0.5 + 0.95/2, len(vals) -1 )
            ci = (mue - t_critical * std_err, mue + t_critical * std_err)
            statistics = (mue, f"[{ci[0]} - {ci[1]}]")
    if normality_flag == 0 or len(vals) < 10:       # extremely samll sample size --> use min / max instead
        return (np.mean(vals), f"[{np.min(vals)} - {np.max(vals)}]")
    return statistics

# Extract distribution out of categorical column 
def get_cat_col_distribution (tier_df, col):
    vals = tier_df.loc[:,col].values
    tot_cnt = len(vals)
    
    uniq_cat, cat_freq = np.unique(vals, return_counts = True, equal_nan=True)
    cat_distr = [str(p) + '%' for p in np.round(cat_freq / tot_cnt * 100,1)]
    uniq_cat = [f"{c} category %" for c in uniq_cat]
    return uniq_cat, cat_distr

# Depict statiscal properties for different customer tier
# lookup --> dict (key: tier_id, value: list of customers)
# cust_df --> orginial input df from supabase (all metrics for each customer)
def depict_tier_profile(lookup, cust_df):
    tierId = [f"{int(t)}" for t in lookup]
    df1 = pd.DataFrame(tierId, columns=['CustTier'])

    # new profile vals and columns
    profileVals = []
    profile_cols = []

    # Iterate over each customer segments
    for tierId, customers in lookup.items():   
        tier_level_features = []
        for col in cust_df.columns[1:]:                       # iterate over col 
            # print(f"calc CI on {col}\n")
            tier_df = cust_df[cust_df.customer.isin(customers)]
            if col != 'cust_cat':
                ci = get_conf_interval_numeric_col(tier_df, col)    # get 95 conf interval for numeric fields
                tier_level_features.extend(ci)
                if tierId == 0:
                    profile_cols.extend([f"{col} (mue)", f"{col} (95% Conf Interval)"])
            # else:                                            # categorical col
            #     cat_cols, cat_values = get_cat_col_distribution(tier_df, col)
            #     tier_level_features.extend(cat_values)
            #     if tierId == 0:
            #         profile_cols.extend(cat_cols)
        profileVals.append(tier_level_features)

    feature_profile = pd.DataFrame(profileVals, columns= profile_cols)
    # display(feature_profile)
    tier_profile = pd.concat([df1, feature_profile], axis = 1)
    # writeCustTier2excel(tier_profile, file_path, tab)
    return tier_profile                             # Return profile statistics dataframe

# Testing Section Below for this module
if __name__ == "__main__":
    cust_df = read_custTier_from_supabase()
    # train kmeans
    kmeans_custTier_df, kmeans_custTier_lookup = k_means_fine_tune(cust_df)
    kmeans_profile_df = depict_tier_profile(kmeans_custTier_lookup, cust_df)
    tier_mapping = rule_based_tier_mapping(kmeans_profile_df, [0.7,0.2,0.1])
    print(f"k-means tier mapping: {tier_mapping}\n")
    kmeans_custTier_df["cust tier"] = kmeans_custTier_df["cust tier"].apply(lambda lab: tier_mapping[int(lab)])
    kmeans_profile_df["CustTier"] = kmeans_profile_df["CustTier"].apply(lambda lab: tier_mapping[int(lab)])
    writeCustTier2excel(kmeans_custTier_df, 'custTier.xlsx', "cust tier (kmeans)")
    writeCustTier2excel(kmeans_profile_df, 'custTier.xlsx', 'tier profile (kmeans)')
    write2txt (kmeans_custTier_lookup, "custTier.txt", "w", "k-means")
    # load the custTier res to supabase
    load_df_2_supbase(kmeans_custTier_df, "kMeans_custTier_output")

    # train gmm 
    gmm_custTier_df, gmm_custTier_lookup = gmm_fine_tune(cust_df)
    gmm_profile_df = depict_tier_profile(gmm_custTier_lookup, cust_df)
    tier_mapping = rule_based_tier_mapping(gmm_profile_df, [0.7,0.2,0.1])
    print(f"GMM tier mapping: {tier_mapping}\n")
    gmm_custTier_df["cust tier"] = gmm_custTier_df["cust tier"].apply(lambda lab: tier_mapping[int(lab)])
    gmm_profile_df["CustTier"] = gmm_profile_df["CustTier"].apply(lambda lab: tier_mapping[int(lab)])
    writeCustTier2excel(gmm_custTier_df, 'custTier.xlsx', "cust tier (gmm)")
    writeCustTier2excel(gmm_profile_df, 'custTier.xlsx', 'tier profile (gmm)')
    write2txt (gmm_custTier_lookup, "custTier.txt", "a", "gmm")
    load_df_2_supbase(gmm_custTier_df, "gmm_custTier_output")

