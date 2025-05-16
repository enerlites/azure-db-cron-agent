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
import urllib
import logging
from shared.azureDBWriter import *                          # import packages and modules

class CustTierClustering:
    # Read model input table from Azure db to Python Memory
    def __init__(self):
        load_dotenv()           # load the .env vars
        ODBC_18_CONN = urllib.parse.quote_plus(
            f"Driver={{ODBC Driver 18 for SQL Server}};"
            f"Server=tcp:{os.getenv('DB_SERVER')},1433;"
            f"Database=enerlitesDB;"
            f"Uid=sqladmin;"
            f"Pwd={os.getenv('DB_PASS')};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )
        self.DB_CONN = f"mssql+pyodbc:///?odbc_connect={ODBC_18_CONN}"
        self.AZ_ENGINE = create_engine(self.DB_CONN)
        self.SQLQuery = 'SELECT * FROM modeling.customer_tier_input;'
        with self.AZ_ENGINE.connect() as conn:
            self.inputDf = pd.read_sql(self.SQLQuery, conn)         # store modeling input (read from db)

        # set up class level logger in Azure Prod env
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    # Load the cust Tier result back to Supabase
    def loadModelResBack2DB(self, outputDf, schema, tableName):
        with self.AZ_ENGINE.connect() as conn:
            outputDf.to_sql(tableName, con = conn, schema=schema, if_exists = "replace", index =False)
        self.logger.info(f"[AZURE] loadModelResBack2DB LOAD {outputDf.shape} 2 \'{tableName}\'!\n")

    # Define model pipeline before model invocation
    def preprocess_pip(self):
        cust_df = self.inputDf
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
    def k_means_fine_tune(self):
        cust_df = self.inputDf.copy()
        cus_series, preprocessor = self.preprocess_pip()
        
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
        self.logger.info(f"k-means model: optimal k = {opt_k} with max silhouette score = {np.max(silscore_stats)}\n")
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
    def gmm_fine_tune(self):
        cust_df = self.inputDf.copy()
        cus_series, preprocessor = self.preprocess_pip()

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
        self.logger.info(f"gmm model: optimal k = {opt_k} with max silhouette score = {np.max(silscore_stats)}\n")
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
    def rule_based_tier_mapping(self, tier_profile_df, rule_percent_chain):
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
    def get_conf_interval_numeric_col (self, tier_df, col):
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
    def get_cat_col_distribution (self, tier_df, col):
        vals = tier_df.loc[:,col].values
        tot_cnt = len(vals)
        
        uniq_cat, cat_freq = np.unique(vals, return_counts = True, equal_nan=True)
        cat_distr = [str(p) + '%' for p in np.round(cat_freq / tot_cnt * 100,1)]
        uniq_cat = [f"{c} category %" for c in uniq_cat]
        return uniq_cat, cat_distr

    # Depict statiscal properties for different customer tier
    # lookup --> dict (key: tier_id, value: list of customers)
    # cust_df --> orginial input df from supabase (all metrics for each customer)
    def depict_tier_profile(self, lookup):
        cust_df = self.inputDf
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
                    ci = self.get_conf_interval_numeric_col(tier_df, col)    # get 95 conf interval for numeric fields
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

    # wrong this func for everything
    def annual_customer_segmentation(self):
        # train kmeans
        kmeans_custTier_df, kmeans_custTier_lookup = self.k_means_fine_tune()
        kmeans_profile_df = self.depict_tier_profile(kmeans_custTier_lookup)
        # Return a lookup table 
        tier_mapping = self.rule_based_tier_mapping(kmeans_profile_df, [0.7,0.2,0.1])
        self.logger.info(f"k-means tier mapping: {tier_mapping}\n")
        kmeans_custTier_df["cust tier"] = kmeans_custTier_df["cust tier"].apply(lambda lab: tier_mapping[int(lab)])
        kmeans_profile_df["CustTier"] = kmeans_profile_df["CustTier"].apply(lambda lab: tier_mapping[int(lab)])

        # load Kmeans result back to db
        self.loadModelResBack2DB(kmeans_custTier_df, "modeling", "kMeans_custTier_output")
        # load Kmeans profile back to db
        self.loadModelResBack2DB(kmeans_profile_df, "modeling", "kMeans_custTier_profile_statistics")

        # train gmm 
        gmm_custTier_df, gmm_custTier_lookup = self.gmm_fine_tune()
        gmm_profile_df = self.depict_tier_profile(gmm_custTier_lookup)
        tier_mapping = self.rule_based_tier_mapping(gmm_profile_df, [0.7,0.2,0.1])
        self.logger.info(f"GMM tier mapping: {tier_mapping}\n")
        gmm_custTier_df["cust tier"] = gmm_custTier_df["cust tier"].apply(lambda lab: tier_mapping[int(lab)])
        gmm_profile_df["CustTier"] = gmm_profile_df["CustTier"].apply(lambda lab: tier_mapping[int(lab)])

        # load gmm result back to db
        self.loadModelResBack2DB(gmm_custTier_df, "modeling", "gmm_custTier_output")
        # load gmm profile back to db
        self.loadModelResBack2DB(gmm_profile_df, "modeling", "gmm_custTier_profile_statistics")

        # Send all model result as attachments in 1 email
        attDfs = [kmeans_custTier_df,kmeans_profile_df,gmm_custTier_df,gmm_profile_df]
        attNames = ['Customer Tier Classificiation (K-means).xlsx','Customer Tier Profile (K-means).xlsx','Customer Tier Classificiation (GMM).xlsx','Customer Tier Profile (GMM).xlsx']
        myDBWriter = AzureDBWriter(None, None)
        myDBWriter.auto_send_email(outputDfs=attDfs, attachmentNames=attNames, emailSubject="Annual Cust Tier NO-REPLY")

# Testing Section Below for this module
if __name__ == "__main__":
    pass
    # myClustering = CustTierClustering()
    # myClustering.annual_customer_segmentation()

