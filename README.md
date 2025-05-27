# azure-db-cron-agent

## 📂 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview 🧠

Shared package defining all OneDrive read/write ETL modules used across the modelling package and 5 cron task packages.

**Usage Notes**:
- When testing a single task (e.g., `modelling` in `dev1`), ensure `runOnStartup` of respective `function.json` is set to `False` for the other 4 tasks.
- For faster testing, lines 36–37 in `advAnalyticsModel.py` are commented out. Uncomment before merging into `main`.
- `advAnalyticsModel.py` defines the base class for modelling. Other models inherit from it and override abstract methods.
---

## 🚀 Features


---

## 🛠 Installation

```bash
# Clone this repository
git clone https://github.com/enerlites/azure-db-cron-agent.git
git checkout dev1

# Install dependencies
pip install -r requirements.txt
```

## Usage

To use this project, you need to set up the Azure Functions and configure the necessary settings in the `local.settings.json` file. Here's a brief overview of the usage:

1. **Set up Azure Functions**: Deploy the functions in this project to your Azure Functions instance.
2. **Configure `local.settings.json`**: This file is used for local development to store settings like connection strings and function app settings. An example file is provided as `local.settings.json.example`. Rename it to `local.settings.json` and update the values as needed.
3. **Run the functions**: You can run the functions locally using the Azure Functions Core Tools or deploy them to Azure and run them in the cloud.

For detailed usage instructions, refer to the README files and documentation within each function's folder.

---

## 🏗️ Project Structure

The project is organized into several folders, each responsible for a specific set of tasks or functionalities:

- **`dailyCompPricingTask/`**: This folder contains the Azure Function responsible for running the daily competitive pricing job. It fetches competitor pricing data and updates the database accordingly.
  - `function.json`: Configuration file for the Azure Function timer trigger, scheduled to run daily.
  - `__init__.py`: Python script that executes the `daily_comp_pricing_job` from the `shared` package.

- **`monthlyCustSegTask/`**: Houses the Azure Function for the monthly customer segmentation task. This task involves running clustering models (K-means, GMM) to segment customers based on their behavior and attributes.
  - `function.json`: Defines the timer trigger for the Azure Function, scheduled monthly.
  - `__init__.py`: Python script that initializes and runs the customer segmentation and demand forecasting models from the `modelling` package.

- **`monthlyERPTask/`**: Contains the Azure Function for the monthly ERP (Enterprise Resource Planning) data integration job.
  - `function.json`: Configuration for the Azure Function's monthly timer trigger.
  - `__init__.py`: Python script that executes the `monthly_netsuite_erp_job` from the `shared` package.

- **`monthlySkuPricingTask/`**: This folder includes the Azure Function for the monthly SKU (Stock Keeping Unit) internal pricing job.
  - `function.json`: Defines the monthly timer trigger for this Azure Function.
  - `__init__.py`: Python script that runs the `monthly_en_internal_pricing_job` from the `shared` package.

- **`monthlySkuPromoTask/`**: Contains the Azure Function responsible for generating monthly SKU promotion brochures.
  - `function.json`: Configuration for the Azure Function's monthly timer trigger.
  - `__init__.py`: Python script that executes the `monthly_promotion_brochure_job` from the `shared` package.

- **`modelling/`**: This directory contains the core data modelling scripts.
  - `advAnalyticsModel.py`: Defines an abstract base class `AdvAnalyticsModel` for advanced analytics models, providing common functionalities like database connection, data loading, and result saving.
  - `custTier.py`: Implements the `CustTierClustering` class, which inherits from `AdvAnalyticsModel`. It performs customer segmentation using K-means and Gaussian Mixture Models (GMM), including preprocessing, model fine-tuning, rule-based tier mapping, and profile depiction.
  - `predictDemand.py`: Implements the `DemandForecast` class, also inheriting from `AdvAnalyticsModel`. It preprocesses sales data and trains an XGBoost Regressor to forecast product demand.

- **`shared/`**: This package (implicitly, as it's imported by tasks and models) likely contains shared utility functions, database interaction modules (like `azureDBWriter.py`), and common job definitions used across different Azure Functions and modelling scripts. For example, it would contain the definitions for `daily_comp_pricing_job`, `monthly_netsuite_erp_job`, etc.

---

## 🔄 Workflow

The `azure-db-cron-agent` project orchestrates a series of automated tasks using Azure Functions. These functions are triggered on predefined schedules (daily or monthly) to perform data extraction, transformation, loading (ETL), and advanced analytics.

**General Flow:**

1.  **Scheduled Trigger**: Azure Functions are initiated by timer triggers as defined in their respective `function.json` files.
2.  **Function Execution**: The `__init__.py` script within each function's folder is executed.
3.  **Core Logic Invocation**:
    *   Most tasks invoke specific job functions from the `shared` package (e.g., `daily_comp_pricing_job`, `monthly_netsuite_erp_job`). These jobs handle interactions with external systems like OneDrive, NetSuite (indirectly), and perform data operations.
    *   The `monthlyCustSegTask` invokes models from the `modelling` package (`CustTierClustering` and `DemandForecast`). These models perform complex calculations, machine learning tasks, and data analysis.
4.  **Data Interaction**:
    *   The `modelling` package's base class, `AdvAnalyticsModel`, handles connections to the Azure SQL Database for fetching input data and storing model results.
    *   The `shared.azureDBWriter` module is likely used by various jobs for database write operations and sending email notifications with attachments.
5.  **Outputs & Results**:
    *   Processed data and model outputs are stored in the Azure SQL Database.
    *   Some tasks generate reports or files (e.g., promotion brochures) which might be sent via email or stored.
    *   Email notifications are sent for certain tasks, often including results as attachments.

**Text-based Visualization:**

```
+------------------------+
| Scheduled Triggers     |
| (NCRONTAB Expressions  |
|  in function.json)     |
+-----------+------------+
            | (Initiates)
            v
+------------------------+     +------------------------------------------------------+
| Azure Functions        |---->| Core Logic Packages                                  |
| (e.g.,                 |     |                                                      |
|  dailyCompPricingTask, |     |   +----------------------+   +---------------------+ |
|  monthlyCustSegTask,   |     |   | modelling/           |   | shared/             | |
|  monthlyERPTask,       |     |   | - CustTierClustering |   | - ETL Jobs (e.g.,   | |
|  monthlySkuPricingTask,|     |   | - DemandForecast     |   |   daily_comp_pricing| |
|  monthlySkuPromoTask)  |     |   | - AdvAnalyticsModel  |   |   _job)             | |
+------------------------+     |   +-------+--------------+   | - azureDBWriter     | |
                               |           | (DB I/O)         |   (DB I/O, Email)   | |
                               |           |                  +--------+------------+ |
                               |           +---------------------------+              |
                               +------------------------------------------------------+
                                           |
                                           | (Data I/O, File Operations, API Calls, Notifications)
                                           v
                  +------------------------+--------------------------+
                  | Data Stores & Outputs  | External Systems         |
                  +------------------------+--------------------------+
                  | - Azure SQL Database   | - OneDrive (via shared)  |
                  | - Email Notifications  | - NetSuite (indirectly   |
                  | - Excel/CSV Reports    |   via ERP job in shared) |
                  +------------------------+--------------------------+
```

---

## Configuration

Configuration settings for the Azure Functions are stored in the `local.settings.json` file. This file is not included in the repository for security reasons. You need to create it based on the provided `local.settings.json.example` file and update the values with your actual configuration.

Here's an overview of the settings you may need to configure:

- **AzureWebJobsStorage**: Connection string for the Azure Storage account used by the Functions host.
- **FUNCTIONS_WORKER_RUNTIME**: The runtime environment for the functions, typically set to `python`.
- **SqlConnectionString**: Connection string for the Azure SQL Database.
- **Other settings**: Depending on the functions and your environment, you may need to configure additional settings like API keys, service endpoints, etc.

For local development, you can use the `AzureWebJobsStorage` setting with a connection string to a local Azure Storage emulator or an actual Azure Storage account. Remember to update the connection strings and sensitive information before deploying to production.

---

## Testing

To test the project locally, follow these steps:

1. **Install Azure Functions Core Tools**: This tool allows you to run and debug Azure Functions locally. Install it following the [official guide](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local).
2. **Start the Storage Emulator** (if using local storage): If you're using the Azure Storage Emulator for local development, start it now.
3. **Run the Functions**: In the project root, run `func start` to start the Azure Functions host. This command will build the project and start all functions.
4. **Test the Endpoints**: Use tools like Postman or curl to test the HTTP endpoints of your functions. For timer-triggered functions, you can manually trigger them using the Azure Functions Core Tools or wait for the scheduled time.

For detailed testing instructions, refer to the README files within each function's folder.

---

## Results

The results of the ETL processes and analytics models are typically stored in an Azure SQL Database or another configured data sink. The exact location and format of the results depend on the specific implementation of each function and model.

For example, the customer segmentation model may store the segmentation results in a SQL table, while the demand forecasting model could save its results as a CSV file in a blob storage.

To access and analyze the results, you can connect to the Azure SQL Database using SQL Server Management Studio, Azure Data Studio, or any other SQL client tool. For blob storage, you can use Azure Storage Explorer or similar tools.

---

## Contributing

Contributions to this project are welcome! To contribute, follow these steps:

1. **Fork the repository**: Create a personal copy of the repository on GitHub by clicking the "Fork" button.
2. **Clone the forked repository**: Clone your forked repository to your local machine using `git clone <your-fork-url>`.
3. **Create a new branch**: Create a new branch for your feature or bug fix using `git checkout -b <branch-name>`.
4. **Make your changes**: Implement your feature or fix the bug in your local repository.
5. **Test your changes**: Thoroughly test your changes to ensure they work as expected and do not break any existing functionality.
6. **Commit and push**: Commit your changes with a clear message describing the update, and push the branch to your forked repository.
7. **Create a pull request**: In the original repository, click on "New pull request" and follow the prompts to create a pull request from your branch.

Please ensure your code adheres to the project's coding standards and includes appropriate tests for any new functionality. For significant changes, consider discussing your plans with the project maintainers before starting implementation.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
