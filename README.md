# azure-db-cron-agent

## 📂 Table of Contents

- [Overview](#overview)
    5 Cron Schedulers to run based out of LA time
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

shared package define all OneDrive read, write, ETL modules that are widely used in modelling package and 5 cron task packages.
a) When testing a single task at dev1 for example modelling functions, make sure that "runOnStartup" is set to False in 4 other tasks.
b) Additionally, for faster testing, line 36-37 of advAnalyticsModel.py is commented. Make sure to uncomment this if you ever want to merge main.
c) advAnalyticsModel.py defines parent class for modelling. The other models inherit from this class and overwrite abstract functions. 
---

## 🚀 Features


---

## 🛠 Installation

```bash
# Clone this repository
git clone https://github.com/your-username/your-project.git
git checkout dev1

# Install dependencies
pip install -r requirements.txt
