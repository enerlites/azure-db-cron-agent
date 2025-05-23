# azure-db-cron-agent

## 📂 Table of Contents

- [Overview](#overview)
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

## Overview 🧠

Shared package defining all OneDrive read/write ETL modules used across the modelling package and 5 cron task packages.

**Usage Notes**:
- When testing a single task (e.g., `modelling` in `dev1`), ensure `runOnStartup` is set to `False` for the other 4 tasks.
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
