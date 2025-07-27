This project implements an advanced Decentralized Finance (DeFi) wallet risk scorer. It integrates real-time transaction history fetching from Etherscan with sophisticated risk assessment algorithms. The system preprocesses transaction data and calculates a comprehensive risk score for each wallet, offering both traditional DeFi metric-based analysis and a novel synthetic scoring approach for wallets with limited on-chain history.

Features
Etherscan Data Integration: Fetches historical transactions for given Ethereum wallet addresses using the Etherscan API.

Compound Protocol Focus: Specifically filters and analyzes transactions related to Compound V2 and V3 (Comptroller and Comet USDC) and key cToken contracts.

Comprehensive Wallet Metrics: Calculates crucial metrics from real transaction data, including:

Total Borrow/Supply Volume

Net Borrow/Supply Ratio

Liquidation Count and Frequency

Maximum Loan-to-Value (LTV) Ratio

Active/Inactive Days

Volatility Exposure based on token types

Transaction Count and Average Transaction Size

Hybrid Risk Scoring Model:

Traditional Scoring: For wallets with sufficient transaction data, risk is assessed based on a weighted sum of liquidation events, borrow/supply ratios, utilization, inactivity, and volatility exposure.

Synthetic Scoring: For wallets with little to no on-chain activity, a novel synthetic risk score is generated using:

Address Entropy Analysis: Measures the randomness and complexity of the wallet address.

Address Pattern Analysis: Identifies suspicious patterns (e.g., consecutive characters, unusual hex digit distribution) within the address.

Distribution Factor: Assesses the address's position relative to a sorted list of other addresses.

Volatility Factor: Adds a controlled, address-seeded random component.

Batch Processing with Concurrency: Efficiently analyzes multiple wallet addresses in parallel using ThreadPoolExecutor, with fallback to sequential processing for robustness.

pip install --r requirements.txt 
Installation
Clone the repository:

Bash

git clone https://github.com/Ashwinxxx/Enhanced_Defi_Wallet_Scorer.git
cd Enhanced_Defi_Wallet_Scorer
Etherscan API Key:
The project currently has an Etherscan API key hardcoded within EnhancedDeFiDataCollector.__init__. For production use or better security practices, it's highly recommended to:

Obtain your own free Etherscan API Key. Which i have  given

Consider passing it as an environment variable or loading it from a configuration file instead of hardcoding it. For instance, you could modify EnhancedDeFiDataCollector to accept an API key during initialization or read it from an environment variable:

Python

# In EnhancedDeFiDataCollector.__init__
self.etherscan_api_key = os.getenv("ETHERSCAN_API_KEY", "YOUR_HARDCODED_KEY_HERE_FOR_DEVELOPMENT")
And then set the environment variable: export ETHERSCAN_API_KEY="YOUR_ACTUAL_KEY"

Usage
To run the wallet risk analysis:

Navigate to the project directory in your terminal.

Execute the main script:

Bash

python risk_scorer.py
The script will:

Start analyzing the predefined list of 100 wallet addresses.

Print progress for each wallet.

Display the top 20 highest-risk wallets with their scores and key metrics.

Show a summary of the risk analysis, including average/median scores and distribution across risk categories.

Save the complete results to a CSV file named defi_risk_analysis_YYYYMMDD_HHMMSS.csv in the project directory.

How It Works
The project is structured into three main classes:

EnhancedDeFiDataCollector
This class is responsible for interacting with the Etherscan API to retrieve blockchain data.

It fetches up to the 100 most recent transactions for a given wallet address, with a fallback start block for efficiency.

It specifically identifies and filters transactions that interact with known Compound V2, V3, and cToken contract addresses.

Includes fallback mechanisms for current block number retrieval and robust error handling for API calls.

HybridRiskScoreCalculator
This class determines the risk score for a wallet using a hybrid approach:

Traditional DeFi Risk Calculation:

Liquidation Risk: Based on the count and frequency of liquidation events.

Borrow/Supply Ratio Risk: Evaluates the ratio of borrowed assets to supplied assets.

Utilization Risk: Reflects the maximum Loan-to-Value (LTV) ratio.

Inactivity Risk: Assesses how long a wallet with borrowed funds has been inactive.

Volatility Exposure Risk: Considers the average volatility score of tokens involved in transactions.
These factors are weighted to produce a final risk score between 0 and 1000.

Synthetic Risk Calculation (for inactive/new wallets):
If a wallet lacks sufficient on-chain data (no significant borrow/supply volume or transactions), a synthetic score is generated using features derived directly from the wallet address itself:

Address Entropy: A measure of the randomness of the hexadecimal characters in the address.

Address Patterns: Detects repeating sequences or unusual distributions of characters.

Distribution Factor: Assigns a risk based on the address's sorted position among other analyzed addresses.

Volatility Factor: Introduces a controlled random element, seeded by the address, to add diversity to scores.

WalletAnalyzer
This is the orchestrator class that ties everything together.

It uses EnhancedDeFiDataCollector to get transaction data.

If real transaction data is available, it calculates detailed WalletMetrics based on on-chain activity.

If real data is scarce, it generates "synthetic" metrics to ensure a score can still be provided.

It then passes these metrics to the HybridRiskScoreCalculator to obtain the final risk score.

The analyze_wallets_batch method enables concurrent analysis of multiple wallets, significantly speeding up the process, with sequential fallback for reliability.

Output
The script generates a comprehensive CSV file (e.g., defi_risk_analysis_20250727_102025.csv) containing the following columns for each analyzed wallet:

wallet_id: The Ethereum wallet address.

score: The calculated risk score (0-1000).

total_borrow_volume: Total volume of assets borrowed (in ETH equivalent).

total_supply_volume: Total volume of assets supplied (in ETH equivalent).

borrow_supply_ratio: Ratio of borrowed to supplied volume.

liquidation_count: Number of liquidation events.

max_ltv_ratio: Maximum Loan-to-Value ratio observed.

transaction_count: Total number of transactions.

volatility_exposure: Average volatility exposure of assets.

active_days: Number of unique days with activity.

The console output provides a quick summary, highlighting the top 20 riskiest wallets and a breakdown of the overall risk distribution.

Detailed Output: Prints top high-risk wallets, summary statistics (average/median score, risk categories), and a risk score distribution.

CSV Export: Saves all analysis results, including detailed metrics and risk scores, to a time-stamped CSV file.

Robust Error Handling: Includes logging and graceful error handling for API calls and data processing.


