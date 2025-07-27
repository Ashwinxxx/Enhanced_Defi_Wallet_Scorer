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

Detailed Output: Prints top high-risk wallets, summary statistics (average/median score, risk categories), and a risk score distribution.

CSV Export: Saves all analysis results, including detailed metrics and risk scores, to a time-stamped CSV file.

Robust Error Handling: Includes logging and graceful error handling for API calls and data processing.


