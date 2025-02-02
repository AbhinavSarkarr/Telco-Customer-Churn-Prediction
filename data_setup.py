import pandas as pd
import sqlite3
from pathlib import Path

def setup_customer_churn_database(csv_path: str, db_path: str = 'customer_churn.db'):
    """
    Set up SQLite database with customer churn data.
    """
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)
    
    # Convert TotalCharges to numeric, handling any whitespace
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    expected_columns = {
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    }

    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {expected_columns}")
    
    print(f"Found {len(df)} customer records")
    
    print("Creating database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop table if exists to avoid duplicate entries
    cursor.execute("DROP TABLE IF EXISTS customer_churn")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_churn (
        customerID TEXT PRIMARY KEY,
        gender TEXT,
        SeniorCitizen INTEGER,
        Partner TEXT,
        Dependents TEXT,
        tenure INTEGER,
        PhoneService TEXT,
        MultipleLines TEXT,
        InternetService TEXT,
        OnlineSecurity TEXT,
        OnlineBackup TEXT,
        DeviceProtection TEXT,
        TechSupport TEXT,
        StreamingTV TEXT,
        StreamingMovies TEXT,
        Contract TEXT,
        PaperlessBilling TEXT,
        PaymentMethod TEXT,
        MonthlyCharges REAL,
        TotalCharges REAL,
        Churn TEXT
    )
    """)
    
    batch_size = 1000
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    
    print("Inserting data into database...")
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        
        batch_data = df.iloc[start_idx:end_idx]
        
        # Removed if_index parameter and kept only the necessary ones
        batch_data.to_sql('customer_churn', 
                         conn, 
                         if_exists='append', 
                         index=False)
        
        print(f"Processed batch {i+1}/{total_batches}")
    
    print("Creating indices...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_churn ON customer_churn(Churn)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_contract ON customer_churn(Contract)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_internet_service ON customer_churn(InternetService)")
    
    cursor.execute("SELECT COUNT(*) FROM customer_churn")
    count = cursor.fetchone()[0]
    print(f"\nVerification: {count} customer records inserted into database")
    
    # Print some useful statistics
    print("\nChurn distribution:")
    cursor.execute("""
    SELECT Churn, COUNT(*) as count, 
           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customer_churn), 2) as percentage
    FROM customer_churn 
    GROUP BY Churn
    """)
    distribution = cursor.fetchall()
    for churn, count, percentage in distribution:
        print(f"{churn}: {count} ({percentage}%)")
        
    print("\nContract type distribution:")
    cursor.execute("""
    SELECT Contract, COUNT(*) as count,
           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customer_churn), 2) as percentage
    FROM customer_churn 
    GROUP BY Contract
    """)
    contract_dist = cursor.fetchall()
    for contract, count, percentage in contract_dist:
        print(f"{contract}: {count} ({percentage}%)")
    
    print("\nAverage monthly charges by contract type:")
    cursor.execute("""
    SELECT Contract, 
           ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges,
           ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100, 2) as churn_rate
    FROM customer_churn 
    GROUP BY Contract
    """)
    charges_dist = cursor.fetchall()
    for contract, avg_charges, churn_rate in charges_dist:
        print(f"{contract}: ${avg_charges} (Churn Rate: {churn_rate}%)")
    
    conn.close()
    print("\nDatabase setup complete!")

if __name__ == "__main__":
    csv_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    setup_customer_churn_database(csv_file)