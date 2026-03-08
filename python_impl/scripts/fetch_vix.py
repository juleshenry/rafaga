import yfinance as yf
import pandas as pd
import argparse
import os

def fetch_vix_data(start_date="2004-01-01", end_date=None, output_file="data/vix_historical.csv"):
    """
    Fetches historical VIX data from Yahoo Finance.
    The CBOE launched VIX futures around 2004.
    """
    print(f"Fetching ^VIX data from {start_date} to {end_date if end_date else 'today'}...")
    
    # Download data
    vix_data = yf.download("^VIX", start=start_date, end=end_date)
    
    if vix_data.empty:
        print("No data fetched. Check your connection or date range.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    vix_data.to_csv(output_file)
    print(f"Successfully saved {len(vix_data)} trading days of VIX data to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical VIX data.")
    parser.add_argument("--start", type=str, default="2004-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="../data/vix_historical.csv", help="Output CSV path")
    
    args = parser.parse_args()
    fetch_vix_data(start_date=args.start, end_date=args.end, output_file=args.output)
