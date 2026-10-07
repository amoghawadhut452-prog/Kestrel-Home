"""CLI scorer: python src/predict.py path/to/orders.csv path/to/out.csv
Works with no other file from data/ (needs model_coefficients.json only)."""
import sys, argparse
sys.path.insert(0, __file__.rsplit('/', 1)[0].rsplit('\\', 1)[0] if False else __file__[:max(__file__.rfind('/'), __file__.rfind('\\'))])
import pandas as pd
from features import fix_value, features
import scorer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_csv')
    ap.add_argument('output_csv')
    a = ap.parse_args()
    df = pd.read_csv(a.input_csv)
    # minimal partners mimics training cleaning: value fix needs products list,
    # so predict also reads products.csv from the same folder as the input
    import os
    prod_path = os.path.join(os.path.dirname(os.path.abspath(a.input_csv)), 'products.csv')
    prod = pd.read_csv(prod_path) if os.path.exists(prod_path) else None
    if prod is not None:
        df = fix_value(df, prod)
        cust_path = os.path.join(os.path.dirname(os.path.abspath(a.input_csv)), 'customers.csv')
        cust = pd.read_csv(cust_path) if os.path.exists(cust_path) else pd.DataFrame(columns=['customer_id','city','state','signup_date','shield_member'])
    else:
        cust = pd.DataFrame(columns=['customer_id','city','state','signup_date','shield_member'])
        prod = pd.DataFrame(columns=['sku','family','model_name','list_price_inr','warranty_months','launch_date'])
        df = df.copy()
        df['order_value_inr'] = df['order_value_inr'].astype(float)
        df['value_fixed'] = 0
    ftr = features(df, cust, prod)
    p = scorer.predict_proba(ftr, scorer.load_coeffs())
    out = pd.DataFrame({'order_id': ftr['order_id'], 'score': p})
    out.to_csv(a.output_csv, index=False)
    print(f'wrote {a.output_csv}: {len(out)} rows')

if __name__ == '__main__':
    main()
