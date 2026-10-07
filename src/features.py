"""Shared cleaning + dispatch-time feature engineering.

Rules: dedupe on order_id (crm row wins), fix Oct-2025 gateway values (x100),
only features knowable at dispatch are produced.
"""
import pandas as pd

LEAK_COLS = ['last_service_event_type', 'pickup_scheduled_at']
DROP_FOR_MODEL = []  # filled below

def load():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test_unlabelled.csv')
    cust = pd.read_csv('data/customers.csv')
    prod = pd.read_csv('data/products.csv')
    return train, test, cust, prod

def dedupe(train):
    # crm row wins; partner_feed rows are re-imports
    t = train.sort_values('source', ascending=True)  # 'crm' < 'partner_feed'
    return t.drop_duplicates('order_id', keep='first').reset_index(drop=True)

def fix_value(df, prod):
    m = df.merge(prod[['sku','list_price_inr']], on='sku', how='left')
    ratio = m['order_value_inr'] / (m['list_price_inr'] * m['qty'])
    bad = ratio > 5
    df = df.copy()
    df['order_value_inr'] = m['order_value_inr'].where(~bad, m['order_value_inr']/100.0)
    df['value_fixed'] = bad.astype(int)
    return df

def features(df, cust, prod):
    d = df.merge(cust, on='customer_id', how='left', validate='m:1')
    d = d.merge(prod, on='sku', how='left', validate='m:1')
    d['order_placed_at'] = pd.to_datetime(d['order_placed_at'])
    out = pd.DataFrame({
        'order_id': d['order_id'],
        'order_placed_at': d['order_placed_at'],
        'returned': d['returned'] if 'returned' in d else None,
        'sales_channel': d['sales_channel'],
        'payment_mode': d['payment_mode'],
        'discount_pct': d['discount_pct'].astype(float),
        'qty': d['qty'].astype(float),
        'order_value_inr': d['order_value_inr'].astype(float),
        'promised_delivery_days': d['promised_delivery_days'].astype(float),
        'default_pincode': (d['delivery_pincode'].astype(str).str.zfill(6) == '000000').astype(int),
        'is_gift': (d['is_gift'] == 'Y').astype(int),
        'customer_prior_orders': d['customer_prior_orders'].astype(float),
        'customer_prior_returns': d['customer_prior_returns'].astype(float),
        'shield_member': (d['shield_member'] == 'Y').astype(int),
        'family': d['family'],
        'state': d['state'],
        'value_ratio': (d['order_value_inr'] / d['list_price_inr']).clip(0, 20).fillna(1.0),
        'prior_return_rate': d['customer_prior_returns'] / (d['customer_prior_orders'] + 1),
    })
    return out

CAT = ['sales_channel', 'payment_mode', 'family', 'state']
NUM = ['discount_pct','qty','order_value_inr','promised_delivery_days','default_pincode',
       'is_gift','customer_prior_orders','customer_prior_returns','shield_member',
       'value_ratio','prior_return_rate']

def design(d, ref_cols=None):
    X = pd.get_dummies(d[CAT+NUM], columns=CAT)
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=0)
    return X
