"""FastAPI scoring service + static HTML page."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Any
import scorer

app = FastAPI(title='Kestrel Return-Risk Scorer')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s_root = os.path.join(_ROOT, 'src')
import sys as _s; _s.path.insert(0, s_root)
COEFFS = scorer.load_coeffs()

def _load_lookup(path, cols):
    if os.path.exists(path):
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame(columns=cols)

CUST = _load_lookup(os.path.join(_ROOT, 'data', 'customers.csv'),
                    ['customer_id','city','state','signup_date','shield_member'])
PROD = _load_lookup(os.path.join(_ROOT, 'data', 'products.csv'),
                    ['sku','family','model_name','list_price_inr','warranty_months','launch_date'])

class Order(BaseModel):
    order_id: Optional[str] = 'UNKNOWN'
    order_placed_at: Optional[Any] = None
    customer_id: Optional[str] = None
    sku: Optional[str] = None
    sales_channel: Optional[str] = None
    payment_mode: Optional[str] = None
    discount_pct: Optional[Any] = 0
    qty: Optional[Any] = 1
    order_value_inr: Optional[Any] = None
    promised_delivery_days: Optional[Any] = 5
    delivery_pincode: Optional[Any] = '110001'
    is_gift: Optional[Any] = 'N'
    customer_prior_orders: Optional[Any] = 0
    customer_prior_returns: Optional[Any] = 0
    shield_member: Optional[Any] = None

def _num(v, default, lo, hi):
    try: x = float(v)
    except (TypeError, ValueError): x = default
    if not np.isfinite(x): x = default
    return min(max(x, lo), hi)

REASON_PHRASES = {
    'payment_mode_cod': 'cash on delivery orders return more often',
    'payment_mode_emi': 'EMI-paid orders return more often',
    'is_gift_1': 'it is marked as a gift, which returns more often',
    'default_pincode': 'no full address was captured (default pincode)',
    'customer_prior_returns': 'this customer has returned before',
    'prior_return_rate': 'a high share of this customer’s past orders came back',
    'shield_member': 'Shield member (free returns, so returns are more likely)',
    'discount_pct': 'a deep discount on the order',
}

def reasons_for(ftr):
    return scorer.reasons(ftr, COEFFS)

@app.post('/score')
def score(order: Order):
    try:
        row = order.model_dump()
        cust_lookup = CUST[CUST.customer_id == row['customer_id']]
        shield = row['shield_member']
        if shield is None or (isinstance(shield, str) and shield.strip() == ''):
            shield = cust_lookup.shield_member.iloc[0] if len(cust_lookup) else 'N'
        shield = 'Y' if str(shield).upper() in ('Y','YES','TRUE','1') else 'N'
        prod_lookup = PROD[PROD.sku == row['sku']]
        list_price = float(prod_lookup.list_price_inr.iloc[0]) if len(prod_lookup) else None
        try: val = float(row['order_value_inr'])
        except (TypeError, ValueError): val = list_price if list_price else 0.0
        qty = _num(row['qty'], 1, 1, 100)
        if list_price and val > 5 * list_price * qty: val /= 100.0  # Oct-style gateway fix
        df = pd.DataFrame([{
            'order_id': row['order_id'] or 'UNKNOWN',
            'order_placed_at': pd.to_datetime(row['order_placed_at'], errors='coerce') if row['order_placed_at'] else pd.Timestamp('2026-01-15'),
            'sales_channel': row['sales_channel'] or 'app',
            'payment_mode': row['payment_mode'] or 'prepaid_upi',
            'discount_pct': _num(row['discount_pct'], 0, 0, 100), 'qty': qty, 'order_value_inr': val,
            'promised_delivery_days': _num(row['promised_delivery_days'], 5, 0, 30),
            'delivery_pincode': str(row['delivery_pincode'] if row['delivery_pincode'] is not None else '000000').zfill(6),
            'is_gift': row['is_gift'],
            'customer_prior_orders': _num(row['customer_prior_orders'], 0, 0, 100000),
            'customer_prior_returns': _num(row['customer_prior_returns'], 0, 0, 100000),
            'shield_member': shield,
            'city': cust_lookup.city.iloc[0] if len(cust_lookup) else 'Unknown',
            'state': cust_lookup.state.iloc[0] if len(cust_lookup) else 'Unknown',
            'family': prod_lookup.family.iloc[0] if len(prod_lookup) else 'Other',
            'list_price_inr': list_price if list_price else (val or 1.0),
        }])
        df['order_placed_at'] = df['order_placed_at'].fillna(pd.Timestamp('2026-01-15'))
        is_gift_y = str(row['is_gift']).upper() in ('Y','YES','TRUE','1')
        ftr = pd.DataFrame({
            'order_id': df.order_id, 'order_placed_at': df.order_placed_at, 'returned': None,
            'sales_channel': df.sales_channel, 'payment_mode': df.payment_mode,
            'discount_pct': df.discount_pct, 'qty': df.qty, 'order_value_inr': df.order_value_inr,
            'promised_delivery_days': df.promised_delivery_days,
            'default_pincode': (df.delivery_pincode.astype(str).str.zfill(6) == '000000').astype(int),
            'is_gift': int(is_gift_y),
            'customer_prior_orders': df.customer_prior_orders,
            'customer_prior_returns': df.customer_prior_returns,
            'shield_member': (df.shield_member == 'Y').astype(int),
            'family': df.family, 'state': df.state,
            'value_ratio': (df.order_value_inr / df.list_price_inr).clip(0, 20).fillna(0),
            'prior_return_rate': df.customer_prior_returns / (df.customer_prior_orders + 1),
        })
        p = float(scorer.predict_proba(ftr, COEFFS)[0])
        T = COEFFS['threshold']
        band = 'high' if p >= T else ('medium' if p >= T*0.42 else 'low')
        action = 'call' if p >= T else 'no action'
        return {'order_id': row['order_id'] or 'UNKNOWN', 'score': round(p, 4), 'risk_band': band,
                'recommended_action': action, 'reasons': reasons_for(ftr)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'could not score order: {e}')

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
@app.get('/', response_class=HTMLResponse)
def home():
    return open(HTML_PATH, encoding='utf-8').read()
