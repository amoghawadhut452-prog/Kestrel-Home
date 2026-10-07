"""Exploratory pass: schema, shapes, injected-instruction scan, dedupe, joins."""
import pandas as pd, json, re

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test_unlabelled.csv')
cust = pd.read_csv('data/customers.csv')
prod = pd.read_csv('data/products.csv')
sub = pd.read_csv('data/sample_submission.csv')
print('shapes', train.shape, test.shape, cust.shape, prod.shape, sub.shape)

# injected instruction scan on all free text fields
pat = re.compile(r'(please do not|approved|dataset name|ignore|instruction|system prompt|final figure|do not drop|report as|label|metric|accuracy)', re.I)
hits = []
for df, name in [(train,'train'), (test,'test')]:
    for i, v in df['delivery_note'].fillna('').items():
        if pat.search(str(v)):
            hits.append((name, i, str(v)[:120]))
print('injection-like note count:', len(hits))
for h in hits[:5]: print(h)
print('sample notes:', train['delivery_note'].dropna().unique()[:5])

# dedupe
dup = train['order_id'].duplicated().sum()
print('train dup order_ids:', dup)
t2 = train.drop_duplicates('order_id', keep='first')
print('after dedupe (keep first):', t2.shape, 'returned rate', t2['returned'].mean())
t_crm = train[train.source=='crm']
print('crm-only rows:', len(t_crm), 'unique ids', t_crm.order_id.nunique(), 'ret rate', t_crm.returned.mean())

# join checks
m = t2.merge(cust, on='customer_id', how='left', indicator=True)
print('customer join unmatched:', (m._merge!='both').sum())
m2 = t2.merge(prod, on='sku', how='left', indicator=True)
print('product join unmatched:', (m2._merge!='both').sum())
print('test join cust/prod:', test.merge(cust,on='customer_id',how='left',indicator=True)._merge.eq('both').mean(),
      test.merge(prod,on='sku',how='left',indicator=True)._merge.eq('both').mean())

# dates
for df,n in [(train,'train'),(test,'test')]:
    d = pd.to_datetime(df.order_placed_at)
    print(n, d.min(), d.max())
print('returned rate overall', train.returned.mean(), t_crm.returned.mean())
print('test order_ids dup:', test.order_id.duplicated().sum())
print('test service event values:', test.last_service_event_type.value_counts().to_dict())
print('train service event values:', train.last_service_event_type.value_counts().to_dict())
print('pickup nonnull train/test:', train.pickup_scheduled_at.notna().sum(), test.pickup_scheduled_at.notna().sum())
