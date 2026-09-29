"""Fixed semantic encoding from the evaluated numeric-count baseline."""
def classify(train, full, policy):
 import numpy as np
 rows=[];discrete=[]
 counts=set(policy['numeric_count_columns']);events=set(policy['event_indicator_columns_require_train_binary']);categories=set(policy['categorical_columns'])
 if (counts&events) or (counts&categories) or (events&categories):raise ValueError('Conflicting semantic roles.')
 for col in train:
  x=full[col].to_numpy(dtype=float);x=x[np.isfinite(x)]
  values=np.unique(x)
  if col in counts:
   if not len(x) or (x<0).any() or not np.isclose(x,np.rint(x),rtol=0,atol=1e-8).all():
    raise ValueError('Count field has unexpected support: '+col)
   role='numeric_count';why='Named protocol count; integer-valued does not imply categorical.'
  elif col in events:
   if not len(values) or not set(values).issubset({0.,1.}):raise ValueError('Expected TRAIN binary indicator has nonbinary support: '+col)
   role='binary';why='Event-indicator role and full TRAIN support restricted to 0/1.';discrete.append(col)
  elif col in categories:
   role='categorical';why='Explicit semantic categorical declaration.';discrete.append(col)
  else:raise ValueError('Unknown field: '+col)
  rows.append({'column':col,'encoding':role,'ctgan_discrete':col in discrete,'TRAIN_unique_values':len(values),
               'fit_subset_unique_values':int(train[col].nunique()),'reason':why})
 return discrete,rows
