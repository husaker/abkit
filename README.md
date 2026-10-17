# jupyter lab runs from L4_ds folder
uv run --project abkit --with jupyter jupyter lab

Data decisions. All 20,692,840 events are kept.

Rows without a session (4,598, none are purchases): 
kept in events and user_months; left out of sessions, because they have no session to belong to.

Negative prices (126 purchases, −0.06% of revenue): 
kept as recorded.

Exact duplicates (1.1M, 89% remove_from_cart; purchases 0.05% of revenue): 
kept. Simulated user-level splits put them in both groups equally. Revisit for cart-based metrics in chapter 5.