import torch

item_seq = torch.tensor([[1., 2., 3., 4., 0.], [2., 4., 6., 8., 0.]], dtype=torch.float32)
behavior_seq = torch.tensor([[1., 2., 1., 2., 0.], [1., 2., 1., 2., 0.]], dtype=torch.float32)
#
batch_size, seq_len = behavior_seq.size()
#
prev_purchase = torch.zeros((batch_size, seq_len), dtype=torch.long)
next_purchase = torch.zeros((batch_size, seq_len), dtype=torch.long)
prev_next_purchase = torch.zeros((batch_size, seq_len, 2), dtype=torch.long)
last_purchase = torch.zeros(batch_size, dtype=torch.long)

for i in range(seq_len):
    # Get the current item and behavior
    item = item_seq[:, i]
    behavior = behavior_seq[:, i]
    purchase_mask = (behavior == 2)
    view_mask = (behavior == 1)

    prev_purchase[:, i] = torch.where(last_purchase != 0, last_purchase, prev_purchase[:, i])

    last_purchase_mask = (last_purchase == 2)
    mask = purchase_mask ^ last_purchase_mask
    last_purchase = torch.where(purchase_mask, item, last_purchase)
    # last_purchase = torch.where(mask, item, last_purchase)

    # Find the next purchase for each item
    sub_item_seq = item_seq[:, i+1:]
    sub_behavior_seq = behavior_seq[:, i+1:]
    next_purchase_mask = (sub_behavior_seq==2) * sub_behavior_seq
    if next_purchase_mask.size(1) != 0:
        next_purchase_indices = next_purchase_mask.max(dim=1).indices
        next_purchase[:, i] = sub_item_seq[torch.arange(len(next_purchase_indices)), next_purchase_indices]

    prev_next_purchase[:, i, 0] = prev_purchase[:, i]
    prev_next_purchase[:, i, 1] = next_purchase[:, i]

print("11111111111111")

