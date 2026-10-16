"""Revalidate receipts against immutable cases and actual review decisions."""
from .contrast_transforms import validate_contrast
from .transforms import validate_deletion
from .reviews import review_receipt


def check_receipt(store,r):
    a=store.object(r['parent_hash']);b=store.object(r['child_hash'])
    if r.get('review_id'):
        if review_receipt(store,r['review_id'])!=r or r['status']!='human_confirmed':raise ValueError('没有有效人工审核')
    elif r.get('transform')=='delete_safe_background_v1':
        if validate_deletion(a,b)!=r or r['status']!='rule_verified':raise ValueError('缩减收据不匹配')
    else:
        factor=r.get('transform','').removesuffix('_v1')
        if validate_contrast(a,b,factor)!=r or r['status']!='rule_verified':raise ValueError('变换收据不匹配')
    return a,b
