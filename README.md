works:

```bash
python ./main.py --optimizer AdamW --lr 1e-4 --conv_type norm1 --norm_type normal --morpho_type none
python ./main.py --optimizer AdamW --lr 1e-3 --conv_type normal --norm_type none --morpho_type none
python ./main.py --optimizer AdamW --lr 1e-3 --conv_type norm1 --norm_type normal --morpho_type infinity
```