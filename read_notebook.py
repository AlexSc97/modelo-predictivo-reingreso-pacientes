import json

with open('src/modelo_produccion_flask.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f'Total code cells: {len(cells)}\n')

for i, c in enumerate(cells[:20]):
    print(f"\n{'='*60}")
    print(f"CELL {i}")
    print('='*60)
    print(''.join(c['source']))
