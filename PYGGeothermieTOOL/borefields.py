import matplotlib.pyplot as plt
import pygfunction as gt
import Data_Base as bd
from pathlib import Path


# -------------------------------------------------------------------------
# Borehole fields
# -------------------------------------------------------------------------

# Rectangular field
rectangle_field = gt.borefield.Borefield.rectangle_field(
    bd.rows_sonden, bd.Columns_sonden, bd.Bx, bd.By, bd.H, bd.D, bd.r)

# U-shaped field
U_shaped_field = gt.borefield.Borefield.U_shaped_field(
    bd.rows_sonden, bd.Columns_sonden, bd.Bx, bd.By, bd.H, bd.D, bd.r)

def borefield():
    # -------------------------------------------------------------------------
    # Draw bore fields
    # -------------------------------------------------------------------------
    for borefield in [
            rectangle_field, U_shaped_field]:
        borefield.visualize_field()
        plt.show()




# --- auto_borefield.py: AUTOGENERIERTE PARAMETER ---
H = 100
Bx = 6
By = 6
rows_sonden = 0
Columns_sonden = 0

# -----------------------------------------------------
# Freiform-Feld (optional, nur wenn CSV vorhanden)
# Quelle: ./autoborehole/borefield_polygon_points.csv (x_m,y_m)
# -----------------------------------------------------
freeform_field = []
try:
    import csv
    from pygfunction.boreholes import Borehole
    root = Path(__file__).resolve().parent
    csv_path = root / 'autoborehole' / 'borefield_polygon_points.csv'

    def _load_freeform_points(path):
        holes_local = []
        with path.open('r', encoding='utf-8') as fh:
            rdr = csv.DictReader(fh)
            xk = yk = None
            if rdr.fieldnames:
                for k in rdr.fieldnames:
                    lk = (k or '').strip().lower()
                    if lk == 'x_m':
                        xk = k
                    if lk == 'y_m':
                        yk = k
            if xk and yk:
                for row in rdr:
                    try:
                        x = float(str(row.get(xk, '0')).replace(',', '.'))
                        y = float(str(row.get(yk, '0')).replace(',', '.'))
                    except Exception:
                        continue
                    holes_local.append(Borehole(float(bd.H), float(bd.D), float(bd.r), x, y))
        return holes_local

    if csv_path.exists():
        freeform_field = _load_freeform_points(csv_path)
except Exception:
    # Import tolerant lassen, falls CSV fehlt/fehlerhaft ist
    freeform_field = []