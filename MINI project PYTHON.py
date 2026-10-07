"""
Car Sales Analysis System  -  Python + NumPy + Matplotlib ONLY (no Flask, no server)

Run :  python car_sales_site.py
It cleans the data, makes all charts and writes a complete website (car_sales_site.html)
next to this file, then opens it in your browser.

Install once:  pip install numpy matplotlib
"""
import base64
import io
import os
import webbrowser

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # draw to memory, no window needed
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Missing packages. Run this first:\n    pip install numpy matplotlib")

# ----------------------------------------------------------------------
# 1. DATA  (model, price in lakh INR, units sold, profit in lakh INR)
#    Contains bad rows on purpose: missing price, negative sales, duplicate.
# ----------------------------------------------------------------------
ROWS = [
    ("Swift", 6.5, 1800, 900), ("Baleno", 7.5, 1500, 880), ("Creta", 12.0, 1300, 1500),
    ("Nexon", 11.0, 1450, 1320), ("City", 13.0, 900, 1250), ("Verna", 12.5, 700, 980),
    ("Innova", 20.0, 800, 1700), ("Fortuner", 38.0, 400, 2400), ("Thar", 14.5, 850, 1350),
    ("Seltos", 13.5, 1100, 1400), ("Punch", 6.8, 1600, 640), ("Brezza", 10.5, 1250, 1180),
    ("Ertiga", 10.0, 1350, 1250), ("i20", 8.0, 1000, 760), ("Venue", 9.5, 1050, 950),
    ("Scorpio", 15.5, 650, 1450), ("XUV700", 18.0, 600, 1650), ("Altroz", 7.2, 780, 600),
    ("Amaze", 7.8, 820, 650), ("Kwid", 4.8, 1700, 380),
    ("Swift", 6.5, 1800, 900),      # duplicate
    ("Duster", None, 500, 400),     # missing price
    ("Glanza", 7.0, -50, 120),      # negative sales
]

SEGMENTS = {
    "All": lambda p: np.ones(len(p), dtype=bool),
    "Budget (<8L)": lambda p: p < 8,
    "Mid (8-15L)": lambda p: (p >= 8) & (p < 15),
    "Premium (>=15L)": lambda p: p >= 15,
}


# ----------------------------------------------------------------------
# 2. NUMPY: cleaning
# ----------------------------------------------------------------------
def load_and_clean():
    models = np.array([r[0] for r in ROWS])
    nums = np.array([r[1:] for r in ROWS], dtype=float)      # None -> nan
    price, sales, profit = nums[:, 0], nums[:, 1], nums[:, 2]

    ok = ~(np.isnan(price) | np.isnan(sales) | np.isnan(profit)) & (price > 0) & (sales > 0)
    models, price, sales, profit = models[ok], price[ok], sales[ok], profit[ok]

    _, first = np.unique(models, return_index=True)          # drop duplicate models
    keep = np.sort(first)
    return models[keep], price[keep], sales[keep], profit[keep]


# ----------------------------------------------------------------------
# 3. MATPLOTLIB: charts -> <img> tags
# ----------------------------------------------------------------------
def to_img(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return '<img src="data:image/png;base64,%s">' % base64.b64encode(buf.getvalue()).decode()


def bar(models, values, title, ylabel, color):
    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models[order], values[order], color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.3)
    return to_img(fig)


def histogram(values, title, xlabel, color):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(8, len(values)), color=color, edgecolor="white")
    ax.axvline(values.mean(), color="red", linestyle="--", label="Mean %.1f" % values.mean())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of models")
    ax.legend()
    return to_img(fig)


def scatter(x, y, labels, xlabel, ylabel, title, color):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(x, y, s=70, color=color, alpha=0.85, edgecolor="white")
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        r = np.corrcoef(x, y)[0, 1]
        ax.plot(xs, slope * xs + intercept, "r--", label="Trend (r = %+.2f)" % r)
        ax.legend()
    for xi, yi, name in zip(x, y, labels):
        ax.annotate(name, (xi, yi), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return to_img(fig)


# ----------------------------------------------------------------------
# 4. ONE SEGMENT = KPIs + insights + 6 charts + table  (as HTML)
# ----------------------------------------------------------------------
def build_segment(m, p, s, f):
    if len(m) == 0:
        return "<p>No cars in this segment.</p>"

    revenue = p * s
    margin = f / revenue * 100

    kpis = [("Models", len(m)), ("Units sold", "{:,}".format(int(s.sum()))),
            ("Revenue (Rs lakh)", "{:,.0f}".format(revenue.sum())),
            ("Profit (Rs lakh)", "{:,.0f}".format(f.sum())),
            ("Margin", "%.1f%%" % (f.sum() / revenue.sum() * 100))]
    kpi_html = "".join('<div class="card"><small>%s</small><b>%s</b></div>' % k for k in kpis)

    ins = ["Top sellers: <b>%s</b>" % ", ".join(m[np.argsort(s)[::-1][:3]]),
           "Most profitable: <b>%s</b>" % ", ".join(m[np.argsort(f)[::-1][:3]]),
           "Highest margin: <b>%s</b> (%.1f%%)" % (m[np.argmax(margin)], margin.max())]
    if len(m) > 2:
        ins.append("Correlation: price vs sales <b>%+.2f</b>, sales vs profit <b>%+.2f</b>, price vs profit <b>%+.2f</b>"
                   % (np.corrcoef(p, s)[0, 1], np.corrcoef(s, f)[0, 1], np.corrcoef(p, f)[0, 1]))
    ins_html = "<ul>" + "".join("<li>%s</li>" % i for i in ins) + "</ul>"

    charts = [
        ("Bar: Units sold", bar(m, s, "Units Sold by Model", "Units sold", "#2a6fdb")),
        ("Bar: Profit", bar(m, f, "Profit by Model", "Profit (lakh INR)", "#1e9e6a")),
        ("Histogram: Sales", histogram(s, "Sales Distribution", "Units sold", "#2a6fdb")),
        ("Histogram: Price", histogram(p, "Price Distribution", "Price (lakh INR)", "#e08a1e")),
        ("Scatter: Price vs Sales", scatter(p, s, m, "Price (lakh INR)", "Units sold", "Price vs Sales", "#2a6fdb")),
        ("Scatter: Sales vs Profit", scatter(s, f, m, "Units sold", "Profit (lakh INR)", "Sales vs Profit", "#1e9e6a")),
    ]
    chart_html = "".join('<div class="card"><h3>%s</h3>%s</div>' % c for c in charts)

    rows = "".join(
        "<tr><td>%s</td><td>%.1f</td><td>{:,}</td><td>{:,.0f}</td><td>{:,.0f}</td><td>%.1f</td></tr>"
        .format(int(s[i]), f[i], revenue[i]) % (m[i], p[i], margin[i])
        for i in np.argsort(s)[::-1])
    table = ('<table><tr><th>Model</th><th>Price</th><th>Units</th><th>Profit</th>'
             '<th>Revenue</th><th>Margin %%</th></tr>%s</table>' % rows)

    return ('<div class="kpis">%s</div><div class="card"><h3>Insights</h3>%s</div>'
            '<div class="grid">%s</div><div class="card"><h3>Data table</h3>'
            '<div class="scroll">%s</div></div>') % (kpi_html, ins_html, chart_html, table)


# ----------------------------------------------------------------------
# 5. PAGE
# ----------------------------------------------------------------------
CSS = """
:root{--bg:#f5f6f8;--card:#fff;--tx:#1b1f27;--mut:#667085;--bd:#e3e6eb;--a:#2a6fdb}
@media(prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#181c24;--tx:#e8eaee;--mut:#98a2b3;--bd:#2a303b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font-family:system-ui,"Segoe UI",Roboto,sans-serif;line-height:1.5}
header{background:linear-gradient(135deg,#2a6fdb,#7b4dff);color:#fff;padding:44px 20px}
header div,main{max-width:1200px;margin:auto}
header h1{margin:0 0 6px;font-size:2rem}header p{margin:0;opacity:.9}
main{padding:20px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px 16px;margin-bottom:14px}
.card h3{margin:0 0 8px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:14px}
.kpis .card{margin:0}.kpis small{color:var(--mut);display:block}.kpis b{font-size:1.5rem}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.tabs button{border:1px solid var(--bd);background:var(--card);color:var(--tx);padding:9px 16px;border-radius:20px;cursor:pointer;font-size:.95rem}
.tabs button.on{background:var(--a);border-color:var(--a);color:#fff}
.seg{display:none}.seg.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px;margin-bottom:14px}
.grid .card{margin:0}
img{max-width:100%;height:auto;background:#fff;border-radius:8px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--bd);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--mut)}
footer{text-align:center;color:var(--mut);padding:24px;font-size:.85rem}
"""

JS = """
function show(i){
  document.querySelectorAll('.seg').forEach((d,k)=>d.classList.toggle('on',k===i));
  document.querySelectorAll('.tabs button').forEach((b,k)=>b.classList.toggle('on',k===i));
}
show(0);
"""


def main():
    models, price, sales, profit = load_and_clean()
    removed = len(ROWS) - len(models)
    print("Cleaned data: %d rows -> %d rows (%d removed)" % (len(ROWS), len(models), removed))

    tabs, sections = [], []
    for i, (name, rule) in enumerate(SEGMENTS.items()):
        mask = rule(price)
        print("Building charts for:", name)
        tabs.append('<button onclick="show(%d)">%s</button>' % (i, name))
        sections.append('<div class="seg">%s</div>' % build_segment(models[mask], price[mask], sales[mask], profit[mask]))

    intro = ("Python + NumPy + Matplotlib &nbsp;|&nbsp; Cleaned %d raw rows, removed %d bad rows. "
             "Price and profit in Rs lakh." % (len(ROWS), removed))
    html = ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Car Sales Analysis System</title><style>' + CSS + '</style></head><body>'
            '<header><div><h1>Car Sales Analysis System</h1><p>' + intro + '</p></div></header>'
            '<main><div class="tabs">' + "".join(tabs) + '</div>' + "".join(sections) + '</main>'
            '<footer>Car Sales Analysis System</footer><script>' + JS + '</script></body></html>')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_sales_site.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("\nWebsite created:", out)
    webbrowser.open("file://" + out.replace("\\", "/") if os.name == "nt" else "file://" + out)


if __name__ == "__main__":
    main()
