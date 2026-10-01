# -*- coding: utf-8 -*-
"""
KG Flood Watch — Streamlit Dashboard
วิเคราะห์สภาพอากาศ ปริมาณฝน น้ำในแม่น้ำ และน้ำทะเลหนุน (ข้อมูลฟรีจาก Open-Meteo)

รันในเครื่อง : pip install -r requirements.txt  แล้ว  streamlit run app.py
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="KG Flood Watch", page_icon="🌊", layout="wide")

# ───────────────────────── ค่าตั้งต้น ─────────────────────────
TZ = "Asia/Bangkok"
# ชื่อ: (lat, lon, sea_lat, sea_lon)  — sea = None คือพื้นที่ไม่ติดทะเล
PRESETS = {
    "กรุงเทพฯ (เจ้าพระยา)": (13.75, 100.50, 13.40, 100.60),
    "สมุทรปราการ": (13.60, 100.60, 13.40, 100.60),
    "นนทบุรี": (13.86, 100.51, 13.40, 100.60),
    "อยุธยา": (14.35, 100.57, 13.40, 100.60),
    "นครสวรรค์": (15.70, 100.14, None, None),
    "ชลบุรี": (13.36, 100.98, 13.30, 100.85),
    "หาดใหญ่ / สงขลา": (7.01, 100.47, 7.20, 100.65),
    "เชียงใหม่ (ปิง)": (18.79, 98.98, None, None),
    "กำหนดเอง": None,
}
RAIN_3D_CRIT = 150.0
WEIGHTS = dict(rain=0.35, river=0.40, tide=0.25)
COMBO_BONUS = 10
LEVELS = [(25, "ปกติ", "#2a9d8f"), (50, "เฝ้าระวัง", "#e9c46a"),
          (75, "เตือนภัย", "#f4a261"), (101, "วิกฤต", "#e63946")]


def level_of(r):
    for cut, name, color in LEVELS:
        if r < cut:
            return name, color
    return LEVELS[-1][1], LEVELS[-1][2]

# ───────────────────────── ดึงข้อมูล (cache 30 นาที) ─────────────────────────
def get_json(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat, lon, past, fc):
    j = get_json("https://api.open-meteo.com/v1/forecast", dict(
        latitude=lat, longitude=lon, timezone=TZ, past_days=past, forecast_days=fc,
        hourly="relative_humidity_2m,pressure_msl",
        daily="temperature_2m_max,temperature_2m_min,precipitation_sum,"
              "precipitation_probability_max,wind_speed_10m_max"))
    d = pd.DataFrame(j["daily"]).rename(columns={
        "time": "date", "temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
        "precipitation_sum": "rain", "precipitation_probability_max": "rain_prob",
        "wind_speed_10m_max": "wind_max"})
    d["date"] = pd.to_datetime(d["date"])
    h = pd.DataFrame(j["hourly"])
    h["date"] = pd.to_datetime(h["time"]).dt.normalize()
    hd = h.groupby("date").agg(rh=("relative_humidity_2m", "mean"),
                               pressure=("pressure_msl", "mean")).reset_index()
    return d.merge(hd, on="date", how="left").set_index("date")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_river(lat, lon, past, fc):
    j = get_json("https://flood-api.open-meteo.com/v1/flood", dict(
        latitude=lat, longitude=lon, past_days=past, forecast_days=fc,
        daily="river_discharge,river_discharge_max"))
    d = pd.DataFrame(j["daily"]).rename(columns={
        "time": "date", "river_discharge": "discharge", "river_discharge_max": "discharge_max"})
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_sea(lat, lon, past, fc):
    j = get_json("https://marine-api.open-meteo.com/v1/marine", dict(
        latitude=lat, longitude=lon, timezone=TZ, past_days=past, forecast_days=fc,
        hourly="sea_level_height_msl"))
    h = pd.DataFrame(j["hourly"]).dropna()
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.normalize()
    g = h.groupby("date")["sea_level_height_msl"]
    out = g.agg(sea_max="max", sea_min="min")
    out["high_tide_time"] = h.loc[g.idxmax()].set_index("date")["time"].dt.strftime("%H:%M")
    out["tide_range"] = out["sea_max"] - out["sea_min"]
    return out, h.set_index("time")["sea_level_height_msl"]


def demo_data(past, fc, with_sea=True):
    rng = np.random.default_rng(7)
    days = pd.date_range(pd.Timestamp(date.today()) - pd.Timedelta(days=past),
                         periods=past + fc, freq="D", name="date")
    n = len(days); t = np.arange(n)
    storm = 60 * np.exp(-((t - n * 0.7) / 4) ** 2)
    rain = np.clip(rng.gamma(0.6, 8, n) + storm, 0, None).round(1)
    q = 1500 + np.convolve(rain, np.ones(10) * 18, "full")[:n] + rng.normal(0, 60, n)
    df = pd.DataFrame(index=days, data=dict(
        tmax=32 + rng.normal(0, 1.2, n), tmin=25 + rng.normal(0, 1, n),
        rain=rain, rain_prob=np.clip(rain * 3 + 20, 0, 100).round(),
        wind_max=10 + rng.normal(0, 3, n), rh=np.clip(70 + rain * 0.6, 0, 100),
        pressure=1008 - storm * 0.05 + rng.normal(0, 1, n),
        discharge=q, discharge_max=q * 1.15))
    hourly = None
    if with_sea:
        hrs = pd.date_range(days[0], periods=n * 24, freq="h")
        th = np.arange(len(hrs)) / 24
        lvl = (0.45 * np.cos(2 * np.pi * th / 1.035) * (0.6 + 0.4 * np.cos(2 * np.pi * th / 14.77))
               + 0.15 * np.cos(2 * np.pi * th / 0.5175) + 0.1)
        hourly = pd.Series(lvl, index=hrs)
        g = hourly.groupby(hourly.index.normalize())
        df["sea_max"] = g.max(); df["sea_min"] = g.min()
        df["high_tide_time"] = g.idxmax().dt.strftime("%H:%M")
        df["tide_range"] = df["sea_max"] - df["sea_min"]
    return df, hourly

# ───────────────────────── วิเคราะห์ ─────────────────────────
def rain_class(mm):
    if pd.isna(mm): return "-"
    if mm < 0.1: return "ไม่มีฝน"
    if mm <= 10: return "ฝนเล็กน้อย"
    if mm <= 35: return "ฝนปานกลาง"
    if mm <= 90: return "ฝนหนัก"
    return "ฝนหนักมาก"


def rel_score(s, crit=None):
    s = s.astype(float)
    base = s.quantile(0.5)
    crit = crit if crit else s.quantile(0.95)
    if pd.isna(crit) or crit <= base:
        return pd.Series(0.0, index=s.index), crit
    return ((s - base) / (crit - base) * 100).clip(0, 100), crit


def analyze(df, river_crit=None, tide_crit=None):
    df = df.copy()
    df["rain_class"] = df["rain"].apply(rain_class)
    df["rain_3d"] = df["rain"].rolling(3, min_periods=1).sum()
    df["rain_7d"] = df["rain"].rolling(7, min_periods=1).sum()
    df["s_rain"] = (df["rain_3d"] / RAIN_3D_CRIT * 100).clip(0, 100)
    crit = {}
    if "discharge" in df and df["discharge"].notna().any():
        df["s_river"], crit["river"] = rel_score(df["discharge"], river_crit)
        df["river_trend"] = df["discharge"].diff(3)
    else:
        df["s_river"] = np.nan
    if "sea_max" in df and df["sea_max"].notna().any():
        df["s_tide"], crit["tide"] = rel_score(df["sea_max"], tide_crit)
    else:
        df["s_tide"] = np.nan
    wsum = sum(WEIGHTS[k] * df[f"s_{k}"].notna() for k in WEIGHTS)
    score = sum(WEIGHTS[k] * df[f"s_{k}"].fillna(0) for k in WEIGHTS) / wsum.replace(0, np.nan)
    df["combo_alert"] = (df["s_river"] >= 60) & (df["s_tide"] >= 60)
    df["risk"] = (score + COMBO_BONUS * df["combo_alert"]).clip(0, 100).round(1)
    df["risk_level"] = df["risk"].apply(lambda r: level_of(r)[0])
    return df, crit

# ───────────────────────── กราฟ ─────────────────────────
def build_chart(df, crit, today, hourly):
    has_river = "discharge" in df and df["discharge"].notna().any()
    has_sea = "sea_max" in df and df["sea_max"].notna().any()
    titles = ["สภาพอากาศ", "ปริมาณฝน", "ปริมาณน้ำในแม่น้ำ (m³/s)",
              "ระดับน้ำทะเล / น้ำทะเลหนุน (m)", "ดัชนีเสี่ยงน้ำท่วม (0-100)"]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        subplot_titles=titles, row_heights=[1, 1.1, 1.1, 1.1, 0.9],
                        specs=[[{"secondary_y": True}], [{"secondary_y": True}],
                               [{}], [{}], [{}]])
    x = df.index
    # 1 อากาศ
    fig.add_trace(go.Scatter(x=x, y=df["tmax"], name="อุณหภูมิสูงสุด", line=dict(color="#e76f51")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df["tmin"], name="อุณหภูมิต่ำสุด", line=dict(color="#f4a261"),
                             fill="tonexty", fillcolor="rgba(244,162,97,.2)"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df["rh"], name="ความชื้น %", line=dict(color="#2a9d8f", dash="dot")),
                  1, 1, secondary_y=True)
    # 2 ฝน
    fig.add_trace(go.Bar(x=x, y=df["rain"], name="ฝนรายวัน (mm)", marker_color="#4895ef",
                         customdata=df["rain_class"],
                         hovertemplate="%{y:.1f} mm<br>%{customdata}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=df["rain_7d"], name="ฝนสะสม 7 วัน", line=dict(color="#1d3557", width=2)),
                  2, 1, secondary_y=True)
    fig.add_hline(y=35, line_dash="dot", line_color="orange", row=2, col=1)
    fig.add_hline(y=90, line_dash="dot", line_color="red", row=2, col=1)
    # 3 แม่น้ำ
    if has_river:
        if "discharge_max" in df:
            fig.add_trace(go.Scatter(x=x, y=df["discharge_max"], name="กรณีสูงสุด",
                                     line=dict(color="rgba(38,70,83,.3)")), 3, 1)
        fig.add_trace(go.Scatter(x=x, y=df["discharge"], name="น้ำท่า", line=dict(color="#264653", width=2.5)), 3, 1)
        if crit.get("river"):
            fig.add_hline(y=crit["river"], line_dash="dash", line_color="red", row=3, col=1)
    # 4 น้ำทะเล
    if has_sea:
        if hourly is not None:
            fig.add_trace(go.Scatter(x=hourly.index, y=hourly.values, name="ระดับน้ำรายชั่วโมง",
                                     line=dict(color="rgba(106,76,147,.35)", width=1)), 4, 1)
        fig.add_trace(go.Scatter(x=x, y=df["sea_max"], name="น้ำขึ้นสูงสุด", mode="lines+markers",
                                 line=dict(color="#6a4c93", width=2), customdata=df["high_tide_time"],
                                 hovertemplate="%{y:+.2f} m เวลา %{customdata}<extra></extra>"), 4, 1)
        if crit.get("tide"):
            fig.add_hline(y=crit["tide"], line_dash="dash", line_color="red", row=4, col=1)
    # 5 ความเสี่ยง
    fig.add_trace(go.Bar(x=x, y=df["risk"], name="ดัชนีเสี่ยง",
                         marker_color=[level_of(r)[1] for r in df["risk"]],
                         customdata=df["risk_level"],
                         hovertemplate="%{y:.0f} — %{customdata}<extra></extra>"), 5, 1)
    fig.update_yaxes(range=[0, 100], row=5, col=1)
    fig.add_vline(x=str(today.date()), line_dash="dash", line_color="gray", row="all", col=1)
    fig.update_layout(height=1250, hovermode="x unified", bargap=0.15,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.03))
    return fig

# ───────────────────────── UI ─────────────────────────
with st.sidebar:
    st.header("ตั้งค่า")
    preset = st.selectbox("พื้นที่", list(PRESETS))
    p = PRESETS[preset] or (13.75, 100.50, 13.40, 100.60)
    custom = preset == "กำหนดเอง"
    c1, c2 = st.columns(2)
    lat = c1.number_input("Lat", value=p[0], format="%.4f", disabled=not custom)
    lon = c2.number_input("Lon", value=p[1], format="%.4f", disabled=not custom)
    use_sea = st.toggle("รวมน้ำทะเลหนุน", value=p[2] is not None)
    if use_sea:
        c3, c4 = st.columns(2)
        sea_lat = c3.number_input("Sea lat", value=p[2] or 13.40, format="%.4f",
                                  help="จุดในทะเล ใกล้ปากแม่น้ำ")
        sea_lon = c4.number_input("Sea lon", value=p[3] or 100.60, format="%.4f")
    past = st.slider("ย้อนหลัง (วัน)", 7, 92, 30)
    fc = st.slider("พยากรณ์ล่วงหน้า (วัน)", 1, 7, 7)
    st.subheader("ระดับวิกฤต")
    river_crit = st.number_input("น้ำท่า m³/s (0 = อัตโนมัติ)", value=0.0, step=100.0)
    tide_crit = st.number_input("น้ำทะเล m (0 = อัตโนมัติ)", value=0.0, step=0.05)
    demo = st.checkbox("ใช้ข้อมูลจำลอง (ทดสอบ)")
    if st.button("รีเฟรชข้อมูล", width="stretch"):
        st.cache_data.clear()
    st.caption("ข้อมูล: Open-Meteo · น้ำท่าเป็นค่าจากแบบจำลอง GloFAS ไม่ใช่สถานีวัดจริง")

st.title("🌊 KG Flood Watch")
st.caption(f"{preset} · ย้อนหลัง {past} วัน · พยากรณ์ {fc} วัน")

# โหลดข้อมูล
hourly = None
errors = []
with st.spinner("กำลังดึงข้อมูล..."):
    if demo:
        df, hourly = demo_data(past, fc, use_sea)
    else:
        frames = []
        for name, fn, args in [("สภาพอากาศ", fetch_weather, (lat, lon)),
                               ("แม่น้ำ", fetch_river, (lat, lon)),
                               ("น้ำทะเล", fetch_sea, (sea_lat, sea_lon) if use_sea else None)]:
            if args is None:
                continue
            try:
                res = fn(*args, past, fc)
                if name == "น้ำทะเล":
                    res, hourly = res
                frames.append(res)
            except Exception as e:
                errors.append(f"ดึงข้อมูล{name}ไม่ได้: {e}")
        if not frames:
            st.error("ไม่มีข้อมูล ตรวจสอบพิกัดหรือการเชื่อมต่ออินเทอร์เน็ต")
            st.stop()
        df = pd.concat(frames, axis=1).sort_index()
        if "rain" in df:
            df = df[df["rain"].notna()]
for e in errors:
    st.warning(e)

df, crit = analyze(df, river_crit or None, tide_crit or None)
today = pd.Timestamp(date.today())
if today not in df.index:
    today = df.index[df.index <= today].max() if (df.index <= today).any() else df.index[0]
r = df.loc[today]

# สถานะวันนี้
name, color = level_of(r["risk"])
combo = " · ⚠️ น้ำเหนือ + น้ำทะเลหนุนพร้อมกัน" if r["combo_alert"] else ""
st.markdown(
    f"""<div style="background:{color};color:#fff;padding:18px 22px;border-radius:14px;margin-bottom:12px">
    <div style="font-size:15px;opacity:.9">ความเสี่ยงน้ำท่วมวันนี้ ({today:%d/%m/%Y})</div>
    <div style="font-size:38px;font-weight:700;line-height:1.2">{name} · {r['risk']:.0f}/100</div>
    <div style="font-size:15px">{combo.strip(' ·') if combo else ''}</div></div>""",
    unsafe_allow_html=True)

m = st.columns(4)
m[0].metric("ฝนวันนี้", f"{r['rain']:.1f} mm", r["rain_class"], delta_color="off")
m[1].metric("ฝนสะสม 3 วัน", f"{r['rain_3d']:.0f} mm", f"7 วัน {r['rain_7d']:.0f} mm", delta_color="off")
if pd.notna(r.get("discharge")):
    m[2].metric("น้ำในแม่น้ำ", f"{r['discharge']:,.0f} m³/s",
                f"{r['river_trend']:+,.0f} ใน 3 วัน", delta_color="inverse")
else:
    m[2].metric("น้ำในแม่น้ำ", "-")
if pd.notna(r.get("sea_max")):
    m[3].metric("น้ำขึ้นสูงสุด", f"{r['sea_max']:+.2f} m", f"เวลา {r['high_tide_time']}", delta_color="off")
else:
    m[3].metric("น้ำขึ้นสูงสุด", "-")

# วันเสี่ยงล่วงหน้า
fc_df = df[df.index > today]
hot = fc_df[fc_df["risk"] >= 50]
if len(hot):
    st.warning("วันที่ควรเฝ้าระวังล่วงหน้า: " +
               ", ".join(f"{d:%d/%m} ({v:.0f} {level_of(v)[0]})" for d, v in hot["risk"].items()))
elif len(fc_df):
    st.success(f"{len(fc_df)} วันข้างหน้า ความเสี่ยงอยู่ในเกณฑ์ปกติถึงเฝ้าระวัง")

tab1, tab2, tab3 = st.tabs(["📈 กราฟ", "📅 พยากรณ์รายวัน", "📍 แผนที่"])
with tab1:
    st.plotly_chart(build_chart(df, crit, today, hourly), width="stretch")
    st.caption(f"เส้นประแดง = ระดับวิกฤต | น้ำท่า {crit.get('river', 0):,.0f} m³/s · "
               f"น้ำทะเล {crit.get('tide', 0):+.2f} m | เส้นประเทา = วันนี้")

with tab2:
    show = df[df.index >= today].copy()
    table = pd.DataFrame({
        "วันที่": show.index.strftime("%a %d/%m"),
        "อุณหภูมิ °C": show.apply(lambda x: f"{x.tmin:.0f}-{x.tmax:.0f}", axis=1),
        "ฝน mm": show["rain"].round(1),
        "โอกาสฝน %": show["rain_prob"],
        "ระดับฝน": show["rain_class"],
        "น้ำท่า m³/s": show.get("discharge", pd.Series(np.nan, index=show.index)).round(0),
        "น้ำขึ้น m": show.get("sea_max", pd.Series(np.nan, index=show.index)).round(2),
        "เวลาน้ำขึ้น": show.get("high_tide_time", pd.Series("-", index=show.index)),
        "ดัชนีเสี่ยง": show["risk"].round(0),
        "ระดับ": show["risk_level"] + np.where(show["combo_alert"], " ⚠️", ""),
    })
    colors = {n: c for _, n, c in LEVELS}
    styled = table.style.map(
        lambda v: f"background-color:{colors.get(str(v).replace(' ⚠️', ''), '')};color:white"
        if str(v).replace(" ⚠️", "") in colors else "", subset=["ระดับ"])
    st.dataframe(styled, hide_index=True, width="stretch")
    st.download_button("ดาวน์โหลด CSV ทั้งหมด", df.round(3).to_csv().encode("utf-8-sig"),
                       file_name=f"flood_{today:%Y%m%d}.csv", mime="text/csv")

with tab3:
    pts = pd.DataFrame({"lat": [lat], "lon": [lon], "color": ["#e63946"], "size": [800]})
    if use_sea:
        pts.loc[1] = [sea_lat, sea_lon, "#6a4c93", 800]
    st.map(pts, latitude="lat", longitude="lon", color="color", size="size", zoom=8)
    st.caption("🔴 จุดวิเคราะห์ฝน/แม่น้ำ  🟣 จุดวัดน้ำทะเล")

with st.expander("วิธีคำนวณดัชนีเสี่ยง"):
    st.markdown(f"""
- **ฝน ({WEIGHTS['rain']:.0%})**: ฝนสะสม 3 วัน เทียบ {RAIN_3D_CRIT:.0f} mm = 100 คะแนน
- **แม่น้ำ ({WEIGHTS['river']:.0%})**: ค่ากลางของช่วงข้อมูล = 0, ระดับวิกฤต = 100
- **น้ำทะเล ({WEIGHTS['tide']:.0%})**: น้ำขึ้นสูงสุดรายวัน ค่ากลาง = 0, ระดับน้ำหนุนสูง = 100
- น้ำเหนือ + น้ำหนุน ≥ 60 คะแนนพร้อมกัน → บวกเพิ่ม {COMBO_BONUS}
- ระดับวิกฤตอัตโนมัติ = เปอร์เซ็นไทล์ 95 ของช่วงข้อมูล (ควรใส่ค่าจริงของพื้นที่ในแถบด้านซ้าย)
- ระดับฝนตามเกณฑ์กรมอุตุนิยมวิทยา: เล็กน้อย ≤10 · ปานกลาง ≤35 · หนัก ≤90 · หนักมาก >90 mm
""")
