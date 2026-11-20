"use client";

import { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid,
  ReferenceDot,
} from "recharts";

type Building = {
  building_id: string;
  primaryspaceusage: string;
  sqm: number;
  site_id: string;
};

type Hour = { timestamp: string; kwh: number | null; forecast_kwh: number | null };
type AnomalyEvent = {
  type: string; start: string; end: string; hours: number;
  peak_z: number; excess_kwh: number;
};
type ShiftResponse = {
  baseline_kwh: number[]; shifted_kwh: number[]; ci_gco2_kwh: number[];
  kwh_moved: number; co2_saved_kg: number; co2_saved_pct: number; provider: string;
};

const fmtDate = (s: string) => s.slice(0, 10);

export default function Home() {
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [selected, setSelected] = useState("");
  const [start, setStart] = useState("2017-12-01");
  const [days, setDays] = useState(7);
  const [hours, setHours] = useState<Hour[]>([]);
  const [events, setEvents] = useState<AnomalyEvent[]>([]);
  const [shift, setShift] = useState<ShiftResponse | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/buildings")
      .then((r) => r.json())
      .then((b: Building[]) => {
        setBuildings(b);
        const joey = b.find((x) => x.building_id === "Hog_office_Joey");
        setSelected(joey ? joey.building_id : b[0]?.building_id ?? "");
      })
      .catch(() => setErr("failed to load buildings — is the API running?"));
  }, []);

  const load = () => {
    if (!selected) return;
    setErr("");
    setLoading(true);
    const end = new Date(start + "T00:00");
    end.setDate(end.getDate() + Number(days));
    const endStr = end.toISOString().slice(0, 10);
    Promise.all([
      fetch(`/api/buildings/${selected}/forecast?start=${start}&end=${endStr}`)
        .then((r) => r.json()),
      fetch(`/api/buildings/${selected}/anomalies?start=${start}&end=${endStr}`)
        .then((r) => r.json()),
    ])
      .then(([f, a]) => {
        setHours(f.hours ?? []);
        setEvents(a.events ?? []);
        setShift(null);
      })
      .catch(() => setErr("failed to load building data"))
      .finally(() => setLoading(false));
  };

  useEffect(load, [selected]);

  const chartData = useMemo(
    () => hours.map((h) => ({
      t: h.timestamp.slice(5, 13), // MM-DD HH
      actual: h.kwh,
      forecast: h.forecast_kwh,
    })),
    [hours]
  );

  const axis = { stroke: "#64748b", fontSize: 11, tickLine: false };
  const tooltip = {
    contentStyle: {
      background: "#121821", border: "1px solid rgba(255,255,255,0.1)",
      borderRadius: 10, fontSize: 12,
    },
    labelStyle: { color: "#94a3b8" },
  };

  // anomaly start markers, matched to the chart x labels
  const markers = useMemo(
    () =>
      events
        .filter((e) => e.type === "residual")
        .map((e) => {
          const row = chartData.find((d) => d.t === e.start.slice(5, 13));
          return row ? { x: row.t, y: row.actual ?? row.forecast ?? 0 } : null;
        })
        .filter(Boolean) as { x: string; y: number }[],
    [events, chartData]
  );

  const runShift = () => {
    setErr("");
    fetch(
      `/api/buildings/${selected}/shift?start=${start}&horizon_h=24&provider=dewa`,
      { method: "POST" }
    )
      .then((r) => r.json())
      .then((j) => {
        if (j.detail) setErr(String(j.detail));
        else setShift(j);
      })
      .catch(() => setErr("shift request failed"));
  };

  const b = buildings.find((x) => x.building_id === selected);

  return (
    <main>
      <header>
        <div className="logo">
          <div className="logo-mark">G</div>
          <h1>
            <span className="brand">Grid</span>Wise
          </h1>
        </div>
        <p className="subtitle">
          building electricity forecast · anomaly detection · carbon-aware load
          shifting
        </p>
        {b && (
          <div className="building-meta">
            <span className="chip"><b>{b.primaryspaceusage}</b></span>
            <span className="chip">{b.sqm?.toLocaleString()} m²</span>
            <span className="chip">site {b.site_id}</span>
          </div>
        )}
      </header>

      <div className="controls">
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          {buildings.map((b) => (
            <option key={b.building_id} value={b.building_id}>
              {b.building_id} ({b.primaryspaceusage})
            </option>
          ))}
        </select>
        <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
        <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
          {[3, 7, 14, 30].map((d) => (
            <option key={d} value={d}>{d} days</option>
          ))}
        </select>
        <button onClick={load} disabled={loading}>
          {loading ? "loading…" : "load"}
        </button>
      </div>

      {err && <p className="error">{err}</p>}

      <div className="cards">
        <div className="stat">
          <div className="label">anomaly events</div>
          <div className="value amber">{events.length}</div>
        </div>
        {shift && (
          <>
            <div className="stat">
              <div className="label">load shifted</div>
              <div className="value blue">
                {shift.kwh_moved.toLocaleString()}
                <span className="unit"> kWh</span>
              </div>
            </div>
            <div className="stat">
              <div className="label">CO₂ avoided</div>
              <div className="value green">
                {shift.co2_saved_kg.toLocaleString()}
                <span className="unit"> kg · {shift.co2_saved_pct}%</span>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="grid">
        <div className="panel">
          <h2>actual vs forecast (kWh)</h2>
          {chartData.length === 0 ? (
            <p className="empty">no data — pick a building and date range</p>
          ) : (
            <LineChart data={chartData} width={720} height={320}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="t" {...axis} interval="preserveStartEnd" />
              <YAxis {...axis} width={56} />
              <Tooltip {...tooltip} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Line dataKey="actual" stroke="#edf2f7" dot={false} strokeWidth={1.8}
                activeDot={{ r: 3, fill: "#edf2f7" }} />
              <Line dataKey="forecast" stroke="#34d399" dot={false}
                strokeWidth={1.8} strokeDasharray="4 3"
                activeDot={{ r: 3, fill: "#34d399" }} />
              {markers.map((m, i) => (
                <ReferenceDot key={i} x={m.x} y={m.y} r={5} fill="#f87171"
                  stroke="#0a0e14" strokeWidth={2} ifOverflow="extendDomain" />
              ))}
            </LineChart>
          )}
          <p className="note">red dots mark residual-anomaly event starts</p>
        </div>

        <div className="panel">
          <h2>anomalies</h2>
          {events.length === 0 ? (
            <p className="empty">no anomalies in this window</p>
          ) : (
            events.slice(0, 20).map((e, i) => (
              <div key={i} className={`anomaly-item ${e.type}`}>
                <strong>{e.type.replace("_", " ")}</strong>
                <div className="meta">
                  {fmtDate(e.start)} → {fmtDate(e.end)} · {e.hours}h · peak z
                  {" "}{e.peak_z} · excess {Math.round(e.excess_kwh)} kWh
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="panel" style={{ marginTop: 18 }}>
        <h2>carbon-aware shift plan — DEWA scenario, 24 h from window start</h2>
        <div style={{ marginBottom: 14 }}>
          <button onClick={runShift}>optimise shift</button>
        </div>
        {shift && (
          <>
            <LineChart
              width={720} height={220} margin={{ top: 16 }}
              data={shift.ci_gco2_kwh.map((ci, i) => ({
                t: i,
                baseline: shift.baseline_kwh[i],
                shifted: shift.shifted_kwh[i],
                ci,
              }))}
            >
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="t" {...axis} />
              <YAxis yAxisId="l" {...axis} width={56} />
              <YAxis yAxisId="r" orientation="right" stroke="#fbbf24"
                fontSize={11} tickLine={false} width={50} />
              <Tooltip {...tooltip} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Line yAxisId="l" dataKey="baseline" stroke="#94a3b8" dot={false}
                strokeWidth={1.8} />
              <Line yAxisId="l" dataKey="shifted" stroke="#34d399" dot={false}
                strokeWidth={2.2} />
              <Line yAxisId="r" dataKey="ci" stroke="#fbbf24" dot={false}
                strokeWidth={1.2} strokeDasharray="2 3" />
            </LineChart>
            <p className="note">
              grey = baseline load, green = shifted load (moves toward the amber
              carbon dips), amber dashed = carbon intensity (gCO₂/kWh).
              Provider: {shift.provider} — UAE scenario, calibrated projection,
              not measured data.
            </p>
          </>
        )}
      </div>
    </main>
  );
}
