"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";

const PollutionMap = dynamic(() => import("@/components/PollutionMap"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-screen bg-slate-950">
      <div className="text-center">
        <div className="w-12 h-12 border-[3px] border-cyan-300/90 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-300 text-sm font-medium">Loading map...</p>
      </div>
    </div>
  ),
});

export interface StationData {
  station_name: string;
  latitude: number;
  longitude: number;
  predicted_source: string;
  confidence: number;
  probabilities: Record<string, number>;
  pm25: number;
  no2: number | null;
  aqi: number;
  aqi_category: string;
  temperature: number;
  wind_speed: number;
  wind_direction: number;
  humidity: number;
  timestamp: string;
  pm25_history: number[];
  time_labels: string[];
}

interface ApiResponse {
  stations: StationData[];
  model_accuracy: number;
  timestamp: string;
  available_range?: { start: string; end: string };
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [data, setData] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedStationName, setSelectedStationName] = useState<string | null>(null);
  const [selectedTime, setSelectedTime] = useState<string | null>(null);
  const [isHistorical, setIsHistorical] = useState(false);

  // Derive the selected station object from the data array
  const selectedStation = data?.stations.find(s => s.station_name === selectedStationName) || null;

  const fetchData = useCallback(async (time?: string) => {
    try {
      const endpoint = time ? `/predict/history?timestamp=${encodeURIComponent(time)}` : "/predict/latest";
      const res = await fetch(`${API_BASE}${endpoint}`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json: ApiResponse = await res.json();

      setData(json);
      setError(null);

      // If no station is selected by name, default to the first one available
      if (!selectedStationName && json.stations.length > 0) {
        setSelectedStationName(json.stations[0].station_name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to connect to API");
    } finally {
      setLoading(false);
    }
  }, [selectedStationName]);

  useEffect(() => {
    fetchData(selectedTime || undefined);

    // Only poll if not in historical mode
    if (!selectedTime) {
      const interval = setInterval(() => fetchData(), 60000);
      return () => clearInterval(interval);
    }
  }, [fetchData, selectedTime]);

  const handleTimeChange = (time: string | null) => {
    setSelectedTime(time);
    setIsHistorical(!!time);
    setLoading(true);
  };

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-screen bg-slate-950">
        <div className="text-center">
          <div className="w-12 h-12 border-[3px] border-cyan-300/90 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-300 text-sm font-medium">Connecting to PIE Engine...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-slate-950">
        <div className="text-center bg-slate-900/85 p-8 rounded-2xl border border-red-500/30 max-w-md shadow-2xl backdrop-blur">
          <div className="text-red-400 text-4xl mb-4">⚠</div>
          <h2 className="text-white text-2xl font-bold tracking-tight mb-2">Unable to reach PIE Engine</h2>
          <p className="text-slate-300 mb-4 text-sm">
            {error}
          </p>
          <p className="text-slate-400 text-sm mb-4 leading-relaxed">
            Make sure the FastAPI backend is running locally:
            <code className="block mt-2 text-cyan-300 bg-slate-800/90 rounded px-3 py-2 text-xs">
              uvicorn main:app --port 8000
            </code>
          </p>
          <button
            onClick={() => { setLoading(true); fetchData(selectedTime || undefined); }}
            className="px-6 py-2.5 bg-cyan-400 hover:bg-cyan-300 text-slate-950 font-semibold rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-slate-950 text-slate-100 flex flex-col overflow-hidden">
      {/* Top app shell / header */}
      <header className="flex-shrink-0 border-b border-slate-800/80 bg-slate-950/75 backdrop-blur-xl z-20">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-300 to-emerald-400 flex items-center justify-center text-base font-extrabold text-slate-950 shadow-[0_0_28px_rgba(34,211,238,0.32)]">
              PIE
            </div>
            <div>
              <p className="text-base font-bold tracking-tight">
                Delhi Pollution Intelligence Engine
              </p>
              <p className="text-sm text-slate-400">
                Real‑time source fingerprinting & AI explanations
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-sm">
            {data && (
              <div className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-full border border-emerald-500/30 bg-emerald-500/10">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-300 font-medium">Model ready</span>
                <span className="text-slate-400">
                  CV accuracy{" "}
                  <span className="font-semibold text-slate-100">
                    {(data.model_accuracy * 100).toFixed(1)}%
                  </span>
                </span>
              </div>
            )}
            {data && (
              <div className="hidden md:flex flex-col items-end text-xs leading-tight text-slate-400">
                <span className="uppercase tracking-wide text-slate-500">
                  Last refreshed
                </span>
                <span className="font-mono text-slate-200">
                  {data.timestamp.replace("T", " ").split(".")[0]}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main content: full-screen map dashboard */}
      <main className="flex-1 relative min-h-0">
        <PollutionMap
          stations={data?.stations || []}
          modelAccuracy={data?.model_accuracy || 0}
          selectedStation={selectedStation}
          onSelectStation={(s) => setSelectedStationName(s?.station_name || null)}
          currentTime={data?.timestamp || ""}
          availableRange={data?.available_range}
          onTimeChange={handleTimeChange}
          isHistorical={isHistorical}
        />
      </main>

      {/* Subtle footer */}
      <footer className="hidden sm:block flex-shrink-0 border-t border-slate-900 bg-slate-950/80 text-xs text-slate-500">
        <div className="max-w-6xl mx-auto px-4 py-2.5 flex items-center justify-between">
          <span>
            Built for Delhi air quality intelligence &amp; policy diagnostics.
          </span>
          <span className="hidden md:inline">
            Data: OpenAQ, Open‑Meteo, FIRMS · Model: XGBoost multi‑class source classifier
          </span>
        </div>
      </footer>
    </div>
  );
}
