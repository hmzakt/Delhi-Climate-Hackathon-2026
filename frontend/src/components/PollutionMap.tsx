"use client";

import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import { useEffect, useState } from "react";
import "leaflet/dist/leaflet.css";
import type { StationData } from "@/app/page";
import ExplanationPanel from "./ExplanationPanel";

/* â”€â”€ source colours â”€â”€ */
const SOURCE_COLORS: Record<string, string> = {
    traffic: "#f59e0b",
    biomass: "#ef4444",
    secondary_aerosols: "#c026d3",
    stagnation: "#8b5cf6",
    winter_inversion: "#3b82f6",
    soil_road_dust: "#78350f",
    firecrackers: "#ec4899",
    urban_mix: "#06b6d4",
    clean_air: "#10b981",
    background: "#6b7280",
};

const SOURCE_ICONS: Record<string, string> = {
    traffic: "🚗",
    biomass: "🔥",
    secondary_aerosols: "🧪",
    stagnation: "🌫️",
    winter_inversion: "❄️",
    soil_road_dust: "🚜",
    firecrackers: "🎇",
    urban_mix: "🏙️",
    clean_air: "🌿",
    background: "📊",
};

function aqiSeverity(aqi: number): string {
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Moderate";
    if (aqi <= 150) return "Unhealthy (S)";
    if (aqi <= 200) return "Unhealthy";
    if (aqi <= 300) return "Very Unhealthy";
    return "Hazardous";
}

function aqiColor(aqi: number): string {
    if (aqi <= 50) return "#10b981"; // Good
    if (aqi <= 100) return "#22c55e"; // Satisfactory
    if (aqi <= 200) return "#f59e0b"; // Moderate
    if (aqi <= 300) return "#f97316"; // Poor
    if (aqi <= 400) return "#ef4444"; // Very Poor
    return "#991b1b"; // Severe
}

/* â”€â”€ Sparkline (mini chart) â”€â”€ */
function Sparkline({ data, width = 200, height = 40 }: { data: number[]; width?: number; height?: number }) {
    if (!data.length) return null;
    const max = Math.max(...data, 1);
    const min = Math.min(...data, 0);
    const range = max - min || 1;
    const points = data
        .map((v, i) => `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * height}`)
        .join(" ");
    return (
        <svg width={width} height={height} className="mt-1">
            <polyline fill="none" stroke="url(#sparkGrad)" strokeWidth="2" points={points} />
            <defs>
                <linearGradient id="sparkGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#06b6d4" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
            </defs>
        </svg>
    );
}

/* â”€â”€ Side panel â”€â”€ */
function StationPanel({
    station,
    onClose,
    onExplainClick,
}: {
    station: StationData;
    onClose: () => void;
    onExplainClick: () => void;
}) {
    const srcColor = SOURCE_COLORS[station.predicted_source] || "#6b7280";
    const srcIcon = SOURCE_ICONS[station.predicted_source] || "📊";

    const sortedProbs = Object.entries(station.probabilities).sort(([, a], [, b]) => b - a);

    return (
        <div className="absolute top-0 right-0 w-[400px] h-full z-[1000] bg-slate-950/95 backdrop-blur-xl border-l border-slate-700/70 overflow-y-auto">
            {/* header */}
            <div className="p-5 border-b border-slate-800/80">
                <div className="flex justify-between items-start">
                    <div>
                        <p className="text-xs text-slate-400 uppercase tracking-[0.16em] mb-1">Station</p>
                        <h2 className="text-slate-100 font-bold text-xl leading-tight">
                            {station.station_name.replace(", Delhi - DPCC", "")}
                        </h2>
                    </div>
                    <button onClick={onClose} className="text-slate-400 hover:text-slate-100 transition p-1 -mt-1 text-lg">×</button>
                </div>
                <p className="text-slate-400 text-sm mt-1">
                    {station.latitude.toFixed(4)}°N, {station.longitude.toFixed(4)}°E &nbsp;·&nbsp; {station.timestamp}
                </p>
            </div>

            {/* Predicted Source (Spotlight) */}
            <div className="p-5 border-b border-slate-800/80 bg-slate-900/35">
                <div className="flex items-center gap-4 mb-3">
                    <div
                        className="w-16 h-16 rounded-2xl flex items-center justify-center text-3xl shadow-xl"
                        style={{ background: `${srcColor}30`, border: `2px solid ${srcColor}50` }}
                    >
                        {srcIcon}
                    </div>
                    <div>
                        <p className="text-white font-extrabold text-2xl tracking-tight uppercase">
                            {station.predicted_source.replace(/_/g, " ")}
                        </p>
                        <p className="text-xs font-mono uppercase tracking-[0.14em]" style={{ color: srcColor }}>
                            Fingerprint Confidence: {(station.confidence * 100).toFixed(1)}%
                        </p>
                    </div>
                </div>

                <div className="mt-4 bg-slate-950 p-4 rounded-xl border border-slate-700/70 flex items-center justify-between">
                    <div>
                        <p className="text-slate-400 text-xs uppercase font-semibold tracking-[0.14em] mb-1">Global AQI (Fetched)</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-4xl font-black text-white">{station.aqi}</span>
                            <span className="text-xs font-semibold px-2 py-0.5 rounded-full"
                                style={{ background: `${aqiColor(station.aqi)}20`, color: aqiColor(station.aqi), border: `1px solid ${aqiColor(station.aqi)}40` }}>
                                {aqiSeverity(station.aqi)}
                            </span>
                        </div>
                    </div>
                    <div className="text-right">
                        <p className="text-slate-400 text-xs uppercase font-semibold mb-1">PM2.5 Dominance</p>
                        <p className="text-white font-mono text-sm">{station.pm25.toFixed(0)} <span className="text-slate-500">µg/m³</span></p>
                    </div>
                </div>
            </div>

            {/* Meteorology & Secondary Metrics */}
            <div className="p-5 border-b border-slate-800/80">
                <p className="text-xs text-slate-400 uppercase tracking-[0.14em] font-semibold mb-4">Live Metrics</p>
                <div className="grid grid-cols-2 gap-3">
                    {[
                        { label: "NO2", value: station.no2 ? `${station.no2.toFixed(1)} µg/m³` : "N/A" },
                        { label: "TEMP", value: `${station.temperature.toFixed(1)}°C` },
                        { label: "WIND", value: `${station.wind_speed.toFixed(1)} km/h` },
                        { label: "HUMID", value: `${station.humidity.toFixed(0)}%` },
                    ].map((item) => (
                        <div key={item.label} className="bg-slate-900/50 p-3 rounded-xl border border-slate-700/70">
                            <p className="text-xs text-slate-400 uppercase font-semibold mb-1">{item.label}</p>
                            <p className="text-white font-mono text-sm">{item.value}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* PM2.5 history */}
            <div className="p-5 border-b border-slate-800/80">
                <div className="flex items-center justify-between mb-4">
                    <p className="text-xs text-slate-400 uppercase font-semibold tracking-[0.14em]">24H PM2.5 Profile</p>
                    <span className="text-slate-500 text-xs font-mono">Real-time Sparkline</span>
                </div>
                <div className="bg-slate-900/30 p-4 rounded-2xl border border-slate-700/60">
                    <Sparkline data={station.pm25_history} width={300} height={60} />
                </div>
            </div>

            {/* probability distribution */}
            <div className="p-5 border-b border-slate-800/80">
                <p className="text-xs text-slate-400 uppercase font-semibold tracking-[0.14em] mb-4">Source Probability Distribution</p>
                <div className="space-y-2.5">
                    {sortedProbs.map(([cls, prob]) => (
                        <div key={cls} className="flex items-center gap-3">
                            <span className="text-slate-300 text-xs w-28 capitalize font-medium flex items-center gap-1.5">
                                <span className="text-xs">{SOURCE_ICONS[cls]}</span>
                                {cls.replace(/_/g, " ")}
                            </span>
                            <div className="flex-1 bg-slate-900/50 rounded-full h-2 overflow-hidden border border-slate-700/70">
                                <div
                                    className="h-full rounded-full transition-all duration-700 ease-out"
                                    style={{
                                        width: `${Math.max(prob * 100, 1)}%`,
                                        background: SOURCE_COLORS[cls] || "#6b7280",
                                        boxShadow: `0 0 10px ${SOURCE_COLORS[cls]}40`
                                    }}
                                />
                            </div>
                            <span className="text-slate-400 font-mono text-xs w-10 text-right">
                                {(prob * 100).toFixed(0)}%
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* AI Explanation Section */}
            <div className="p-5">
                <p className="text-xs text-slate-400 uppercase font-semibold tracking-[0.14em] mb-3">AI-Powered Insights</p>
                <div className="flex flex-col gap-2">
                    <button
                        onClick={onExplainClick}
                        className="w-full px-4 py-3 bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 border border-cyan-500/30 rounded-xl hover:from-cyan-500/30 hover:to-emerald-500/30 transition-all group"
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <span className="text-xl">🧠</span>
                                <div className="text-left">
                                    <p className="text-white font-semibold text-sm">Explain This Spike</p>
                                    <p className="text-slate-300 text-xs">Get AI-powered explanation</p>
                                </div>
                            </div>
                            <span className="text-slate-400 group-hover:text-cyan-300 transition text-lg">→</span>
                        </div>
                    </button>
                    <p className="text-slate-500 text-xs mt-1 text-center">
                        Choose between public-friendly or policy-level insights
                    </p>
                </div>
            </div>
        </div>
    );
}

/* â”€â”€ Fit bounds helper â”€â”€ */
function FitBounds({ stations }: { stations: StationData[] }) {
    const map = useMap();
    useEffect(() => {
        if (stations.length) {
            const bounds = stations.map((s) => [s.latitude, s.longitude] as [number, number]);
            map.fitBounds(bounds, { padding: [80, 80], maxZoom: 13 });
        }
    }, [stations, map]);
    return null;
}

/* â”€â”€ Main Map Component â”€â”€ */
export default function PollutionMap({
    stations,
    modelAccuracy,
    selectedStation,
    onSelectStation,
    currentTime,
    availableRange,
    onTimeChange,
    isHistorical,
}: {
    stations: StationData[];
    modelAccuracy: number;
    selectedStation: StationData | null;
    onSelectStation: (s: StationData | null) => void;
    currentTime: string;
    availableRange?: { start: string; end: string };
    onTimeChange: (time: string | null) => void;
    isHistorical: boolean;
}) {
    // Guard against SSR / hydration issues with Leaflet
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    // Local state for the date/time inputs with safety checks
    const [inputDate, setInputDate] = useState(currentTime?.includes("T") ? currentTime.split("T")[0] : "");
    const [inputTime, setInputTime] = useState(currentTime?.includes("T") ? currentTime.split("T")[1].substring(0, 5) : "");
    const [showExplanation, setShowExplanation] = useState(false);

    // Update local inputs when currentTime changes
    useEffect(() => {
        if (currentTime?.includes("T")) {
            setInputDate(currentTime.split("T")[0]);
            setInputTime(currentTime.split("T")[1].substring(0, 5));
        }
    }, [currentTime]);

    const handleTravel = () => {
        const iso = `${inputDate}T${inputTime}:00.000Z`;
        onTimeChange(iso);
    };

    const jumpHour = (hours: number) => {
        const currentMs = new Date(currentTime).getTime();
        const targetMs = currentMs + (hours * 1000 * 60 * 60);
        onTimeChange(new Date(targetMs).toISOString());
    };

    // Avoid rendering map until mounted to prevent Leaflet DOM errors
    if (!mounted) {
        return null;
    }

    return (
        <div className="relative w-full h-full min-h-0">
            {/* Mode / legend overlay */}
            <div className="absolute top-4 left-4 z-[1000] bg-slate-950/85 backdrop-blur-xl rounded-2xl border border-slate-700/70 px-4 py-3.5 max-w-sm shadow-2xl space-y-3">
                <div className="flex items-center justify-between gap-3">
                    <div className="flex flex-col gap-1">
                        <p className="text-xs uppercase tracking-[0.18em] text-slate-400 font-semibold">
                            View Mode
                        </p>
                        <p className="flex items-center gap-2 text-sm text-slate-200">
                            <span className={isHistorical ? "h-2 w-2 rounded-full bg-amber-400" : "h-2 w-2 rounded-full bg-emerald-400"}>
                                &nbsp;
                            </span>
                            <span className="font-medium">
                                {isHistorical ? "Historical snapshot" : "Live now"}
                            </span>
                        </p>
                        <p className="text-xs text-slate-400 font-mono">
                            {currentTime.replace("T", " ").split(".")[0]}
                        </p>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                        {isHistorical && (
                            <button
                                onClick={() => onTimeChange(null)}
                                className="text-xs px-2.5 py-1 bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 rounded-full hover:bg-cyan-500/20 transition"
                            >
                                Reset to live
                            </button>
                        )}
                        <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-400">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                            <span>Model accuracy</span>
                            <span className="font-mono text-slate-200">
                                {(modelAccuracy * 100).toFixed(1)}%
                            </span>
                        </div>
                    </div>
                </div>

                {/* Legend */}
                <div className="pt-1 border-t border-slate-800 mt-2">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-400 mb-1">
                        Source legend
                    </p>
                    <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1.5">
                        {Object.entries(SOURCE_COLORS).map(([src, color]) => (
                            <div key={src} className="flex items-center gap-1.5">
                                <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                                <span className="text-slate-300 text-xs capitalize tracking-tight">
                                    {src.replace(/_/g, " ")}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Time Travel Control Panel */}
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-[1000] w-[95%] max-w-2xl bg-slate-950/90 backdrop-blur-xl rounded-2xl border border-slate-700/70 p-5 shadow-2xl">
                <div className="flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                        <span className="text-slate-300 text-xs uppercase font-bold tracking-widest">Temporal Navigation</span>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => jumpHour(-1)}
                                className="px-2 py-1.5 bg-slate-900 border border-slate-700 rounded-lg hover:bg-slate-800 text-slate-300 transition"
                            >
                                ← 1h
                            </button>
                            <span className="text-cyan-300 font-mono text-sm">{currentTime.replace("T", " ").split(".")[0]}</span>
                            <button
                                onClick={() => jumpHour(1)}
                                className="px-2 py-1.5 bg-slate-900 border border-slate-700 rounded-lg hover:bg-slate-800 text-slate-300 transition"
                            >
                                1h →
                            </button>
                        </div>
                    </div>

                    <div className="flex items-center gap-3">
                        <div className="flex-1 flex gap-2">
                            <input
                                type="date"
                                value={inputDate}
                                onChange={(e) => setInputDate(e.target.value)}
                                min={availableRange?.start.split("T")[0]}
                                max={availableRange?.end.split("T")[0]}
                                className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-cyan-400/60 transition-colors"
                            />
                            <input
                                type="time"
                                value={inputTime}
                                onChange={(e) => setInputTime(e.target.value)}
                                className="w-32 bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-white text-sm focus:outline-none focus:border-cyan-400/60 transition-colors"
                            />
                        </div>
                        <button
                            onClick={handleTravel}
                            className="bg-cyan-400 hover:bg-cyan-300 text-slate-950 px-6 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-[0_0_20px_rgba(34,211,238,0.28)] active:scale-95 whitespace-nowrap"
                        >
                            TRAVEL
                        </button>
                    </div>

                    {availableRange && (
                        <p className="text-xs text-slate-500 text-center italic">
                            Available Data: {availableRange.start.split("T")[0]} to {availableRange.end.split("T")[0]}
                        </p>
                    )}
                </div>
            </div>

            {/* Map */}
            <MapContainer
                center={[28.62, 77.16]}
                zoom={12}
                className="w-full h-full"
                zoomControl={false}
                style={{ background: "#0a0f1e" }}
            >

                <TileLayer
                    attribution='&copy; <a href="https://carto.com">CARTO</a>'
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />
                <FitBounds stations={stations} />

                {stations.map((s) => {
                    return (
                        <CircleMarker
                            key={s.station_name}
                            center={[s.latitude, s.longitude]}
                            radius={10 + (s.aqi / 40)}
                            fillColor={aqiColor(s.aqi)}
                            color="white"
                            weight={selectedStation?.station_name === s.station_name ? 3 : 1}
                            fillOpacity={0.8}
                            eventHandlers={{
                                click: () => onSelectStation(s),
                            }}
                        >
                            <Popup>
                                <div className="font-sans tracking-tight">
                                    <p className="font-bold text-slate-100 border-b border-slate-700 pb-1.5 mb-1.5">{s.station_name.split(",")[0]}</p>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xl font-bold" style={{ color: aqiColor(s.aqi) }}>{s.aqi}</span>
                                        <span className="text-xs text-slate-400 uppercase tracking-[0.08em]">US AQI (Fetched)</span>
                                    </div>
                                    <p className="text-xs text-slate-300 mt-1 capitalize">Primary Source: {s.predicted_source.replace(/_/g, " ")}</p>
                                </div>
                            </Popup>
                        </CircleMarker>
                    );
                })}
            </MapContainer>

            {/* Side panel */}
            {selectedStation && (
                <>
                    <StationPanel
                        station={selectedStation}
                        onClose={() => onSelectStation(null)}
                        onExplainClick={() => setShowExplanation(true)}
                    />
                    {/* Explanation Panel Modal */}
                    {showExplanation && (
                        <ExplanationPanel
                            station={selectedStation}
                            onClose={() => setShowExplanation(false)}
                        />
                    )}
                </>
            )}
        </div>
    );
}

