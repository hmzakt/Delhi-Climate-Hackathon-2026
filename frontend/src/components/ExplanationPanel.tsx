"use client";

import { useState } from "react";
import type { StationData } from "@/app/page";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface ExplanationResponse {
    explanation: string;
    context: Record<string, unknown>;
    audience: string;
    model_used: string;
}

interface ExplanationPanelProps {
    station: StationData;
    onClose: () => void;
}

export default function ExplanationPanel({ station, onClose }: ExplanationPanelProps) {
    const [explanation, setExplanation] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [audience, setAudience] = useState<"public" | "authority">("public");
    const [modelUsed, setModelUsed] = useState<string>("");

    const fetchExplanation = async (audienceType: "public" | "authority") => {
        setLoading(true);
        setError(null);
        setAudience(audienceType);
        
        try {
            const response = await fetch(`${API_BASE}/explain`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    station_name: station.station_name,
                    timestamp: station.timestamp,
                    audience: audienceType,
                }),
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data: ExplanationResponse = await response.json();
            setExplanation(data.explanation);
            setModelUsed(data.model_used);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to fetch explanation");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[2000] bg-slate-950/70 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-slate-950 border border-slate-700/70 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col shadow-2xl">
                {/* Header */}
                <div className="p-6 border-b border-slate-800/80 flex items-center justify-between">
                    <div>
                        <h2 className="text-slate-100 font-bold text-2xl tracking-tight mb-1">
                            AI Explanation: {station.station_name.replace(", Delhi - DPCC", "")}
                        </h2>
                        <p className="text-slate-400 text-sm">
                            {station.timestamp.replace("T", " ").split(".")[0]}
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-slate-400 hover:text-slate-100 transition p-2 hover:bg-slate-800 rounded-lg text-lg"
                    >
                        ×
                    </button>
                </div>

                {/* Audience Selection */}
                <div className="p-6 border-b border-slate-800/80 bg-slate-900/35">
                    <p className="text-slate-300 text-sm mb-3.5">Select explanation type:</p>
                    <div className="flex gap-3">
                        <button
                            onClick={() => fetchExplanation("public")}
                            disabled={loading}
                            className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
                                audience === "public" && explanation
                                    ? "bg-cyan-500/20 text-cyan-300 border-2 border-cyan-500/50"
                                    : "bg-slate-900 text-slate-200 border border-slate-700 hover:bg-slate-800"
                            } ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
                        >
                            👥 Explain to Me (Public)
                        </button>
                        <button
                            onClick={() => fetchExplanation("authority")}
                            disabled={loading}
                            className={`flex-1 px-4 py-3 rounded-xl font-semibold transition-all ${
                                audience === "authority" && explanation
                                    ? "bg-emerald-500/20 text-emerald-300 border-2 border-emerald-500/50"
                                    : "bg-slate-900 text-slate-200 border border-slate-700 hover:bg-slate-800"
                            } ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
                        >
                         Policy-Level Insight
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 text-[15px]">
                    {loading && (
                        <div className="flex flex-col items-center justify-center py-12">
                            <div className="w-12 h-12 border-[3px] border-cyan-300 border-t-transparent rounded-full animate-spin mb-4" />
                            <p className="text-slate-300">Generating explanation...</p>
                        </div>
                    )}

                    {error && (
                        <div className="bg-red-500/10 border border-red-500/40 rounded-xl p-4 mb-4">
                            <p className="text-red-300 text-sm">{error}</p>
                        </div>
                    )}

                    {explanation && !loading && (
                        <div className="space-y-4">
                            <div className="max-w-none">
                                <div className="text-slate-200 leading-relaxed whitespace-pre-wrap text-[15px]">
                                    {explanation.split("\n").map((line, idx) => {
                                        // Format headings
                                        if (line.startsWith("**") && line.endsWith("**")) {
                                            return (
                                                <h3 key={idx} className="text-slate-100 font-bold text-lg mt-6 mb-3.5">
                                                    {line.replace(/\*\*/g, "")}
                                                </h3>
                                            );
                                        }
                                        // Format bullet points
                                        if (line.trim().startsWith("-")) {
                                            return (
                                                <li key={idx} className="ml-5 mb-2 list-disc">
                                                    {line.trim().substring(1).trim()}
                                                </li>
                                            );
                                        }
                                        // Regular paragraphs
                                        if (line.trim()) {
                                            return (
                                                <p key={idx} className="mb-3.5">
                                                    {line}
                                                </p>
                                            );
                                        }
                                        return <br key={idx} />;
                                    })}
                                </div>
                            </div>

                            {modelUsed && (
                                <div className="mt-6 pt-4 border-t border-slate-800/80">
                                    <p className="text-slate-400 text-xs">
                                        Generated using: <span className="text-slate-300 font-mono">{modelUsed}</span>
                                    </p>
                                </div>
                            )}
                        </div>
                    )}

                    {!explanation && !loading && !error && (
                        <div className="text-center py-12 text-slate-400">
                            <p className="mb-4">Select an explanation type above to get started.</p>
                            <p className="text-sm text-slate-500">
                                Get AI-powered insights about this pollution spike tailored to your needs.
                            </p>
                        </div>
                    )}
                </div>

                {/* Footer with Feedback */}
                {explanation && (
                    <div className="p-6 border-t border-slate-800/80 bg-slate-900/35">
                        <p className="text-slate-400 text-sm mb-3.5">Was this explanation helpful?</p>
                        <div className="flex gap-2">
                            <button
                                onClick={async () => {
                                    try {
                                        await fetch(`${API_BASE}/explain/feedback`, {
                                            method: "POST",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ helpful: true }),
                                        });
                                        alert("Thank you for your feedback! ðŸ™");
                                    } catch {
                                        // Silent fail
                                    }
                                }}
                                className="flex-1 px-4 py-2 bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 rounded-lg hover:bg-emerald-500/30 transition text-sm font-semibold"
                            >
                                ✓ Helpful
                            </button>
                            <button
                                onClick={async () => {
                                    try {
                                        await fetch(`${API_BASE}/explain/feedback`, {
                                            method: "POST",
                                            headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ helpful: false }),
                                        });
                                        alert("Thank you for your feedback! We'll work to improve. ðŸ™");
                                    } catch {
                                        // Silent fail
                                    }
                                }}
                                className="flex-1 px-4 py-2 bg-rose-500/20 text-rose-300 border border-rose-500/40 rounded-lg hover:bg-rose-500/30 transition text-sm font-semibold"
                            >
                                ✗ Not Helpful
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

