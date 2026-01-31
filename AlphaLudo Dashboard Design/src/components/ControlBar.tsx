import { StopCircle, Plus, Minus } from 'lucide-react';

interface ControlBarProps {
  actorCount: number;
  sessionTime: number;
  onSpawnActor: () => void;
  onKillActor: () => void;
  onStop: () => void;
}

export function ControlBar({
  actorCount,
  sessionTime,
  onSpawnActor,
  onKillActor,
  onStop,
}: ControlBarProps) {
  const formatTime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="glass-panel border-b border-white/10 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-8">
        <h1 className="text-xl tracking-tight text-white/95">AlphaLudo</h1>
        <div className="flex items-center gap-3 px-4 py-1.5 bg-white/5 backdrop-blur-xl rounded-full border border-white/10">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
          <span className="font-mono text-sm text-white/70">{formatTime(sessionTime)}</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Actor Controls */}
        <div className="flex items-center gap-3 px-4 py-1.5 bg-white/5 backdrop-blur-xl rounded-full border border-white/10">
          <span className="text-xs text-white/50">Actors</span>
          <span className="font-mono text-sm text-white/90">{actorCount}</span>
          <div className="flex gap-0.5">
            <button
              onClick={onSpawnActor}
              disabled={actorCount >= 32}
              className="p-1 hover:bg-white/10 rounded-full disabled:opacity-30 disabled:cursor-not-allowed transition-all active:scale-95"
              title="Spawn Actor"
            >
              <Plus className="w-3.5 h-3.5 text-white/70" />
            </button>
            <button
              onClick={onKillActor}
              disabled={actorCount <= 1}
              className="p-1 hover:bg-white/10 rounded-full disabled:opacity-30 disabled:cursor-not-allowed transition-all active:scale-95"
              title="Kill Actor"
            >
              <Minus className="w-3.5 h-3.5 text-white/70" />
            </button>
          </div>
        </div>

        {/* Stop Button */}
        <button
          onClick={onStop}
          className="flex items-center gap-2 px-4 py-1.5 bg-red-500/90 backdrop-blur-xl hover:bg-red-500 rounded-full transition-all active:scale-95 border border-red-400/30 shadow-[0_0_20px_rgba(239,68,68,0.3)]"
        >
          <StopCircle className="w-4 h-4 text-white" />
          <span className="text-sm text-white">Stop</span>
        </button>
      </div>
    </div>
  );
}