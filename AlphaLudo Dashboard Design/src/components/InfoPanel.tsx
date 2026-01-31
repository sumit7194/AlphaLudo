import { Trophy, Target, Bot, Hash, Gamepad2, Database, Cpu, HardDrive, Activity } from 'lucide-react';
import { LeaderboardEntry, SystemStat } from '../hooks/useTrainingData';

interface InfoPanelProps {
  leaderboard: LeaderboardEntry[];
  iteration: number;
  gamesPlayed: number;
  bufferSize: number;
  systemStats: SystemStat[];
  deviceInfo: string;
  mctsThreshold: number;
  onUpdateMctsThreshold: (value: number) => void;
}

export function InfoPanel({
  leaderboard,
  iteration,
  gamesPlayed,
  bufferSize,
  systemStats,
  deviceInfo,
  mctsThreshold,
  onUpdateMctsThreshold,
}: InfoPanelProps) {
  const formatNumber = (num: number) => num.toLocaleString('en-US');
  
  const avgCpu = systemStats.reduce((sum, stat) => sum + stat.cpu, 0) / systemStats.length;
  const avgRam = systemStats.reduce((sum, stat) => sum + stat.ram, 0) / systemStats.length;

  const getIcon = (type: LeaderboardEntry['type'], index: number) => {
    if (index === 0) return <Trophy className="w-3.5 h-3.5 text-yellow-500 drop-shadow-[0_0_6px_rgba(234,179,8,0.6)]" />;
    if (type === 'bot') return <Bot className="w-3.5 h-3.5 text-white/30" />;
    return <Target className="w-3.5 h-3.5 text-blue-400" />;
  };

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Leaderboard */}
      <div className="glass-panel flex-shrink-0">
        <div className="px-5 pt-4 pb-3 border-b border-white/10">
          <h3 className="text-xs tracking-wide uppercase text-white/40">Leaderboard</h3>
        </div>
        <div className="p-3 space-y-1.5">
          {leaderboard.map((entry, index) => (
            <div
              key={entry.name}
              className={`flex items-center justify-between px-3 py-2 rounded-xl transition-all ${
                entry.type === 'current' 
                  ? 'bg-blue-500/15 border-l-2 border-blue-500 shadow-[0_0_12px_rgba(59,130,246,0.2)]' 
                  : 'bg-white/5 hover:bg-white/10 border border-white/5'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-white/30 text-xs w-4">{index + 1}</span>
                {getIcon(entry.type, index)}
                <span className={`text-xs ${entry.type === 'current' ? 'text-white/90' : 'text-white/60'}`}>
                  {entry.name}
                </span>
              </div>
              <span className="font-mono text-xs text-white/70">{entry.elo}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Metrics */}
      <div className="glass-panel flex-shrink-0">
        <div className="px-5 pt-4 pb-3 border-b border-white/10">
          <h3 className="text-xs tracking-wide uppercase text-white/40">Metrics</h3>
        </div>
        <div className="p-3 space-y-1.5">
          <div className="flex items-center justify-between px-3 py-2 bg-white/5 rounded-xl border border-white/5">
            <div className="flex items-center gap-2">
              <Hash className="w-3.5 h-3.5 text-white/30" />
              <span className="text-xs text-white/60">Iteration</span>
            </div>
            <span className="font-mono text-xs text-white/70">{formatNumber(iteration)}</span>
          </div>
          <div className="flex items-center justify-between px-3 py-2 bg-white/5 rounded-xl border border-white/5">
            <div className="flex items-center gap-2">
              <Gamepad2 className="w-3.5 h-3.5 text-white/30" />
              <span className="text-xs text-white/60">Games</span>
            </div>
            <span className="font-mono text-xs text-white/70">{formatNumber(gamesPlayed)}</span>
          </div>
          <div className="flex items-center justify-between px-3 py-2 bg-white/5 rounded-xl border border-white/5">
            <div className="flex items-center gap-2">
              <Database className="w-3.5 h-3.5 text-white/30" />
              <span className="text-xs text-white/60">Buffer</span>
            </div>
            <span className="font-mono text-[10px] text-white/70">{formatNumber(bufferSize)}</span>
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="glass-panel flex-1 flex flex-col overflow-hidden">
        <div className="px-5 pt-4 pb-3 border-b border-white/10">
          <h3 className="text-xs tracking-wide uppercase text-white/40">System Health</h3>
        </div>
        <div className="p-3 space-y-1.5 flex-shrink-0">
          <div className="bg-white/5 rounded-xl p-3 border border-white/5">
            <div className="flex items-center gap-2 text-white/40 mb-2">
              <Cpu className="w-3.5 h-3.5" />
              <span className="text-xs">CPU</span>
            </div>
            <div className="text-lg font-light text-white/90">{avgCpu.toFixed(1)}%</div>
            <div className="mt-2 bg-white/5 rounded-full h-1.5 overflow-hidden border border-white/10">
              <div
                className={`h-full transition-all rounded-full ${avgCpu > 80 ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]' : avgCpu > 60 ? 'bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.6)]' : 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]'}`}
                style={{ width: `${avgCpu}%` }}
              />
            </div>
          </div>

          <div className="bg-white/5 rounded-xl p-3 border border-white/5">
            <div className="flex items-center gap-2 text-white/40 mb-2">
              <HardDrive className="w-3.5 h-3.5" />
              <span className="text-xs">RAM</span>
            </div>
            <div className="text-lg font-light text-white/90">{avgRam.toFixed(0)} MB</div>
            <div className="mt-2 bg-white/5 rounded-full h-1.5 overflow-hidden border border-white/10">
              <div
                className="bg-blue-500 h-full transition-all rounded-full shadow-[0_0_8px_rgba(59,130,246,0.6)]"
                style={{ width: `${(avgRam / 1024) * 100}%` }}
              />
            </div>
          </div>

          <div className="bg-white/5 rounded-xl p-3 border border-white/5">
            <div className="flex items-center gap-2 text-white/40 mb-2">
              <Activity className="w-3.5 h-3.5" />
              <span className="text-xs">Processes</span>
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-lg font-light text-white/90">{systemStats.length}</div>
              <div className="text-xs text-white/40">{deviceInfo}</div>
            </div>
          </div>
        </div>

        {/* Process List - Scrollable */}
        <div className="flex-1 overflow-auto px-3 pb-3">
          <div className="space-y-1">
            {systemStats.map((stat) => (
              <div
                key={stat.processId}
                className="bg-white/5 rounded-lg px-3 py-2 hover:bg-white/10 transition-colors border border-white/5"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] font-mono text-white/40">{stat.processId}</span>
                  <span className="text-[10px] text-green-500 drop-shadow-[0_0_4px_rgba(34,197,94,0.6)]">●</span>
                </div>
                <div className="flex gap-2">
                  <div className="flex-1">
                    <div className="bg-white/5 rounded-full h-1 overflow-hidden border border-white/10">
                      <div
                        className={`h-full ${stat.cpu > 80 ? 'bg-red-500' : stat.cpu > 60 ? 'bg-yellow-500' : 'bg-green-500'}`}
                        style={{ width: `${stat.cpu}%` }}
                      />
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="bg-white/5 rounded-full h-1 overflow-hidden border border-white/10">
                      <div
                        className="bg-blue-500 h-full"
                        style={{ width: `${(stat.ram / 1024) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* MCTS Control */}
      <div className="glass-panel flex-shrink-0 p-4">
        <div className="text-xs text-white/40 mb-2 tracking-wide uppercase">MCTS Threshold</div>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min="0.5"
            max="1.0"
            step="0.01"
            value={mctsThreshold}
            onChange={(e) => onUpdateMctsThreshold(parseFloat(e.target.value))}
            className="flex-1 h-1.5 appearance-none bg-white/5 rounded-full outline-none slider border border-white/10"
            style={{
              background: `linear-gradient(to right, #0a84ff 0%, #0a84ff ${((mctsThreshold - 0.5) / 0.5) * 100}%, rgba(255,255,255,0.05) ${((mctsThreshold - 0.5) / 0.5) * 100}%, rgba(255,255,255,0.05) 100%)`
            }}
          />
          <span className="font-mono text-xs text-white/70 w-10 text-right">
            {mctsThreshold.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
