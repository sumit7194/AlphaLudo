import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { EloDataPoint, LossDataPoint } from '../hooks/useTrainingData';

interface MetricsHeroProps {
  mainElo: number;
  eloHistory: EloDataPoint[];
  loss: number;
  lossHistory: LossDataPoint[];
  winRate: number;
}

export function MetricsHero({ mainElo, eloHistory, loss, lossHistory, winRate }: MetricsHeroProps) {
  const eloTrend = eloHistory.length > 1 
    ? eloHistory[eloHistory.length - 1].value - eloHistory[eloHistory.length - 10]?.value 
    : 0;
  const lossTrend = lossHistory.length > 1 
    ? lossHistory[lossHistory.length - 1].value - lossHistory[lossHistory.length - 10]?.value 
    : 0;

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="grid grid-cols-3 gap-4 h-64">
      {/* Main Elo */}
      <div className="col-span-2 glass-panel overflow-hidden">
        <div className="p-6 h-full flex flex-col">
          <div className="flex items-start justify-between mb-4">
            <div>
              <div className="text-xs text-white/40 mb-1 tracking-wide uppercase">Main Elo</div>
              <div className="flex items-baseline gap-3">
                <span className="text-5xl font-light tracking-tight text-white/95">{Math.round(mainElo)}</span>
                <div className={`flex items-center gap-1 text-xs px-2 py-1 rounded-full ${eloTrend >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                  {eloTrend >= 0 ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                  <span>{eloTrend >= 0 ? '+' : ''}{eloTrend.toFixed(1)}</span>
                </div>
              </div>
            </div>
          </div>
          <div className="flex-1 -mx-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={eloHistory}>
                <defs>
                  <linearGradient id="eloGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#0a84ff" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#0a84ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff" strokeOpacity={0.05} />
                <XAxis 
                  dataKey="time" 
                  tickFormatter={formatTime}
                  stroke="#ffffff"
                  tick={{ fontSize: 11, fill: '#ffffff60' }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis 
                  domain={['dataMin - 20', 'dataMax + 20']}
                  stroke="#ffffff"
                  tick={{ fontSize: 11, fill: '#ffffff60' }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'rgba(28, 28, 30, 0.95)', 
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '12px',
                    backdropFilter: 'blur(20px)',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
                    color: '#f5f5f7'
                  }}
                  labelFormatter={formatTime}
                  formatter={(value: number) => [value.toFixed(1), 'Elo']}
                />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#0a84ff" 
                  strokeWidth={3} 
                  dot={false}
                  fill="url(#eloGradient)"
                  filter="drop-shadow(0 0 8px rgba(10, 132, 255, 0.5))"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Loss and Win Rate */}
      <div className="flex flex-col gap-4">
        {/* Loss */}
        <div className="glass-panel flex-1 overflow-hidden">
          <div className="p-5 h-full flex flex-col">
            <div className="text-xs text-white/40 mb-1 tracking-wide uppercase">Loss</div>
            <div className="flex items-baseline gap-2 mb-3">
              <span className="text-3xl font-light tracking-tight text-white/95">{loss.toFixed(4)}</span>
              <div className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${lossTrend <= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                {lossTrend <= 0 ? <TrendingDown className="w-3 h-3" /> : <TrendingUp className="w-3 h-3" />}
              </div>
            </div>
            <div className="flex-1 -mx-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff" strokeOpacity={0.03} />
                  <XAxis dataKey="time" hide />
                  <YAxis domain={['dataMin - 0.005', 'dataMax + 0.005']} hide />
                  <Tooltip
                    contentStyle={{ 
                      backgroundColor: 'rgba(28, 28, 30, 0.95)', 
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      borderRadius: '12px',
                      backdropFilter: 'blur(20px)',
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
                      color: '#f5f5f7'
                    }}
                    labelFormatter={formatTime}
                    formatter={(value: number) => [value.toFixed(4), 'Loss']}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#ff453a" 
                    strokeWidth={2.5} 
                    dot={false}
                    filter="drop-shadow(0 0 6px rgba(255, 69, 58, 0.5))"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Win Rate */}
        <div className="glass-panel flex-1 p-5">
          <div className="text-xs text-white/40 mb-1 tracking-wide uppercase">Win Rate</div>
          <div className="text-3xl font-light tracking-tight text-white/95 mb-3">{(winRate * 100).toFixed(1)}%</div>
          <div className="bg-white/5 rounded-full h-2 overflow-hidden backdrop-blur-sm border border-white/10">
            <div
              className="bg-gradient-to-r from-blue-500 to-blue-400 h-full transition-all duration-500 rounded-full shadow-[0_0_12px_rgba(59,130,246,0.6)]"
              style={{ width: `${winRate * 100}%` }}
            />
          </div>
          <div className="text-xs text-white/40 mt-2">
            {winRate < 0.4 && 'Low performance'}
            {winRate >= 0.4 && winRate <= 0.7 && 'Optimal range'}
            {winRate > 0.7 && 'High dominance'}
          </div>
        </div>
      </div>
    </div>
  );
}
