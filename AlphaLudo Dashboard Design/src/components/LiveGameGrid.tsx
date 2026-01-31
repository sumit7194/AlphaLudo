import { LiveGame } from '../hooks/useTrainingData';

interface LiveGameGridProps {
  games: LiveGame[];
}

export function LiveGameGrid({ games }: LiveGameGridProps) {
  const getTypeColor = (type: LiveGame['type']) => {
    switch (type) {
      case 'ghost':
        return 'bg-purple-500/10 border-purple-400/30';
      case 'eval':
        return 'bg-blue-500/10 border-blue-400/30';
      case 'self-play':
        return 'bg-green-500/10 border-green-400/30';
    }
  };

  const getTypeDot = (type: LiveGame['type']) => {
    switch (type) {
      case 'ghost':
        return 'bg-purple-500 shadow-[0_0_8px_rgba(168,85,247,0.6)]';
      case 'eval':
        return 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]';
      case 'self-play':
        return 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]';
    }
  };

  const getStatusRing = (status: LiveGame['status']) => {
    return status === 'active' 
      ? 'ring-2 ring-green-400/40 ring-offset-2 ring-offset-black/50' 
      : 'ring-1 ring-white/10';
  };

  return (
    <div className="glass-panel h-full flex flex-col">
      <div className="px-6 pt-5 pb-4 border-b border-white/10">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm tracking-wide uppercase text-white/40">Live Games</h2>
            <div className="text-xs text-white/30 mt-0.5">
              {games.filter(g => g.status === 'active').length} active
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.6)]" />
              <span className="text-white/40">Self-Play</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_6px_rgba(59,130,246,0.6)]" />
              <span className="text-white/40">Eval</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-purple-500 shadow-[0_0_6px_rgba(168,85,247,0.6)]" />
              <span className="text-white/40">Ghost</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-auto">
        <div className="grid grid-cols-8 gap-3 h-full content-start">
          {games.map((game) => (
            <div
              key={game.id}
              className={`aspect-square rounded-2xl transition-all hover:scale-105 ${getTypeColor(game.type)} ${getStatusRing(game.status)} relative group cursor-pointer backdrop-blur-sm border`}
            >
              <div className="absolute inset-0 flex flex-col items-center justify-center p-2">
                <div className={`w-2 h-2 rounded-full ${getTypeDot(game.type)} mb-1`} />
                <span className="text-[10px] font-mono text-white/50">{game.id}</span>
              </div>
              
              {/* Tooltip on hover */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                <div className="glass-panel px-3 py-2 text-xs whitespace-nowrap shadow-xl">
                  <div className="text-white/40 mb-1 text-[10px] tracking-wide uppercase">Game {game.id}</div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <div className={`w-1.5 h-1.5 rounded-full ${getTypeDot(game.type)}`} />
                    <span className="capitalize text-white/80">{game.type}</span>
                  </div>
                  <div className="text-white/60 text-[11px]">
                    {game.player1} vs {game.player2}
                  </div>
                  <div className="text-white/40 text-[10px] mt-1">
                    Moves: {game.moveCount}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
