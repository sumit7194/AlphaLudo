import { useState, useEffect, useCallback } from 'react';

export interface EloDataPoint {
  time: number;
  value: number;
}

export interface LossDataPoint {
  time: number;
  value: number;
}

export interface LeaderboardEntry {
  name: string;
  elo: number;
  type: 'current' | 'previous' | 'bot';
}

export interface LiveGame {
  id: number;
  type: 'ghost' | 'eval' | 'self-play';
  status: 'active' | 'finished';
  player1: string;
  player2: string;
  moveCount: number;
}

export interface SystemStat {
  processId: number;
  cpu: number;
  ram: number;
  status: string;
}

export function useTrainingData() {
  const [mainElo, setMainElo] = useState(1523);
  const [eloHistory, setEloHistory] = useState<EloDataPoint[]>([]);
  const [loss, setLoss] = useState(0.0342);
  const [lossHistory, setLossHistory] = useState<LossDataPoint[]>([]);
  const [winRate, setWinRate] = useState(0.573);
  const [actorCount, setActorCount] = useState(16);
  const [sessionTime, setSessionTime] = useState(0);
  const [iteration, setIteration] = useState(12450);
  const [gamesPlayed, setGamesPlayed] = useState(24891);
  const [bufferSize, setBufferSize] = useState(1048576);
  const [mctsThreshold, setMctsThreshold] = useState(0.75);
  const [deviceInfo] = useState('MPS (Apple Silicon)');

  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([
    { name: 'Main Model', elo: 1523, type: 'current' },
    { name: 'Gen 47', elo: 1498, type: 'previous' },
    { name: 'Gen 46', elo: 1476, type: 'previous' },
    { name: 'Baseline Bot', elo: 1420, type: 'bot' },
    { name: 'Gen 45', elo: 1401, type: 'previous' },
    { name: 'Random Bot', elo: 1156, type: 'bot' },
  ]);

  const [liveGames, setLiveGames] = useState<LiveGame[]>(() =>
    Array.from({ length: 24 }, (_, i) => ({
      id: i,
      type: ['ghost', 'eval', 'self-play'][Math.floor(Math.random() * 3)] as LiveGame['type'],
      status: 'active' as const,
      player1: i % 3 === 0 ? 'Main' : `Gen ${47 - (i % 5)}`,
      player2: i % 2 === 0 ? 'Baseline' : `Gen ${46 - (i % 4)}`,
      moveCount: Math.floor(Math.random() * 80) + 20,
    }))
  );

  const [systemStats, setSystemStats] = useState<SystemStat[]>(() =>
    Array.from({ length: 16 }, (_, i) => ({
      processId: 1000 + i,
      cpu: 45 + Math.random() * 30,
      ram: 512 + Math.random() * 256,
      status: 'Running',
    }))
  );

  // Initialize history
  useEffect(() => {
    const now = Date.now();
    const initialEloHistory: EloDataPoint[] = [];
    const initialLossHistory: LossDataPoint[] = [];
    
    for (let i = 60; i >= 0; i--) {
      initialEloHistory.push({
        time: now - i * 1000,
        value: 1480 + Math.random() * 40 + (60 - i) * 0.7,
      });
      initialLossHistory.push({
        time: now - i * 1000,
        value: 0.05 - (60 - i) * 0.0003 + Math.random() * 0.01,
      });
    }
    
    setEloHistory(initialEloHistory);
    setLossHistory(initialLossHistory);
  }, []);

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();

      // Update Elo
      setMainElo((prev) => {
        const newElo = prev + (Math.random() - 0.48) * 2;
        setEloHistory((hist) => [...hist.slice(-60), { time: now, value: newElo }]);
        return newElo;
      });

      // Update Loss
      setLoss((prev) => {
        const newLoss = Math.max(0.01, prev - 0.00001 + (Math.random() - 0.52) * 0.002);
        setLossHistory((hist) => [...hist.slice(-60), { time: now, value: newLoss }]);
        return newLoss;
      });

      // Update Win Rate
      setWinRate((prev) => Math.max(0.3, Math.min(0.85, prev + (Math.random() - 0.5) * 0.02)));

      // Update iteration and games
      setIteration((prev) => prev + Math.floor(Math.random() * 3));
      setGamesPlayed((prev) => prev + Math.floor(Math.random() * 5));

      // Update session time
      setSessionTime((prev) => prev + 1);

      // Randomly update game statuses
      setLiveGames((games) =>
        games.map((game) => {
          if (Math.random() < 0.05) {
            return {
              ...game,
              status: game.status === 'active' ? 'finished' : 'active',
              moveCount: game.status === 'active' ? game.moveCount : Math.floor(Math.random() * 80) + 20,
            };
          }
          return {
            ...game,
            moveCount: game.status === 'active' ? game.moveCount + 1 : game.moveCount,
          };
        })
      );

      // Update system stats
      setSystemStats((stats) =>
        stats.map((stat) => ({
          ...stat,
          cpu: Math.max(20, Math.min(95, stat.cpu + (Math.random() - 0.5) * 10)),
          ram: Math.max(300, Math.min(900, stat.ram + (Math.random() - 0.5) * 50)),
        }))
      );
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const spawnActor = useCallback(() => {
    if (actorCount < 32) {
      setActorCount((prev) => prev + 1);
      setSystemStats((stats) => [
        ...stats,
        {
          processId: 1000 + stats.length,
          cpu: 45 + Math.random() * 30,
          ram: 512 + Math.random() * 256,
          status: 'Running',
        },
      ]);
      setLiveGames((games) => [
        ...games,
        {
          id: games.length,
          type: ['ghost', 'eval', 'self-play'][Math.floor(Math.random() * 3)] as LiveGame['type'],
          status: 'active',
          player1: 'Main',
          player2: `Gen ${46 - (games.length % 5)}`,
          moveCount: Math.floor(Math.random() * 80) + 20,
        },
      ]);
    }
  }, [actorCount]);

  const killActor = useCallback(() => {
    if (actorCount > 1) {
      setActorCount((prev) => prev - 1);
      setSystemStats((stats) => stats.slice(0, -1));
      setLiveGames((games) => games.slice(0, -1));
    }
  }, [actorCount]);

  const stopTraining = useCallback(() => {
    console.log('Training stopped gracefully');
    alert('Training session stopped. Data has been saved.');
  }, []);

  const updateMctsThreshold = useCallback((value: number) => {
    setMctsThreshold(value);
  }, []);

  return {
    mainElo,
    eloHistory,
    loss,
    lossHistory,
    winRate,
    actorCount,
    sessionTime,
    iteration,
    gamesPlayed,
    leaderboard,
    liveGames,
    systemStats,
    mctsThreshold,
    bufferSize,
    deviceInfo,
    spawnActor,
    killActor,
    stopTraining,
    updateMctsThreshold,
  };
}
