import { useState, useEffect } from 'react';
import { MetricsHero } from './components/MetricsHero';
import { ControlBar } from './components/ControlBar';
import { LiveGameGrid } from './components/LiveGameGrid';
import { InfoPanel } from './components/InfoPanel';
import { useTrainingData } from './hooks/useTrainingData';

export default function App() {
  const {
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
  } = useTrainingData();

  // Safety check for initial render
  if (!eloHistory.length || !lossHistory.length) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 via-gray-100 to-gray-50">
        <div className="text-gray-500">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="h-screen overflow-hidden bg-gradient-to-br from-black via-zinc-950 to-black flex flex-col">
      {/* Control Bar */}
      <ControlBar
        actorCount={actorCount}
        sessionTime={sessionTime}
        onSpawnActor={spawnActor}
        onKillActor={killActor}
        onStop={stopTraining}
      />

      {/* Main Content - No Scroll */}
      <div className="flex-1 flex gap-4 p-4 overflow-hidden">
        {/* Left Column - Main Metrics */}
        <div className="flex-1 flex flex-col gap-4 overflow-hidden">
          {/* Hero Metrics */}
          <MetricsHero
            mainElo={mainElo}
            eloHistory={eloHistory}
            loss={loss}
            lossHistory={lossHistory}
            winRate={winRate}
          />

          {/* Live Game Grid */}
          <div className="flex-1 overflow-hidden">
            <LiveGameGrid games={liveGames} />
          </div>
        </div>

        {/* Right Sidebar - Info Panel */}
        <div className="w-80 overflow-hidden">
          <InfoPanel
            leaderboard={leaderboard}
            iteration={iteration}
            gamesPlayed={gamesPlayed}
            bufferSize={bufferSize}
            systemStats={systemStats}
            deviceInfo={deviceInfo}
            mctsThreshold={mctsThreshold}
            onUpdateMctsThreshold={updateMctsThreshold}
          />
        </div>
      </div>
    </div>
  );
}