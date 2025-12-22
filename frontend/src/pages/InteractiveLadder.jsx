import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, TrendingUp, TrendingDown, Minus, Info } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';
import { getPrediction } from '../services/api';
import getTeamLogo from '../utils/teamLogos';

const InteractiveLadder = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('position');
  const [selectedTeam, setSelectedTeam] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getPrediction();
        setPredictions(data.predictions);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const sortedPredictions = [...predictions].sort((a, b) => {
    if (sortBy === 'position') return a.predicted_position - b.predicted_position;
    if (sortBy === 'points') return (b.wins * 4) - (a.wins * 4);
    if (sortBy === 'percentage') return b.percentage - a.percentage;
    if (sortBy === 'wins') return b.wins - a.wins;
    return 0;
  });

  // Mock team stats for radar chart
  const getTeamStats = (team) => [
    { attribute: 'Attack', value: 70 + Math.random() * 30, fullMark: 100 },
    { attribute: 'Defense', value: 60 + Math.random() * 40, fullMark: 100 },
    { attribute: 'Midfield', value: 65 + Math.random() * 35, fullMark: 100 },
    { attribute: 'Clearances', value: 55 + Math.random() * 45, fullMark: 100 },
    { attribute: 'Tackles', value: 60 + Math.random() * 40, fullMark: 100 },
  ];

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-afl-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 max-w-7xl mx-auto">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-5xl font-bold text-white mb-2">AFL Ladder 2025</h1>
        <p className="text-gray-400">Predicted Season Standings · Live Rankings</p>
      </motion.div>

      {/* Sort Controls */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mb-6 flex flex-wrap gap-3"
      >
        {[
          { key: 'position', label: 'Position' },
          { key: 'points', label: 'Points' },
          { key: 'percentage', label: 'Percentage' },
          { key: 'wins', label: 'Wins' }
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setSortBy(key)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              sortBy === key 
                ? 'bg-afl-primary text-white shadow-lg shadow-afl-primary/50' 
                : 'bg-slate text-gray-400 hover:text-white hover:border-afl-primary/50 border border-white/10'
            }`}
          >
            {label}
          </button>
        ))}
      </motion.div>

      {/* Ladder Table */}
      <motion.div 
        layout
        className="bg-slate rounded-2xl border border-white/10 overflow-hidden"
      >
        {/* Table Header */}
        <div className="grid grid-cols-12 gap-4 p-4 bg-slate-light border-b border-white/10 text-xs uppercase text-gray-400 font-semibold">
          <div className="col-span-1">Pos</div>
          <div className="col-span-4">Team</div>
          <div className="col-span-1 text-center">P</div>
          <div className="col-span-1 text-center">W</div>
          <div className="col-span-1 text-center">L</div>
          <div className="col-span-1 text-center">D</div>
          <div className="col-span-2 text-center">%</div>
          <div className="col-span-1 text-center">Pts</div>
        </div>

        {/* Table Body with Animations */}
        <AnimatePresence>
          {sortedPredictions.map((team, index) => {
            const played = team.wins + team.losses + team.draws;
            const points = (team.wins * 4) + (team.draws * 2);
            const prevPosition = predictions.findIndex(t => t.team === team.team) + 1;
            const positionChange = prevPosition - (index + 1);

            return (
              <motion.div
                key={team.team}
                layout
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className={`grid grid-cols-12 gap-4 p-4 border-b border-white/5 hover:bg-slate-light/50 transition-all cursor-pointer group ${
                  index < 8 ? 'border-l-4 border-l-green-500' : ''
                }`}
                onClick={() => setSelectedTeam(team)}
              >
                {/* Position with change indicator */}
                <div className="col-span-1 flex items-center space-x-2">
                  <span className="text-xl font-bold text-white">{index + 1}</span>
                  {positionChange > 0 && <TrendingUp className="h-4 w-4 text-green-500" />}
                  {positionChange < 0 && <TrendingDown className="h-4 w-4 text-red-500" />}
                  {positionChange === 0 && <Minus className="h-4 w-4 text-gray-500" />}
                </div>

                {/* Team */}
                <div className="col-span-4 flex items-center space-x-3">
                  <img src={getTeamLogo(team.team)} alt={team.team} className="h-10 w-10 object-contain" />
                  <span className="font-semibold text-white group-hover:text-afl-primary transition-colors">
                    {team.team}
                  </span>
                </div>

                {/* Stats */}
                <div className="col-span-1 flex items-center justify-center text-gray-300">{played}</div>
                <div className="col-span-1 flex items-center justify-center text-green-400 font-semibold">{team.wins}</div>
                <div className="col-span-1 flex items-center justify-center text-red-400">{team.losses}</div>
                <div className="col-span-1 flex items-center justify-center text-gray-400">{team.draws}</div>
                <div className="col-span-2 flex items-center justify-center text-white font-semibold">
                  {team.percentage.toFixed(1)}%
                </div>
                <div className="col-span-1 flex items-center justify-center">
                  <span className="bg-afl-primary text-white px-3 py-1 rounded-full font-bold text-sm">
                    {points}
                  </span>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </motion.div>

      {/* Team Analysis Modal */}
      <AnimatePresence>
        {selectedTeam && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/80 backdrop-blur-sm z-40"
              onClick={() => setSelectedTeam(null)}
            />
            
            {/* Modal */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 50 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 50 }}
              transition={{ type: "spring", damping: 25 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-6"
            >
              <div className="bg-slate rounded-2xl border border-white/10 p-8 max-w-3xl w-full max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-4">
                    <img src={getTeamLogo(selectedTeam.team)} alt={selectedTeam.team} className="h-16 w-16 object-contain" />
                    <div>
                      <h2 className="text-3xl font-bold text-white">{selectedTeam.team}</h2>
                      <p className="text-gray-400">2025 Season Analysis</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedTeam(null)}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-4 gap-4 mb-8">
                  <div className="bg-slate-light rounded-lg p-4 text-center">
                    <p className="text-2xl font-bold text-white">{selectedTeam.predicted_position.toFixed(1)}</p>
                    <p className="text-xs text-gray-400 uppercase">Predicted Pos</p>
                  </div>
                  <div className="bg-slate-light rounded-lg p-4 text-center">
                    <p className="text-2xl font-bold text-green-400">{selectedTeam.wins}</p>
                    <p className="text-xs text-gray-400 uppercase">Wins</p>
                  </div>
                  <div className="bg-slate-light rounded-lg p-4 text-center">
                    <p className="text-2xl font-bold text-white">{selectedTeam.percentage.toFixed(1)}%</p>
                    <p className="text-xs text-gray-400 uppercase">Percentage</p>
                  </div>
                  <div className="bg-slate-light rounded-lg p-4 text-center">
                    <p className="text-2xl font-bold text-afl-primary">{(selectedTeam.wins * 4) + (selectedTeam.draws * 2)}</p>
                    <p className="text-xs text-gray-400 uppercase">Points</p>
                  </div>
                </div>

                {/* Radar Chart */}
                <div className="bg-slate-light rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                    <Info className="h-5 w-5 text-afl-primary" />
                    <span>Team Attributes vs League Average</span>
                  </h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={getTeamStats(selectedTeam.team)}>
                        <PolarGrid stroke="rgba(255,255,255,0.1)" />
                        <PolarAngleAxis 
                          dataKey="attribute" 
                          tick={{ fill: '#9ca3af', fontSize: 12 }}
                        />
                        <PolarRadiusAxis 
                          angle={90} 
                          domain={[0, 100]}
                          tick={{ fill: '#6b7280', fontSize: 10 }}
                        />
                        <Radar 
                          name="Team" 
                          dataKey="value" 
                          stroke="#ED1B2F" 
                          fill="#ED1B2F" 
                          fillOpacity={0.6} 
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1a1a1a', 
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: '8px',
                            color: '#fff'
                          }}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default InteractiveLadder;
