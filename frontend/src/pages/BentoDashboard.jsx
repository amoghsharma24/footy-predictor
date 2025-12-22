import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Trophy, TrendingUp, Target, Zap, BarChart3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { getPrediction, getFeatures } from '../services/api';
import getTeamLogo from '../utils/teamLogos';

const BentoDashboard = () => {
  const [prediction, setPrediction] = useState(null);
  const [features, setFeatures] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [predData, featData] = await Promise.all([
          getPrediction(),
          getFeatures()
        ]);
        setPrediction(predData);
        setFeatures(featData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-afl-primary"></div>
      </div>
    );
  }

  const topTeam = prediction?.predictions[0];
  const top5 = prediction?.predictions.slice(0, 5) || [];

  return (
    <div className="min-h-screen p-6 max-w-7xl mx-auto">
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-5xl font-bold text-white mb-2">Command Center</h1>
        <p className="text-gray-400">AFL 2025 Season Predictions · Real-time Analytics</p>
      </motion.div>

      {/* Bento Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Model Accuracy Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="bg-slate rounded-2xl p-6 border border-white/10 hover:border-afl-primary/50 transition-all"
        >
          <div className="flex items-center justify-between mb-4">
            <Target className="h-8 w-8 text-afl-primary" />
            <span className="text-xs text-gray-400 uppercase tracking-wide">Accuracy</span>
          </div>
          <h3 className="text-4xl font-bold text-white mb-1">{prediction?.model_mae.toFixed(3)}</h3>
          <p className="text-sm text-gray-400">Mean Absolute Error</p>
          <div className="mt-4 h-1 bg-slate-light rounded-full overflow-hidden">
            <div className="h-full bg-afl-primary w-3/4"></div>
          </div>
        </motion.div>

        {/* Premier Favorite Card */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-slate rounded-2xl p-6 border border-white/10 hover:border-afl-primary/50 transition-all"
        >
          <div className="flex items-center justify-between mb-4">
            <Trophy className="h-8 w-8 text-yellow-500" />
            <span className="text-xs text-gray-400 uppercase tracking-wide">Premier</span>
          </div>
          <div className="flex items-center space-x-3 mb-2">
            <img src={getTeamLogo(topTeam?.team)} alt={topTeam?.team} className="h-12 w-12 object-contain" />
            <div>
              <h3 className="text-2xl font-bold text-white">{topTeam?.team}</h3>
              <p className="text-sm text-gray-400">Predicted Winner</p>
            </div>
          </div>
        </motion.div>

        {/* Match of the Week - Orbital Animation */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="bg-slate rounded-2xl p-6 border border-white/10 hover:border-afl-primary/50 transition-all lg:col-span-2"
        >
          <div className="flex items-center justify-between mb-4">
            <Zap className="h-8 w-8 text-afl-primary" />
            <span className="text-xs text-gray-400 uppercase tracking-wide">Match of the Week</span>
          </div>
          <div className="relative h-32 flex items-center justify-center">
            {/* Orbital Animation */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="absolute w-full h-full"
            >
              <div className="absolute left-0 top-1/2 -translate-y-1/2">
                <img src={getTeamLogo(top5[0]?.team)} alt="" className="h-16 w-16 object-contain" />
              </div>
              <div className="absolute right-0 top-1/2 -translate-y-1/2">
                <img src={getTeamLogo(top5[1]?.team)} alt="" className="h-16 w-16 object-contain" />
              </div>
            </motion.div>
            <div className="relative z-10 text-center">
              <p className="text-xl font-bold text-white">VS</p>
              <p className="text-xs text-gray-400 mt-1">Round 1 · MCG</p>
            </div>
          </div>
        </motion.div>

        {/* Mini Ladder - Top 5 */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-slate rounded-2xl p-6 border border-white/10 hover:border-afl-primary/50 transition-all lg:col-span-2"
        >
          <div className="flex items-center justify-between mb-4">
            <Trophy className="h-8 w-8 text-afl-primary" />
            <span className="text-xs text-gray-400 uppercase tracking-wide">Top 5 Preview</span>
          </div>
          <div className="space-y-3">
            {top5.map((team, index) => (
              <motion.div
                key={team.team}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
                className="flex items-center justify-between p-3 bg-slate-light rounded-lg hover:bg-slate-light/80 transition-all"
              >
                <div className="flex items-center space-x-3">
                  <span className="text-2xl font-bold text-afl-primary w-6">{index + 1}</span>
                  <img src={getTeamLogo(team.team)} alt={team.team} className="h-8 w-8 object-contain" />
                  <span className="font-semibold text-white">{team.team}</span>
                </div>
                <span className="text-sm text-gray-400">{team.predicted_position.toFixed(1)}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Feature Importance Chart */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="bg-slate rounded-2xl p-6 border border-white/10 hover:border-afl-primary/50 transition-all lg:col-span-2"
        >
          <div className="flex items-center justify-between mb-4">
            <BarChart3 className="h-8 w-8 text-afl-primary" />
            <span className="text-xs text-gray-400 uppercase tracking-wide">Model Architecture</span>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={features?.features.slice(0, 6)} layout="vertical">
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="feature" 
                  type="category" 
                  width={100}
                  tick={{ fill: '#9ca3af', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1a1a1a', 
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    color: '#fff'
                  }}
                />
                <Bar dataKey="importance" fill="#ED1B2F" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-gray-400 mt-2 text-center">Feature weights driving 2025 predictions</p>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.6 }}
          className="bg-slate rounded-2xl p-6 border border-white/10 hover:border-afl-primary/50 transition-all"
        >
          <div className="flex items-center justify-between mb-4">
            <TrendingUp className="h-8 w-8 text-green-500" />
            <span className="text-xs text-gray-400 uppercase tracking-wide">Model</span>
          </div>
          <h3 className="text-2xl font-bold text-white mb-1">{prediction?.model_name}</h3>
          <p className="text-sm text-gray-400">Ensemble Learning</p>
          <div className="mt-4 space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Training Data</span>
              <span className="text-white">2010-2024</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Features</span>
              <span className="text-white">{features?.total_features || 0}</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default BentoDashboard;
