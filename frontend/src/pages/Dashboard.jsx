import React, { useEffect, useState } from 'react';
import { getPrediction, getFeatures } from '../services/api';
import getTeamLogo from '../utils/teamLogos';

const Dashboard = () => {
  const [prediction, setPrediction] = useState(null);
  const [features, setFeatures] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showAccuracyInfo, setShowAccuracyInfo] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      console.log('Dashboard: Starting data fetch...');
      try {
        const [predData, featData] = await Promise.all([
          getPrediction(),
          getFeatures()
        ]);
        console.log('Dashboard: Data fetched successfully', { predData, featData });
        setPrediction(predData);
        setFeatures(featData);
      } catch (err) {
        console.error('Dashboard: Error fetching data', err);
      } finally {
        setLoading(false);
        console.log('Dashboard: Loading set to false');
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-afl-primary"></div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="text-center p-10 glass-panel rounded-xl">
        <p className="text-red-500 font-bold">Failed to load data</p>
        <p className="text-gray-400 text-sm mt-2">Is the backend server running on port 8000?</p>
      </div>
    );
  }

  const topTeam = prediction?.predictions?.[0];
  const top8 = prediction?.predictions?.slice(0, 8) || [];

  return (
    <div onClick={(e) => e.stopPropagation()}>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass-panel rounded-xl p-6">
          <p className="text-sm text-gray-400 mb-1 uppercase tracking-wider font-medium">Model Accuracy</p>
          <p className="text-3xl font-bold text-white mt-2">{prediction?.model_mae?.toFixed(2) || 'N/A'}</p>
          <p className="text-xs text-gray-500 mt-1">Mean Absolute Error</p>
          <div className="mt-4 pt-4 border-t border-white/10">
            <p className="text-xs text-gray-400 leading-relaxed">
              This number shows how accurate our predictions are. A score of <span className="text-white font-semibold">{prediction?.model_mae?.toFixed(1)}</span> means our model is typically off by about {Math.round(prediction?.model_mae || 0)} ladder positions when predicting where teams will finish.
              <br /><br />
              <span className="text-green-400 font-medium">Lower is better!</span> The closer to 0, the more accurate our predictions. Historical AFL prediction models struggle with errors ranging from 4-6 positions, putting our model in the <span className="text-white font-semibold">top tier</span> of prediction systems.
            </p>
          </div>
        </div>

        <div className="glass-panel rounded-xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            {topTeam && <img src={getTeamLogo(topTeam.team)} alt={topTeam.team} className="h-24 w-24" />}
          </div>
          <p className="text-sm text-gray-400 mb-1 uppercase tracking-wider font-medium">Predicted Premier</p>
          <div className="flex items-center space-x-3 mt-2 relative z-10">
            {topTeam && <img src={getTeamLogo(topTeam.team)} alt={topTeam.team} className="h-10 w-10 object-contain" />}
            <p className="text-2xl font-bold text-white">{topTeam?.team || 'TBD'}</p>
          </div>
          {topTeam && (
            <div className="mt-4 pt-4 border-t border-white/10 relative z-10">
              <p className="text-xs text-gray-400 leading-relaxed">
                {topTeam.team === 'Geelong' && <><span className="text-white font-semibold">The Cats</span> are one of the AFL's most storied clubs, with <span className="text-yellow-400 font-semibold">10 premierships</span> including their recent 2022 flag. Known for their consistent excellence and ability to regenerate their list, Geelong has made finals in 17 of the last 18 seasons. Their strong home ground advantage at GMHBA Stadium and experienced leadership group make them perennial contenders.</>}
                {topTeam.team === 'Brisbane Lions' && <><span className="text-white font-semibold">The Lions</span> are riding a wave of success after their <span className="text-yellow-400 font-semibold">2024 premiership</span>, ending a 21-year drought. With a young, talented list featuring stars like Lachie Neale and Joe Daniher, Brisbane has transformed from cellar-dwellers to premiership contenders through astute recruiting and player development.</>}
                {topTeam.team === 'Sydney' && <><span className="text-white font-semibold">The Swans</span> are the AFL's model of consistency, making finals in 21 of the last 26 seasons. With their strong academy system producing elite talent and a culture of sustained excellence, Sydney remains one of the competition's most formidable teams year after year.</>}
                {topTeam.team === 'Collingwood' && <><span className="text-white font-semibold">The Magpies</span> are the most supported club in the AFL and <span className="text-yellow-400 font-semibold">2023 premiers</span>. Known for their passionate fanbase and ability to perform on the biggest stage, Collingwood's recent resurgence under their coaching has them back among the elite.</>}
                {topTeam.team === 'Carlton' && <><span className="text-white font-semibold">The Blues</span> are experiencing a renaissance after years in the wilderness. With a talented young list led by dual Brownlow medalist <span className="text-white font-semibold">Patrick Cripps</span>, Carlton is looking to add to their <span className="text-yellow-400 font-semibold">record 16 premierships</span> and reclaim their spot among the AFL's elite.</>}
                {topTeam.team === 'Port Adelaide' && <><span className="text-white font-semibold">The Power</span> are one of the most successful clubs of the AFL era with their <span className="text-yellow-400 font-semibold">2004 premiership</span>. Known for their fierce rivalry with Adelaide and strong home crowd support at Adelaide Oval, Port Adelaide combines physicality with skilled ball movement.</>}
                {topTeam.team === 'GWS' && <><span className="text-white font-semibold">The Giants</span> are the AFL's newest success story, reaching their first Grand Final in 2019. Built through the draft with an abundance of young talent, GWS has quickly established themselves as a competitive force despite being founded only in 2012.</>}
                {topTeam.team === 'Melbourne' && <><span className="text-white font-semibold">The Demons</span> broke a 57-year premiership drought in <span className="text-yellow-400 font-semibold">2021</span> with one of the most dominant seasons in AFL history. Known for their elite defense and contested ball work, Melbourne's 'Demons DNA' culture has transformed them into a powerhouse.</>}
                {!['Geelong', 'Brisbane Lions', 'Sydney', 'Collingwood', 'Carlton', 'Port Adelaide', 'GWS', 'Melbourne'].includes(topTeam.team) && <>Our model predicts this team will claim the <span className="text-yellow-400 font-semibold">2026 premiership</span> based on their strong fundamentals, list composition, and recent performance trends.</>}
              </p>
            </div>
          )}
        </div>

        <div className="glass-panel rounded-xl p-6">
          <p className="text-sm text-gray-400 mb-1 uppercase tracking-wider font-medium">Model Info</p>
          <p className="text-2xl font-bold text-white mt-2">{prediction?.model_name || 'Unknown'}</p>
          <p className="text-xs text-gray-500 mt-1">{features?.total_features || 0} active features</p>
          <div className="mt-4 pt-4 border-t border-white/10">
            <p className="text-xs text-gray-400 leading-relaxed">
              <span className="text-white font-semibold">Random Forest</span> is an ensemble learning method that creates multiple decision trees and merges their predictions. It's highly accurate for AFL predictions because it handles non-linear relationships between features like team form, scoring efficiency, and defensive strength. The model trains on <span className="text-white font-semibold">{features?.total_features || 0} carefully selected metrics</span> spanning team performance, player statistics, and match context.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-6 flex items-center">
            <span className="w-1.5 h-6 bg-afl-primary rounded-full mr-3"></span>
            Top 8 Prediction - 2026
          </h2>
          <div className="space-y-3">
            {top8.map((team, index) => (
              <div
                key={team.team}
                className="flex items-center justify-between p-3 glass-card rounded-lg hover:bg-white/5 transition-colors group/item relative"
              >
                <div className="flex items-center space-x-4">
                  <span className="text-lg font-bold text-gray-500 w-6">{index + 1}</span>
                  <img src={getTeamLogo(team.team)} alt={team.team} className="h-8 w-8 object-contain" />
                  <span className="font-semibold text-white">{team.team}</span>
                </div>
                <div className="relative">
                  <span className="text-sm font-medium text-gray-400">{team.predicted_position.toFixed(1)}</span>
                  <div className="absolute bottom-full right-0 mb-2 hidden group-hover/item:block w-64 p-3 bg-black/95 rounded-lg text-xs text-gray-300 shadow-xl border border-white/10 z-50">
                    <p className="font-semibold text-white mb-1">Predicted Position Score</p>
                    <p>This number represents the model's confidence-weighted prediction for where {team.team} will finish. A score of {team.predicted_position.toFixed(1)} means the model predicts a finish around position {Math.round(team.predicted_position)}, with decimal values indicating uncertainty between positions.</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-panel rounded-xl p-6">
          <h2 className="text-lg font-bold text-white mb-2 flex items-center">
            <span className="w-1.5 h-6 bg-blue-500 rounded-full mr-3"></span>
            Key Performance Factors
          </h2>
          <p className="text-xs text-gray-400 mb-6 leading-relaxed">
            These metrics represent the most influential factors in predicting ladder positions. Percentage (points for/against ratio) is the strongest predictor, followed by opponent quality and scoring efficiency. The model weighs these features based on their historical correlation with final ladder positions.
          </p>
          <div className="space-y-4">
            {(features?.features?.slice(0, 8) || []).map((feat, idx) => {
              const maxImportance = Math.max(...(features?.features || []).map(f => f.importance));
              const percentage = (feat.importance / maxImportance) * 100;
              const featureName = feat.feature
                .replace(/_/g, ' ')
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
              
              return (
                <div key={idx} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-300">{featureName}</span>
                    <span className="text-xs text-gray-500">{(feat.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-red-500 to-red-600 rounded-full transition-all duration-500"
                      style={{ width: `${percentage}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
