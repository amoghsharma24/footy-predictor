import React from 'react';
import { getPredictedLadder, getHistoricalLadder } from '../services/api';
import getTeamLogo from '../utils/teamLogos';

class InteractiveLadder extends React.Component {
  state = { 
    ladderData: [], 
    loading: true, 
    sortBy: 'position',
    selectedYear: 2026,
    availableYears: [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015]
  };

  componentDidMount() {
    this.fetchLadderData(2026);
  }

  fetchLadderData = (year) => {
    console.log(`Ladder: Fetching data for ${year}...`);
    this.setState({ loading: true, selectedYear: year });
    
    if (year === 2026) {
      // Fetch predicted ladder with normalized W/L/D
      getPredictedLadder()
        .then(data => {
          console.log('Ladder: Predicted 2026 ladder', data);
          const formatted = data.ladder.map(team => ({
            team: team.team,
            predicted_position: team.position,
            current_position: team.position,
            change: '→',
            wins: team.wins,
            draws: team.draws,
            losses: team.losses,
            percentage: team.percentage,
            points: team.premiership_points
          }));
          this.setState({ ladderData: formatted, loading: false });
        })
        .catch(err => {
          console.error('Ladder: Error fetching 2026 prediction', err);
          this.setState({ loading: false });
        });
    } else {
      // Fetch historical data
      getHistoricalLadder(year)
        .then(data => {
          console.log(`Ladder: Historical data for ${year}`, data);
          const formatted = data.ladder.map(team => ({
            team: team.team,
            predicted_position: team.position,
            current_position: team.position,
            change: '→',
            wins: team.wins,
            draws: team.draws,
            losses: team.losses,
            percentage: team.percentage,
            points: team.premiership_points
          }));
          this.setState({ ladderData: formatted, loading: false });
        })
        .catch(err => {
          console.error(`Ladder: Error fetching ${year} data`, err);
          this.setState({ loading: false });
        });
    }
  }

  render() {
    const { ladderData, loading, sortBy, selectedYear, availableYears } = this.state;
    
    if (loading) {
      return (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-afl-primary"></div>
        </div>
      );
    }

    if (!ladderData || ladderData.length === 0) {
      return (
        <div className="text-center p-10 glass-panel rounded-xl">
          <p className="text-gray-400">No ladder data available.</p>
        </div>
      );
    }

    const sorted = [...ladderData].sort((a, b) => {
      if (sortBy === 'position') return (a.predicted_position || 0) - (b.predicted_position || 0);
      if (sortBy === 'points') {
        const ptsA = (a.points || ((a.wins || 0) * 4 + (a.draws || 0) * 2));
        const ptsB = (b.points || ((b.wins || 0) * 4 + (b.draws || 0) * 2));
        if (ptsB !== ptsA) return ptsB - ptsA; // Sort by points DESC
        return (b.percentage || 0) - (a.percentage || 0); // Then by percentage DESC
      }
      if (sortBy === 'percentage') return (b.percentage || 0) - (a.percentage || 0);
      if (sortBy === 'wins') return (b.wins || 0) - (a.wins || 0);
      return 0;
    });

    return (
      <div>
        <div className="mb-6 flex items-center justify-between">
          <div className="flex gap-2">
            <select
              value={selectedYear}
              onChange={(e) => {
                e.preventDefault();
                this.fetchLadderData(parseInt(e.target.value));
              }}
              className="px-4 py-2 rounded-lg text-sm font-medium glass-panel backdrop-blur-md border border-white/20 text-white hover:bg-white/10 transition-colors cursor-pointer"
            >
              {availableYears.map(year => (
                <option key={year} value={year} className="bg-neutral-900">
                  {year === 2026 ? '2026 (Predicted)' : year}
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
          {['position', 'points', 'percentage', 'wins'].map(key => (
            <button
              key={key}
              onClick={() => this.setState({ sortBy: key })}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                sortBy === key 
                  ? 'bg-afl-primary text-white shadow-lg shadow-red-900/20' 
                  : 'glass-card text-gray-400 hover:text-white hover:bg-white/10'
              }`}
            >
              {key.charAt(0).toUpperCase() + key.slice(1)}
            </button>
          ))}
          </div>
        </div>

        <div className="glass-panel rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10 bg-white/5">
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Pos</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-400 uppercase tracking-wider">Team</th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-gray-400 uppercase tracking-wider">W</th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-gray-400 uppercase tracking-wider">L</th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-gray-400 uppercase tracking-wider">D</th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-gray-400 uppercase tracking-wider">%</th>
                  <th className="px-6 py-4 text-center text-xs font-semibold text-gray-400 uppercase tracking-wider">Pts</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {sorted.map((team, idx) => {
                  const wins = team.wins || 0;
                  const draws = team.draws || 0;
                  const losses = team.losses || 0;
                  const percentage = team.percentage || 0;
                  const pts = team.points || ((wins * 4) + (draws * 2));
                  
                  // Percentage heatmap: Green > 100%, Red < 100%
                  const getPercentageColor = (pct) => {
                    if (pct >= 120) return 'text-green-400 font-bold';
                    if (pct >= 100) return 'text-green-300';
                    if (pct >= 85) return 'text-yellow-300';
                    if (pct >= 70) return 'text-orange-400';
                    return 'text-red-400';
                  };
                  
                  const totalGames = wins + losses + draws;
                  
                  return (
                    <tr 
                      key={team.team || idx} 
                      className={`hover:bg-white/5 transition-colors ${
                        idx < 8 ? 'bg-green-500/10 border-l-4 border-green-500' : ''
                      }`}
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className={`flex items-center justify-center w-8 h-8 rounded-full font-bold text-sm ${
                          idx < 8 ? 'bg-green-500/20 text-green-400' : 'text-gray-500'
                        }`}>
                          {idx + 1}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <img src={getTeamLogo(team.team)} alt="" className="h-10 w-10 object-contain mr-4" />
                          <span className="font-bold text-white text-base">{team.team}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-gray-300 font-medium">{wins}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-gray-400">{losses}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-center text-gray-400">{draws}</td>
                      <td className={`px-6 py-4 whitespace-nowrap text-center font-medium ${getPercentageColor(percentage)}`}>
                        {percentage.toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className="inline-flex items-center justify-center px-3 py-1 rounded-md bg-white/10 text-white font-bold border border-white/10 min-w-[3rem]">
                          {pts}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }
}

export default InteractiveLadder;
