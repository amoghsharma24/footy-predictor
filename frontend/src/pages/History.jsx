import React, { useState, useEffect } from 'react';
import { getHistoricalLadder } from '../services/api';
import { Calendar, Search } from 'lucide-react';

const History = () => {
  const [year, setYear] = useState(2025);
  const [ladder, setLadder] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchLadder = async () => {
      setLoading(true);
      try {
        const data = await getHistoricalLadder(year);
        setLadder(data.ladder);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchLadder();
  }, [year]);

  const years = Array.from({ length: 11 }, (_, i) => 2025 - i);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900">Historical Ladders</h1>
            <p className="text-sm text-gray-500 mt-1">View past season results and standings</p>
          </div>
          
          <div className="relative">
            <select
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              className="pl-4 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-afl-primary focus:border-transparent outline-none appearance-none bg-white text-sm font-medium"
            >
              {years.map((y) => (
                <option key={y} value={y}>{y} Season</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        {loading ? (
          <div className="p-12 flex justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-afl-primary"></div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Pos</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Team</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">P</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">W</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">D</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">L</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">PF</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">PA</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">%</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Pts</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {ladder?.map((team) => (
                  <tr key={team.team} className={team.position <= 8 ? 'bg-blue-50/40' : ''}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center justify-center h-8 w-8 rounded-full text-sm font-semibold ${
                        team.position <= 4 ? 'bg-afl-primary text-white' :
                        team.position <= 8 ? 'bg-blue-500 text-white' :
                        'bg-gray-200 text-gray-700'
                      }`}>
                        {team.position}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-3">
                        <span className="text-sm font-medium text-gray-900">{team.team}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {team.wins + team.draws + team.losses}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600 font-medium">
                      {team.wins}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {team.draws}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-medium">
                      {team.losses}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {team.points_for}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {team.points_against}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">
                      {team.percentage.toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">
                      {team.premiership_points}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;
