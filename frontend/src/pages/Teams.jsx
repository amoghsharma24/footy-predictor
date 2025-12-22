import React, { useState, useEffect } from 'react';
import { getTeams } from '../services/api';
import { Users } from 'lucide-react';
import getTeamLogo from '../utils/teamLogos';

const Teams = () => {
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTeams = async () => {
      try {
        const data = await getTeams();
        setTeams(data.teams);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchTeams();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-afl-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h1 className="text-2xl font-semibold text-gray-900">AFL Teams</h1>
        <p className="text-sm text-gray-500 mt-1">{teams.length} teams competing in the 2025 season</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {teams.map((team) => (
          <div 
            key={team.name} 
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 hover:border-afl-primary transition-colors flex items-center space-x-4"
          >
            <img 
              src={getTeamLogo(team.name)} 
              alt={team.name}
              className="h-12 w-12 object-contain"
            />
            <div>
              <h3 className="text-lg font-medium text-gray-900">{team.name}</h3>
              <span className="text-xs text-gray-500">AFL</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Teams;
