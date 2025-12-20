import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Trophy, BarChart2, History, Activity } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-3">
              <Trophy className="h-7 w-7 text-afl-primary" />
              <div>
                <span className="font-semibold text-lg text-gray-900">AFL Ladder Predictor</span>
              </div>
            </Link>
          </div>
          
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-1">
              <Link
                to="/"
                className={`px-4 py-2 text-sm font-medium flex items-center space-x-2 transition-colors ${
                  isActive('/') 
                    ? 'text-afl-primary border-b-2 border-afl-primary' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Activity className="h-4 w-4" />
                <span>Dashboard</span>
              </Link>
              
              <Link
                to="/history"
                className={`px-4 py-2 text-sm font-medium flex items-center space-x-2 transition-colors ${
                  isActive('/history') 
                    ? 'text-afl-primary border-b-2 border-afl-primary' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <History className="h-4 w-4" />
                <span>History</span>
              </Link>
              
              <Link
                to="/teams"
                className={`px-4 py-2 text-sm font-medium flex items-center space-x-2 transition-colors ${
                  isActive('/teams') 
                    ? 'text-afl-primary border-b-2 border-afl-primary' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <BarChart2 className="h-4 w-4" />
                <span>Teams</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
