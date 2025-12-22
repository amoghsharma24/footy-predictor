import React from 'react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Ladder from './pages/Ladder';

function App() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-12">
        <section id="dashboard">
          <Dashboard />
        </section>
        <section id="ladder">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">Live Ladder</h2>
          </div>
          <Ladder />
        </section>
      </main>
    </div>
  );
}

export default App;
