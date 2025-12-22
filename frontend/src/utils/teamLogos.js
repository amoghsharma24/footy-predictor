const teamLogos = {
  'Adelaide': '/logos/AdelaideCrows_2024.svg',
  'Brisbane': '/logos/Brisbane_Lions.svg',
  'Brisbane Lions': '/logos/Brisbane_Lions.svg',
  'Carlton': '/logos/CarltonFC_2019.svg',
  'Collingwood': '/logos/Collingwood_Football_Club_Logo_(2017–present).svg',
  'Essendon': '/logos/Essendon_FC_logo.svg',
  'Fremantle': '/logos/Fremantle_FC_logo.svg',
  'Geelong': '/logos/Geelong_Cats.svg',
  'Gold Coast': '/logos/GCSuns_2024.svg',
  'Greater Western Sydney': '/logos/GWS_Giants_logo.svg',
  'GWS': '/logos/GWS_Giants_logo.svg',
  'Hawthorn': '/logos/Hawthorn-football-club-brand.svg',
  'Melbourne': '/logos/MelbourneFC_2016.svg',
  'North Melbourne': '/logos/North_Melbourne_FC_logo.svg',
  'Port Adelaide': '/logos/PortAdelaideFootballClub_2019.svg',
  'Richmond': '/logos/Richmond_Tigers_logo.svg',
  'St Kilda': '/logos/StKildaFC_2024.svg',
  'Sydney': '/logos/SydneySwans_2020.svg',
  'West Coast': '/logos/West_Coast_Eagles_logo_2017.svg',
  'West Coast Eagles': '/logos/West_Coast_Eagles_logo_2017.svg',
  'Western Bulldogs': '/logos/Western_Bulldogs_logo.svg',
};

const getTeamLogo = (teamName) => {
  // Try exact match first
  if (teamLogos[teamName]) {
    return teamLogos[teamName];
  }
  
  // Try partial match
  for (const [key, value] of Object.entries(teamLogos)) {
    if (teamName.includes(key) || key.includes(teamName)) {
      return value;
    }
  }
  
  // Fallback to a generic AFL logo
  return 'https://squiggle.com.au/images/afl.svg';
};

export default getTeamLogo;
