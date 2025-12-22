import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getPrediction = async () => {
  console.log('Fetching prediction...');
  try {
    const response = await api.get('/predict');
    console.log('Prediction response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching prediction:', error);
    throw error;
  }
};

export const getPredictedLadder = async () => {
  console.log('Fetching predicted ladder with full stats...');
  try {
    const response = await api.get('/predict/ladder');
    console.log('Predicted ladder response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching predicted ladder:', error);
    throw error;
  }
};

export const getHistoricalLadder = async (year) => {
  console.log(`Fetching historical ladder for ${year}...`);
  try {
    const response = await api.get(`/historical/${year}`);
    console.log('Historical ladder response:', response.data);
    return response.data;
  } catch (error) {
    console.error(`Error fetching historical ladder for ${year}:`, error);
    throw error;
  }
};

export const getTeams = async () => {
  console.log('Fetching teams...');
  try {
    const response = await api.get('/teams');
    console.log('Teams response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching teams:', error);
    throw error;
  }
};

export const getTeamHistory = async (team) => {
  console.log(`Fetching history for ${team}...`);
  try {
    const response = await api.get(`/teams/${team}/history`);
    console.log('Team history response:', response.data);
    return response.data;
  } catch (error) {
    console.error(`Error fetching history for ${team}:`, error);
    throw error;
  }
};

export const getTeamPrediction = async (team) => {
  console.log(`Fetching prediction for ${team}...`);
  try {
    const response = await api.get(`/predict/team/${team}`);
    console.log('Team prediction response:', response.data);
    return response.data;
  } catch (error) {
    console.error(`Error fetching prediction for ${team}:`, error);
    throw error;
  }
};

export const getModelComparison = async () => {
  console.log('Fetching model comparison...');
  try {
    const response = await api.get('/compare');
    console.log('Model comparison response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching model comparison:', error);
    throw error;
  }
};

export const getFeatures = async () => {
  console.log('Fetching features...');
  try {
    const response = await api.get('/features');
    console.log('Features response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching features:', error);
    throw error;
  }
};
