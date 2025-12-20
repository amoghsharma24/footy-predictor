import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getPrediction = async () => {
  const response = await api.get('/predict');
  return response.data;
};

export const getHistoricalLadder = async (year) => {
  const response = await api.get(`/historical/${year}`);
  return response.data;
};

export const getTeams = async () => {
  const response = await api.get('/teams');
  return response.data;
};

export const getTeamHistory = async (team) => {
  const response = await api.get(`/teams/${team}/history`);
  return response.data;
};

export const getTeamPrediction = async (team) => {
  const response = await api.get(`/predict/team/${team}`);
  return response.data;
};

export const getModelComparison = async () => {
  const response = await api.get('/compare');
  return response.data;
};

export const getFeatures = async () => {
  const response = await api.get('/features');
  return response.data;
};
