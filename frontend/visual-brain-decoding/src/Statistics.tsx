import React, { useEffect, useState } from "react";
import { fetchStatistics } from "./components/services/api-service";
import { Box, Typography, CircularProgress } from "@mui/material";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LabelList
} from "recharts";

const Statistics: React.FC = () => {
  const [stats, setStats] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getStats = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchStatistics();
        setStats(data);
      } catch (err: any) {
        setError(err.message || "Failed to fetch statistics");
      } finally {
        setLoading(false);
      }
    };
    getStats();
  }, []);

  const chartData =
    stats
      ? Object.entries(stats).map(([name, value]) => ({
          name,
          value,
        }))
      : [];

  return (
    <Box sx={{ width: "100%", maxWidth: 700, mx: "auto", mt: 4 }}>
      <Typography variant="h5" fontWeight={600} gutterBottom>
        Reconstruction Statistics
      </Typography>
      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
          <CircularProgress />
        </Box>
      )}
      {error && (
        <Typography color="error" sx={{ my: 2 }}>
          {error}
        </Typography>
      )}
      {!loading && !error && stats && (
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-20} textAnchor="end" interval={0} height={60} />
            <YAxis allowDecimals={false} domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="value" fill="#1976d2">
              <LabelList dataKey="value" position="top" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </Box>
  );
};

export default Statistics;
