import { useAuth } from "react-oidc-context";
import { useEffect, useState } from "react";
import { Container, Typography, Paper, Box, CircularProgress, Alert } from '@mui/material';
import { fetchUserAttributes } from "./components/services/api-service";

const Profile = () => {
  const auth = useAuth();
  const [attributes, setAttributes] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!auth.isAuthenticated) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    fetchUserAttributes()
      .then((data) => {
        setAttributes(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Failed to fetch user attributes");
        setLoading(false);
      });
  }, [auth.isAuthenticated]);

  if (!auth.isAuthenticated) {
    return <Typography>Please sign in to view your profile.</Typography>;
  }

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="30vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Container>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container>
      <Paper sx={{ padding: 3, marginTop: 3 }}>
        <Typography variant="h4" gutterBottom>
          User Profile
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {attributes && Object.entries(attributes).map(([key, value]) => (
            <Box key={key}>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                {key.charAt(0).toUpperCase() + key.slice(1)}:
              </Typography>
              <Typography variant="body1">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </Typography>
            </Box>
          ))}
        </Box>
      </Paper>
    </Container>
  );
};

export default Profile;
