import { useAuth } from "react-oidc-context";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import styled from "styled-components";
import Recordings from "./Recordings";
import Recording from "./Recording";
import Statistics from "./Statistics";
import { AppBar, Toolbar, Typography, Button, Box, Paper } from '@mui/material';
import { AuthProvider } from "./contexts/AuthContext";
import Profile from "./Profile";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 2rem;
  background-color: #f5f5f5;
`;

const WelcomeMessage = styled.h1`
  font-size: 2.5rem;
  color: #333;
  margin-bottom: 2rem;
  text-align: center;
`;

const StyledButton = styled.button`
  padding: 0.8rem 1.5rem;
  font-size: 1rem;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: #0056b3;
  }
`;

const NavigationBar = () => {
  const auth = useAuth();
  
  const signOutRedirect = () => {
    const clientId = import.meta.env.VITE_CLIENT_ID;
    const logoutUri = import.meta.env.VITE_LOGOUT_URI;
    const cognitoDomain = import.meta.env.COGNITO_DOMAIN;
    window.location.href = `${cognitoDomain}/logout?client_id=${clientId}&logout_uri=${encodeURIComponent(logoutUri)}`;
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Visual Brain Decoding Tool
        </Typography>
        {auth.isAuthenticated && (
          <Box>
            <Button color="inherit" component={Link} to="/">Recordings</Button>
            <Button color="inherit" component={Link} to="/statistics">Statistics</Button>
            <Button color="inherit" component={Link} to="/profile">Profile</Button>
            <Button color="inherit" onClick={() => signOutRedirect()}>Sign out</Button>
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
};

const LandingPage = () => {
  const auth = useAuth();

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg,rgb(155, 195, 235) 0%,rgb(145, 172, 202) 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 2,
      }}
    >
      <Paper
        elevation={6}
        sx={{
          maxWidth: 420,
          width: "100%",
          bgcolor: "white",
          borderRadius: 3,
          p: 5,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <WelcomeMessage>
          🧠 Visual Brain Decoding Tool
        </WelcomeMessage>
        <p style={{ fontSize: "1.3rem", textAlign: "center", marginBottom: "2rem", color: "#333" }}>
          Reconstruct and visualize seen images from your fMRI scans.<br />
        </p>
        {!auth.isAuthenticated && (
          <StyledButton onClick={() => auth.signinRedirect()}>
            Sign in
          </StyledButton>
        )}
      </Paper>
    </Box>
  );
};

function HomeRoute() {
  const auth = useAuth();
  if (auth.isAuthenticated) {
    return <Recordings />;
  }
  return <LandingPage />;
}

function App() {
  const auth = useAuth();

  if (auth.isLoading) {
    return (
      <Container>
        <p>Loading...</p>
      </Container>
    );
  }

  if (auth.error) {
    return (
      <Container>
        <p>Error: {auth.error.message}</p>
      </Container>
    );
  }

  return (
    <AuthProvider>
      <Router>
        <NavigationBar />
        <Routes>
          <Route path="/" element={<HomeRoute />} />
          <Route path="/recordings/:id" element={<Recording />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/statistics" element={<Statistics />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
