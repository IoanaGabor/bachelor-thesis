import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from "react-oidc-context";
import { websocketService } from '../components/services/websocket-service';
import { Snackbar, Alert } from '@mui/material';

interface AuthContextType {
  userId: string | null;
  isAuthenticated: boolean;
  toastOpen: boolean;
  toastMessage: string;
  setToastOpen: (open: boolean) => void;
}

const AuthContext = createContext<AuthContextType>({
  userId: null,
  isAuthenticated: false,
  toastOpen: false,
  toastMessage: '',
  setToastOpen: () => {},
});

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const auth = useAuth();
  const [userId, setUserId] = useState<string | null>(null);
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  
  useEffect(() => {
    if (auth.isAuthenticated && auth.user?.profile?.sub) {
      const userSubject = auth.user.profile.sub;
      setUserId(userSubject);
      
      websocketService.connect(userSubject);
      
      const handleReconstructionComplete = (data: any) => {
        console.log("Reconstruction complete:", data);
        setToastMessage("A new reconstruction is ready!");
        setToastOpen(true);
      };
      
      websocketService.addListener('reconstruction_notification', handleReconstructionComplete);
      console.log("WebSocket service connected. listener added");
      
      return () => {
        websocketService.removeListener('reconstruction_notification', handleReconstructionComplete);
        websocketService.disconnect();
      };
    }
  }, [auth.isAuthenticated, auth.user]);
  
  return (
    <AuthContext.Provider value={{ 
      userId, 
      isAuthenticated: auth.isAuthenticated,
      toastOpen,
      toastMessage,
      setToastOpen
    }}>
      {children}
      <Snackbar
        open={toastOpen}
        autoHideDuration={3000}
        onClose={() => setToastOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setToastOpen(false)} severity="info" sx={{ width: '100%' }}>
          {toastMessage}
        </Alert>
      </Snackbar>
    </AuthContext.Provider>
  );
};

export const useAuthContext = () => useContext(AuthContext); 