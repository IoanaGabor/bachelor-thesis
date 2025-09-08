import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { fetchRecordingById, fetchReconstructionsForRecording, requestReconstruction } from './components/services/api-service';
import { useAuthContext } from './contexts/AuthContext';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import RecordingImageCard from './components/recording/RecordingImageCard';
import ReconstructionList from './components/recording/ReconstructionList';
import type { Recording, Reconstruction } from './components/recording/types';
import { websocketService } from './components/services/websocket-service';

function RecordingDetail() {
  const { id } = useParams<{ id: string }>();
  const [recording, setRecording] = useState<Recording | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isReconstructing, setIsReconstructing] = useState<boolean>(false);
  const [reconstructions, setReconstructions] = useState<Reconstruction[]>([]);
  const { userId } = useAuthContext();

  const [toastOpen, setToastOpen] = useState(false);
  const [toastMsg, setToastMsg] = useState('');
  const [toastSeverity, setToastSeverity] = useState<'success' | 'error'>('success');

  const [selectedReconstruction, setSelectedReconstruction] = useState<Reconstruction | null>(null);

  useEffect(() => {
    if (!id) return;

    const fetchData = async () => {
      try {
        const [recordingData, reconstructionsData] = await Promise.all([
          fetchRecordingById(String(id)),
          fetchReconstructionsForRecording(String(id))
        ]);
        setRecording(recordingData);
        setReconstructions(reconstructionsData);
      } catch (err) {
        setError('Recording not found');
      } finally {
        setLoading(false);
      }
    };
    const handleReconstructionComplete = (data: any) => {
      console.log("Reconstruction complete refresh data:", data);
      fetchData();
    };
    
    websocketService.addListener('reconstruction_notification', handleReconstructionComplete);
    fetchData();
  }, [id]);

  const handleReconstruction = async (numberOfSteps: number) => {
    if (!id) return;
    setIsReconstructing(true);
    setError(null);

    try {
      await requestReconstruction(id, numberOfSteps);
      setToastMsg('Reconstruction request sent successfully!');
      setToastSeverity('success');
      setToastOpen(true);
    } catch (err) {
      setError('Reconstruction failed');
      setToastMsg('Failed to send reconstruction request.');
      setToastSeverity('error');
      setToastOpen(true);
    } finally {
      setIsReconstructing(false);
    }
  };

  const handleToastClose = (
    event?: React.SyntheticEvent | Event,
    reason?: string
  ) => {
    if (reason === 'clickaway') {
      return;
    }
    setToastOpen(false);
  };

  const handleSelectReconstruction = (reconstruction: Reconstruction) => {
    setSelectedReconstruction(reconstruction);
  };

  const handleDialogClose = () => {
    setSelectedReconstruction(null);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh">
        <Typography>Loading...</Typography>
      </Box>
    );
  }
  if (error) {
    return (
      <Box p={4} textAlign="center">
        <Typography color="error" variant="h6">
          Error: {error}
        </Typography>
      </Box>
    );
  }
  if (!recording) return null;

  return (
    <Box maxWidth="lg" mx="auto" p={3}>
      <Typography variant="h4" fontWeight={700} mb={4}>
        {recording.description}
      </Typography>
      <Paper elevation={3} sx={{ p: { xs: 2, md: 4 } }}>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
          <Box sx={{ flex: 1 }}>
            <RecordingImageCard
              recording={{ ...recording, png_file: `${import.meta.env.VITE_PUBLIC_ENDPOINT}${recording.png_file}` }}
              isReconstructing={isReconstructing}
              onReconstruct={handleReconstruction}
            />
          </Box>
          <Box sx={{ flex: 1 }}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="h6" fontWeight={600}>
                Reconstructions
              </Typography>
              <Divider sx={{ mt: 1, mb: 2 }} />
            </Box>
            <ReconstructionList reconstructions={reconstructions} onSelect={handleSelectReconstruction} />
          </Box>
        </Box>
      </Paper>
      <Snackbar
        open={toastOpen}
        autoHideDuration={4000}
        onClose={handleToastClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleToastClose} severity={toastSeverity} sx={{ width: '100%' }}>
          {toastMsg}
        </Alert>
      </Snackbar>
      <Dialog
        open={!!selectedReconstruction}
        onClose={handleDialogClose}
        maxWidth="md"
        fullWidth
        aria-labelledby="reconstruction-dialog-title"
      >
        <DialogTitle sx={{ pr: 5 }}>
          Reconstruction Details
          <IconButton
            aria-label="close"
            onClick={handleDialogClose}
            sx={{
              position: 'absolute',
              right: 8,
              top: 8,
              color: (theme) => theme.palette.grey[500],
            }}
            size="large"
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          {selectedReconstruction && (
            <Box display="flex" flexDirection={{ xs: 'column', md: 'row' }} gap={4}>
              <Box flex={1} textAlign="center">
                <Typography variant="subtitle1" fontWeight={500} mb={1}>
                  Original Image
                </Typography>
                <Box
                  component="img"
                  src={`${import.meta.env.VITE_PUBLIC_ENDPOINT}${recording.png_file}`}
                  alt="Original"
                  sx={{
                    width: 275,
                    height: 275,
                    borderRadius: 2,
                    objectFit: 'cover',
                    boxShadow: 1,
                    maxWidth: '100%',
                  }}
                />
              </Box>
              <Box flex={1} textAlign="center">
                <Typography variant="subtitle1" fontWeight={500} mb={1}>
                  Reconstruction
                </Typography>
                <Box
                  component="img"
                  src={`${import.meta.env.VITE_PUBLIC_ENDPOINT}${selectedReconstruction.reconstruction_png_path}`}
                  alt="Reconstruction"
                  sx={{
                    width: 275,
                    height: 275,
                    borderRadius: 2,
                    objectFit: 'cover',
                    boxShadow: 1,
                    maxWidth: '100%',
                  }}
                />
              </Box>
            </Box>
          )}
          {selectedReconstruction && (
            <Box mt={3}>
              <Typography variant="body2" color="text.secondary">
                <strong>Number of Steps:</strong> {selectedReconstruction.number_of_steps}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                <strong>Metrics:</strong>
              </Typography>
              {(() => {
                let metrics: Record<string, any> = {};
                try {
                  metrics = JSON.parse(selectedReconstruction.metrics_json || '{}');
                } catch (e) {
                  metrics = {};
                }
                return Object.keys(metrics).length > 0 ? (
                  <Box mt={0.5}>
                    {Object.entries(metrics).map(([key, value]) => (
                      <Typography
                        key={key}
                        variant="caption"
                        color="text.secondary"
                        display="block"
                        sx={{ ml: 0.5 }}
                      >
                        {key}: {typeof value === 'number' ? value.toFixed(4) : String(value)}
                      </Typography>
                    ))}
                  </Box>
                ) : null;
              })()}
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                <strong>Uploaded at:</strong> {selectedReconstruction.uploaded_at ? new Date(selectedReconstruction.uploaded_at).toLocaleString() : ''}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose} color="primary" variant="contained">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default RecordingDetail;
