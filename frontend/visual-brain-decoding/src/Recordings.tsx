import { useEffect, useState, useRef } from 'react';
import { Card, CardContent, CardMedia, Typography, CircularProgress, Container, Paper, Dialog, DialogTitle, DialogContent, Fab, Box, IconButton, DialogActions, Button } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import CloseIcon from '@mui/icons-material/Close';
import type { Recording } from './components/types/recording';
import { useNavigate } from 'react-router-dom';
import useRecordingStore from './store';
import RecordingUploadForm from './components/forms/RecordingUploadForm';
import { useAuthContext } from './contexts/AuthContext';

const Recordings = () => {
  const navigate = useNavigate();
  const { recordings, loading, error, fetchAllRecordings, deleteRecording } = useRecordingStore();
  const [isUploadDialogOpen, setIsUploadDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [recordingToDelete, setRecordingToDelete] = useState<number | null>(null);
  const paperRef = useRef<HTMLDivElement>(null);
  const { userId } = useAuthContext();

  useEffect(() => {
    fetchAllRecordings(); 
  }, [fetchAllRecordings]);

  const handleUploadSuccess = () => {
    setIsUploadDialogOpen(false);
    fetchAllRecordings();
  };
  
  const [fabRight, setFabRight] = useState<number | null>(null);

  useEffect(() => {
    const updateFabPosition = () => {
      if (paperRef.current) {
        const rect = paperRef.current.getBoundingClientRect();
        setFabRight(window.innerWidth - rect.right);
      }
    };
    updateFabPosition();
    window.addEventListener('resize', updateFabPosition);
    return () => window.removeEventListener('resize', updateFabPosition);
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="40vh" data-testid="loading">
        <CircularProgress />
      </Box>
    );
  }
  if (error) return <Typography color="error">Error: {error}</Typography>;

  return (
    <Container sx={{ position: 'relative', minHeight: '100vh', pb: 8 }}>
      <Paper sx={{ padding: 3 }} ref={paperRef}>
        <div style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center', marginBottom: '20px' }}>
          <Typography variant="h4">
            Recordings
          </Typography>
        </div>
        {recordings.length === 0 ? (
          <Typography>No recordings found.</Typography>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {recordings.map((recording: Recording) => (
              <div
                key={recording.id}
                style={{ position: 'relative' }}
                data-testid="recording-card"
              >
                <Paper elevation={3}>
                  <Card sx={{ display: 'flex', flexDirection: 'row', position: 'relative' }}>
                    <IconButton
                      aria-label="remove"
                      size="small"
                      data-testid="delete-recording"
                      onClick={(e) => {
                        e.stopPropagation();
                        setRecordingToDelete(Number(recording.id));
                        setDeleteDialogOpen(true);
                      }}
                      sx={{
                        position: 'absolute',
                        top: 4,
                        right: 4,
                        zIndex: 2,
                        background: 'rgba(255,255,255,0.7)',
                        '&:hover': { background: 'rgba(255,0,0,0.15)' }
                      }}
                    >
                      <CloseIcon fontSize="small" />
                    </IconButton>
                    <div
                      onClick={() => navigate(`/recordings/${recording.id}`)}
                      style={{ cursor: 'pointer', display: 'flex', flex: 1 }}
                    >
                      <CardMedia
                        component="img"
                        sx={{ width: 150, objectFit: 'cover' }}
                        image={`${import.meta.env.VITE_PUBLIC_ENDPOINT}${recording.png_file}`}
                        alt="Brain Recording Thumbnail"
                      />
                      <CardContent sx={{ flex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="h6">{recording.description}</Typography>
                      </CardContent>
                    </div>
                  </Card>
                </Paper>
              </div>
            ))}
          </div>
        )}
      </Paper>
      {fabRight !== null && (
        <Fab
          color="primary"
          aria-label="add"
          data-testid="upload-fab"
          onClick={() => setIsUploadDialogOpen(true)}
          sx={{
            position: 'fixed',
            bottom: 32,
            right: `calc(${fabRight}px + 0px)`,   
            zIndex: 1200,
            boxShadow: 6,
          }}
        >
          <AddIcon />
        </Fab>
      )}

      <Dialog 
        open={isUploadDialogOpen} 
        onClose={() => setIsUploadDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ pb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          Upload New Recording
          <IconButton
            aria-label="close"
            data-testid="close-upload-dialog"
            onClick={() => setIsUploadDialogOpen(false)}
            edge="end"
            sx={{ ml: 2 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent sx={{ pt: 0 }}>
          <Box mt={2}>
            <RecordingUploadForm onSuccess={handleUploadSuccess} />
          </Box>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Recording</DialogTitle>
        <DialogContent>
          <Typography>Are you sure you want to delete this recording?</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={() => {
              if (recordingToDelete) {
                deleteRecording(recordingToDelete);
                setDeleteDialogOpen(false);
                setRecordingToDelete(null);
              }
            }} 
            color="error"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Recordings;
