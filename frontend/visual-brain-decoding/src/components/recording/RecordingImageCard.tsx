import { useState } from 'react';
import { Card, CardContent, CardMedia, Typography, Button, CircularProgress, Slider, Box } from '@mui/material';
import type { Recording } from './types';

interface Props {
  recording: Recording;
  isReconstructing: boolean;
  onReconstruct: (numberOfSteps: number) => void;
}

const DEFAULT_STEPS = 100;
const MIN_STEPS = 10;
const MAX_STEPS = 200;

const RecordingImageCard: React.FC<Props> = ({ recording, isReconstructing, onReconstruct }) => {
  const [steps, setSteps] = useState<number>(DEFAULT_STEPS);

  const handleSliderChange = (_: Event, value: number | number[]) => {
    setSteps(Array.isArray(value) ? value[0] : value);
  };

  const handleReconstructClick = () => {
    onReconstruct(steps);
  };

  return (
    <Card
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        boxShadow: 0,
        width: { xs: '100%', md: 500 },
        mx: 'auto',
      }}
    >
      <CardMedia
        component="img"
        image={recording.png_file}
        alt="Brain Recording"
        sx={{
          width: { xs: 250, md: 500 },
          height: { xs: 250, md: 370 },
          objectFit: 'cover',
          borderRadius: 2,
          mx: 'auto',
          mt: 2,
        }}
      />
      <CardContent sx={{ textAlign: 'center', width: '100%' }}>
        <Typography variant="subtitle1" fontWeight={500}>
          Original Image
        </Typography>
        <Box sx={{ mt: 3, mb: 2, px: 2 }}>
          <Typography gutterBottom variant="body2" color="text.secondary">
            Number of reconstruction steps: <b>{steps}</b>
          </Typography>
          <Slider
            value={steps}
            min={MIN_STEPS}
            max={MAX_STEPS}
            step={1}
            onChange={handleSliderChange}
            valueLabelDisplay="auto"
            disabled={isReconstructing}
            sx={{ width: '90%', mx: 'auto' }}
          />
        </Box>
        <Button
          variant="contained"
          color="primary"
          onClick={handleReconstructClick}
          disabled={isReconstructing}
          sx={{ mt: 1, width: 1, maxWidth: 300 }}
        >
          {isReconstructing ? (
            <>
              <CircularProgress size={20} sx={{ mr: 1 }} /> Reconstructing...
            </>
          ) : (
            'Request New Reconstruction'
          )}
        </Button>
      </CardContent>
    </Card>
  );
};

export default RecordingImageCard;
