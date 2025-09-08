import { ListItem, ListItemAvatar, ListItemText, Avatar, Typography, Box } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import type { Reconstruction } from './types';

interface Props {
  reconstruction: Reconstruction;
  index: number;
  onClick?: () => void;
  clickable?: boolean;
}

const ReconstructionListItem: React.FC<Props> = ({ reconstruction, index, onClick, clickable }) => (
  <ListItem
    alignItems="flex-start"
    sx={{
      mb: 2,
      borderRadius: 2,
      boxShadow: 1,
      bgcolor: 'grey.50',
      '&:last-child': { mb: 0 },
      cursor: clickable ? 'pointer' : 'default',
      '&:hover': clickable ? { boxShadow: 3, bgcolor: 'grey.100' } : undefined,
    }}
    onClick={clickable ? onClick : undefined}
    secondaryAction={
      <Box sx={{ minWidth: 90, textAlign: 'right' }}>
        <Typography variant="caption" color="text.secondary">
          Steps: {reconstruction.number_of_steps}
        </Typography>
      </Box>
    }
  >
    <ListItemAvatar>
      <Avatar
        variant="rounded"
        src={`${import.meta.env.VITE_PUBLIC_ENDPOINT}${reconstruction.reconstruction_png_path}`}
        alt={`Reconstruction ${index + 1}`}
        sx={{
          width: 80, 
          height: 80,
          mr: 2,
          bgcolor: 'grey.200',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <ImageIcon sx={{ fontSize: 48 }} /> 
      </Avatar>
    </ListItemAvatar>
    <ListItemText
      secondary={
        <>
          <Typography variant="caption" color="text.secondary" display="block">
            {new Date(reconstruction.uploaded_at).toLocaleString()}
          </Typography>
          {(() => {
            let metrics: Record<string, any> = {};
            try {
              metrics = JSON.parse(reconstruction.metrics_json || '{}');
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
        </>
      }
    />
  </ListItem>
);

export default ReconstructionListItem; 