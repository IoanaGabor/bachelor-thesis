import { Box, List, Typography } from '@mui/material';
import type { Reconstruction } from './types';
import ReconstructionListItem from './ReconstructionListItem';

interface Props {
  reconstructions: Reconstruction[];
  onSelect?: (reconstruction: Reconstruction) => void;
}

const ReconstructionList: React.FC<Props> = ({ reconstructions, onSelect }) => {
  if (reconstructions.length === 0) {
    return (
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        sx={{
          bgcolor: 'grey.50',
          borderRadius: 2,
          boxShadow: 1,
          p: 4,
          minHeight: 250,
          minWidth: 300,
          width: { xs: '100%', sm: 400, md: 500 },
          height: 'auto',
          mx: 'auto',
        }}
      >
        <Box
          display="flex"
          alignItems="center"
          justifyContent="center"
          width="100%"
          height="100%"
          flex={1}
        >
          <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center' }}>
            No reconstructions yet.
          </Typography>
        </Box>
      </Box>
    );
  }

  const sortedReconstructions = [...reconstructions].sort(
    (a, b) => new Date(b.uploaded_at).getTime() - new Date(a.uploaded_at).getTime()
  );

  return (
    <Box sx={{ maxHeight: 500, overflowY: 'auto' }}>
      <List>
        {sortedReconstructions.map((reconstruction, index) => (
          <ReconstructionListItem
            key={reconstruction.id}
            reconstruction={reconstruction}
            index={index}
            onClick={onSelect ? () => onSelect(reconstruction) : undefined}
            clickable={!!onSelect}
          />
        ))}
      </List>
    </Box>
  );
};

export default ReconstructionList; 