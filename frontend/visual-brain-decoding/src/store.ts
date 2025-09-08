import { create } from 'zustand';
import { fetchRecordings, fetchRecordingById, fetchReconstructionsForRecording } from './components/services/api-service';
import type { Recording } from './components/types/recording';

interface Reconstruction {
  id: number;
  brain_recording_id: number;
  reconstruction_png: string;
  created_at: string;
}

interface RecordingStore {
  recordings: Recording[];
  currentRecording: Recording | null;
  reconstructions: Reconstruction[];
  loading: boolean;
  error: string | null;
  fetchAllRecordings: () => Promise<void>;
  fetchRecording: (id: string) => Promise<void>;
  fetchReconstructions: (id: string) => Promise<void>;
  deleteRecording: (id: number) => Promise<void>;
}

const useRecordingStore = create<RecordingStore>((set) => ({
  recordings: [],
  currentRecording: null,
  reconstructions: [],
  loading: false,
  error: null,

  fetchAllRecordings: async () => {
    set({ loading: true, error: null });
    try {
      const data = await fetchRecordings();
      set({ recordings: data, loading: false });
    } catch (err) {
      set({ error: 'Failed to fetch recordings', loading: false });
    }
  },

  fetchRecording: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const data = await fetchRecordingById(id);
      set({ currentRecording: data, loading: false });
    } catch (err) {
      set({ error: 'Failed to fetch recording', loading: false });
    }
  },

  fetchReconstructions: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const data = await fetchReconstructionsForRecording(id);
      set({ reconstructions: data, loading: false });
    } catch (err) {
      set({ error: 'Failed to fetch reconstructions', loading: false });
    }
  },

  deleteRecording: async (id: number) => {
    set({ loading: true, error: null });
    try {
      await import('./components/services/api-service').then(mod => mod.deleteRecording(id));
      set((state) => ({
        recordings: state.recordings.filter(recording => Number(recording.id) !== id),
        loading: false
      }));
    } catch (err) {
      set({ error: 'Failed to delete recording', loading: false });
    }
  },
}));

export default useRecordingStore;
