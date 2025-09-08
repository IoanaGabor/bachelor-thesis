export interface Recording {
  id: number;
  description: string;
  png_file: string;
}

export interface Reconstruction {
  id: number;
  brain_recording_id: number;
  reconstruction_png_path: string;
  uploaded_at: string;
  number_of_steps: number;
  status: string;
  metrics_json: string;
} 