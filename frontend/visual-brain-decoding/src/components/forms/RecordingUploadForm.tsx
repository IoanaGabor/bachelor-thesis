import { useState } from "react";
import { uploadRecording } from "../services/api-service";
import { Box, TextField, Button, Alert } from "@mui/material";
import type { ChangeEvent } from "react"; 

interface RecordingUploadFormProps {
  onSuccess?: () => void;
}

const RecordingUploadForm = ({ onSuccess }: RecordingUploadFormProps) => {
  const [niftiFile, setNiftiFile] = useState<File | null>(null);
  const [pngFile, setPngFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>, setFile: (file: File | null) => void, allowedTypes: string[]) => {
    const file = e.target.files?.[0] || null;
    setFile(file);
    setError("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!niftiFile || !pngFile) {
      setError("Both files are required.");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("nifti_file", niftiFile);
    formData.append("png_file", pngFile);
    formData.append("description", description);

    try {
      await uploadRecording(formData);
      onSuccess?.();
    } catch (err) {
      console.error(err);
      setError("Upload failed.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 p-4 border rounded-lg shadow-md">
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <TextField
          type="file"
          onChange={(e: ChangeEvent<HTMLInputElement>) => handleFileChange(e, setNiftiFile, [
            "application/gzip",
            "application/x-gzip",
            "application/octet-stream",
            "application/x-npy",
            "text/plain",
          ])}
          label="fMRI voxel data file"
          InputLabelProps={{ shrink: true }}
          fullWidth
        />
        <TextField
          type="file"
          onChange={(e: ChangeEvent<HTMLInputElement>) => handleFileChange(e, setPngFile, ["image/png"])}
          label="PNG File"
          InputLabelProps={{ shrink: true }}
          fullWidth
        />
        <TextField
          multiline
          rows={2} 
          placeholder="Description (optional)"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          fullWidth
        />
        {error && (
          <Alert severity="error" sx={{ mt: 1 }}>
            {error}
          </Alert>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={isLoading}
            sx={{ minWidth: 120 }}
          >
            {isLoading ? "Uploading..." : "Upload"}
          </Button>
        </Box>
      </Box>
    </form>
  );
};

export default RecordingUploadForm;